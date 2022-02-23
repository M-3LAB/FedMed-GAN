from re import X
import torch

from tools.utilize import *
from model.unit.unit import *
from arch_base.base import Base

import itertools
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


__all__ = ['Unit']

class Unit(Base):
    def __init__(self, config, train_loader, valid_loader, assigned_loader, 
                 device, file_path, batch_limit_weight=1.0):
        super(Unit, self).__init__(config=config, train_loader=train_loader, valid_loader=valid_loader, assigned_loader=assigned_loader, 
                 device=device, file_path=file_path, batch_limit_weight=batch_limit_weight)

        self.config = config

        # Dimensionality (channel-wise) of image embedding
        shared_dim = self.config['dim'] * 2 ** self.config['n_downsample'] 

        # model, two modality, 1 and 2, the aim is to generate 2 from 1
        shared_E = ResidualBlock(features=shared_dim).to(device)       
        self.generator_from_a_to_b_enc = Encoder(in_channels=self.config['input_dim'], dim=self.config['dim'], 
                 n_downsample=self.config['n_downsample'], 
                 shared_block=shared_E).to(device)

        self.generator_from_b_to_a_enc = Encoder(in_channels=self.config['input_dim'], dim=self.config['dim'], 
                 n_downsample=self.config['n_downsample'], 
                 shared_block=shared_E).to(device)
        shared_G = ResidualBlock(features=shared_dim).to(device)
        self.generator_from_a_to_b_dec = Generator(out_channels=self.config['input_dim'], dim=self.config['dim'], 
                   n_upsample=self.config['n_downsample'], 
                   shared_block=shared_G).to(device)

        self.generator_from_b_to_a_dec = Generator(out_channels=self.config['input_dim'], dim=self.config['dim'], 
                   n_upsample=self.config['n_downsample'], 
                   shared_block=shared_G).to(device)

        input_shape = (1, self.config['size'], self.config['size'])
        self.discriminator_from_a_to_b = Discriminator(input_shape, auxiliary_rotation=self.config['auxiliary_rotation'], 
                                                auxiliary_translation=self.config['auxiliary_translation'], auxiliary_scaling=self.config['auxiliary_scaling'], 
                                                num_rot_label=4, num_translate_label=5, num_scaling_label=4).to(self.device)
        self.discriminator_from_b_to_a = Discriminator(input_shape, auxiliary_rotation=self.config['auxiliary_rotation'], 
                                                auxiliary_translation=self.config['auxiliary_translation'], auxiliary_scaling=self.config['auxiliary_scaling'], 
                                                num_rot_label=4, num_translate_label=5, num_scaling_label=4).to(self.device)  
        # optimizer
        self.optimizer_generator = torch.optim.Adam(itertools.chain(self.generator_from_a_to_b_enc.parameters(),
                                                            self.generator_from_a_to_b_dec.parameters(),
                                                            self.generator_from_b_to_a_enc.parameters(),
                                                            self.generator_from_b_to_a_dec.parameters()),
                                            lr=self.config['lr'], betas=(self.config['beta1'], self.config['beta2']))
        self.optimizer_discriminator_from_a_to_b = torch.optim.Adam(self.discriminator_from_a_to_b.parameters(),
                                                    lr=self.config['lr'], betas=(self.config['beta1'], self.config['beta2']))
        self.optimizer_discriminator_from_b_to_a = torch.optim.Adam(self.discriminator_from_b_to_a.parameters(),
                                                    lr=self.config['lr'], betas=(self.config['beta1'], self.config['beta2']))

        self.lr_scheduler_generator = torch.optim.lr_scheduler.LambdaLR(self.optimizer_generator, 
                        lr_lambda=LambdaLR(self.config['num_epoch'], 0, self.config['decay_epoch']).step)
        self.lr_scheduler_discriminator_from_a_to_b = torch.optim.lr_scheduler.LambdaLR(self.optimizer_discriminator_from_a_to_b, 
                        lr_lambda=LambdaLR(self.config['num_epoch'], 0, self.config['decay_epoch']).step)
        self.lr_scheduler_discriminator_from_b_to_a = torch.optim.lr_scheduler.LambdaLR(self.optimizer_discriminator_from_b_to_a, 
                        lr_lambda=LambdaLR(self.config['num_epoch'], 0, self.config['decay_epoch']).step)                                                              

        # differential privacy
        if self.config['diff_privacy']:
            self.discriminator_from_a_to_b.models[0].register_backward_hook(self.master_hook_adder)
            self.discriminator_from_b_to_a.models[0].register_backward_hook(self.master_hook_adder)


    def collect_generated_images(self, batch):
        real_a = batch[self.config['source_domain']].to(self.device) 
        real_b = batch[self.config['target_domain']].to(self.device)           
        # get shared latent representation
        mu1, Z1 = self.generator_from_a_to_b_enc(real_a, self.device)
        mu2, Z2 = self.generator_from_a_to_b_enc(real_b, self.device)

        # reconstruct images
        fake_fake_a = self.generator_from_a_to_b_dec(Z1)
        fake_fake_b = self.generator_from_b_to_a_dec(Z2)

        # translate images
        fake_a = self.generator_from_a_to_b_dec(Z2)
        fake_b = self.generator_from_b_to_a_dec(Z1)

        # cycle translation
        mu1_, Z1_ = self.generator_from_a_to_b_enc(fake_a, self.device)
        mu2_, Z2_ = self.generator_from_b_to_a_enc(fake_b, self.device)
        cycle_a = self.generator_from_a_to_b_dec(Z2_)
        cycle_b = self.generator_from_b_to_a_dec(Z1_)

        imgs = [real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b]
        tmps = [cycle_a, cycle_b, mu1, mu2, mu1_, mu2_]
        return imgs, tmps

    def calculate_basic_gan_loss(self, images):
        real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b= images[0]
        cycle_a, cycle_b, mu1, mu2, mu1_, mu2_ = images[1] 

        loss_GAN_1 = self.config['lambda_gan'] * self.discriminator_from_a_to_b.compute_loss(fake_a, self.valid)
        loss_GAN_2 = self.config['lambda_gan'] * self.discriminator_from_b_to_a.compute_loss(fake_b, self.valid)
        loss_KL_1 = self.config['lambda_kl'] * compute_kl(mu1)
        loss_KL_2 = self.config['lambda_kl'] * compute_kl(mu2)
        loss_ID_1 = self.config['lambda_identity'] * self.criterion_pixel(fake_fake_a, real_a)
        loss_ID_2 = self.config['lambda_identity'] * self.criterion_pixel(fake_fake_b, real_b)
        loss_KL_1_ = self.config['lambda_kl_translated'] * compute_kl(mu1_)
        loss_KL_2_ = self.config['lambda_kl_translated'] * compute_kl(mu2_)
        loss_cyc_1 = self.config['lambda_cycle'] * self.criterion_pixel(cycle_a, real_a)
        loss_cyc_2 = self.config['lambda_cycle'] * self.criterion_pixel(cycle_b, real_b)

        loss_gan_basic = (loss_KL_1 + loss_KL_2 + loss_ID_1 + loss_ID_2 + loss_GAN_1 + loss_GAN_2
                                + loss_KL_1_ + loss_KL_2_ + loss_cyc_1 + loss_cyc_2)
        return loss_gan_basic


