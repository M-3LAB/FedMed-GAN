import torch
import itertools

from model.munit.munit import *
from arch_centralized.base import Base
from tools.utilize import *

__all__ = ['Munit']

class Munit(Base):
    def __init__(self, config, train_loader, valid_loader, assigned_loader, 
                 device, file_path, batch_limit_weight=1.0, atl=False):
        super(Munit, self).__init__(config=config, train_loader=train_loader, valid_loader=valid_loader, assigned_loader=assigned_loader, 
                 device=device, file_path=file_path, batch_limit_weight=batch_limit_weight, atl=atl)

        self.config = config

        # model, two modality, 1 and 2, the aim is to generate 2 from 1
        self.generator_from_a_to_b_enc = Encoder(in_channels=self.config['input_dim'], dim=self.config['dim'],
                            n_downsample=self.config['n_downsample'], n_residual=self.config['n_res'],
                            style_dim=self.config['style_dim']).to(self.device)

        self.generator_from_b_to_a_enc = Encoder(in_channels=self.config['input_dim'], dim=self.config['dim'],
                            n_downsample=self.config['n_downsample'], n_residual=self.config['n_res'],
                            style_dim=self.config['style_dim']).to(self.device)

        self.generator_from_a_to_b_dec = Decoder(out_channels=self.config['input_dim'], dim=self.config['dim'],
                                                 n_residual=self.config['n_res'], n_upsample=self.config['n_upsample'],
                                                style_dim=self.config['style_dim']).to(self.device)

        self.generator_from_b_to_a_dec = Decoder(out_channels=self.config['input_dim'], dim=self.config['dim'],
                                                n_residual=self.config['n_res'], n_upsample=self.config['n_upsample'],
                                                style_dim=self.config['style_dim']).to(self.device)

        self.discriminator_from_a_to_b = Discriminator(auxiliary_rotation=self.config['auxiliary_rotation'], 
                                                auxiliary_translation=self.config['auxiliary_translation'], auxiliary_scaling=self.config['auxiliary_scaling'], 
                                                num_rot_label=4, num_translate_label=5, num_scaling_label=4).to(self.device)
        self.discriminator_from_b_to_a = Discriminator(auxiliary_rotation=self.config['auxiliary_rotation'], 
                                                auxiliary_translation=self.config['auxiliary_translation'], auxiliary_scaling=self.config['auxiliary_scaling'], 
                                                num_rot_label=4, num_translate_label=5, num_scaling_label=4).to(self.device)  

        # optimizer
        self.optimizer_generator = torch.optim.Adam(itertools.chain(self.generator_from_a_to_b_enc.parameters(),
                                                            self.generator_from_a_to_b_dec.parameters(),
                                                            self.generator_from_b_to_a_enc.parameters(),
                                                            self.generator_from_b_to_a_dec.parameters()),
                                            lr=self.config['lr'], betas=(self.config['beta1'], self.config['beta2']))

        self.optimizer_discriminator_from_a_to_b = torch.optim.Adam(self.discriminator_from_a_to_b.parameters(), lr=self.config['lr'], betas=(self.config['beta1'], self.config['beta2']))
        self.optimizer_discriminator_from_b_to_a = torch.optim.Adam(self.discriminator_from_b_to_a.parameters(), lr=self.config['lr'], betas=(self.config['beta1'], self.config['beta2']))

        self.lr_scheduler_generator = torch.optim.lr_scheduler.LambdaLR(self.optimizer_generator, 
                        lr_lambda=LambdaLR(self.config['num_epoch'], 0, self.config['decay_epoch']).step)

        self.lr_scheduler_discriminator_from_a_to_b = torch.optim.lr_scheduler.LambdaLR(self.optimizer_discriminator_from_a_to_b, 
                        lr_lambda=LambdaLR(self.config['num_epoch'], 0, self.config['decay_epoch']).step)
        self.lr_scheduler_discriminator_from_b_to_a = torch.optim.lr_scheduler.LambdaLR(self.optimizer_discriminator_from_b_to_a, 
                        lr_lambda=LambdaLR(self.config['num_epoch'], 0, self.config['decay_epoch']).step)                                                              

        # differential privacy
        if self.config['diff_privacy']:
            self.discriminator_from_a_to_b.models.disc_0[0].register_backward_hook(self.master_hook_adder)
            self.discriminator_from_b_to_a.models.disc_0[0].register_backward_hook(self.master_hook_adder)


    def collect_generated_images(self, batch):
        real_a = batch[self.config['source_domain']].to(self.device) 
        real_b = batch[self.config['target_domain']].to(self.device)           

        style_1 = torch.randn(real_a.size(0), self.config['style_dim'], 1, 1).type(torch.cuda.FloatTensor).to(self.device)
        style_2 = torch.randn(real_a.size(0), self.config['style_dim'], 1, 1).type(torch.cuda.FloatTensor).to(self.device)

        # get shared latent representation
        c_code_1, s_code_1 = self.generator_from_a_to_b_enc(real_a)
        c_code_2, s_code_2 = self.generator_from_b_to_a_enc(real_b)

        # reconstruct images
        fake_fake_a = self.generator_from_a_to_b_dec(c_code_1, s_code_1)
        fake_fake_b = self.generator_from_b_to_a_dec(c_code_2, s_code_2)

        # translate images
        fake_a = self.generator_from_a_to_b_dec(c_code_2, style_1)
        fake_b = self.generator_from_b_to_a_dec(c_code_1, style_2)

        # cycle translation
        c_code_21, s_code_21 = self.generator_from_a_to_b_enc(fake_a)
        c_code_12, s_code_12 = self.generator_from_b_to_a_enc(fake_b)

        cycle_a = self.generator_from_a_to_b_dec(c_code_12, s_code_1) if self.config['lambda_cycle'] > 0 else 0
        cycle_b= self.generator_from_b_to_a_dec(c_code_21, s_code_2) if self.config['lambda_cycle'] > 0 else 0

        imgs = [real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b]
        tmps = [cycle_a, cycle_b, c_code_1, c_code_2, c_code_12, c_code_21, s_code_21, s_code_12]
        return imgs, tmps
             

    def calculate_basic_gan_loss(self, images):
        real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b = images[0]
        cycle_a, cycle_b, c_code_1, c_code_2, c_code_12, c_code_21, s_code_21, s_code_12 = images[1] 

        style_1 = torch.randn(real_a.size(0), self.config['style_dim'], 1, 1).type(torch.cuda.FloatTensor).to(self.device)
        style_2 = torch.randn(real_a.size(0), self.config['style_dim'], 1, 1).type(torch.cuda.FloatTensor).to(self.device)

        loss_generator_totalAN_1 = self.config['lambda_gan'] * self.discriminator_from_a_to_b.compute_loss(fake_a, self.valid)
        loss_generator_totalAN_2 = self.config['lambda_gan'] * self.discriminator_from_b_to_a.compute_loss(fake_b, self.valid)
        loss_s_1 = self.config['lambda_style'] * self.criterion_recon(s_code_21, style_1)
        loss_s_2 = self.config['lambda_style'] * self.criterion_recon(s_code_12, style_2)
        loss_c_1 = self.config['lambda_content']  * self.criterion_recon(c_code_12, c_code_1.detach())
        loss_c_2 = self.config['lambda_content'] * self.criterion_recon(c_code_21, c_code_2.detach())
        loss_cyc_1 = self.config['lambda_cycle'] * self.criterion_recon(cycle_a, real_a) if self.config['lambda_cycle'] > 0 else 0
        loss_cyc_2 = self.config['lambda_cycle'] * self.criterion_recon(cycle_b, real_b) if self.config['lambda_cycle'] > 0 else 0

        loss_gan_basic = loss_generator_totalAN_1 + loss_generator_totalAN_2 + loss_s_1 + loss_s_2\
             + loss_c_1 + loss_c_2 + loss_cyc_1 + loss_cyc_2

        return loss_gan_basic
