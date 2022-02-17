from fnmatch import translate
import torch
import torch.nn.functional as F

from torch.autograd.variable import Variable
from model.reg.reg import Reg
from model.reg.transformer import Reg_Transformer
from model.reg.loss import smooothing_loss
from tools.utilize import *
from model.unit.unit import *
from metrics.metrics import mae, psnr, ssim, fid
from evaluation.common import concate_tensor_lists, average
import random
import numpy as np
from loss_function.simclr_loss import simclr_loss
from loss_function.supercon_loss import supercon_loss

#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('Agg')

import kornia.geometry.transform as kt

__all__ = ['Base']

class Base():
    def __init__(self, config, train_loader, valid_loader, assigned_loader,
                 device, file_path, batch_limit_weight=1.0):

        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.assigned_loader = assigned_loader
        self.device = device
        self.file_path = file_path
        self.batch_limit_weight = batch_limit_weight
        self.angle_list = config['angle_list']
        self.translation_list = config['translation_list']
        self.scaling_list = config['scaling_list']
        self.batch_size = config['batch_size']
        self.valid = 1
        self.fake = 0

        # fid stats
        self.fid_stats = '{}/{}/{}_{}_fid_stats.npz'.format(
            self.config['fid_dir'], self.config['dataset'], self.config['source_domain'], self.config['target_domain'])

        if self.config['reg_gan']:
            self.reg = Reg(self.config['size'], self.device).to(self.device)
            self.spatial_transformer = Reg_Transformer().to(self.device)

        # differential privacy
        if self.config['diff_privacy']:
            self.clip_bound = self.config['clip_bound']
            self.sensitivity = self.config['sensitivity']
            self.noise_multiplier = self.config['noise_multiplier']

        # model, two modality, 1 and 2, the aim is to generate 2 from 1
        self.generator_from_a_to_b_enc = None
        self.generator_from_b_to_a_enc = None
        self.generator_from_a_to_b_dec = None
        self.generator_from_b_to_a_dec = None
        self.discriminator_from_a_to_b = None
        self.discriminator_from_b_to_a = None

        # loss
        self.criterion_recon = torch.nn.L1Loss().to(self.device)
        self.criterion_pixel = torch.nn.L1Loss().to(device)
        self.criterion_gan_from_a_to_b = torch.nn.MSELoss().to(device)
        self.criterion_gan_from_b_to_a = torch.nn.MSELoss().to(device)
        self.criterion_pixelwise_from_a_to_b = torch.nn.L1Loss().to(device)
        self.criterion_pixelwise_from_b_to_a = torch.nn.L1Loss().to(device)
        self.criterion_identity = torch.nn.L1Loss().to(device)
        self.criterion_sr = torch.nn.L1Loss().to(device)
        self.criterion_rotation = torch.nn.BCEWithLogitsLoss().to(device)
        self.criterion_rotation = torch.nn.BCEWithLogitsLoss().to(device)
        self.criterion_translation = torch.nn.BCEWithLogitsLoss().to(device)
        self.criterion_scaling = torch.nn.BCEWithLogitsLoss().to(device)

        if self.config['reg_gan']:
            self.reg = Reg(self.config['size'], self.device).to(self.device)
            self.spatial_transformer = Reg_Transformer().to(self.device)

        # differential privacy
        if self.config['diff_privacy']:
            pass

        # optimizer
        self.optimizer_generator = None
        self.optimizer_discriminator_from_a_to_b = None
        self.optimizer_discriminator_from_b_to_a = None

        self.lr_scheduler_generator = None
        self.lr_scheduler_discriminator_from_a_to_b = None
        self.lr_scheduler_discriminator_from_b_to_a = None

        if self.config['reg_gan']:
            self.optimizer_reg = torch.optim.Adam(self.reg.parameters(),
                                lr=self.config['lr'], betas=[self.config['beta1'], self.config['beta2']])
        # other loss
        if self.config['auxiliary_rotation']:
            self.rot_labels = self.create_rotation_labels(num_augmentation=self.config['num_augmentation'])

        if self.config['auxiliary_translation']:
            self.translation_labels = self.create_translation_labels(num_augmentation=self.config['num_augmentation'])

        if self.config['auxiliary_scaling']:
            self.scale_labels = self.create_scaling_labels(num_augmentation=self.config['num_augmentation'])

        self.batch_limit = int(self.config['data_num'] * self.batch_limit_weight / self.config['batch_size'])
        if self.config['debug']:
            self.batch_limit = 2

    def collect_generated_images(self, batch):
        pass

    def calculate_basic_gan_loss(self, images):
        pass

    def train_epoch(self, inf=''):
        for i, batch in enumerate(self.train_loader):
            if i > self.batch_limit:
                break

            """
            Train Generators
            """
            # differential privacy, train with real
            if self.config['diff_privacy']:
                self.dynamic_hook_function = self.dummy_hook

            self.optimizer_generator.zero_grad()

            imgs, tmps = self.collect_generated_images(batch=batch)
            real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b = imgs

            # gan loss
            loss_gan_basic = self.calculate_basic_gan_loss([imgs, tmps])

            # total loss
            loss_generator_total = loss_gan_basic

            # reg loss
            if self.config['reg_gan']:
                self.optimizer_reg.zero_grad()
                reg_trans = self.reg(fake_b, real_b)
                sysregist_from_a_to_b = self.spatial_transformer(fake_b, reg_trans, self.device)
                loss_sr = self.criterion_sr(sysregist_from_a_to_b, real_b)
                loss_sm = smooothing_loss(reg_trans)

            # idn loss
            if self.config['identity']:
                loss_identity_fake_b = self.criterion_identity(fake_b, fake_fake_a)
                loss_identity_fake_a = self.criterion_identity(fake_a, fake_fake_b)
                loss_identity = self.config['lambda_identity'] * (loss_identity_fake_a + loss_identity_fake_b)
                loss_generator_total += loss_identity

            # reg loss
            if self.config['reg_gan']:
                loss_reg = self.config['lambda_corr'] * loss_sr + self.config['lambda_smooth'] * loss_sm
                loss_generator_total += loss_reg

            # auxiliary loss
            if self.config['auxiliary_rotation']:
                loss_auxiliary_rotation_g, rot_real_a, rot_real_b = self.calulate_generator_auxiliary_rotation(real_a, real_b)
                loss_generator_total += loss_auxiliary_rotation_g

            if self.config['auxiliary_translation']:
                loss_auxiliary_translate_g, translate_real_a, translate_real_b = self.calulate_generator_auxiliary_translation(real_a, real_b)
                loss_generator_total += loss_auxiliary_translate_g

            if self.config['auxiliary_scaling']:
                loss_auxiliary_scale_g, scale_real_a, scale_real_b = self.calulate_generator_auxiliary_scaling(real_a, real_b)
                loss_generator_total += loss_auxiliary_scale_g

            loss_generator_total.backward()

            # differential privacy
            if self.config['diff_privacy']:
                # sanitize the gradients passed to the generator
                self.dynamic_hook_function = self.diff_privacy_conv_hook

            # torch.nn.utils.clip_grad_norm_(self.generator_from_a_to_b_enc.parameters(), max_norm=5, norm_type=2)
            # torch.nn.utils.clip_grad_norm_(self.generator_from_a_to_b_dec.parameters(), max_norm=5, norm_type=2)
            # torch.nn.utils.clip_grad_norm_(self.generator_from_b_to_a_enc.parameters(), max_norm=5, norm_type=2)
            # torch.nn.utils.clip_grad_norm_(self.generator_from_b_to_a_dec.parameters(), max_norm=5, norm_type=2)

            self.optimizer_generator.step()

            if self.config['reg_gan']:
                self.optimizer_reg.step()

            """
            Train Discriminator
            """
            self.optimizer_discriminator_from_a_to_b.zero_grad()
            self.optimizer_discriminator_from_b_to_a.zero_grad()

            loss_discriminator_from_a_to_b = self.discriminator_from_a_to_b.compute_loss(
                real_a, self.valid) + self.discriminator_from_a_to_b.compute_loss(fake_a.detach(), self.fake)
            loss_discriminator_from_b_to_a = self.discriminator_from_b_to_a.compute_loss(
                real_b, self.valid) + self.discriminator_from_b_to_a.compute_loss(fake_b.detach(), self.fake)

            loss_discriminator_from_a_to_b_total = loss_discriminator_from_a_to_b
            loss_discriminator_from_b_to_a_total = loss_discriminator_from_b_to_a

            # auxiliary loss
            if self.config['auxiliary_rotation']:
                loss_auxiliary_rotation_d = self.calulate_discriminator_auxiliary_rotation(
                    fake_a, fake_b, rot_real_a, rot_real_b)
                loss_discriminator_from_a_to_b_total += loss_auxiliary_rotation_d
                loss_discriminator_from_b_to_a_total += loss_auxiliary_rotation_d

            if self.config['auxiliary_translation']:
                loss_auxiliary_translate_d = self.calulate_discriminator_auxiliary_translation(
                    fake_a, fake_b, translate_real_a, translate_real_b)
                loss_discriminator_from_a_to_b_total += loss_auxiliary_translate_d
                loss_discriminator_from_b_to_a_total += loss_auxiliary_translate_d

            if self.config['auxiliary_scaling']:
                loss_auxiliary_scale_d = self.calulate_discriminator_auxiliary_scaling(
                    fake_a, fake_b, scale_real_a, scale_real_b)
                loss_discriminator_from_a_to_b_total += loss_auxiliary_scale_d
                loss_discriminator_from_b_to_a_total += loss_auxiliary_scale_d

            loss_discriminator_from_a_to_b_total.backward(retain_graph=True)
            loss_discriminator_from_b_to_a_total.backward(retain_graph=True)

            # torch.nn.utils.clip_grad_norm_(self.discriminator_from_a_to_b.parameters(), max_norm=5, norm_type=2)
            # torch.nn.utils.clip_grad_norm_(self.discriminator_from_b_to_a.parameters(), max_norm=5, norm_type=2)

            self.optimizer_discriminator_from_b_to_a.step()
            self.optimizer_discriminator_from_a_to_b.step()

            # print log
            infor = '\r{}[Batch {}/{}] [Gen loss: {:.4f}] [Dis loss: {:.4f}, {:.4f}]'.format(inf, i, self.batch_limit,
                        loss_generator_total.item(), loss_discriminator_from_a_to_b.item(), loss_discriminator_from_b_to_a.item())

            if self.config['identity']:
                infor = '{} [Idn Loss: {:.4f}]'.format(infor, loss_identity.item())

            if self.config['reg_gan']:
                infor = '{} [Reg Loss: {:.4f}]'.format(infor, loss_reg.item())

            if self.config['auxiliary_rotation']:
                infor = '{} [Rot Loss: {:.4f}, {:.4f}]'.format(
                    infor, loss_auxiliary_rotation_g.item(), loss_auxiliary_rotation_d.item())

            if self.config['auxiliary_translation']:
                infor = '{} [Trans Loss: {:.4f}, {:.4f}]'.format(
                    infor, loss_auxiliary_translate_g.item(), loss_auxiliary_translate_d.item())

            if self.config['auxiliary_scaling']:
                infor = '{} [Scali Loss: {:.4f}, {:.4f}]'.format(
                    infor, loss_auxiliary_scale_g.item(), loss_auxiliary_scale_d.item())

            print(infor, flush=True, end=' ')

        # update learning rates
        self.lr_scheduler_generator.step()
        self.lr_scheduler_discriminator_from_a_to_b.step()
        self.lr_scheduler_discriminator_from_b_to_a.step()

    def evaluation(self):
        # initialize fake_b_list
        fake_b_list = torch.randn(self.config['batch_size'], 1, self.config['size'], self.config['size'])
        # to reduce gpu memory for evaluation
        mae_list = []
        psnr_list = []
        ssim_list = []

        with torch.no_grad():
            for i, batch in enumerate(self.valid_loader):
                imgs, tmps = self.collect_generated_images(batch=batch)
                real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b = imgs

                if self.config['fid']:
                    fake_b_list = concate_tensor_lists(fake_b_list, fake_b, i)

                mae_value = mae(real_b, fake_b) 
                psnr_value = psnr(real_b, fake_b)
                ssim_value = ssim(real_b, fake_b)

                mae_list.append(mae_value)
                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)

            if self.config['fid']:
                fid_value = fid(fake_b_list, self.config['batch_size_inceptionV3'],
                                self.config['target_domain'], self.fid_stats, self.device)
            else:
                fid_value = 0

        return average(mae_list), average(psnr_list), average(ssim_list), fid_value     

    def get_model(self, description='centralized'):
        return self.generator_from_a_to_b_enc, self.generator_from_b_to_a_enc, self.generator_from_a_to_b_dec,\
             self.generator_from_b_to_a_dec, self.discriminator_from_a_to_b, self.discriminator_from_b_to_a

    def set_model(self, gener_from_a_to_b_enc, gener_from_a_to_b_dec, gener_from_b_to_a_enc,\
                gener_from_b_to_a_dec, discr_from_a_to_b, discr_from_b_to_a):
        self.generator_from_a_to_b_enc = gener_from_a_to_b_enc
        self.generator_from_a_to_b_dec = gener_from_a_to_b_dec
        self.generator_from_b_to_a_enc = gener_from_b_to_a_enc
        self.generator_from_b_to_a_dec = gener_from_b_to_a_dec
        self.discriminator_from_a_to_b = discr_from_a_to_b
        self.discriminator_from_b_to_a = discr_from_b_to_a

    def infer_images(self, save_img_path, data_loader):
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                imgs, tmps = self.collect_generated_images(batch=batch)
                real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b = imgs
                if i <= self.config['num_img_save']:
                    img_path = '{}/{}-slice-{}'.format(
                        save_img_path, batch['name_a'][0], batch['slice_num'].numpy()[0])

                    mae_value = mae(real_b, fake_b).item() 
                    psnr_value = psnr(real_b, fake_b).item()
                    ssim_value = ssim(real_b, fake_b).item()
                    
                    img_all = torch.cat((real_a, real_b, fake_a, fake_b, fake_fake_a, fake_fake_b), 0)
                    save_image(img_all, 'all_m_{:.4f}_p_{:.4f}_s_{:.4f}.png'.format(mae_value, psnr_value, ssim_value), img_path)

                    save_image(real_a, 'real_a.png', img_path)
                    save_image(real_b, 'real_b.png', img_path)
                    save_image(fake_a, 'fake_a.png', img_path)
                    save_image(fake_b, 'fake_b.png', img_path)
                    save_image(fake_fake_a, 'fake_fake_a.png', img_path)
                    save_image(fake_fake_b, 'fake_fake_b.png', img_path)

    def master_hook_adder(self, module, grad_input, grad_output):
        # global dynamic_hook_function
        return self.dynamic_hook_function(module, grad_input, grad_output)

    def dummy_hook(self, module, grad_input, grad_output):
        pass

    # differential privacy, train with real
    def modify_gradnorm_conv_hook(self, module, grad_input, grad_output):
        # get grad wrt. input (image)
        grad_wrt_image = grad_input[0]
        grad_input_shape = grad_wrt_image.size()
        batchsize = grad_input_shape[0]
        clip_bound_ = self.clip_bound / batchsize  # account for the 'sum' operation in GP

        grad_wrt_image = grad_wrt_image.view(batchsize, -1)
        grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

        # clip
        clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
        clip_coef = clip_coef.unsqueeze(-1)
        grad_wrt_image = clip_coef * grad_wrt_image
        grad_input_new = [grad_wrt_image.view(grad_input_shape)]
        for i in range(len(grad_input)-1):
            grad_input_new.append(grad_input[i+1])

        return tuple(grad_input_new)

    # differential privacy, train with real
    def diff_privacy_conv_hook(self, module, grad_input, grad_output):
        # global noise_multiplier
        # get grad wrt. input (image)
        grad_wrt_image = grad_input[0]
        grad_input_shape = grad_wrt_image.size()
        batchsize = grad_input_shape[0]
        clip_bound_ = self.clip_bound / batchsize

        grad_wrt_image = grad_wrt_image.view(batchsize, -1)
        grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

        # clip
        clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
        clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
        clip_coef = clip_coef.unsqueeze(-1)
        grad_wrt_image = clip_coef * grad_wrt_image

        # add noise
        noise = clip_bound_ * self.noise_multiplier * self.sensitivity * torch.randn_like(grad_wrt_image)
        grad_wrt_image = grad_wrt_image + noise
        grad_input_new = [grad_wrt_image.view(grad_input_shape)]
        for i in range(len(grad_input)-1):
            grad_input_new.append(grad_input[i+1])

        return tuple(grad_input_new)

    # create labels for rotation loss
    def create_rotation_labels(self, num_augmentation='four'):
        #TODO: rotation angle list subpass 3
        if len(self.angle_list) > 3:
            raise NotImplementedError('Roation Label Number > 4 Have Not Been Implemented Yet')
        
        if num_augmentation == 'four':
            rot_labels = torch.zeros(4 * self.batch_size).to(self.device)
            for i in range(4 * self.batch_size):
                if i < self.batch_size:
                    rot_labels[i] = 0
                elif i < 2 * self.batch_size:
                    rot_labels[i] = 1
                elif i < 3 * self.batch_size:
                    rot_labels[i] = 2
                else:
                    rot_labels[i] = 3
            return F.one_hot(rot_labels.to(torch.int64), 4).float()
        
        elif num_augmentation == 'two':
            rot_labels = torch.zeros(3 * self.batch_size).to(self.device)
            for i in range(3 * self.batch_size):
                if i < self.batch_size:
                    rot_labels[i] = 0
                elif i < 2 * self.batch_size:
                    rot_labels[i] = 1
                else:
                    rot_labels[i] = 2
            return F.one_hot(rot_labels.to(torch.int64), 3).float()

        elif num_augmentation == 'one':
            rot_labels = torch.zeros(2 * self.batch_size).to(self.device)
            for i in range(2 * self.batch_size):
                if i < self.batch_size:
                    rot_labels[i] = 0
                else:
                    rot_labels[i] = 1
            return F.one_hot(rot_labels.to(torch.int64), 2).float()
        
        else:
            raise NotImplementedError('Augmentation Number Not Implemented Yet')


    def create_translation_labels(self, num_augmentation='four'):
        #TODO: translation list subpass 1
        if len(self.translation_list) > 1:
            raise NotImplementedError('Translation Label Number > 1 Have Not Been Implemented Yet')

        if num_augmentation == 'four': 
            translation_labels = torch.zeros(5 * self.batch_size).to(self.device)
            for i in range(5 * self.batch_size):
                if i < self.batch_size:
                    translation_labels[i] = 0
                elif i < 2 * self.batch_size:
                    translation_labels[i] = 1
                elif i < 3 * self.batch_size:
                    translation_labels[i] = 2
                elif i < 4 * self.batch_size:
                    translation_labels[i] = 3
                else:
                    translation_labels[i] = 4
            return F.one_hot(translation_labels.to(torch.int64), 5).float()

        elif num_augmentation == 'two': 
            translation_labels = torch.zeros(3 * self.batch_size).to(self.device)
            for i in range(3 * self.batch_size):
                if i < self.batch_size:
                    translation_labels[i] = 0
                elif i < 2 * self.batch_size:
                    translation_labels[i] = 1
                else:
                    translation_labels[i] = 2
            return F.one_hot(translation_labels.to(torch.int64), 3).float()

        elif num_augmentation == 'one': 
            translation_labels = torch.zeros(2 * self.batch_size).to(self.device)
            for i in range(2 * self.batch_size):
                if i < self.batch_size:
                    translation_labels[i] = 0
                else:
                    translation_labels[i] = 1
            return F.one_hot(translation_labels.to(torch.int64), 2).float()
        
        else:
            raise NotImplementedError('Augmentation Number Not Implemented Yet')

    def create_scaling_labels(self, num_augmentation='four'):
        #TODO: scaling list subpass 3
        if len(self.scaling_list) > 3:
            raise NotImplementedError('Scaling Label Number > 4 Have Not Been Implemented Yet')

        if num_augmentation == 'four':
            scaling_labels = torch.zeros(4 * self.batch_size).to(self.device)
            for i in range(4 * self.batch_size):
                if i < self.batch_size:
                    scaling_labels[i] = 0
                elif i < 2 * self.batch_size:
                    scaling_labels[i] = 1
                elif i < 3 * self.batch_size:
                    scaling_labels[i] = 2
                else:
                    scaling_labels[i] = 3

            return F.one_hot(scaling_labels.to(torch.int64), 4).float()

        elif num_augmentation == 'two':
            scaling_labels = torch.zeros(3 * self.batch_size).to(self.device)
            for i in range(3 * self.batch_size):
                if i < self.batch_size:
                    scaling_labels[i] = 0
                elif i < 2 * self.batch_size:
                    scaling_labels[i] = 1
                else:
                    scaling_labels[i] = 2

            return F.one_hot(scaling_labels.to(torch.int64), 3).float()

        elif num_augmentation == 'one':
            scaling_labels = torch.zeros(2 * self.batch_size).to(self.device)
            for i in range(2 * self.batch_size):
                if i < self.batch_size:
                    scaling_labels[i] = 0
                else:
                    scaling_labels[i] = 1

            return F.one_hot(scaling_labels.to(torch.int64), 2).float()

        else:
            raise NotImplementedError('Augmentation Number Not Implemented Yet')
        

    # rotate images and return [x, x_90, x_180, x_270]
    def rotate_images(self, x, num_augmentation='four', contraD=False):
        """
        Augmentation Number Choices, Four, Two, One
        If Four, then 90 degree, 180 degree and 270 degree
        If Two, then random pick two from 90 degree, 180 degree, 270 degree
        If One, then random pick one from 90 degree, 180 degree, 270 degree
        """
        out = x
        if num_augmentation == 'four':
            for i in range(len(self.angle_list)):
                angle = torch.tensor([self.angle_list[i] for _ in range(self.batch_size)]).to(self.device)
                rot_x = kt.rotate(x, angle)
                out = torch.cat((out, rot_x), 0)

        elif num_augmentation == 'one':
            angle_list = random.sample(self.angle_list, k=1)
            assert len(angle_list) == 1
            angle = torch.tensor([angle_list[0] for _ in range(self.batch_size)]).to(self.device)
            rot_x = kt.rotate(x, angle)
            if contraD:
                return rot_x
            else: 
                out = torch.cat((out, rot_x), 0)

        elif num_augmentation == 'two':
            # Random Output is 2 or 3 views
            random_angle_list = random.sample(self.angle_list, k=2)
            assert random_angle_list[0] != random_angle_list[1]
            angle_1 = torch.tensor([random_angle_list[0] for _ in range(self.batch_size)]).to(self.device)
            angle_2 = torch.tensor([random_angle_list[1] for _ in range(self.batch_size)]).to(self.device)
            rot_x_1 = kt.rotate(x, angle_1)
            rot_x_2 = kt.rotate(x, angle_2)
            if contraD:
                return rot_x_1, rot_x_2
            else:
                out = torch.cat((out, rot_x_1), 0)
                out = torch.cat((out, rot_x_2), 0)
        else:
            raise NotImplementedError('Augmentation Number have not Implemented Yet')

        return out

    def translate_images(self, x, num_augmentation='four', contraD=False):
        """
        Augmentation Number
        If Augmentation Number is Four, Then Pick 5 Views included the original view
        If Augmentation Number is Two, Then random choose two quadrant 
        If Augmentation Number is One, Then random choose one quadrant
        """
        #Now it only supports one translation distance
        assert len(self.translation_list) == 1
        out = x
        quadrant_list = np.array([[1,1], [1, -1], [-1, 1], [-1, 1]]) 

        if num_augmentation == 'four':
            for i in range(len(self.translation_list)):
                # Translate into Upper Right Corner
                ur_translation_factor = torch.tensor([[self.translation_list[i],
                                                       self.translation_list[i]]
                                                       for _ in range(self.batch_size)]).to(self.device)
                ur_translate_x = kt.translate(x, ur_translation_factor)
                out = torch.cat((out, ur_translate_x), 0)

                #Translate into Upper Left Corner
                ul_translation_factor = torch.tensor([[-self.translation_list[i],
                                                       self.translation_list[i]]
                                                       for _ in range(self.batch_size)]).to(self.device)
                ul_translate_x = kt.translate(x, ul_translation_factor)
                out = torch.cat((out, ul_translate_x), 0)

                #Translate into Bottom Right Corner
                br_translation_factor = torch.tensor([[self.translation_list[i],
                                                       -self.translation_list[i]]
                                                       for _ in range(self.batch_size)]).to(self.device)
                br_translate_x = kt.translate(x, br_translation_factor)
                out = torch.cat((out, br_translate_x), 0)

                #Translate into Bottom Left Corner
                bl_translation_factor = torch.tensor([[-self.translation_list[i],
                                                       -self.translation_list[i]]
                                                       for _ in range(self.batch_size)]).to(self.device)
                bl_translate_x = kt.translate(x, bl_translation_factor)
                out = torch.cat((out, bl_translate_x), 0)

        elif num_augmentation == 'one':
            random_quadrant_list = random.sample(list(quadrant_list), k=1)
            translate_1 = torch.tensor([list(self.translation_list[0] * random_quadrant_list[0]) 
                                        for _ in range(self.batch_size)]).to(self.device)
            translate_1_x = kt.translate(x, translate_1.type(torch.float))
            if contraD:
                return translate_1_x
            else:
                out = torch.cat((out, translate_1_x), dim=0)

        elif num_augmentation == 'two': 
            random_quadrant_list = random.sample(list(quadrant_list), k=2)
            translate_1 = torch.tensor([list(self.translation_list[0] * random_quadrant_list[0]) 
                                        for _ in range(self.batch_size)]).to(self.device)
            translate_2 = torch.tensor([list(self.translation_list[0] * random_quadrant_list[1]) 
                                        for _ in range(self.batch_size)]).to(self.device)
            translate_1_x = kt.translate(x, translate_1.type(torch.float))
            translate_2_x = kt.translate(x, translate_2.type(torch.float))
            
            if contraD:
                return translate_1_x, translate_2_x
            else:
                out = torch.cat((out, translate_1_x), dim=0)
                out = torch.cat((out, translate_2_x), dim=0)
        else:
            raise NotImplementedError('Augmentation Number have not Implemented Yet')

        return out

    def scaling_images(self, x, num_augmentation='four', contraD=False):
        out = x
        if num_augmentation == 'four':
            for i in range(len(self.scaling_list)):
                scale_factor = torch.tensor([[self.scaling_list[i],
                                              self.scaling_list[i]]
                                              for _ in range(self.batch_size)]).to(self.device)
                scale_x = kt.scale(x, scale_factor)
                out = torch.cat((out, scale_x), 0)

        elif num_augmentation == 'one':
            random_scaling_list = random.sample(self.scaling_list, k=1)
            assert len(random_scaling_list) == 1
            scale_factor = torch.tensor([[random_scaling_list[0], random_scaling_list[0]] 
                                         for _ in range(self.batch_size)]).to(self.device)
            scale_x = kt.scale(x, scale_factor)
            if contraD:
                return scale_x
            else:
                out = torch.cat((out, scale_x), 0)
            
        elif num_augmentation == 'two':
            random_scaling_list = random.sample(self.scaling_list, k=2)
            assert len(random_scaling_list) == 2

            scale_factor_1 = torch.tensor([[random_scaling_list[0], random_scaling_list[0]] 
                                            for _ in range(self.batch_size)]).to(self.device)

            scale_factor_2 = torch.tensor([[random_scaling_list[1], random_scaling_list[1]] 
                                            for _ in range(self.batch_size)]).to(self.device)
            scale_1_x = kt.scale(x, scale_factor_1)
            scale_2_x = kt.scale(x, scale_factor_2)
            
            if contraD:
                return scale_1_x, scale_2_x
            else:
                out = torch.cat((out, scale_1_x), dim=0)
                out = torch.cat((out, scale_2_x), dim=0)
        else:
            raise NotImplementedError('Augmentation Number have not Implemented Yet')

        return out

    def calulate_generator_auxiliary_rotation(self, real_a, real_b):
        rot_real_a = Variable(self.rotate_images(real_a, num_augmentation=self.config['num_augmentation'], contraD=False))
        rot_real_b = Variable(self.rotate_images(real_b, num_augmentation=self.config['num_augmentation'], contraD=False))

        if self.config['contraD']:
            rot_real_a_logits = self.discriminator_from_a_to_b(rot_x=rot_real_a, projection_head='discriminator_head')
            rot_real_b_logits = self.discriminator_from_b_to_a(rot_x=rot_real_b, projection_head='discriminator_head')
        else:
            rot_real_a_logits = self.discriminator_from_a_to_b(rot_x=rot_real_a)
            rot_real_b_logits = self.discriminator_from_b_to_a(rot_x=rot_real_b)

        rot_real_a_loss = torch.sum(self.criterion_rotation(rot_real_a_logits, self.rot_labels))
        rot_real_b_loss = torch.sum(self.criterion_rotation(rot_real_b_logits, self.rot_labels))

        loss_auxiliary_rotation_g = rot_real_a_loss + rot_real_b_loss
        loss_auxiliary_rotation_g = self.config['weight_rotation_loss_g'] * loss_auxiliary_rotation_g

        return loss_auxiliary_rotation_g, rot_real_a, rot_real_b

    def calulate_generator_auxiliary_translation(self, real_a, real_b):
        translate_real_a = Variable(self.translate_images(real_a, num_augmentation=self.config['num_augmentation'], contraD=False))
        translate_real_b = Variable(self.translate_images(real_b, num_augmentation=self.config['num_augmentation'], contraD=False))

        if self.config['contraD']:
            translate_real_a_logits = self.discriminator_from_a_to_b(translate_x=translate_real_a, projection_head='discriminator_head')
            translate_real_b_logits = self.discriminator_from_b_to_a(translate_x=translate_real_b, projection_head='discriminator_head')
        else:
            translate_real_a_logits = self.discriminator_from_a_to_b(translate_x=translate_real_a)
            translate_real_b_logits = self.discriminator_from_b_to_a(translate_x=translate_real_b)


        translate_real_a_loss = torch.sum(self.criterion_translation(translate_real_a_logits, self.translation_labels))
        translate_real_b_loss = torch.sum(self.criterion_translation(translate_real_b_logits, self.translation_labels))

        loss_auxiliary_translate_g = translate_real_a_loss + translate_real_b_loss
        loss_auxiliary_translate_g = self.config['weight_translation_loss_g'] * loss_auxiliary_translate_g

        return loss_auxiliary_translate_g, translate_real_a, translate_real_b

    def calulate_generator_auxiliary_scaling(self, real_a, real_b):
        scale_real_a = Variable(self.scaling_images(real_a, num_augmentation=self.config['num_augmentation'], contraD=False))
        scale_real_b = Variable(self.scaling_images(real_b, num_augmentation=self.config['num_augmentation'], contraD=False))

        if self.config['contraD']:
            scale_real_a_logits = self.discriminator_from_a_to_b(scale_x=scale_real_a, projection_head='discriminator_head')
            scale_real_b_logits = self.discriminator_from_b_to_a(scale_x=scale_real_b, projection_head='discriminator_head')
        else:
            scale_real_a_logits = self.discriminator_from_a_to_b(scale_x=scale_real_a)
            scale_real_b_logits = self.discriminator_from_b_to_a(scale_x=scale_real_b)


        scale_real_a_loss = torch.sum(self.criterion_scaling(scale_real_a_logits, self.scale_labels))
        scale_real_b_loss = torch.sum(self.criterion_scaling(scale_real_b_logits, self.scale_labels))

        loss_auxiliary_scale_g = scale_real_a_loss + scale_real_b_loss
        loss_auxiliary_scale_g = self.config['weight_scaling_loss_g'] * loss_auxiliary_scale_g

        return loss_auxiliary_scale_g, scale_real_a, scale_real_b

    def calulate_discriminator_auxiliary_rotation(self, fake_a, fake_b, rot_real_a, rot_real_b):
        # auxiliary rotation loss
        rot_fake_b = Variable(self.rotate_images(fake_b, num_augmentation=self.config['num_augmentation'], contraD=False))
        rot_fake_a = Variable(self.rotate_images(fake_a, num_augmentation=self.config['num_augmentation'], contraD=False))

        if self.config['contraD']:
            rot_real_a_logits = self.discriminator_from_a_to_b(rot_x=rot_real_a, projection_head='discriminator_head')
            rot_real_b_logits = self.discriminator_from_b_to_a(rot_x=rot_real_b, projection_head='discriminator_head')
            rot_fake_b_logits = self.discriminator_from_a_to_b(rot_x=rot_fake_b, projection_head='discriminator_head')
            rot_fake_a_logits = self.discriminator_from_b_to_a(rot_x=rot_fake_a, projection_head='discriminator_head')
        else:
            rot_real_a_logits = self.discriminator_from_a_to_b(rot_x=rot_real_a)
            rot_real_b_logits = self.discriminator_from_b_to_a(rot_x=rot_real_b)
            rot_fake_b_logits = self.discriminator_from_a_to_b(rot_x=rot_fake_b)
            rot_fake_a_logits = self.discriminator_from_b_to_a(rot_x=rot_fake_a)

        rot_real_a_loss = torch.sum(self.criterion_rotation(rot_real_a_logits, self.rot_labels))
        rot_real_b_loss = torch.sum(self.criterion_rotation(rot_real_b_logits, self.rot_labels))
        rot_fake_a_loss = torch.sum(self.criterion_rotation(rot_fake_a_logits, self.rot_labels))
        rot_fake_b_loss = torch.sum(self.criterion_rotation(rot_fake_b_logits, self.rot_labels))

        loss_auxiliary_rotation_d = rot_real_a_loss + rot_real_b_loss + \
                                    rot_fake_a_loss + rot_fake_b_loss
        loss_auxiliary_rotation_d = self.config['weight_rotation_loss_d'] * loss_auxiliary_rotation_d

        return loss_auxiliary_rotation_d

    def calulate_discriminator_auxiliary_translation(self, fake_a, fake_b, translate_real_a, translate_real_b):
        # auxiliary translation loss
        translate_fake_b = Variable(self.translate_images(fake_b, num_augmentation=self.config['num_augmentation'], contraD=False))
        translate_fake_a = Variable(self.translate_images(fake_a, num_augmentation=self.config['num_augmentation'], contraD=False))

        if self.config['contraD']:
            translate_real_a_logits = self.discriminator_from_a_to_b(translate_x=translate_real_a, projection_head='discriminator_head')
            translate_real_b_logits = self.discriminator_from_b_to_a(translate_x=translate_real_b, projection_head='discriminator_head')
            translate_fake_b_logits = self.discriminator_from_a_to_b(translate_x=translate_fake_b, projection_head='discriminator_head')
            translate_fake_a_logits = self.discriminator_from_b_to_a(translate_x=translate_fake_a, projection_head='discriminator_head')
        else:
            translate_real_a_logits = self.discriminator_from_a_to_b(translate_x=translate_real_a)
            translate_real_b_logits = self.discriminator_from_b_to_a(translate_x=translate_real_b)
            translate_fake_b_logits = self.discriminator_from_a_to_b(translate_x=translate_fake_b)
            translate_fake_a_logits = self.discriminator_from_b_to_a(translate_x=translate_fake_a)

        translate_real_a_loss = torch.sum(self.criterion_translation(translate_real_a_logits, self.translation_labels))
        translate_real_b_loss = torch.sum(self.criterion_translation(translate_real_b_logits, self.translation_labels))
        translate_fake_a_loss = torch.sum(self.criterion_translation(translate_fake_a_logits, self.translation_labels))
        translate_fake_b_loss = torch.sum(self.criterion_translation(translate_fake_b_logits, self.translation_labels))

        loss_auxiliary_translate_d = translate_real_a_loss + translate_real_b_loss + \
                                    translate_fake_a_loss + translate_fake_b_loss
        loss_auxiliary_translate_d = self.config['weight_translation_loss_d'] * loss_auxiliary_translate_d

        return loss_auxiliary_translate_d

    def calulate_discriminator_auxiliary_scaling(self, fake_a, fake_b, scale_real_a, scale_real_b):
        # auxiliary scaling loss
        scale_fake_b = Variable(self.scaling_images(fake_b, num_augmentation=self.config['num_augmentation'], contraD=False))
        scale_fake_a = Variable(self.scaling_images(fake_a, num_augmentation=self.config['num_augmentation'], contraD=False))
        
        if self.config['contraD']:
            scale_real_a_logits = self.discriminator_from_a_to_b(scale_x=scale_real_a, projection_head='discriminator_head')
            scale_real_b_logits = self.discriminator_from_b_to_a(scale_x=scale_real_b, projection_head='discriminator_head')
            scale_fake_b_logits = self.discriminator_from_a_to_b(scale_x=scale_fake_b, projection_head='discriminator_head')
            scale_fake_a_logits = self.discriminator_from_b_to_a(scale_x=scale_fake_a, projection_head='discriminator_head')
        else:
            scale_real_a_logits = self.discriminator_from_a_to_b(scale_x=scale_real_a)
            scale_real_b_logits = self.discriminator_from_b_to_a(scale_x=scale_real_b)
            scale_fake_b_logits = self.discriminator_from_a_to_b(scale_x=scale_fake_b)
            scale_fake_a_logits = self.discriminator_from_b_to_a(scale_x=scale_fake_a)

        scale_real_a_loss = torch.sum(self.criterion_scaling(scale_real_a_logits, self.scale_labels))
        scale_real_b_loss = torch.sum(self.criterion_scaling(scale_real_b_logits, self.scale_labels))
        scale_fake_a_loss = torch.sum(self.criterion_scaling(scale_fake_a_logits, self.scale_labels))
        scale_fake_b_loss = torch.sum(self.criterion_scaling(scale_fake_b_logits, self.scale_labels))

        loss_auxiliary_scale_d = scale_real_a_loss + scale_real_b_loss + \
                                        scale_fake_a_loss + scale_fake_b_loss
        loss_auxiliary_scale_d = self.config['weight_scaling_loss_d'] * loss_auxiliary_scale_d

        return loss_auxiliary_scale_d

    def calculate_simclr_loss(self, real_a, real_b):
        rot_real_a = Variable(self.rotate_images(real_a, num_augmentation='one', contraD=True))
        translate_real_a = Variable(self.translate_images(real_a, num_augmentation='one', contraD=True))
        rot_real_a_logit, translate_real_a_logit = self.discriminator_from_a_to_b(real_x1=rot_real_a, 
                                                                                  real_x2=translate_real_a,
                                                                                  projection_head='real_head')
        simclr_real_a_loss = simclr_loss(rot_real_a_logit, translate_real_a_logit, 
                                         temperature=self.config['temp'],
                                         normalize=True)

        rot_real_b = Variable(self.rotate_images(real_b, num_augmentation='one', contraD=True))
        translate_real_b = Variable(self.translate_images(real_b, num_augmentation='one', contraD=True))
        rot_real_b_logit, translate_real_b_logit = self.discriminator_from_b_to_a(real_x1=rot_real_b, 
                                                                                  real_x2=translate_real_b,
                                                                                  projection_head='real_head')
        simclr_real_b_loss = simclr_loss(rot_real_b_logit, translate_real_b_logit, 
                                         temperature=self.config['temp'],
                                         normalize=True)
        simclr_auxiliary_loss = self.config['weight_simclr_loss'] * (simclr_real_a_loss + simclr_real_b_loss)
        return simclr_auxiliary_loss 

    def calculate_superconf_loss(self, fake_a, fake_b, real_a, real_b):
        rot_real_a = Variable(self.rotate_images(real_a, num_augmentation='one', contraD=True))
        translate_real_a = Variable(self.translate_images(real_a, num_augmentation='one', contraD=True))
        rot_real_a_logit, translate_real_a_logit, fake_b_simclr_logit = self.discriminator_from_a_to_b(fake_x=fake_b,
                                                                                                       real_x1=rot_real_a, 
                                                                                                       real_x2=translate_real_a,
                                                                                                       projection_head='fake_head')
        supercon_real_a_loss = supercon_loss(out1=rot_real_a_logit, out2=translate_real_a_logit, 
                                             others=fake_b_simclr_logit, 
                                             temperature=self.config['temp'], 
                                             normalize=True)

        rot_real_b = Variable(self.rotate_images(real_b, num_augmentation='one', contraD=True))
        translate_real_b = Variable(self.translate_images(real_b, num_augmentation='one', contraD=True))

        rot_real_b_logit, translate_real_b_logit, fake_a_simclr_logit = self.discriminator_from_a_to_b(fake_x=fake_a,
                                                                                                       real_x1=rot_real_b, 
                                                                                                       real_x2=translate_real_b,
                                                                                                       projection_head='fake_head')
        supercon_real_b_loss = supercon_loss(out1=rot_real_b_logit, out2=translate_real_b_logit, 
                                             others=fake_a_simclr_logit, 
                                             temperature=self.config['temp'], 
                                             normalize=True)
        
        supercon_auxiliary_loss = supercon_real_a_loss + supercon_real_b_loss 
        supercon_auxiliary_loss = self.config['weight_supercon_loss'] * supercon_auxiliary_loss
        
        return supercon_auxiliary_loss 
