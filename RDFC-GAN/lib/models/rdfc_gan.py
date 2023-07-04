
import itertools
import os

import imageio
import lib.utils.save_vis as vis
import numpy as np
import torch
import torch.nn.functional as F
from helper import addPepperNoise,norm_normalize
from lib.losses.gan_loss import (GANLoss, L1_loss, manhattan_loss,nor_loss)

from .base import Base
from .discriminator import build_discriminator
from .generator import build_generator
from .generator.normal_net.NNET import NNET
from .generator.label_net.labels_model import PSPNet
from .init_weights import init_weights


class RDFCGAN(Base):
    def __init__(self,
                 device,
                 distributed,
                 args,
                 is_train=True,
                 fixed_training_samples=None,
                 fixed_testing_samples=None,
                 num_classes = 14,
                 out_height = 224,
                 out_width = 304,
                 label_wall = 12,
                 label_floor = 5,
                 label_ceiling = 3):
        super(RDFCGAN, self).__init__(
            device=device,
            distributed=distributed,
            args=args,
            is_train=is_train,
        )

        self.num_classes = num_classes
        self.label_wall = label_wall
        self.label_floor = label_floor
        self.label_ceiling = label_ceiling
        self.out_height = out_height
        self.out_width = out_width
        self.init_model()

        if self.is_train:
            self.init_optimizer()
            self.init_lr_scheduler()

            self.criterionGAN = GANLoss(args.gan_loss_type).to(device)
            self.criterionCycle = L1_loss
            self.criterionL1 = L1_loss
            self.manhattan_loss = manhattan_loss
            self.nor_loss = nor_loss
            self.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

            # for vis
            self.fixed_training_samples = fixed_training_samples
            self.fixed_testing_samples = fixed_testing_samples
            

    def set_input(self, data):
        self.real_A, self.real_B = data['rgb'].to(self.device), data['gt_depth'].to(self.device)
        self.aux_A = data['raw_depth'].to(self.device)
        self.gt_normal = data['gt_normal'].to(self.device)
        self.gt_label = data['labels'].to(self.device)
        self.norm_mask = data['normal_masks'].to(self.device)
        if 'depth_masks' in data:
            self.mask = data['depth_masks'].to(self.device)
        else:
            self.mask = torch.ones_like(self.real_B).to(self.device)

        self.image_loss_weight = self.mask / (self.mask.sum() + 1e-6)    # useful for L1 loss actually.

        #***
    def init_model(self):
        """Init generators and discriminators"""
        self.G_A2B = build_generator(self.args.model.G_A2B)
        self.G_B2A = build_generator(self.args.model.G_B2A)
        self.disc_A = build_discriminator(self.args.model.D_A)
        self.disc_B = build_discriminator(self.args.model.D_B)
        
        self.G_normal = NNET(self.out_height,self.out_width)

        self.G_label =PSPNet(self.num_classes)

        
        if not self.distributed:
            self.G_A2B = self.G_A2B.to(self.device)
            self.G_B2A = self.G_B2A.to(self.device)
            self.disc_A = self.disc_A.to(self.device)
            self.disc_B = self.disc_B.to(self.device)

            self.G_normal = self.G_normal.to(self.device)
            self.G_label = self.G_label.to(self.device)

        else:
            self.G_A2B = torch.nn.parallel.DistributedDataParallel(self.G_A2B.to(self.device),
                                                                   device_ids=[self.local_rank],
                                                                   # find_unused_parameters=True
                                                                   )
            self.G_B2A = torch.nn.parallel.DistributedDataParallel(self.G_B2A.to(self.device),
                                                                   device_ids=[self.local_rank],
                                                                   # find_unused_parameters=True
                                                                   )
            self.disc_A = torch.nn.parallel.DistributedDataParallel(self.disc_A.to(self.device),
                                                                    device_ids=[self.local_rank])
            self.disc_B = torch.nn.parallel.DistributedDataParallel(self.disc_B.to(self.device),
                                                                    device_ids=[self.local_rank])
            
            self.G_normal = torch.nn.parallel.DistributedDataParallel(self.G_normal.to(self.device),
                                                                   device_ids=[self.local_rank])
                                                    
            self.G_label = torch.nn.parallel.DistributedDataParallel(self.G_label.to(self.device),
                                                                     device_ids=[self.local_rank])
        init_weights(self.G_A2B)
        init_weights(self.G_B2A)
        init_weights(self.disc_A)
        init_weights(self.disc_B)



        self.models = dict(G_A2B=self.G_A2B,
                           G_B2A=self.G_B2A,
                           disc_A=self.disc_A,
                           disc_B=self.disc_B,
                           G_normal = self.G_normal,
                           G_label = self.G_label)

    def init_optimizer(self):
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()),
                                            lr=self.args.lr,
                                            betas=(self.args.beta1, self.args.beta2))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.disc_A.parameters(), self.disc_B.parameters()),
                                            lr=self.args.lr,
                                            betas=(self.args.beta1, self.args.beta2))
        self.optimizer_Normal = torch.optim.AdamW(self.G_normal.parameters(),
                                            weight_decay=self.args.weight_decay,
                                            lr=self.args.lr * 0.5,
                                            betas=(self.args.beta1, self.args.beta2))
        self.optimizer_Label = torch.optim.Adam(self.G_label.parameters(),
                                            lr=self.args.lr,
                                            betas=(self.args.beta1, self.args.beta2))

        self.optimizers = dict(G=self.optimizer_G,
                               D=self.optimizer_D,
                               Nor=self.optimizer_Normal,
                               label = self.optimizer_Label)

    def init_lr_scheduler(self):
        lr_scheduler = self.get_lr_scheduler()
        self.lr_scheduler_G = lr_scheduler(optimizer=self.optimizer_G)
        self.lr_scheduler_D = lr_scheduler(optimizer=self.optimizer_D)
       
        self.lr_scheduler_Nor = lr_scheduler(optimizer=self.optimizer_Normal)

        self.lr_scheduler_Label = lr_scheduler(optimizer=self.optimizer_Label)

        self.lr_schedulers = dict(G=self.lr_scheduler_G,
                                  D=self.lr_scheduler_D,
                                  Nor=self.lr_scheduler_Nor,
                                  label=self.lr_scheduler_Label)

    def forward_test(self, **kwargs):
        rgb = kwargs['rgb'].to(self.device)
        raw_depth = kwargs['raw_depth'].to(self.device)
        
        pred_label = self.G_label(rgb)
        pred_normal = self.G_normal(rgb)[: ,0:3 ,: ,:]
        pred_normal = norm_normalize(pred_normal)
        ret = self.G_A2B(rgb, raw_depth, pred_normal)
        pred_depth = ret['pred_depth']

        ret = dict(pred_depth = pred_depth,
                pred_normal = pred_normal,
                pred_label = pred_label)

        return ret

    def forward(self):
        """Get fake images & reconstructed images"""

        self.fake_label_realA_list = self.G_label(self.real_A)
        assert self.fake_label_realA_list[0].size()[2:] == self.gt_label.size()[1:]  
        assert self.fake_label_realA_list[0].size()[1] == self.num_classes

        self.fake_normal_realA = self.G_normal(self.real_A)[:, 0:3, :, :]
        self.fake_normal_realA = norm_normalize(self.fake_normal_realA)

        self.fake_B = self.G_A2B(self.real_A, self.aux_A, self.fake_normal_realA)['pred_depth']

        self.fake_B = addPepperNoise(self.fake_B)
        self.rec_A = self.G_B2A(self.fake_B)


        self.fake_A = self.G_B2A(self.real_B)
        self.fake_label_fakeA_list = self.G_label(self.fake_A)
        assert self.fake_label_fakeA_list[0].size()[2:] == self.gt_label.size()[1:]  
        assert self.fake_label_fakeA_list[0].size()[1] == self.num_classes

        self.fake_normal_fakeA = self.G_normal(self.fake_A)[:, 0:3, :, :]
        self.fake_normal_fakeA = norm_normalize(self.fake_normal_fakeA)
        self.rec_B = self.G_A2B(self.fake_A, self.aux_A, self.fake_normal_fakeA)['pred_depth']

    def backward_G(self):
        lambda_A = self.args.lambda_A
        lambda_B = self.args.lambda_B
        lambda_l1 = self.args.lambda_L1
        # identity loss, not use

        # GAN loss
        # GAN loss disc_A(G_A2B(A))  & GAN loss disc_B(G_B2A(B))
        loss_G_A2B = self.criterionGAN(self.disc_A(self.fake_B), True) # * lambda_l1
        loss_G_B2A = self.criterionGAN(self.disc_B(self.fake_A), True) # * lambda_l1

        # L1 loss(pred)
        loss_A2B_L1 = self.criterionL1(self.fake_B, self.real_B,
                                            weight=self.image_loss_weight) * lambda_l1 * 5
        loss_B2A_L1 = self.criterionL1(self.fake_A, self.real_A,
                                            weight=self.image_loss_weight) * lambda_l1 * 3

        # cycle loss(pred)
        # Forward cycle loss || G_B2A(G_A2B(A)) - A ||
        rec_A = self.G_B2A(self.fake_B)  # G_B2A(G_A2B(A))
        loss_cycle_A2B = self.criterionCycle(rec_A, self.real_A) * lambda_A

        # Backward cycle loss || G_A2B(G_B2A(B)) - B ||
        fakeA_normal = self.G_normal(self.fake_A)[:, 0:3, :, :]
        fakeA_normal = norm_normalize(fakeA_normal)
        rec_B = self.G_A2B(self.fake_A, self.aux_A, fakeA_normal)['pred_depth']  # B_A2B(G_B2A(B))
        loss_cycle_B2A = self.criterionCycle(rec_B, self.real_B,weight=self.image_loss_weight) * lambda_B

        #semantic label loss
        loss_label_A2B = (self.CrossEntropyLoss(self.fake_label_realA_list[0],self.gt_label) + self.CrossEntropyLoss(self.fake_label_realA_list[1],self.gt_label) * 0.4) * lambda_l1 
        loss_label_B2A = (self.CrossEntropyLoss(self.fake_label_fakeA_list[0],self.gt_label) + self.CrossEntropyLoss(self.fake_label_fakeA_list[1],self.gt_label) * 0.4) * lambda_l1


        # normal Loss
        loss_normal_A2B = self.nor_loss(self.fake_normal_realA,self.gt_normal,self.norm_mask) * lambda_l1 * 2
        floor_loss_A2B,wall_loss_A2B,ceiling_loss_A2B = self.manhattan_loss(self.fake_normal_realA ,self.fake_label_realA_list[0], self.norm_mask,
                                                                            self.label_wall, self.label_floor, self.label_ceiling,lambda_l1) 
 

        loss_normal_B2A = self.nor_loss(self.fake_normal_fakeA,self.gt_normal,self.norm_mask) * lambda_l1 * 2
        floor_loss_B2A,wall_loss_B2A,ceiling_loss_B2A = self.manhattan_loss(self.fake_normal_fakeA , self.fake_label_fakeA_list[0],self.norm_mask,
                                                                            self.label_wall, self.label_floor, self.label_ceiling,lambda_l1)


        loss_G =loss_normal_A2B + \
                loss_normal_B2A + \
                loss_G_A2B + \
                loss_cycle_A2B + \
                loss_cycle_B2A + \
                loss_G_B2A + \
                loss_A2B_L1 + \
                loss_B2A_L1 + \
                loss_label_A2B + \
                loss_label_B2A + \
                floor_loss_A2B + \
                wall_loss_A2B + \
                ceiling_loss_A2B + \
                floor_loss_B2A + \
                wall_loss_B2A + \
                ceiling_loss_B2A

        loss_G.backward()

        return dict(loss_G=loss_G,
                    loss_G_A2B=loss_G_A2B,
                    loss_G_B2A=loss_G_B2A,
                    loss_B2A_L1=loss_B2A_L1,
                    loss_A2B_L1=loss_B2A_L1,
                    loss_cycle_A=loss_cycle_A2B,
                    loss_cycle_B=loss_cycle_B2A,
                    loss_label_A2B = loss_label_A2B,
                    loss_label_B2A = loss_label_B2A,
                    loss_normal_A2B = loss_normal_A2B,
                    loss_normal_B2A = loss_normal_B2A,
                    floor_loss_A2B = floor_loss_A2B,
                    floor_loss_B2A = floor_loss_B2A,
                    wall_loss_A2B = wall_loss_A2B,
                    wall_loss_B2A = wall_loss_B2A,
                    ceiling_loss_A2B = ceiling_loss_A2B,
                    ceiling_loss_B2A = ceiling_loss_B2A
                    )


    def backward_D_basic(self, model_D, real, fake):
        pred_real = model_D(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = model_D(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()

        return loss_D_real, loss_D_fake, loss_D

    def backward_D_A2B(self):
        loss_D_A_real, loss_D_A_fake, loss_D_A = self.backward_D_basic(self.disc_A, self.real_B, self.fake_B)

        return dict(
            loss_D_A=loss_D_A,
            loss_D_A_real=loss_D_A_real,
            loss_D_A_fake=loss_D_A_fake)

    def backward_D_B2A(self):
        loss_D_B_real, loss_D_B_fake, loss_D_B = self.backward_D_basic(self.disc_B, self.real_A, self.fake_A)

        return dict(
            loss_D_B=loss_D_B,
            loss_D_B_real=loss_D_B_real,
            loss_D_B_fake=loss_D_B_fake)

    def optimize_parameters(self):
        loss_stats = {}
        self.forward()
        self.set_requires_grad([self.disc_A, self.disc_B], requires_grad=False)

        self.optimizer_G.zero_grad()
        self.optimizer_Normal.zero_grad()
        self.optimizer_Label.zero_grad()
        loss_stats.update(self.backward_G())
        self.optimizer_G.step()
        self.optimizer_Normal.step()
        self.optimizer_Label.step()

        self.set_requires_grad([self.disc_A, self.disc_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        loss_stats.update(self.backward_D_A2B())  # calculate gradients for D_A2B
        loss_stats.update(self.backward_D_B2A())  # calculate graidents for D_B2A
        self.optimizer_D.step()

        if self.distributed:
            torch.distributed.barrier()

        return self.reduce_loss(loss_stats)

    def _save_samples(self, domain_A, domain_B,domain_C,domain_D, aux_A=None, batch_size=8, savename_prefix=None):

        palette = [(0,0,0), (0,0,255), (232,88,47),
                    (0,217,0), (148,0,240), (222,241,23),
                    (255,205,205), (0,223,228), (106,135,204), (116,28,41),
                    (240,35,235), (0,166,156), (249,139,0),(225,228,194)]
        self.G_A2B.eval()
        self.G_B2A.eval()
        self.G_normal.eval()
        self.G_label.eval()

        with torch.no_grad():
            fake_label_A = self.G_label(domain_A)
            fake_normal_A = self.G_normal(domain_A)[:,0:3,:,:]
            fake_normal_A = norm_normalize(fake_normal_A)
            fake_B = self.G_A2B(domain_A)['pred_depth'] if aux_A is None else self.G_A2B(domain_A, aux_A, fake_normal_A)['pred_depth']
            rec_A = self.G_B2A(fake_B)

            fake_A = self.G_B2A(domain_B)
            fake_label_B = self.G_label(fake_A)
            fake_normal_B = self.G_normal(fake_A)[:,0:3,:,:]
            fake_normal_B = norm_normalize(fake_normal_B)
            rec_B = self.G_A2B(fake_A)['pred_depth'] if aux_A is None else self.G_A2B(fake_A, aux_A, fake_normal_B)['pred_depth']
            

        domain_A = vis.to_data(domain_A)
        fake_A = vis.to_data(fake_A)
        aux_A, fake_B = vis.to_data(aux_A), vis.to_data(fake_B)
        rec_A, rec_B = vis.to_data(rec_A), vis.to_data(rec_B)
        domain_C,fake_normal_A,fake_normal_B = vis.to_data(domain_C),vis.to_data(fake_normal_A),vis.to_data(fake_normal_B)

        gt_depth = vis.to_data(domain_B)

        gt_seg_color = vis.color_label(domain_D,palette,self.num_classes)
        fake_label_A = F.softmax(fake_label_A,dim=1).argmax(1)
        fake_label_A_color = vis.color_label(fake_label_A,palette,self.num_classes)
        fake_label_B = F.softmax(fake_label_B,dim=1).argmax(1)
        fake_label_B_color = vis.color_label(fake_label_B,palette,self.num_classes)
        
        # merge and visual results
        merged = vis.merge_images([domain_A, fake_A, rec_A], num_imgs_per_scene=3, batch_size=batch_size)
        imageio.imsave(f"{savename_prefix}-B-A.jpg", merged.astype(np.uint8))
        merged = vis.merge_images([aux_A, gt_depth, fake_B, rec_B], num_imgs_per_scene=4, batch_size=batch_size)
        imageio.imsave(f"{savename_prefix}-A-B.jpg", merged.astype(np.uint8))
        merged = vis.merge_images([domain_C, fake_normal_A, fake_normal_B], num_imgs_per_scene=3, batch_size=batch_size)
        imageio.imsave(f"{savename_prefix}-normal.jpg", merged.astype(np.uint8))
        merged = vis.merge_images([gt_seg_color,fake_label_A_color,fake_label_B_color], num_imgs_per_scene=3, batch_size=batch_size)
        imageio.imsave(f"{savename_prefix}-label.jpg", merged.astype(np.uint8))

        self.G_A2B.train()
        self.G_B2A.train()
        self.G_normal.train()
        self.G_label.train()
        torch.cuda.empty_cache()

    def save_samples(self, save_dir, iters):
        self._save_samples(self.fixed_testing_samples['rgb'].to(self.device),
                           self.fixed_testing_samples['gt_depth'].to(self.device),
                           self.fixed_testing_samples['gt_normal'].to(self.device),
                           self.fixed_testing_samples['labels'].to(self.device),
                           aux_A=self.fixed_testing_samples['raw_depth'].to(self.device), batch_size=8,
                           savename_prefix=os.path.join(save_dir, 'sample-test-{:06d}'.format(iters)))

        self._save_samples(self.fixed_training_samples['rgb'].to(self.device),
                           self.fixed_training_samples['gt_depth'].to(self.device),
                           self.fixed_training_samples['gt_normal'].to(self.device),
                           self.fixed_training_samples['labels'].to(self.device),
                           aux_A=self.fixed_training_samples['raw_depth'].to(self.device), batch_size=8,
                           savename_prefix=os.path.join(save_dir, 'sample-train-{:06d}'.format(iters)))

    def generator(self):
        return self.G_A2B
