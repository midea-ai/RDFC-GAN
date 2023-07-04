"""
    Author: Wang mingyuan
    Description:
"""
import torch
from .generator import build_generator
from .discriminator import build_discriminator
from .init_weights import init_weights
from .base import Base
from lib.losses.gan_loss import GANLoss, L1_loss
from torch.autograd import grad
import lib.utils.save_vis as vis
import imageio
import numpy as np
import os


class RDFGAN(Base):
    def __init__(self,
                 device,
                 distributed,
                 args,
                 is_train=True,
                 fixed_training_samples=None,
                 fixed_testing_samples=None):
        super(RDFGAN, self).__init__(
            device=device,
            distributed=distributed,
            args=args,
            is_train=is_train
        )
        self.init_model()

        if self.is_train:
            self.init_optimizer()
            self.init_lr_scheduler()

            self.criterionGAN = GANLoss(args.gan_loss_type).to(device)
            self.criterionL1 = L1_loss

            # for vis
            self.fixed_training_samples = fixed_training_samples
            self.fixed_testing_samples = fixed_testing_samples

    def init_model(self):
        """Init generators and discriminators"""
        self.G = build_generator(self.args.model.G)
        self.D = build_discriminator(self.args.model.D)

        if not self.distributed:
            self.G = self.G.to(self.device)
            self.D = self.D.to(self.device)
        else:
            self.G = torch.nn.parallel.DistributedDataParallel(self.G.to(self.device),
                                                               device_ids=[self.local_rank],
                                                               # find_unused_parameters=True
                                                               )
            self.D = torch.nn.parallel.DistributedDataParallel(self.D.to(self.device),
                                                               device_ids=[self.local_rank],
                                                               # find_unused_parameters=True
                                                               )
        init_weights(self.G)
        init_weights(self.D)

        self.models = dict(G=self.G,
                           D=self.D)

    def init_optimizer(self):
        optimizer = self.get_optimizer()
        self.optimizer_G = optimizer(params=self.G.parameters())
        self.optimizer_D = optimizer(params=self.D.parameters())

        self.optimizers = dict(G=self.optimizer_G,
                               D=self.optimizer_D)

    def init_lr_scheduler(self):
        lr_scheduler = self.get_lr_scheduler()
        self.lr_scheduler_G = lr_scheduler(optimizer=self.optimizer_G)
        self.lr_scheduler_D = lr_scheduler(optimizer=self.optimizer_D)
        self.lr_schedulers = dict(G=self.lr_scheduler_G,
                                  D=self.lr_scheduler_D)

    def set_input(self, data):
        # A: rgb domain
        # B: depth domain
        self.real_A, self.real_B = data['rgb'].to(self.device), data['gt_depth'].to(self.device)
        self.corrupted_B = data['raw_depth'].to(self.device)
        if 'depth_mask' in data:
            self.mask = data['depth_mask'].to(self.device)
        else:
            self.mask = torch.ones_like(self.real_B).to(self.device)

        self.image_loss_weight = self.mask / (self.mask.sum() + 1e-6)

    def forward_test(self, **kwargs):
        rgb = kwargs['rgb'].to(self.device)
        raw_depth = kwargs['raw_depth'].to(self.device)
        ret = self.G(rgb, raw_depth)

        return ret

    def forward(self):
        ret = self.G(self.real_A, self.corrupted_B)
        self.fake_B_rgb_branch, self.conf_map_rgb_branch = ret['depth_map_1'], ret['confidence_map_1']
        self.fake_B_depth_branch, self.conf_map_depth_branch = ret['depth_map_2'], ret['confidence_map_2']
        self.final_depth = ret['pred_depth']

        # self.fake_B_rgb_branch, \
        # self.conf_map_rgb_branch, \
        # self.fake_B_depth_branch, \
        # self.conf_map_depth_branch, \
        # self.final_depth = self.G(self.real_A, self.corrupted_B)

    def wgangp_case(self):
        # I didn't implement gardient-penalty in GANLoss helper class
        # DistributedDataParallel not work with torch.autograd.grad()
        real_img, fake_img = self.real_B, self.fake_B_rgb_branch
        eps = torch.rand(real_img.shape[0], 1, 1, 1).to(self.device)
        x_hat = eps * real_img.data + (1 - eps) * fake_img.data
        x_hat.requires_grad = True
        hat_predict = self.D(x_hat)
        grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
        ).mean()
        grad_penalty = 10 * grad_penalty  # the lambda coefficient equals to 10 in wgan-gp
        grad_penalty.backward()

        return {'grad_penalty': grad_penalty}

    def wgan_case(self):
        for param in self.D.parameters():
            param.data.clamp_(-self.args.wgan_clip_value, self.args.wgan_clip_value)

    def backward_D(self):
        fake_AB = self.fake_B_rgb_branch
        # fake_AB = torch.cat([self.real_A, self.fake_B_rgb_branch], dim=1)
        pred_fake = self.D(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = self.real_B
        # real_AB = torch.cat([self.real_A, self.real_B], dim=1)
        pred_real = self.D(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()

        ret = dict(loss_D=loss_D,
                   loss_D_real=loss_D_real,
                   loss_D_fake=loss_D_fake)

        if self.args.gan_loss_type == 'wgan':
            self.wgan_case()
        elif self.args.gan_loss_type == 'wgangp':
            ret.update(self.wgangp_case())

        return ret

    def backward_G(self):
        lambda_l1_rgb_branch = self.args.lambda_l1_rgb_branch
        lambda_l1_depth_branch = self.args.lambda_l1_depth_branch
        lambda_l1_fusion = self.args.lambda_l1_fusion

        fake_AB = self.fake_B_rgb_branch
        # fake_AB = torch.cat([self.real_A, self.fake_B_rgb_branch], dim=1)
        pred_fake = self.D(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # l1 loss term
        loss_L1_rgb_branch = self.criterionL1(self.fake_B_rgb_branch,
                                              self.real_B,
                                              weight=self.image_loss_weight) * lambda_l1_rgb_branch
        loss_L1_depth_branch = self.criterionL1(self.fake_B_depth_branch,
                                                self.real_B,
                                                weight=self.image_loss_weight) * lambda_l1_depth_branch
        loss_L1_fusion = self.criterionL1(self.final_depth,
                                          self.real_B,
                                          weight=self.image_loss_weight) * lambda_l1_fusion

        # DO NOT FORGET BACKPROPAGATION
        loss_G = loss_G_GAN + loss_L1_rgb_branch + loss_L1_depth_branch + loss_L1_fusion
        loss_G.backward()

        return dict(loss_G_GAN=loss_G_GAN,
                    loss_L1_rgb_branch=loss_L1_rgb_branch,
                    loss_L1_depth_branch=loss_L1_depth_branch,
                    loss_L1_fusion=loss_L1_fusion)

    def optimize_parameters(self):
        loss_stats = {}
        self.forward()
        # update D
        self.set_requires_grad(self.D, requires_grad=True)
        self.optimizer_D.zero_grad()
        loss_stats.update(self.backward_D())
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.D, requires_grad=False)
        self.optimizer_G.zero_grad()
        loss_stats.update(self.backward_G())
        self.optimizer_G.step()

        if self.distributed:
            torch.distributed.barrier()

        return self.reduce_loss(loss_stats)

    def _save_samples(self, real_A, corrupted_B, real_B, batch_size=16, savename_prefix=None):
        self.G.eval()
        with torch.no_grad():
            ret = self.G(real_A, corrupted_B)
            fake_B_rgb_branch, conf_map_rgb_branch = ret['depth_map_1'], ret['confidence_map_1']
            fake_B_depth_branch, conf_map_depth_branch = ret['depth_map_2'], ret['confidence_map_2']
            final_depth = ret['pred_depth']
        self.G.train()

        # device to host & colored
        real_A_colored = vis.to_data(real_A.cpu())
        corrupted_B_colored = vis.to_data(corrupted_B.cpu())
        fake_B_rgb_branch_colored = vis.to_data(fake_B_rgb_branch.cpu())
        fake_B_depth_branch_colored = vis.to_data(fake_B_depth_branch.cpu())
        final_depth_colored = vis.to_data(final_depth.cpu())
        real_B_colored = vis.to_data(real_B.cpu())
        merged = vis.merge_images([real_A_colored, corrupted_B_colored, fake_B_depth_branch_colored,
                                   fake_B_rgb_branch_colored, final_depth_colored, real_B_colored],
                                  num_imgs_per_scene=6, batch_size=batch_size)
        imageio.imsave(f"{savename_prefix}.jpg", merged.astype(np.uint8))

        # FIXME: save confidence map
        # conf_map_rgb_branch = conf_map_rgb_branch.cpu()
        # conf_map_depth_branch = conf_map_depth_branch.cpu()
        torch.cuda.empty_cache()

    def save_samples(self, save_dir, iters):
        self._save_samples(self.fixed_testing_samples['rgb'].to(self.device),
                           self.fixed_testing_samples['raw_depth'].to(self.device),
                           self.fixed_testing_samples['gt_depth'].to(self.device),
                           batch_size=16,
                           savename_prefix=os.path.join(save_dir, 'sample-test-{:06d}'.format(iters)))

        self._save_samples(self.fixed_training_samples['rgb'].to(self.device),
                           self.fixed_training_samples['raw_depth'].to(self.device),
                           self.fixed_training_samples['gt_depth'].to(self.device),
                           batch_size=16,
                           savename_prefix=os.path.join(save_dir, 'sample-train-{:06d}'.format(iters)))

    def generator(self):

        return self.G
