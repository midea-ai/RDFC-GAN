"""
    Author: Wang mingyuan
    Description: Base class for GAN-based DC.
"""

import functools
from abc import ABCMeta, abstractmethod

import torch
import torch.distributed as dist
import torch.optim as optim
from lib.utils.checkpoint import load_checkpoint, resume_from, save_checkpoint
from torch.optim import lr_scheduler


class Base:
    def __init__(self,
                 device,
                 distributed,
                 args,
                 is_train=True):
        self.device = device
        self.distributed = distributed
        self.local_rank = args.local_rank
        self.args = args
        self.is_train = is_train

    @abstractmethod 
    def set_input(self, data):
        pass

    @abstractmethod
    def init_model(self):
        pass 

    def get_optimizer(self):
        optimizer_type = self.args.optimizer.lower()
        
        if optimizer_type == 'adam':
            return functools.partial(optim.Adam, lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        elif optimizer_type == 'sgd':
            return functools.partial(optim.SGD, lr=self.args.lr)
        elif optimizer_type == 'rmsprop':
            return functools.partial(optim.RMSprop, lr=self.args.lr)
        else:
            raise NotImplementedError(f'Only Adam, SGD, RMSprop optimizers are supported, but got {optimizer_type}')

    def init_optimizer(self):
        pass

    def get_lr_scheduler(self):
        scheduler_type = self.args.scheduler.lower()
        if scheduler_type == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + self.args.epoch - self.args.decay_epoch) / (
                            self.args.n_epochs - self.args.decay_epoch)

                return lr_l

            return functools.partial(lr_scheduler.LambdaLR, lr_lambda=lambda_rule)
        elif scheduler_type == 'step':

            return functools.partial(lr_scheduler.MultiStepLR, milestones=self.args.lr_decay_epochs, gamma=self.args.lr_decay_rate)
            # return functools.partial(lr_scheduler.StepLR, step_size=self.args.step_size, gamma=self.args.step_gamma)
        elif scheduler_type == 'cosine':

            return functools.partial(lr_scheduler.CosineAnnealingLR, T_max=self.args.n_epochs, eta_min=0)
        else:
            raise NotImplementedError

    def init_lr_scheduler(self):
        pass

    def load_from(self, load_path):
        # logger.log(f'load checkpoint from {load_path}')
        checkpoint = load_checkpoint(model=self.models,
                                     filename=load_path,
                                     map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
        print(checkpoint['meta']['epoch'])

    def resume_from(self, resume_path):
        start_epoch = resume_from(models=self.models,
                                  optimizers=self.optimizers,
                                  lr_schedulers=self.lr_schedulers,
                                  filename=resume_path)
        # assert self.args.epoch == start_epoch
        # self.args.epoch = start_epoch       # necessary for lambdaLR

        return start_epoch

    def save_ckpt(self, filename, meta=None):
        save_checkpoint(module=self.models,
                        filename=filename,
                        optimizer=self.optimizers,
                        lr_scheduler=self.lr_schedulers,
                        meta=meta)

    @abstractmethod
    def save_samples(self, save_dir, iters):
        pass

    def eval(self):
        """Set models to eval mode."""
        for _, model in self.models.items():
            model.eval()

    def train(self):
        """Set models to training mode."""
        for _, model in self.models.items():
            model.train()

    def set_requires_grad(self, models, requires_grad=False):
        if not isinstance(models, list):
            models = [models]

        for m in models:
            if m is not None:
                for param in m.parameters():
                    param.requires_grad = requires_grad

    def reduce_loss(self, loss):
        reduced_loss = dict()
        if dist.is_available() and dist.is_initialized():
            for k, v in loss.items():
                v = v.data.clone()
                dist.all_reduce(v.div_(dist.get_world_size()))
                reduced_loss[k] = v.item()

        else:
            for k, v in loss.items():
                reduced_loss[k] = float(v)
        return reduced_loss

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        # old_lr = self.optimizers[0].param_groups[0]['lr']
        for _, scheduler in self.lr_schedulers.items():
                scheduler.step()

        # lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate %.7f -> %.7f' % (old_lr, lr))

    @abstractmethod
    def generator(self):
        """The G that used to generate target domain"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    @abstractmethod
    def forward_test(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        return self.forward_test(**kwargs)
