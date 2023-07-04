import os
from abc import ABCMeta, abstractmethod

import numpy as np
from torch.utils.data import Dataset
from lib.dataset.transforms import Compose
import collections
import torchvision
import warnings
import torch
import matplotlib.pyplot as plt
from PIL import Image
from lib.metrics.rdf_gan_metric import RDFGANMetric


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 data_root,
                 mode='train',
                 rgb_mean=[0.485, 0.456, 0.406],
                 rgb_std=[0.229, 0.224, 0.225],
                 max_depth=10.0,
                 depth_mean=[5.0],
                 depth_std=[5.0],
                 preprocessor=None):
        self.data_root = data_root
        self.mode = mode
        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        if isinstance(preprocessor, collections.abc.Sequence):
            warnings.warn(f'Generate preprocessor from config will be deprecated in the future.')
            # Generate transform from config
            self.preprocessor = Compose(preprocessor)
        elif isinstance(preprocessor, torchvision.transforms.transforms.Compose):
            self.preprocessor = preprocessor
        else:
            self.preprocessor = lambda x: x

        # before performing normalization, rgb values in the range [0., 1.]
        self._rgb_mean = rgb_mean
        self._rgb_mean_tensor = torch.tensor(rgb_mean).view(1, 3, 1, 1)
        self._rgb_std = rgb_std
        self._rgb_std_tensor = torch.tensor(rgb_std).view(1, 3, 1, 1)

        # before performing normalization, depth values in the range [0., max_depth]
        self._max_depth = max_depth
        self._depth_mean_tensor = torch.tensor(depth_mean).view(1, 1, 1, 1)
        self._depth_std_tensor = torch.tensor(depth_std).view(1, 1, 1, 1)
        self._depth_mean = depth_mean
        self._depth_std = depth_std


    @property
    def rgb_mean(self):
        # tensor
        return self._rgb_mean

    @property
    def rgb_std(self):
        return self._rgb_std

    @property
    def depth_mean(self):
        return self._depth_mean

    @property
    def depth_std(self):
        return self._depth_std

    @property
    def max_depth(self):
        return self._max_depth

    @abstractmethod
    def __len__(self):
        pass

    def __getitem__(self, idx):
        if self.mode == 'val' or self.mode == 'test':
            return self.get_test_data(idx)
        else:
            while True:
                data = self.get_train_data(idx)
                if data is None:
                    idx = self.get_another_id()
                    continue
                return data

    @abstractmethod
    def get_train_data(self, idx):
        pass

    @abstractmethod
    def get_test_data(self, idx):
        pass

    def get_another_id(self):
        return np.random.randint(0, len(self))

    @abstractmethod
    def load_file(self):
        """Load the rgb & depth image file path"""
        pass

    @abstractmethod
    def load_image(self, idx):
        pass

    @abstractmethod
    def load_depth(self, idx):
        pass

    # A workaround for a pytorch bug
    # https://github.com/pytorch/vision/issues/2194
    class ToNumpy:
        # used in NYUDepthv2 dataset
        def __call__(self, sample):
            return np.array(sample)

    def evaluate(self,
                 results,
                 logger=None):

        METRIC = RDFGANMetric()
        METRIC.evaluate_all(results, logger)
        del METRIC

    def show(self, samples, iters=None, save_dir=None, max_show_num=6):
        """use uint8 type to save images"""

        output_dir = os.path.join(save_dir, f'{iters:010d}')
        os.makedirs(output_dir, exist_ok=True)

        rgb = samples['rgb'].detach()
        raw_depth = samples['raw_depth']
        pred_depth = samples['pred_depth'].detach()
        gt_depth = samples['gt_depth'].detach()

        # Un-normalization
        rgb = rgb.mul(self._rgb_std_tensor.type_as(rgb)).add(self._rgb_mean_tensor.type_as(rgb))
        raw_depth = raw_depth.mul(self._depth_std_tensor.type_as(raw_depth)).add(self._depth_mean_tensor.type_as(raw_depth))

        # a werid bug, pred_depth and gt_depth have different memory address, but change one affects another
        # pred_depth.mul_(self._depth_std_tensor.type_as(pred_depth)).add_(self._depth_mean_tensor.type_as(pred_depth))
        # gt_depth.mul_(self._depth_std_tensor.type_as(gt_depth)).add_(self._depth_mean_tensor.type_as(gt_depth))

        pred_depth = pred_depth.mul(self._depth_std_tensor.type_as(pred_depth)).add(self._depth_mean_tensor.type_as(pred_depth))
        gt_depth = gt_depth.mul(self._depth_std_tensor.type_as(gt_depth)).add(self._depth_mean_tensor.type_as(gt_depth))

        rgb = 255.0 * np.transpose(rgb, (0, 2, 3, 1))  # (b, h, w, c)  real color
        raw_depth = raw_depth / self.max_depth
        pred_depth = pred_depth / self.max_depth
        pred_gray = pred_depth
        gt_depth = gt_depth / self.max_depth

        batch_size = min(rgb.shape[0], max_show_num)

        cm = plt.get_cmap('plasma')

        for i in range(batch_size):
            _rgb = rgb[i, :, :, :].data.cpu().numpy()  # (h, w, 3)
            raw_dep = raw_depth[i, 0, :, :].data.cpu().numpy()  # (h, w)
            pred_dep = pred_depth[i, 0, :, :].data.cpu().numpy()  # (h, w)
            gt_dep = gt_depth[i, 0, :, :].data.cpu().numpy()  # (h, w)
            _pred_gray = pred_gray[i, 0, :, :].data.cpu().numpy()  # (h, w)

            # painting
            _rgb = _rgb.astype(np.uint8)
            raw_dep = (255.0 * cm(raw_dep)).astype(np.uint8)
            pred_dep = (255.0 * cm(pred_dep)).astype(np.uint8)
            _pred_gray = (255.0 * _pred_gray).astype(np.uint8)
            gt_dep = (255.0 * cm(gt_dep)).astype(np.uint8)

            _rgb = Image.fromarray(_rgb, 'RGB')
            raw_dep = Image.fromarray(raw_dep[:, :, :3], 'RGB')
            pred_dep = Image.fromarray(pred_dep[:, :, :3], 'RGB')
            gt_dep = Image.fromarray(gt_dep[:, :, :3], 'RGB')
            _pred_gray = Image.fromarray(_pred_gray)

            _rgb.save('{}/{}_rgb.png'.format(output_dir, i))
            raw_dep.save('{}/{}_raw_depth.png'.format(output_dir, i))
            pred_dep.save('{}/{}_pred_depth.png'.format(output_dir, i))
            gt_dep.save('{}/{}_gt_depth.png'.format(output_dir, i))
            _pred_gray.save('{}/{}_pred_gray.png'.format(output_dir, i))

        del rgb, raw_depth, pred_depth, gt_depth, pred_gray
