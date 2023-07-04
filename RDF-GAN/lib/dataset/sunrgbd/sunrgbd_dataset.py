import os
from .sunrgbd_base import SUNRGBDBase
from ..base import BaseDataset
import pickle
import torchvision.transforms as T
from ..pseudo_hole import (MaskBlacks, SegmentationHighLight, DeterministicPseudoHole,
                           Spatter)
from ..preprocessing import (RandomRescale, RandomCrop, RandomFlip,
                             ToTensor, Normalize, Rescale)
import cv2
import numpy as np
import random


class SUNRGBDPseudoDataset(SUNRGBDBase, BaseDataset):
    def __init__(self,
                 data_root,
                 mode='train',
                 preprocessor=None,
                 rgb_mean=[0.485, 0.456, 0.406],
                 rgb_std=[0.229, 0.224, 0.225],
                 max_depth=10.0,
                 depth_mean=[5.0],
                 depth_std=[5.0],
                 ):
        super(SUNRGBDPseudoDataset, self).__init__(
            data_root=data_root,
            mode=mode,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            max_depth=max_depth,
            depth_mean=depth_mean,
            depth_std=depth_std,
            preprocessor=preprocessor)

        assert mode in self.MODES, \
            f'parameter mode must be one of {self.MODES}, but got {mode}'

        # regard raw mode as raw_depth and refined mode as gt_depth by default.
        self.rgb, \
        self.raw_depth, \
        self.bfx_depth, \
        self.label = self.load_file()
        assert len(self.rgb) == len(self.raw_depth)
        assert len(self.rgb) == len(self.bfx_depth)
        assert len(self.rgb) == len(self.label)

        crop_size = (480, 640)
        self.crop_size = crop_size

        # This property is necessary when you need to resize the predicted image to the original size
        self.with_input_origin = True


    def __len__(self):

        if self.mode == 'train':
            return len(self.rgb)
        else:
            return len(self.rgb)

    def load_label(self, idx):
        with open(os.path.join(self.data_root, self.label[idx]), 'rb') as f:
            label = pickle.load(f)

        return label

    def get_train_data(self, idx):
        # do not use bfx_depth during training
        rgb, image_path = self.load_image(idx)  # (h, w, 3)
        raw_depth, gt_depth = self.load_depth(idx)
        label = self.load_label(idx)

        # -------------- perform pseudo operations -------------- #
        masks = []
        pseudo_sample = {'rgb': rgb, 'raw_depth': raw_depth, 'label': label}
        prop = np.random.uniform(0.0, 1.0)
        if prop > 0.3:
            t = SegmentationHighLight()
            masks.append(t(sample=pseudo_sample))

        prop = np.random.uniform(0.0, 1.0)
        if prop > 0.3:
            t = Spatter()
            masks.append(t(sample=pseudo_sample))

        prop = np.random.uniform(0.0, 1.0)
        if prop > 0.3:
            t = MaskBlacks()
            masks.append(t(sample=pseudo_sample))

        prop = np.random.uniform(0.0, 1.0)
        if prop > 0.5:
            t = DeterministicPseudoHole(label=random.sample([30, 19, 9, 25], 2),
                                        random_filtered_label_nums=2)
            masks.append(t(sample=pseudo_sample))

        # combine all pseudo masks
        pseudo_masks = np.zeros_like(raw_depth, dtype=bool)
        for m in masks:
            pseudo_masks |= m
        pseudo_depth = raw_depth.copy()
        pseudo_depth[pseudo_masks] = 0.0
        # -------------- perform pseudo operations -------------- #

        sample = {
            'rgb': rgb,
            'raw_depth': pseudo_depth,
            'gt_depth': raw_depth,
            'depth_masks': []
        }

        t = T.Compose([
            RandomRescale([1.0, 1.4]),
            RandomCrop(crop_height=self.crop_size[0],
                       crop_width=self.crop_size[1]),
            RandomFlip(),
            ToTensor(),
            Normalize(depth_mean=self.depth_mean,
                      depth_std=self.depth_std,
                      rgb_mean=self.rgb_mean,
                      rgb_std=self.rgb_std)
        ])

        sample = t(sample)

        return sample


    def get_test_data(self, idx):
        rgb, image_path = self.load_image(idx)  # (h, w, 3)

        raw_depth, gt_depth = self.load_depth(idx)

        origin_h, origin_w = gt_depth.shape[:2]


        sample = {
            'rgb': rgb,
            'raw_depth': raw_depth,
            'gt_depth': gt_depth,
            'depth_masks': []
        }

        t = T.Compose([
            Rescale(height=self.crop_size[0],
                    width=self.crop_size[1]),
            ToTensor(),
            Normalize(depth_mean=self.depth_mean,
                      depth_std=self.depth_std,
                      rgb_mean=self.rgb_mean,
                      rgb_std=self.rgb_std)
        ])

        sample = t(sample)

        if self.with_input_origin:
            sample.update({'origin_h': str(origin_h),
                        'origin_w': str(origin_w),
                        'real_idx': self.label[idx].split(os.sep)[-1].split('.')[0]})

        return sample


    def load_image(self, idx):
        file = os.path.join(self.data_root, self.rgb[idx])
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32), file

    def get_depthInpaint(self, depth):
        """get real depth"""
        depthInpaint = np.bitwise_or(np.right_shift(depth, 3), np.left_shift(depth, 16 - 3))
        depthInpaint = depthInpaint.astype(np.float32) / 1000
        # depthInpaint[np.where(depthInpaint > 8)] = 8

        return depthInpaint

    def load_depth(self, idx):
            # depth as raw, depth_bfx as gt
        depth_file = os.path.join(self.data_root, self.raw_depth[idx])
        depth_bfx_file = os.path.join(self.data_root, self.bfx_depth[idx])
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        depth_bfx = cv2.imread(depth_bfx_file, cv2.IMREAD_UNCHANGED)

        return self.get_depthInpaint(depth), self.get_depthInpaint(depth_bfx)

    def load_image_path_to_list(self, filename):
        with open(filename, 'r') as f:
            file_list = f.read().splitlines()

        return file_list

    def load_file(self):
        file_prefix = 'train_' if self.mode == 'train' else 'test_'
        # file_prefix = 'train_'

        rgb_image_file_path = os.path.join(self.data_root, f'{file_prefix}rgb.txt')
        depth_image_file_path = os.path.join(self.data_root, f'{file_prefix}depth.txt')
        depth_bfx_image_file_path = os.path.join(self.data_root, f'{file_prefix}depth_bfx.txt')

        label_file_path = os.path.join(self.data_root, f'{file_prefix}seg_label.txt')

        rgb_image_file_list = self.load_image_path_to_list(rgb_image_file_path)
        depth_image_file_list = self.load_image_path_to_list(depth_image_file_path)
        depth_bfx_image_file_list = self.load_image_path_to_list(depth_bfx_image_file_path)
        label_file_list = self.load_image_path_to_list(label_file_path)

        return rgb_image_file_list, depth_image_file_list, depth_bfx_image_file_list, label_file_list


    def show(self, samples, iters=None, save_dir=None):
        pass
