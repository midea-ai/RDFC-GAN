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
from PIL import Image
import torchvision.transforms.functional as TF
import torch


class SUNRGBDPseudoDataset(SUNRGBDBase, BaseDataset):
    def __init__(self,
                 data_root,
                 mode='train',
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
            depth_std=depth_std)

        assert mode in self.MODES, \
            f'parameter mode must be one of {self.MODES}, but got {mode}'

        # regard raw mode as raw_depth and refined mode as gt_depth by default.
        self.rgb, \
        self.raw_depth, \
        self.bfx_depth, \
        self.label,\
        self.gt_normal = self.load_file()
        assert len(self.rgb) == len(self.raw_depth)
        assert len(self.rgb) == len(self.bfx_depth)
        assert len(self.rgb) == len(self.label)
        assert len(self.rgb) == len(self.gt_normal)

        crop_size = (256, 256)
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
        gt_normal,_ = self.load_normal(idx)
        raw_depth, gt_depth = self.load_depth(idx)
        labels = self.load_label(idx)
        depth_valid_mask = gt_depth > 0.0001

        # sample = {
        #     'rgb': rgb,
        #     'raw_depth': raw_depth,
        #     'gt_depth': gt_depth,
        #     'labels': label,
        #     'depth_masks': depth_valid_mask,
        #     'gt_normal' : gt_normal
        # }

        # t = T.Compose([
        #     RandomRescale([0.4, 0.6]),
        #     RandomCrop(crop_height=self.crop_size[0],
        #                crop_width=self.crop_size[1]),
        #     ToTensor(),
        #     Normalize(depth_mean=self.depth_mean,
        #               depth_std=self.depth_std,
        #               rgb_mean=self.rgb_mean,
        #               rgb_std=self.rgb_std)
        # ])

        # sample = t(sample)

        rgb = Image.fromarray(rgb, mode='RGB')
        raw_depth = Image.fromarray(raw_depth, mode='F')
        gt_depth =  Image.fromarray(gt_depth, mode='F')
        gt_normal = Image.fromarray(gt_normal,mode="RGB")
        labels = Image.fromarray(labels,mode='L')

        degree = np.random.uniform(-5.0, 5.0)

        rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)
        raw_depth = TF.rotate(raw_depth, angle=degree, resample=Image.NEAREST)
        gt_depth = TF.rotate(gt_depth, angle=degree, resample=Image.NEAREST)
        gt_normal = TF.rotate(gt_normal,angle=degree, resample=Image.NEAREST)
        labels = TF.rotate(labels,angle=degree,resample=Image.NEAREST)

        t_rgb = T.Compose([T.Resize(self.crop_size),
                           self.ToNumpy(),
                           T.ToTensor(),
                           ])
        t_depth = T.Compose([T.Resize(self.crop_size),
                             self.ToNumpy(),
                             T.ToTensor(),
                             ])
        t_labels = T.Compose([T.Resize(self.crop_size,interpolation=0),
                            self.ToNumpy()])
        
        rgb = t_rgb(rgb)
        gt_normal = t_rgb(gt_normal)

        norm_valid_mask = (gt_normal[0,: ,:] > 0) | (gt_normal[1,: ,:] > 0) | (gt_normal[2,:,:] > 0)
        norm_valid_mask = norm_valid_mask.squeeze(1)

        labels = t_labels(labels)
        labels = torch.from_numpy(labels).long()

        raw_depth = t_depth(raw_depth)
        gt_depth = t_depth(gt_depth)
        
        #labels = torch.from_numpy(np.array(labels, dtype=np.float32)).long()
        #zero_num_5 = (labels == 0).sum()

        rgb = T.Normalize(self.rgb_mean, self.rgb_std)(rgb)
        gt_normal = T.Normalize(0.5,0.5)(gt_normal)

        depth_valid_mask = gt_depth > 0.0001

        gt_depth = T.Normalize(self.depth_mean,self.depth_std)(gt_depth)
        raw_depth = T.Normalize(self.depth_mean,self.depth_std)(raw_depth)

        sample = {'rgb': rgb,
                  'raw_depth': raw_depth,
                  'gt_depth': gt_depth,
                  'depth_masks': depth_valid_mask,
                  'gt_normal':gt_normal,
                  'normal_masks':norm_valid_mask,
                  'labels':labels
                  }
        return sample


    def get_test_data(self, idx):
        rgb, image_path = self.load_image(idx)  # (h, w, 3)

        raw_depth, gt_depth = self.load_depth(idx)

        gt_normal,_ = self.load_normal(idx)
        labels = self.load_label(idx)

        origin_h, origin_w = gt_depth.shape[:2]


        # sample = {
        #     'rgb': rgb,
        #     'raw_depth': raw_depth,
        #     'gt_depth': gt_depth,
        #     'depth_masks': []
        # }

        # t = T.Compose([
        #     Rescale(height=self.crop_size[0],
        #             width=self.crop_size[1]),
        #     ToTensor(),
        #     Normalize(depth_mean=self.depth_mean,
        #               depth_std=self.depth_std,
        #               rgb_mean=self.rgb_mean,
        #               rgb_std=self.rgb_std)
        # ])

        rgb = Image.fromarray(rgb, mode='RGB')
        raw_depth = Image.fromarray(raw_depth, mode='F')
        gt_depth = Image.fromarray(gt_depth, mode='F')
        gt_normal = Image.fromarray(gt_normal,mode='RGB')
        labels = Image.fromarray(labels, mode = 'L')

        t_rgb = T.Compose([
            T.Resize(self.crop_size),
            self.ToNumpy(),
            T.ToTensor(),
            ])
        t_depth = T.Compose([
            T.Resize(self.crop_size),
            self.ToNumpy(),
            T.ToTensor(),
            ])
        t_labels = T.Compose([
                T.Resize(self.crop_size,interpolation=0),
                self.ToNumpy(),])
        
        rgb = t_rgb(rgb)
        gt_normal = t_rgb(gt_normal)

        norm_valid_mask = (gt_normal[0,: ,:] > 0) | (gt_normal[1,: ,:] > 0) | (gt_normal[2,:,:] > 0)
        norm_valid_mask = norm_valid_mask.squeeze(1)

        labels = t_labels(labels)
        labels = torch.from_numpy(labels).long()
        raw_depth = t_depth(raw_depth)
        gt_depth = t_depth(gt_depth)

        rgb = T.Normalize(self.rgb_mean, self.rgb_std)(rgb)
        gt_normal = T.Normalize(0.5, 0.5)(gt_normal)

        valid_mask = gt_depth > 0.0001

        gt_depth = T.Normalize(self.depth_mean,self.depth_std)(gt_depth)
        raw_depth = T.Normalize(self.depth_mean,self.depth_std)(raw_depth)
        sample = {}
        sample.update({'rgb': rgb,
                       'raw_depth': raw_depth,
                        'gt_depth': gt_depth,
                       'depth_masks': valid_mask,
                       'normal_masks':norm_valid_mask,
                       'gt_normal':gt_normal,
                       'labels':labels
                       })

        if self.with_input_origin:
            sample.update({'origin_h': str(origin_h),
                        'origin_w': str(origin_w),
                        'real_idx': self.label[idx].split(os.sep)[-1].split('.')[0]})

        return sample


    def load_image(self, idx):
        file = os.path.join(self.data_root, self.rgb[idx])
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, file

    def load_normal(self, idx):
        file = os.path.join(self.data_root, self.gt_normal[idx])
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, file

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
        normal_file_path = os.path.join(self.data_root, f'{file_prefix}normal.txt')

        rgb_image_file_list = self.load_image_path_to_list(rgb_image_file_path)
        depth_image_file_list = self.load_image_path_to_list(depth_image_file_path)
        depth_bfx_image_file_list = self.load_image_path_to_list(depth_bfx_image_file_path)
        label_file_list = self.load_image_path_to_list(label_file_path)
        normal_file_list = self.load_image_path_to_list(normal_file_path)

        return rgb_image_file_list, depth_image_file_list, depth_bfx_image_file_list, label_file_list, normal_file_list


    def show(self, samples, iters=None, save_dir=None):
        pass
