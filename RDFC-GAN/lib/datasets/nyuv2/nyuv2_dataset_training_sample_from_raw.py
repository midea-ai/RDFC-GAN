import os
import pickle

import cv2
import imageio
import lib.utils.save_vis as vis
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from ..base import BaseDataset
from ..preprocessing import CutOffBlackBorder


class NYUV21400Dataset(BaseDataset):
    def __init__(self,
                 data_root,
                 mode='train',
                 preprocessor=None,
                 rgb_mean=[0.485, 0.456, 0.406],
                 rgb_std=[0.229, 0.224, 0.225],
                 max_depth=10.0,
                 depth_mean=[5.0],
                 depth_std=[5.0],
                 num_sample=500  # number of sparse sample
                 ):
        super(NYUV21400Dataset, self).__init__(
            data_root=data_root,
            mode=mode,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            max_depth=max_depth,
            depth_mean=depth_mean,
            depth_std=depth_std,
            )
        

        height, width = (480 - 12, 640 - 15)
        #crop_size = (228, 304)
        crop_size = (256, 256)
        self.height = height
        self.width = width
        self.crop_size = crop_size

        self.num_sample = num_sample
        self.rgb, self.raw_depth,self.gt_depth, self.gt_normal ,self.labels = self.load_file()

        assert len(self.rgb) == len(self.raw_depth)
        assert len(self.rgb) == len(self.gt_depth)
        assert len(self.rgb) == len(self.gt_normal)

    def get_train_data(self, idx):
        # load rgb image

        rgb = cv2.imread(os.path.join(self.data_root, self.rgb[idx]), cv2.IMREAD_UNCHANGED)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        #load normal image
        gt_normal = cv2.imread(os.path.join(self.data_root,self.gt_normal[idx]),cv2.IMREAD_UNCHANGED)
        gt_normal = cv2.cvtColor(gt_normal,cv2.COLOR_BGR2RGB)

        # load raw depth image
        raw_depth = cv2.imread(os.path.join(self.data_root, self.raw_depth[idx]), cv2.IMREAD_UNCHANGED).astype(np.float32)

        # load gt depth image
        gt_depth = cv2.imread(os.path.join(self.data_root, self.gt_depth[idx]), cv2.IMREAD_UNCHANGED).astype(np.float32)

        #load labels image
        labels = cv2.imread(os.path.join(self.data_root,self.labels[idx]),cv2.IMREAD_UNCHANGED)

        raw_depth = raw_depth / 1000.0
        gt_depth = gt_depth / 1000.0

        cut_off_black_border = CutOffBlackBorder()
        rgb = cut_off_black_border(rgb)
        raw_depth = cut_off_black_border(raw_depth)
        gt_depth = cut_off_black_border(gt_depth)
        gt_normal = cut_off_black_border(gt_normal)
        labels = cut_off_black_border(labels)

        #zero_num_1 = np.sum(labels == 0)

        rgb = Image.fromarray(rgb, mode='RGB')
        raw_depth = Image.fromarray(raw_depth, mode='F')
        gt_depth =  Image.fromarray(gt_depth, mode='F')
        gt_normal = Image.fromarray(gt_normal,mode="RGB")
        labels = Image.fromarray(labels,mode='L')


        degree = np.random.uniform(-5.0, 5.0)

        #flip = np.random.uniform(0.0, 1.0)


        rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)
        raw_depth = TF.rotate(raw_depth, angle=degree, resample=Image.NEAREST)
        gt_depth = TF.rotate(gt_depth, angle=degree, resample=Image.NEAREST)
        gt_normal = TF.rotate(gt_normal,angle=degree, resample=Image.NEAREST)
        labels = TF.rotate(labels,angle=degree,resample=Image.NEAREST)



        t_rgb = T.Compose([T.Resize(self.crop_size),
                           #T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
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

        rgb = T.Normalize(self.rgb_mean, self.rgb_std)(rgb)
        gt_normal = T.Normalize(0.5,0.5)(gt_normal)
        

        raw_depth = t_depth(raw_depth)
        gt_depth = t_depth(gt_depth)

        #sample 500 from raw depth
        depth_sp = self.get_sparse_depth(raw_depth,self.num_sample)

        depth_valid_mask = gt_depth > 0.0001
        gt_depth = T.Normalize(self.depth_mean,self.depth_std)(gt_depth)

        raw_depth_zero_mask = depth_sp==0.0
        depth_sp = T.Normalize(self.depth_mean,self.depth_std)(depth_sp)
        depth_sp[raw_depth_zero_mask] = 0.0
        

        #depth_sp = self.get_sparse_depth(raw_depth, self.num_sample)

        #raw_depth_zero_masks = depth_sp == 0
        #gt_depth = T.Normalize(self.depth_mean,self.depth_std)(gt_depth)
        #raw_depth = T.Normalize(self.depth_mean,self.depth_std)(depth_sp)

        #raw_depth[raw_depth_zero_masks] = 0



        sample = {'rgb': rgb,
                  'raw_depth': depth_sp,
                  'gt_depth': gt_depth,
                  'depth_masks': depth_valid_mask,
                  'gt_normal':gt_normal,
                  'normal_masks':norm_valid_mask,
                  'labels':labels
                  }

        return sample


    def load_pkl_file(self, filename):
        with open(filename, 'rb') as f:
            ret = pickle.load(f)
        return ret


    def get_test_data(self, idx):
        sample = {}
        # load rgb image
        rgb = cv2.imread(os.path.join(self.data_root, self.rgb[idx]), cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        #load gt normal
        gt_normal = cv2.imread(os.path.join(self.data_root,self.gt_normal[idx]),cv2.IMREAD_UNCHANGED)
        gt_normal = cv2.cvtColor(gt_normal,cv2.COLOR_BGR2RGB)

        # load raw depth image
        raw_depth = cv2.imread(os.path.join(self.data_root, self.raw_depth[idx]), cv2.IMREAD_UNCHANGED).astype(
            np.float32)

        # load gt depth image
        gt_depth = cv2.imread(os.path.join(self.data_root, self.gt_depth[idx]), cv2.IMREAD_UNCHANGED).astype(np.float32)

        labels = cv2.imread(os.path.join(self.data_root,self.labels[idx]),cv2.IMREAD_UNCHANGED)

        raw_depth = raw_depth / 1000.0
        gt_depth = gt_depth / 1000.0

        cut_off_black_border = CutOffBlackBorder()

        rgb = cut_off_black_border(rgb)
        raw_depth = cut_off_black_border(raw_depth)
        gt_depth = cut_off_black_border(gt_depth)
        gt_normal = cut_off_black_border(gt_normal)
        labels = cut_off_black_border(labels)

        rgb = Image.fromarray(rgb, mode='RGB')
        raw_depth = Image.fromarray(raw_depth, mode='F')
        gt_depth = Image.fromarray(gt_depth, mode='F')
        gt_normal = Image.fromarray(gt_normal,mode='RGB')
        labels = Image.fromarray(labels, mode = 'L')

        t_rgb = T.Compose([
            T.Resize(self.crop_size),
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
        #sample 500 from raw depth
        depth_sp = self.get_sparse_depth(raw_depth, self.num_sample)
        raw_depth_zero_masks = depth_sp == 0

        gt_depth = T.Normalize(self.depth_mean,self.depth_std)(gt_depth)
        depth_sp = T.Normalize(self.depth_mean,self.depth_std)(depth_sp)
        depth_sp[raw_depth_zero_masks] = 0
        # sample['gt_depth_origin'] = gt_depth.numpy().copy().squeeze()


        # depth_sp = self.get_sparse_depth(raw_depth, self.num_sample)
        # raw_depth_zero_masks = depth_sp == 0
        # raw_depth = T.Normalize(self.depth_mean,self.depth_std)(depth_sp)
        # raw_depth[raw_depth_zero_masks] = 0

        sample.update({'rgb': rgb,
                       'raw_depth': depth_sp,
                        'gt_depth': gt_depth,
                       'depth_masks': valid_mask,
                       'normal_masks':norm_valid_mask,
                       'gt_normal':gt_normal,
                       'labels':labels
                       })


        return sample



    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        return dep_sp

    def load_image(self, idx):
        pass

    def load_depth(self, idx):
        pass

    def load_image_path_to_list(self, filename):
        with open(filename, 'r') as f:
            file_list = f.read().splitlines()

        return file_list

    def load_file(self):
        if self.mode == 'train':
            filename_prefix = 'train'
            indices_file = os.path.join(self.data_root, 'train.txt')
        else:
            filename_prefix = 'test'
            indices_file = os.path.join(self.data_root, 'test.txt')

        indices = self.load_image_path_to_list(indices_file)

        rgb_file_list = [f'{filename_prefix}/rgb/{idx}.png' for idx in indices]
        raw_depth_file_list = [f'{filename_prefix}/depth/{idx}.png' for idx in indices]
        depth_file_list = [f'{filename_prefix}/depth/{idx}.png' for idx in indices]
        
        normal_file_list = [f'{filename_prefix}/norm_v2/{idx}.png' for idx in indices]
        labels_file_list = [f'{filename_prefix}/labels_13/{idx}.png' for idx in indices]
        return rgb_file_list, raw_depth_file_list, depth_file_list,normal_file_list,labels_file_list


    def stat_depth(self):
        """Statistical ground truth depth map

            795/795
            min depth: 713, max depth: 9995
            mean: [2841.94941273], std: [993.28365066]

            val_dataset:654/654
            min depth: 713, max depth: 9986
            mean: [2739.73168995], std: [971.36949092]
        """
        _min, _max = np.inf, -np.inf

        mean, std = np.zeros(1), np.zeros(1)
        for i in range(len(self)):
            file = os.path.join(self.data_root,
                                self.gt_depth[i])
            depth = cv2.imread(file, cv2.IMREAD_UNCHANGED).astype(np.float32)

            depth = depth / 1000.0

            cur_min, cur_max = depth.min(), depth.max()
            if cur_min < _min:
                _min = cur_min
            if cur_max > _max:
                _max = cur_max

            mean[0] += depth.mean()
            std[0] += depth.std()

            print(f'\r{i + 1}/{len(self)}', end='')

        mean /= len(self)
        std /= len(self)

        print('\nmin depth: {}, max depth: {}'.format(_min, _max))
        print(f'mean: {mean}, std: {std}')

    def __len__(self):

        return len(self.rgb)

