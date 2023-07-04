import os
from ..base import BaseDataset
import csv
import cv2
import numpy as np
from ..pseudo_hole import (MaskBlacks, Spatter, SegmentationHighLight)
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from ..preprocessing_v2 import (RandomCrop, RandomRescale, ToNumpy, ToTensor,
                                CenterCrop, Rescale)
import torch
import pickle
import matplotlib.pyplot as plt


class DDRNetHumanDataset(BaseDataset):
    def __init__(self,
                 data_root,
                 mode='train',
                 preprocessor=None,
                 rgb_mean=[0.485, 0.456, 0.406],
                 rgb_std=[0.229, 0.224, 0.225],
                 max_depth=3.0,
                 depth_mean=[1.5],
                 depth_std=[1.5],
                 times=1):
        super(DDRNetHumanDataset, self).__init__(
            data_root=data_root,
            mode=mode,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            max_depth=max_depth,
            depth_mean=depth_mean,
            depth_std=depth_std,
            preprocessor=preprocessor)

        crop_size = (480, 640)
        self.crop_size = crop_size
        self.rgb, self.raw_depth, self.gt_depth, self.mask = self.load_file()

        self._origin_len = len(self.rgb)
        self.times = times if self.mode == 'train' else 1

    def __getitem__(self, idx):
        # act_idx = idx % self._origin_len
        if self.mode == 'val' or self.mode == 'test':
            return self.get_test_data(idx % self._origin_len)
        else:
            while True:
                data = self.get_train_data(idx % self._origin_len)
                if data is None:
                    idx = self.get_another_id()
                    continue
                return data

    def __len__(self):
        return self.times * self._origin_len

    def load_file(self):
        csvfile = os.path.join(self.data_root, f'{self.mode}.csv')
        assert os.path.exists(csvfile)

        rgb, raw_depth, gt_depth, mask = [], [], [], []

        with open(csvfile) as f:
            reader = csv.reader(f)

            for row in reader:
                rgb.append(row[0])
                raw_depth.append(row[1])
                gt_depth.append(row[2])
                mask.append(row[3])

        return rgb, raw_depth, gt_depth, mask


    def load_images_from_file(self, idx):
        rgb = cv2.imread(self.rgb[idx], cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        raw_depth = cv2.imread(self.raw_depth[idx], cv2.IMREAD_UNCHANGED)
        gt_depth = cv2.imread(self.gt_depth[idx], cv2.IMREAD_UNCHANGED)
        human_mask = cv2.imread(self.mask[idx], cv2.IMREAD_UNCHANGED)

        return rgb, raw_depth, gt_depth, human_mask

    def get_train_data(self, idx):
        rgb, \
        raw_depth, \
        gt_depth, \
        human_mask = self.load_images_from_file(idx)

        raw_depth = raw_depth / 1000.0
        gt_depth = gt_depth / 1000.0

        #######################################
        valid_region = human_mask > 0.
        valid_rgb = rgb.copy()
        valid_rgb[~valid_region] = 0.
        valid_raw_depth = raw_depth.copy()
        valid_raw_depth[~valid_region] = 0.

        rgb, raw_depth = valid_rgb, valid_raw_depth

        #######################################

        # -------------------- perform pseudo operations -------------------- #
        masks = []
        pseudo_sample = {'rgb': rgb, 'raw_depth': gt_depth}

        prop = np.random.uniform(0.0, 1.0)
        if prop > 0.5:
            t = SegmentationHighLight()
            masks.append(t(sample=pseudo_sample))

        prop = np.random.uniform(0.0, 1.0)
        if prop > 0.5:
            t = MaskBlacks()
            masks.append(t(sample=pseudo_sample))

        prop = np.random.uniform(0.0, 1.0)
        if prop > 0.5:
            t = Spatter()
            masks.append(t(sample=pseudo_sample))

        pseudo_masks = np.zeros_like(raw_depth, dtype=bool)
        for m in masks:
            pseudo_masks |= m

        pseudo_depth = gt_depth.copy()
        pseudo_depth[pseudo_masks] = 0.0
        # -------------------- perform pseudo operations -------------------- #


        rgb = Image.fromarray(rgb, mode='RGB')
        depth = Image.fromarray(gt_depth.astype('float32'), mode='F')
        pseudo_depth = Image.fromarray(pseudo_depth.astype('float32'), mode='F')
        human_mask = Image.fromarray(human_mask.astype('float32'), mode='F')

        # data augment

        # for convenience, do flip & rotate by TF lib
        degree = np.random.uniform(-5.0, 5.0)
        flip = np.random.uniform(0.0, 1.0)

        rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)
        depth = TF.rotate(depth, angle=degree, resample=Image.NEAREST)
        pseudo_depth = TF.rotate(pseudo_depth, angle=degree, resample=Image.NEAREST)
        human_mask = TF.rotate(human_mask, angle=degree, resample=Image.NEAREST)

        if flip > 0.5:
            rgb = TF.hflip(rgb)
            depth = TF.hflip(depth)
            pseudo_depth = TF.hflip(pseudo_depth)
            human_mask = TF.hflip(human_mask)


        sample = {'rgb': rgb,
                  'raw_depth': pseudo_depth,
                  'gt_depth': depth,
                  'human_mask': human_mask
                  }
        t = T.Compose([
            ToNumpy(keys=['rgb', 'raw_depth', 'gt_depth', 'human_mask']),
            RandomRescale([1.0, 1.4], keys=['rgb', 'raw_depth', 'gt_depth', 'human_mask']),
            CenterCrop(crop_height=self.crop_size[0],
                       crop_width=self.crop_size[1],
                       keys=['rgb', 'raw_depth', 'gt_depth', 'human_mask']),
            ToTensor(keys=['rgb', 'raw_depth', 'gt_depth', 'human_mask'])
        ])

        sample = t(sample)

        # apply normalization
        sample['rgb'] = sample['rgb'] / 255.0
        sample['rgb'] = T.Normalize(self.rgb_mean, self.rgb_std)(sample['rgb'])

        # valid_mask = sample['gt_depth'] > 0.0001 & sample['human_mask'] > 0.
        valid_mask = torch.logical_and(sample['gt_depth'] > 0.0001, sample['human_mask'] > 0.)

        raw_depth_zero_masks = sample['raw_depth'] == 0.0
        raw_depth = T.Normalize(self.depth_mean, self.depth_std)(sample['raw_depth'])
        raw_depth[raw_depth_zero_masks] = 0
        sample['raw_depth'] = raw_depth
        sample['gt_depth'] = T.Normalize(self.depth_mean, self.depth_std)(sample['gt_depth'])

        sample.update(
            {'depth_masks': valid_mask}
        )

        return sample

    def get_test_data(self, idx):
        rgb, \
        raw_depth, \
        gt_depth, \
        human_mask = self.load_images_from_file(idx)

        #######################################
        valid_region = human_mask > 0.
        valid_rgb = rgb.copy()
        valid_rgb[~valid_region] = 0.
        valid_raw_depth = raw_depth.copy()
        valid_raw_depth[~valid_region] = 0.

        rgb, raw_depth = valid_rgb, valid_raw_depth

        #######################################

        raw_depth = raw_depth / 1000.0
        gt_depth = gt_depth / 1000.0

        sample = {'rgb': rgb,
                  'raw_depth': raw_depth.astype(np.float32),
                  'gt_depth': gt_depth.astype(np.float32),
                  'human_mask': human_mask}
        t = T.Compose([
            Rescale(height=self.crop_size[0],
                    width=self.crop_size[1],
                    keys=['rgb', 'raw_depth', 'gt_depth', 'human_mask']),
            ToTensor(keys=['rgb', 'raw_depth', 'gt_depth', 'human_mask'])
        ])

        sample = t(sample)

        # apply normalization
        sample['rgb'] = sample['rgb'] / 255.0
        sample['rgb'] = T.Normalize(self.rgb_mean, self.rgb_std)(sample['rgb'])

        valid_mask = torch.logical_and(sample['gt_depth'] > 0.0001, sample['human_mask'] > 0.)

        raw_depth_zero_masks = sample['raw_depth'] == 0.0
        raw_depth = T.Normalize(self.depth_mean, self.depth_std)(sample['raw_depth'])
        raw_depth[raw_depth_zero_masks] = 0
        sample['raw_depth'] = raw_depth
        sample['gt_depth'] = T.Normalize(self.depth_mean, self.depth_std)(sample['gt_depth'])

        sample.update(
            {'depth_masks': valid_mask,
             'evaluate_mask': sample['human_mask'] > 0.
             }
        )

        return sample

    def load_depth(self, idx):
        pass

    def load_image(self, idx):
        pass

    def stat_depth(self):
        """Statistical ground truth depth map & raw depth map

            20/20
            raw:
                min depth: 0.0, max depth: 2000.0
                mean: [238.48206253], std: [526.32961121]
            gt:
                min depth: 0.0, max depth: 1617.0
                mean: [217.10910263], std: [500.65497894]
            2/2
            raw:
                min depth: 0.0, max depth: 1966.0
                mean: [373.97032166], std: [697.68545532]
            gt:
                min depth: 0.0, max depth: 1795.0
                mean: [160.36244202], std: [490.92971802]
        """

        _min, _max = np.inf, -np.inf

        mean, std = np.zeros(1), np.zeros(1)
        for i in range(len(self)):
            # file = self.gt_depth[i]
            file = self.raw_depth[i]
            depth = cv2.imread(file, cv2.IMREAD_UNCHANGED).astype(np.float32)

            # depth = depth / 1000.0

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

    def show(self, samples, iters=None, save_dir=None, max_show_num=6, save_array=False):
        """use uint8 type to save images"""
        if not save_array:
            super(DDRNetHumanDataset, self).show(samples, iters, save_dir, max_show_num)
        else:
            output_dir = os.path.join(save_dir, f'{iters:010d}')
            os.makedirs(output_dir, exist_ok=True)

            rgb = samples['rgb'].detach()
            raw_depth = samples['raw_depth']
            pred_depth = samples['pred_depth'].detach()
            gt_depth = samples['gt_depth'].detach()

            depth_map_rgb = samples.get('depth_map_rgb', None)
            depth_map_depth = samples.get('depth_map_depth', None)
            confidence_map_rgb = samples.get('confidence_map_rgb', None)
            confidence_map_depth = samples.get('confidence_map_depth', None)

            if depth_map_rgb is not None:
                depth_map_rgb = depth_map_rgb.mul(self._depth_std_tensor.type_as(pred_depth)).add(
                    self._depth_mean_tensor.type_as(pred_depth))
                depth_map_depth = depth_map_depth.mul(self._depth_std_tensor.type_as(pred_depth)).add(
                    self._depth_mean_tensor.type_as(pred_depth))

            # Un-normalization
            rgb = rgb.mul(self._rgb_std_tensor.type_as(rgb)).add(self._rgb_mean_tensor.type_as(rgb))
            raw_depth = raw_depth.mul(self._depth_std_tensor.type_as(raw_depth)).add(
                self._depth_mean_tensor.type_as(raw_depth))

            pred_depth = pred_depth.mul(self._depth_std_tensor.type_as(pred_depth)).add(
                self._depth_mean_tensor.type_as(pred_depth))
            gt_depth = gt_depth.mul(self._depth_std_tensor.type_as(gt_depth)).add(
                self._depth_mean_tensor.type_as(gt_depth))

            # save confidence map
            with open(f'{output_dir}/confidence_map.pkl', 'wb') as f:
                pickle.dump(
                    torch.cat([confidence_map_rgb[:max_show_num, ...], confidence_map_depth[:max_show_num, ...]],
                              dim=1).numpy(), f)

            # save rgb branch and depth branch initial prediction
            with open(f'{output_dir}/depth_map.pkl', 'wb') as f:
                pickle.dump(
                    torch.cat([depth_map_rgb[:max_show_num, ...], depth_map_depth[:max_show_num, ...]], dim=1).numpy(),
                    f)

            # save gt and final predicted depth map
            with open(f'{output_dir}/pred_depth.pkl', 'wb') as f:
                pickle.dump(pred_depth[:max_show_num, ...].numpy(), f)
            with open(f'{output_dir}/gt_depth.pkl', 'wb') as f:
                pickle.dump(gt_depth[:max_show_num, ...].numpy(), f)

            rgb = 255.0 * np.transpose(rgb, (0, 2, 3, 1))  # (b, h, w, c)  real color
            raw_depth = raw_depth / self.max_depth
            pred_depth = pred_depth / self.max_depth
            pred_gray = pred_depth
            gt_depth = gt_depth / self.max_depth

            depth_map_rgb = depth_map_rgb / self.max_depth
            depth_map_depth = depth_map_depth / self.max_depth

            # -----------------
            raw_gray = raw_depth
            gt_gray = gt_depth
            depth_map_rgb_gray = depth_map_rgb
            depth_map_depth_gray = depth_map_depth
            # -----------------

            batch_size = min(rgb.shape[0], max_show_num)

            # cm = plt.get_cmap('plasma')
            cm = plt.get_cmap('inferno')

            for i in range(batch_size):
                _rgb = rgb[i, :, :, :].data.cpu().numpy()  # (h, w, 3)
                raw_dep = raw_depth[i, 0, :, :].data.cpu().numpy()  # (h, w)
                pred_dep = pred_depth[i, 0, :, :].data.cpu().numpy()  # (h, w)
                gt_dep = gt_depth[i, 0, :, :].data.cpu().numpy()  # (h, w)
                _pred_gray = pred_gray[i, 0, :, :].data.cpu().numpy()  # (h, w)

                # ---------------
                _raw_gray = raw_gray[i, 0, :, :].data.cpu().numpy()
                _gt_gray = gt_gray[i, 0, :, :].data.cpu().numpy()
                _depth_map_rgb_gray = depth_map_rgb_gray[i, 0, :, :].data.cpu().numpy()
                _depth_map_depth_gray = depth_map_depth_gray[i, 0, :, :].data.cpu().numpy()
                # ---------------

                depth_map_rgb_i = depth_map_rgb[i, 0, :, :].data.cpu().numpy()
                depth_map_depth_i = depth_map_depth[i, 0, :, :].data.cpu().numpy()

                # painting
                _rgb = _rgb.astype(np.uint8)
                raw_dep = (255.0 * cm(raw_dep)).astype(np.uint8)
                pred_dep = (255.0 * cm(pred_dep)).astype(np.uint8)
                _pred_gray = (255.0 * _pred_gray).astype(np.uint8)
                gt_dep = (255.0 * cm(gt_dep)).astype(np.uint8)

                # -------------------
                _raw_gray = (255.0 * _raw_gray).astype(np.uint8)
                _raw_gray = Image.fromarray(_raw_gray)
                _gt_gray = (255.0 * _gt_gray).astype(np.uint8)
                _gt_gray = Image.fromarray(_gt_gray)
                _depth_map_rgb_gray = (255.0 * _depth_map_rgb_gray).astype(np.uint8)
                _depth_map_rgb_gray = Image.fromarray(_depth_map_rgb_gray)
                _depth_map_depth_gray = (255.0 * _depth_map_depth_gray).astype(np.uint8)
                _depth_map_depth_gray = Image.fromarray(_depth_map_depth_gray)
                # -------------------

                depth_map_rgb_i = (255.0 * cm(depth_map_rgb_i)).astype(np.uint8)
                depth_map_depth_i = (255.0 * cm(depth_map_depth_i)).astype(np.uint8)

                _rgb = Image.fromarray(_rgb, 'RGB')
                raw_dep = Image.fromarray(raw_dep[:, :, :3], 'RGB')
                pred_dep = Image.fromarray(pred_dep[:, :, :3], 'RGB')
                gt_dep = Image.fromarray(gt_dep[:, :, :3], 'RGB')
                _pred_gray = Image.fromarray(_pred_gray)

                depth_map_rgb_i = Image.fromarray(depth_map_rgb_i[:, :, :3], 'RGB')
                depth_map_depth_i = Image.fromarray(depth_map_depth_i[:, :, :3], 'RGB')

                _rgb.save('{}/{}_rgb.png'.format(output_dir, i))
                raw_dep.save('{}/{}_raw_depth.png'.format(output_dir, i))
                pred_dep.save('{}/{}_pred_depth.png'.format(output_dir, i))
                gt_dep.save('{}/{}_gt_depth.png'.format(output_dir, i))
                _pred_gray.save('{}/{}_pred_gray.png'.format(output_dir, i))

                depth_map_rgb_i.save('{}/{}_depth_map_rgb.png'.format(output_dir, i))
                depth_map_depth_i.save('{}/{}_depth_map_depth.png'.format(output_dir, i))

                _raw_gray.save('{}/{}_raw_gray.png'.format(output_dir, i))
                _gt_gray.save('{}/{}_gt_gray.png'.format(output_dir, i))
                _depth_map_rgb_gray.save('{}/{}_depth_map_rgb_gray.png'.format(output_dir, i))
                _depth_map_depth_gray.save('{}/{}_depth_map_depth_gray.png'.format(output_dir, i))

            del rgb, raw_depth, pred_depth, gt_depth, pred_gray, depth_map_rgb, depth_map_depth
