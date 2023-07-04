import os
import warnings
from ..base import BaseDataset
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import json
import h5py
import matplotlib.pyplot as plt
import pickle
from ..pseudo_hole import (CutOffBlackBorder, SegmentationHighLight,
                           MaskBlacks, Spatter)
import cv2

warnings.filterwarnings("ignore", category=UserWarning)


class NYUV2R2RDataset(BaseDataset):
    """
    4w image pairs for training
    """
    def __init__(self,
                 data_root,     # meta file path
                 mode='train',
                 preprocessor=None,
                 rgb_mean=[0.485, 0.456, 0.406],
                 rgb_std=[0.229, 0.224, 0.225],
                 max_depth=10.0,
                 depth_mean=[5.0],
                 depth_std=[5.0],
                 num_sample=500    # number of sparse sample
                 ):
        super(NYUV2R2RDataset, self).__init__(data_root=data_root,
                                              mode=mode,
                                              rgb_mean=rgb_mean,
                                              rgb_std=rgb_std,
                                              max_depth=max_depth,
                                              depth_mean=depth_mean,
                                              depth_std=depth_std,
                                              preprocessor=preprocessor)
        self.num_sample = num_sample
        self.split_json = os.path.join(self.data_root, 'nyu.json')
        if not os.path.exists(self.split_json):
            raise FileNotFoundError('Please put the nyu.json file under the nyudepthv2 root directory')
        self.sample_list = self.load_file()

        # For NYUDepthV2, crop size is fixed
        height, width = (480 - 60, 640 - 85)
        crop_size = (256, 320)
        self.height = height
        self.width = width
        self.crop_size = crop_size

        self.test_rgb, self.test_raw_depth, self.test_gt_depth = self.load_test_file()

    def load_test_file(self):
        with open('./data/nyuv2/test.txt', 'r') as f:
            indices = f.read().splitlines()

        rgb_file_list = [f'./data/nyuv2/test/rgb/{idx}.png' for idx in indices]
        raw_depth_file_list = [f'./data/nyuv2/test/depth_raw/{idx}.png' for idx in indices]
        depth_file_list = [f'./data/nyuv2/test/depth/{idx}.png' for idx in indices]

        return rgb_file_list, raw_depth_file_list, depth_file_list

    def load_file(self):
        with open(self.split_json) as json_file:
            json_data = json.load(json_file)

        return json_data[self.mode]

    def get_train_data(self, idx):
        path_file = os.path.join(self.data_root, self.sample_list[idx]['filename'])

        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)   # (h, w, c)
        depth_h5 = f['depth'][:]             # (h, w)

        cut_off_black_border = CutOffBlackBorder()
        rgb = cut_off_black_border(rgb_h5)
        depth_h5 = cut_off_black_border(depth_h5)

        # -------------- perform pseudo operations -------------- #
        masks = []
        pseudo_sample = {'rgb': rgb, 'raw_depth': depth_h5}
        prop = np.random.uniform(0.0, 1.0)
        if prop > 0.5:
            t = SegmentationHighLight()
            masks.append(t(sample=pseudo_sample))

        prop = np.random.uniform(0.0, 1.0)
        if prop > 0.5:
            t = Spatter()
            masks.append(t(sample=pseudo_sample))

        prop = np.random.uniform(0.0, 1.0)
        if prop > 0.5:
            t = MaskBlacks()
            masks.append(t(sample=pseudo_sample))

        # combine all pseudo masks
        pseudo_maks = np.zeros_like(depth_h5, dtype=bool)

        for m in masks:
            pseudo_maks |= m

        pseudo_depth = depth_h5.copy()
        pseudo_depth[pseudo_maks] = 0.0
        # -------------- perform pseudo operations -------------- #

        rgb = Image.fromarray(rgb_h5, mode='RGB')
        depth = Image.fromarray(depth_h5.astype('float32'), mode='F')
        pseudo_depth = Image.fromarray(pseudo_depth.astype('float32'), mode='F')

        # data augment
        degree = np.random.uniform(-5.0, 5.0)
        flip = np.random.uniform(0.0, 1.0)

        if flip > 0.5:
            rgb = TF.hflip(rgb)
            depth = TF.hflip(depth)
            pseudo_depth = TF.hflip(pseudo_depth)


        rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)
        depth = TF.rotate(depth, angle=degree, resample=Image.NEAREST)
        pseudo_depth = TF.rotate(pseudo_depth, angle=degree, resample=Image.NEAREST)

        t_rgb = T.Compose([T.Resize(self.crop_size),
                           T.ToTensor(),
                           ])
        t_depth = T.Compose([T.Resize(self.crop_size),
                             self.ToNumpy(),
                             T.ToTensor(),
                             ])

        rgb = t_rgb(rgb)
        rgb = T.Normalize(self.rgb_mean, self.rgb_std)(rgb)
        depth = t_depth(depth)
        pseudo_depth = t_depth(pseudo_depth)

        # record mask before apply normalization
        valid_mask = depth > 0.0001
        depth = T.Normalize(self.depth_mean,
                            self.depth_std)(depth)
        pseudo_depth_zero_masks = pseudo_depth == 0.0
        pseudo_depth = T.Normalize(self.depth_mean,
                                   self.depth_std)(pseudo_depth)
        pseudo_depth[pseudo_depth_zero_masks] = 0


        sample = {'rgb': rgb,
                  'raw_depth': pseudo_depth,
                  'gt_depth': depth,
                  'depth_masks': valid_mask
                  }

        return sample


    def get_test_data(self, idx):
        sample = {}
        # load rgb image
        rgb = cv2.imread(self.test_rgb[idx], cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # load raw depth image
        raw_depth = cv2.imread(self.test_raw_depth[idx], cv2.IMREAD_UNCHANGED).astype(
            np.float32)

        # load gt depth image
        gt_depth = cv2.imread(self.test_gt_depth[idx], cv2.IMREAD_UNCHANGED).astype(np.float32)

        raw_depth = raw_depth / 1000.0
        gt_depth = gt_depth / 1000.0

        cut_off_black_border = CutOffBlackBorder()
        rgb = cut_off_black_border(rgb)
        raw_depth = cut_off_black_border(raw_depth)
        gt_depth = cut_off_black_border(gt_depth)

        rgb = Image.fromarray(rgb, mode='RGB')
        raw_depth = Image.fromarray(raw_depth, mode='F')
        gt_depth = Image.fromarray(gt_depth, mode='F')

        t_rgb = T.Compose([T.Resize(self.crop_size),
                           T.ToTensor(),
                           ])
        t_depth = T.Compose([T.Resize(self.crop_size),
                             self.ToNumpy(),
                             T.ToTensor(),
                             ])
        rgb = t_rgb(rgb)
        rgb = T.Normalize(self.rgb_mean, self.rgb_std)(rgb)
        raw_depth = t_depth(raw_depth)
        gt_depth = t_depth(gt_depth)
        valid_mask = gt_depth > 0.0001
        sample['gt_depth_origin'] = gt_depth.numpy().copy().squeeze()

        raw_depth_zero_masks = raw_depth == 0

        gt_depth = T.Normalize(self.depth_mean,
                               self.depth_std)(gt_depth)

        raw_depth = T.Normalize(self.depth_mean,
                                self.depth_std)(raw_depth)

        raw_depth[raw_depth_zero_masks] = 0

        sample.update({'rgb': rgb,
                       'raw_depth': raw_depth,
                       'gt_depth': gt_depth,
                       'depth_masks': valid_mask,
                       # 'evaluate_mask': raw_depth_zero_masks
                       })

        return sample



    def load_image(self, idx):
        pass

    def load_depth(self, idx):
        pass

    def __len__(self):
        if self.mode == 'train':
            return len(self.sample_list)
        else:
            return len(self.test_rgb)

    def show(self, samples, iters=None, save_dir=None, max_show_num=6, save_array=False):
        """use uint8 type to save images"""
        if not save_array:
            super(NYUV2R2RDataset, self).show(samples, iters, save_dir, max_show_num)
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


