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

warnings.filterwarnings("ignore", category=UserWarning)


"""
NYUDepthV2 json file has a following format:
{
    "train": [
        {
            "filename": "train/bedroom_0078/00066.h5"
        }, ...
    ],
    "val": [
        {
            "filename": "train/study_0008/00351.h5"
        }, ...
    ],
    "test": [
        {
            "filename": "val/official/00001.h5"
        }, ...
    ]
}
Reference : https://github.com/XinJCheng/CSPN/blob/master/nyu_dataset_loader.py
"""

# max_depth: 10.0

class NYUV2S2DDataset(BaseDataset):
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
        super(NYUV2S2DDataset, self).__init__(data_root=data_root,
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
        height, width = (240, 320)
        crop_size = (228, 304)

        self.height = height
        self.width = width
        self.crop_size = crop_size

        # Camera intrinsics [fx, fy, cx, cy]
        # self.K = torch.Tensor([
        #     5.1885790117450188e+02 / 2.0,
        #     5.1946961112127485e+02 / 2.0,
        #     3.2558244941119034e+02 / 2.0 - 8.0,
        #     2.5373616633400465e+02 / 2.0 - 6.0
        # ])


    def load_file(self):
        with open(self.split_json) as json_file:
            json_data = json.load(json_file)

        return json_data[self.mode]

    def load_batch_image(self):
        for i in range(2):
            path = os.path.join(self.data_root, self.sample_list[i]['filename'])

            f = h5py.File(path, 'r')
            rgb = f['rgb'][:].transpose(1, 2, 0)
            depth = f['depth'][:]

            with open(f'vis_tmp/{self.mode}_{i}_rgb.pkl', 'wb') as f:
                pickle.dump(rgb, f)

            with open(f'vis_tmp/{self.mode}_{i}_depth.pkl', 'wb') as f:
                pickle.dump(depth, f)



    def get_train_data(self, idx):
        path_file = os.path.join(self.data_root, self.sample_list[idx]['filename'])

        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)   # (h, w, c)
        depth_h5 = f['depth'][:]             # (h, w)

        rgb = Image.fromarray(rgb_h5, mode='RGB')
        depth = Image.fromarray(depth_h5.astype('float32'), mode='F')

        # data augment
        _scale = np.random.uniform(1.0, 1.5)
        scale = np.int(self.height * _scale)
        degree = np.random.uniform(-5.0, 5.0)
        flip = np.random.uniform(0.0, 1.0)

        if flip > 0.5:
            rgb = TF.hflip(rgb)
            depth = TF.hflip(depth)

        rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)
        depth = TF.rotate(depth, angle=degree, resample=Image.NEAREST)

        t_rgb = T.Compose([T.Resize(scale),
                           T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                           T.CenterCrop(self.crop_size),
                           T.ToTensor(),
                           ])
        t_depth = T.Compose([T.Resize(scale),
                             T.CenterCrop(self.crop_size),
                             self.ToNumpy(),
                             T.ToTensor(),
                             ])


        rgb = t_rgb(rgb)
        rgb = T.Normalize(self.rgb_mean, self.rgb_std)(rgb)
        depth = t_depth(depth)
        depth = depth / _scale

        depth_sp = self.get_sparse_depth(depth, self.num_sample)
        # record mask before apply normalization
        valid_mask = depth > 0.0001
        depth = T.Normalize(self.depth_mean,
                            self.depth_std)(depth)
        raw_depth_zero_masks = depth_sp == 0.0
        depth_sp = T.Normalize(self.depth_mean,
                               self.depth_std)(depth_sp)
        depth_sp[raw_depth_zero_masks] = 0

        # K = self.K.clone()
        # K[0] = K[0] * _scale
        # K[1] = K[1] * _scale

        sample = {'rgb': rgb,
                  'raw_depth': depth_sp,
                  'gt_depth': depth,
                  'depth_masks': valid_mask
                  # 'K': K          # intrinsic matrix is not used actually. https://github.com/zzangjinsun/NLSPN_ECCV20/issues/26
                  }

        return sample

    def get_test_data(self, idx):
        sample = {}
        path_file = os.path.join(self.data_root, self.sample_list[idx]['filename'])

        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
        depth_h5 = f['depth'][:]

        rgb = Image.fromarray(rgb_h5, mode='RGB')
        depth = Image.fromarray(depth_h5.astype('float32'), mode='F')

        # data augment
        t_rgb = T.Compose([T.Resize(self.height),
                           T.CenterCrop(self.crop_size),
                           T.ToTensor(),
                           # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                           ])
        t_depth = T.Compose([T.Resize(self.height),
                             T.CenterCrop(self.crop_size),
                             self.ToNumpy(),
                             T.ToTensor(),
                             # T.Normalize((0.5 * self.max_depth), (0.5 * self.max_depth))
                             ])

        rgb = t_rgb(rgb)
        rgb = T.Normalize(self.rgb_mean, self.rgb_std)(rgb)
        depth = t_depth(depth)
        # I just don't want to modify some code in evaluator.
        sample['gt_depth_origin'] = depth.numpy().copy().squeeze()
        depth_sp = self.get_sparse_depth(depth, self.num_sample)
        # record mask before apply normalization
        valid_mask = depth > 0.0001
        depth = T.Normalize(self.depth_mean,
                            self.depth_std)(depth)

        raw_depth_zero_masks = depth_sp == 0.0
        depth_sp = T.Normalize(self.depth_mean,
                               self.depth_std)(depth_sp)
        depth_sp[raw_depth_zero_masks] = 0

        # K = self.K.clone()

        sample.update({'rgb': rgb,
                       'raw_depth': depth_sp,
                       'gt_depth': depth,
                       'depth_masks': valid_mask
                       # 'K': K
                       })

        return sample


    def get_sparse_depth(self, depth, num_sample):
        c, h, w = depth.shape

        assert c == 1

        # Pytorchv1.2+
        idx_nzz = torch.nonzero(depth.view(-1) > 0.0001, as_tuple=False)

        # idx_nzz = torch.where(depth.view(-1) > 0.0001)[0].view(-1, 1)

        num_idx = len(idx_nzz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nzz = idx_nzz[idx_sample[:]]

        mask = torch.zeros((c * h * w))
        mask[idx_nzz] = 1.0
        mask = mask.view(c, h, w)

        depth_sp = depth * mask.type_as(depth)

        return depth_sp

    def load_image(self, idx):
        pass

    def load_depth(self, idx):
        pass

    def __len__(self):
        return len(self.sample_list)

    def show(self, samples, iters=None, save_dir=None, max_show_num=6, save_array=False):
        """use uint8 type to save images"""
        if not save_array:
            super(NYUV2S2DDataset, self).show(samples, iters, save_dir, max_show_num)
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
            # with open(f'{output_dir}/confidence_map.pkl', 'wb') as f:
            #     pickle.dump(
            #         torch.cat([confidence_map_rgb[:max_show_num, ...], confidence_map_depth[:max_show_num, ...]],
            #                   dim=1).numpy(), f)

            # save rgb branch and depth branch initial prediction
            # with open(f'{output_dir}/depth_map.pkl', 'wb') as f:
            #     pickle.dump(
            #         torch.cat([depth_map_rgb[:max_show_num, ...], depth_map_depth[:max_show_num, ...]], dim=1).numpy(),
            #         f)

            # save gt and final predicted depth map
            # with open(f'{output_dir}/pred_depth.pkl', 'wb') as f:
            #     pickle.dump(pred_depth[:max_show_num, ...].numpy(), f)
            # with open(f'{output_dir}/gt_depth.pkl', 'wb') as f:
            #     pickle.dump(gt_depth[:max_show_num, ...].numpy(), f)

            rgb = 255.0 * np.transpose(rgb, (0, 2, 3, 1))  # (b, h, w, c)  real color
            # raw_depth = raw_depth / self.max_depth
            # pred_depth = pred_depth / self.max_depth
            # pred_gray = pred_depth
            # gt_depth = gt_depth / self.max_depth
            #
            # depth_map_rgb = depth_map_rgb / self.max_depth
            # depth_map_depth = depth_map_depth / self.max_depth

            def norm(arr, _min, _max):
                return (arr - _min) / (_max - _min)

            raw_depth = norm(raw_depth, raw_depth.min(), raw_depth.max())
            pred_depth = norm(pred_depth, min(pred_depth.min(), gt_depth.min()), max(pred_depth.max(), gt_depth.max()))
            pred_gray = pred_depth
            gt_depth = norm(gt_depth, gt_depth.min(), gt_depth.max())

            depth_map_rgb = norm(depth_map_rgb, min(depth_map_rgb.min(), gt_depth.min()), max(depth_map_rgb.max(), gt_depth.max()))
            depth_map_depth = norm(depth_map_depth, min(depth_map_depth.min(), gt_depth.min()), max(depth_map_depth.max(), gt_depth.max()))


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

    def stat_depth(self):
        """
            45205/45205
            min depth: 0.0, max depth: 9.999999046325684
            mean: [2.70625685], std: [1.16382526]
            2379/2379
            min depth: 0.0, max depth: 9.999992370605469
            mean: [2.7112556], std: [1.18139173]
            654/654
            min depth: 0.7133004665374756, max depth: 9.986645698547363
            mean: [2.74023268], std: [0.97136956]
        """
        _min, _max = np.inf, -np.inf

        mean, std = np.zeros(1), np.zeros(1)
        for i in range(len(self)):
            file = os.path.join(self.data_root,
                                self.sample_list[i]['filename'])
            f = h5py.File(file, 'r')
            depth = f['depth'][:]

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
