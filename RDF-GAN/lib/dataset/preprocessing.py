import numpy as np
import torchvision
import cv2
import torch
import numba


class Normalize:
    def __init__(self, depth_mean, depth_std, rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225]):
        # depth maps are normalized based on real depth
        # rgb maps are normalized based on values between 0 and 1
        self._depth_mean = depth_mean
        self._depth_std = depth_std
        self._rgb_mean = rgb_mean
        self._rgb_std = rgb_std

    def __call__(self, sample):
        image, gt_depth = sample['rgb'], sample['gt_depth']
        image = image / 255

        image = torchvision.transforms.Normalize(
            mean=self._rgb_mean, std=self._rgb_std)(image)
        sample['rgb'] = image

        # normalize gt depth, do not forget to record mask
        if 'depth_masks' in sample:
            mask = gt_depth > 1e-4
            sample['depth_masks'] = mask
        gt_depth = torchvision.transforms.Normalize(
            mean=self._depth_mean, std=self._depth_std)(gt_depth)
        sample['gt_depth'] = gt_depth

        # normalize raw depth
        raw_depth = sample['raw_depth']
        # TODO
        raw_depth_zero_masks = raw_depth == 0.0

        raw_depth = torchvision.transforms.Normalize(
            mean=self._depth_mean, std=self._depth_std)(raw_depth)

        raw_depth[raw_depth_zero_masks] = 0
        sample['raw_depth'] = raw_depth

        return sample


class ToTensor:

    def __call__(self, sample):
        rgb, gt_depth = sample['rgb'], sample['gt_depth']

        rgb = rgb.transpose((2, 0, 1))
        gt_depth = np.expand_dims(gt_depth, 0).astype(np.float32)

        sample['rgb'] = torch.from_numpy(rgb).float()
        sample['gt_depth'] = torch.from_numpy(gt_depth).float()

        if 'raw_depth' in sample:
            raw_depth = sample['raw_depth']
            raw_depth = np.expand_dims(raw_depth, 0).astype(np.float32)
            sample['raw_depth'] = torch.from_numpy(raw_depth).float()

        return sample

class Rescale:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, sample):
        rgb, gt_depth = sample['rgb'], sample['gt_depth']

        rgb = cv2.resize(rgb, (self.width, self.height),
                         interpolation=cv2.INTER_LINEAR)
        gt_depth = cv2.resize(gt_depth, (self.width, self.height),
                              interpolation=cv2.INTER_NEAREST)

        sample['rgb'] = rgb
        sample['gt_depth'] = gt_depth

        if 'raw_depth' in sample:
            raw_depth = sample['raw_depth']
            raw_depth = cv2.resize(raw_depth, (self.width, self.height),
                                   interpolation=cv2.INTER_NEAREST)
            sample['raw_depth'] = raw_depth

        if 'label' in sample:
            label = sample['label']
            label = cv2.resize(label, (self.width, self.height),
                               interpolation=cv2.INTER_NEAREST)
            sample['label'] = label

        return sample


class RandomRescale:
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        rgb, gt_depth = sample['rgb'], sample['gt_depth']

        target_scale = np.random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * rgb.shape[0]))
        target_width = int(round(target_scale * rgb.shape[1]))

        rgb = cv2.resize(rgb, (target_width, target_height),
                         interpolation=cv2.INTER_LINEAR)
        gt_depth = cv2.resize(gt_depth, (target_width, target_height),
                              interpolation=cv2.INTER_NEAREST)
        if 'raw_depth' in sample:
            raw_depth = sample['raw_depth']
            raw_depth = cv2.resize(raw_depth, (target_width, target_height),
                                   interpolation=cv2.INTER_NEAREST)
            sample['raw_depth'] = raw_depth

        if 'label' in sample:
            label = sample['label']
            label = cv2.resize(label, (target_width, target_height),
                               interpolation=cv2.INTER_NEAREST)
            sample['label'] = label

        sample['rgb'] = rgb
        sample['gt_depth'] = gt_depth

        return sample


class RandomCrop:
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.rescale = Rescale(self.crop_height, self.crop_width)

    def __call__(self, sample):
        rgb, gt_depth = sample['rgb'], sample['gt_depth']
        h = rgb.shape[0]
        w = rgb.shape[1]
        if h <= self.crop_height or w <= self.crop_width:
            # simply rescale instead of random crop as image is not large enough
            sample = self.rescale(sample)
        else:
            i = np.random.randint(0, h - self.crop_height)
            j = np.random.randint(0, w - self.crop_width)
            rgb = rgb[i:i + self.crop_height, j:j + self.crop_width, :]
            gt_depth = gt_depth[i:i + self.crop_height, j:j + self.crop_width]
            if 'raw_depth' in sample:
                raw_depth = sample['raw_depth']
                raw_depth = raw_depth[i:i + self.crop_height, j:j + self.crop_width]
                sample['raw_depth'] = raw_depth
            sample['rgb'] = rgb
            sample['gt_depth'] = gt_depth

            if 'label' in sample:
                label = sample['label']
                label = label[i:i + self.crop_height, j:j + self.crop_width]
                sample['label'] = label

        return sample


class RandomFlip:
    def __call__(self, sample):
        rgb, gt_depth = sample['rgb'], sample['gt_depth']
        if np.random.rand() > 0.5:
            rgb = np.fliplr(rgb).copy()
            gt_depth = np.fliplr(gt_depth).copy()

            if 'raw_depth' in sample:
                raw_depth = sample['raw_depth']
                raw_depth =np.fliplr(raw_depth).copy()
                sample['raw_depth'] = raw_depth

            if 'label' in sample:
                label = sample['label']
                label = np.fliplr(label).copy()
                sample['label'] = label

        sample['rgb'] = rgb
        sample['gt_depth'] = gt_depth

        return sample


@numba.jit()
def hole_image_jit(image, width, height, nums):
    # (H, W, C)
    image_height, image_width = image.shape[:2]
    rows, columns = (image_height - height + 1), (image_width - width + 1)

    #

    idx = np.random.choice(rows * columns - 1, nums)
    masks = np.ones_like(image)
    x, y = idx // columns, idx % columns
    for i in range(nums):
        masks[x[i]:x[i] + width, y[i]:y[i] + height] = 0

    return image * masks


@numba.jit()
def hole_image_jit_v2(image, width, height, nums):
    """Occlusion areas do not overlap"""
    image_height, image_width = image.shape[:2]
    # 480, 640
    rows, columns = (image_height - height + 1), (image_width - width + 1)
    optional_region = np.ones((rows * columns))
    masks = np.ones_like(image)

    for k in range(nums):
        # random select a region, (upper left corner)
        idx = np.random.choice(np.where(optional_region == 1)[0])

        # mask selected region
        x, y = idx // columns, idx % columns
        masks[x:x + width, y:y + height] = 0

        # set the nearby area unobstructed
        left_bound, right_bound = x - width + 1, x + width - 1
        upper_bound, bottom_bound = y - height + 1, y + height - 1
        for i in range(max(0, left_bound), min(right_bound, columns)):
            for j in range(max(0, upper_bound), min(bottom_bound, rows)):
                # x, y = j, i
                optional_region[j * columns + i] = 0

    return image * masks


class DeterministicPseudoHole:
    def __init__(self, label):
        assert isinstance(label, (list, tuple))
        self.label = label

    def __call__(self, sample):
        labels = sample['label']
        raw_depth = sample['gt_depth'].copy()

        # valid means to be holed
        valid_mask = np.zeros_like(labels, bool)

        for l in self.label:
            mask = labels == l
            valid_mask |= mask

        raw_depth[valid_mask] = 0
        sample['raw_depth'] = raw_depth
        sample.pop('label')
        del  labels

        return sample


class PseudoHole:
    def __init__(self,
                 hole_height,
                 hole_width,
                 hole_num):
        self.hole_height = hole_height
        self.hole_width = hole_width
        self.hole_num = hole_num

    def __call__(self, sample):
        gt_depth = sample['gt_depth'].copy()
        img_with_holes = hole_image_jit_v2(image=gt_depth,
                                        width=self.hole_width,
                                        height=self.hole_height,
                                        nums=self.hole_num)
        sample['raw_depth'] = img_with_holes

        return sample


class Identity:
    def __call__(self ,sample):

        return sample


class CutOffBlackBorder:
    def __init__(self,
                 top_pixel_num=45,
                 bottom_pixel_num=15,
                 left_pixel_num=45,
                 right_pixel_num=40):
        self.top_pixel_num = top_pixel_num
        self.bottom_pixel_num = bottom_pixel_num
        self.left_pixel_num = left_pixel_num
        self.right_pixel_num = right_pixel_num

    def __call__(self, img):
        # np.ndarry
        croped_img = img[self.top_pixel_num:-self.bottom_pixel_num, self.left_pixel_num:-self.right_pixel_num]

        return croped_img
