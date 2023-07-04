import cv2
import numpy as np
import numba
import torch
import collections


def hole_image(image, width, height, nums):
    image_height, image_width = image.shape[:2]
    rows, columns = (image_height - height + 1), (image_width - width + 1)

    idx = np.random.choice(rows * columns - 1, nums)
    masks = np.ones_like(image)
    x, y = idx // columns, idx % columns
    for i in range(nums):
        masks[x[i]:x[i] + width, y[i]:y[i] + height] = 0

    return image * masks


@numba.jit()
def hole_image_jit(image, width, height, nums):
    # (H, W, C)
    image_height, image_width = image.shape[:2]
    rows, columns = (image_height - height + 1), (image_width - width + 1)

    idx = np.random.choice(rows * columns - 1, nums)
    masks = np.ones_like(image)
    x, y = idx // columns, idx % columns
    for i in range(nums):
        masks[x[i]:x[i] + width, y[i]:y[i] + height] = 0

    return image * masks


class HoleImage:
    def __init__(self,
                 rgb_patch_width,
                 rgb_patch_height,
                 rgb_patch_nums,
                 depth_patch_width,
                 depth_patch_height,
                 depth_patch_nums):
        # image(h, w, c)
        self.rgb_patch_width = rgb_patch_width
        self.rgb_patch_height = rgb_patch_height
        self.rgb_patch_nums = rgb_patch_nums

        self.depth_patch_width = depth_patch_width
        self.depth_patch_height = depth_patch_height
        self.depth_patch_nums = depth_patch_nums

    def __call__(self, results):
        # result['image'] = hole_image(image=result['image'],
        #                              width=self.rgb_patch_width,
        #                              height=self.rgb_patch_height,
        #                              nums=self.rgb_patch_nums)
        # result['depth'] = hole_image(image=result['depth'],
        #                              width=self.depth_patch_width,
        #                              height=self.depth_patch_height,
        #                              nums=self.depth_patch_nums)

        # use jit instead.
        results['hole_image'] = hole_image_jit(image=results['image'],
                                               width=self.rgb_patch_width,
                                               height=self.rgb_patch_height,
                                               nums=self.rgb_patch_nums)
        results['hole_depth'] = hole_image_jit(image=results['depth'],
                                               width=self.depth_patch_width,
                                               height=self.depth_patch_height,
                                               nums=self.depth_patch_nums)
        return results


class RandomFlipV2:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, results):
        image, depth, label = results['image'], results['depth'], results['label']

        if np.random.rand() > self.prob:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        results['image'] = image
        results['depth'] = depth
        results['label'] = label
        return results


class RandomFlip:
    def __init__(self, prob=None, direction='horizontal', keys=None):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert 0 <= prob < 1
        assert direction in ['horizontal', 'vertical']
        self.keys = keys

    @staticmethod
    def imflip(img, direction='horizontal'):
        """Flip an image horizontally or vertically.

        Args:
            img (ndarray): Image to be flipped.
            direction (str): The flip direction, either "horizontal" or
                "vertical" or "diagonal".

        Returns:
            ndarray: The flipped image.
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        if direction == 'horizontal':
            return np.flip(img, axis=1)
        elif direction == 'vertical':
            return np.flip(img, axis=0)
        else:
            return np.flip(img, axis=(0, 1))

    def __call__(self, results):
        if self.keys is None:
            image, depth, label = results['image'], results['depth'], results['label']
            if np.random.rand() > self.prob:
                # use copy() to make numpy stride positive
                image = self.imflip(image, direction=self.direction).copy()
                depth = self.imflip(depth, direction=self.direction).copy()
                label = self.imflip(label, direction=self.direction).copy()
            results['image'] = image
            results['depth'] = depth
            results['label'] = label
        else:
            for k in self.keys:
                results[k] = self.imflip(results[k], direction=self.direction).copy()

        return results


class Normalize:
    def __init__(self, rgb_mean, rgb_std,
                 depth_mean, depth_std,
                 label_mean, label_std,
                 depth_mode='refined',
                 normalize_label=True):
        self.normalize_label = normalize_label
        self._rgb_mean = np.array(rgb_mean, dtype=np.float32)
        self._rgb_std = np.array(rgb_std, dtype=np.float32)
        self._depth_mean = np.array(depth_mean, dtype=np.float32)
        self._depth_std = np.array(depth_std, dtype=np.float32)
        self._label_mean = np.array(label_mean, dtype=np.float32)
        self._label_std = np.array(label_std, dtype=np.float32)
        self._depth_mode = depth_mode

    @staticmethod
    def imnormalize(img, mean, std, to_rgb=False):
        img = img.copy().astype(np.float32)
        assert img.dtype != np.uint8
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img

    def __call__(self, results):
        image, depth, label = results['image'], results['depth'], results['label']
        results['image'] = self.imnormalize(image, self._rgb_mean, self._rgb_std)

        if self.normalize_label:
            results['label'] = self.imnormalize(label, self._label_mean, self._label_std)

        if self._depth_mode == 'raw':
            # TODO: Check whether the depth map generated by GAN is affected
            # FIXME: Note: zero values are need to be ignored when reconstructing the depth map
            depth_0 = depth == 0
            depth = self.imnormalize(depth, self._depth_mean, self._depth_std)

            # set invalid values to zeros again
            depth[depth_0] = 0
        else:
            depth = self.imnormalize(depth, self._depth_mean, self._depth_std)

        results['depth'] = depth

        if 'label_colored' in results.keys():
            results['label_colored'] = self.imnormalize(results['label_colored'],
                                                        mean=self._rgb_mean,
                                                        std=self._rgb_std)
        return results


class RandomCrop:
    """
    Assuming that the image and depth have same size, such as SUN RGBD DataSet.
    """
    def __init__(self, crop_size, cat_max_ratio=None, keys=None):
        """
        Args:
            crop_size (tuple): Expected size after cropping, (h, w).
            cat_max_ratio (float): The maximum ratio that single category could occupy.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.keys = keys

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]
        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        return img

    @staticmethod
    def rescale(img, width, height):
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    def __call__(self, results):
        if self.keys is None:
            image, depth, label = results['image'], results['depth'], results['label']
            h, w = image.shape[0], image.shape[1]
            if h <= self.crop_size[0] or w <= self.crop_size[1]:
                image = self.rescale(image, width=self.crop_size[1], height=self.crop_size[0])
                depth = self.rescale(depth, width=self.crop_size[1], height=self.crop_size[0])
                label = self.rescale(label, width=self.crop_size[1], height=self.crop_size[0])
            else:
                crop_bbox = self.get_crop_bbox(image)

                if self.cat_max_ratio is not None:
                    if self.cat_max_ratio < 1.:
                        # Repeat 10 times
                        for _ in range(10):
                            seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                            labels, cnt = np.unique(seg_temp, return_counts=True)
                            cnt = cnt[labels != self.ignore_index]
                            if len(cnt) > 1 and np.max(cnt) / np.sum(
                                    cnt) < self.cat_max_ratio:
                                break
                            crop_bbox = self.get_crop_bbox(image)

                image = self.crop(image, crop_bbox)
                depth = self.crop(depth, crop_bbox)
                label = self.crop(label, crop_bbox)
            results['image'] = image
            results['depth'] = depth
            results['label'] = label
        else:
            raise NotImplementedError

        return results


class DimConversion:
    """
    (H, W, C) ---> (C, H, W)
    """

    def __call__(self, results):
        image, depth, label = results['image'], results['depth'], results['label']
        results['image'] = np.transpose(image, (2, 0, 1))
        results['image'] = np.transpose(image, (2, 0, 1))
        results['depth'] = np.transpose(depth, (2, 0, 1))
        results['label'] = np.transpose(label, (2, 0, 1))
        return results


class Resize:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, results):
        image, depth = results['image'], results['depth']

        image = cv2.resize(image, (self.width, self.height),
                           interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.width, self.height),
                           interpolation=cv2.INTER_NEAREST)

        results['image'] = image
        results['depth'] = depth

        if 'label' in results.keys():
            label = results['label']
            label = cv2.resize(label, (self.width, self.height),
                               interpolation=cv2.INTER_NEAREST)
            results['label'] = label

        return results


class RandomRescale:
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, results):
        image, depth, label = results['image'], results['depth'], results['label']

        target_scale = np.random.uniform(self.scale_low, self.scale_high)

        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))

        image = cv2.resize(image, (target_width, target_height),
                           interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (target_width, target_height),
                           interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (target_width, target_height),
                           interpolation=cv2.INTER_NEAREST)
        results['image'] = image
        results['depth'] = depth
        results['label'] = label

        return results


class MultiScaleLabel:
    def __init__(self, downsampling_rates=None):
        if downsampling_rates is None:
            # default
            self.downsampling_rates = [8, 16, 32]
        else:
            self.downsampling_rates = downsampling_rates

    def __call__(self, results):
        label = results['label'].numpy()
        # print(f'id: {id(label)}')
        if len(label.shape) == 3:
            label = label.squeeze()

        h, w = label.shape
        results['label_down'] = dict()

        # Nearest neighbor interpolation
        for rate in self.downsampling_rates:
            label_down = cv2.resize(label, (w // rate, h // rate),
                                    interpolation=cv2.INTER_NEAREST)
            # print(f'id: {id(label_down)}')
            results['label_down'][rate] = torch.from_numpy(label_down)

        return results
