import numpy as np
import cv2
import numba
from skimage.filters import gaussian


# use cv2 instead
@numba.jit()
def hole_image_jit(image, width, height, nums):
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


class RandomPseudoHole:
    """Apply random holes to gt_depth map to generate raw_depth map"""
    def __init__(self,
                 hole_height,
                 hole_width,
                 hole_num):
        self.hole_height = hole_height
        self.hole_width = hole_width
        self.hole_num = hole_num

    def __call__(self, sample):
        gt_depth = sample['gt_depth'].copy()
        img_with_holes = hole_image_jit(image=gt_depth,
                                        width=self.hole_width,
                                        height=self.hole_height,
                                        nums=self.hole_num)
        sample['raw_depth'] = img_with_holes

        return sample


class DeterministicPseudoHole:
    """set depth value to zeros according to the specifically given ground truth segmentation label"""
    def __init__(self, label, random_filtered_label_nums=2):
        assert isinstance(label, (list, tuple))
        self.label = label
        self.random_filtered_label_nums = random_filtered_label_nums

    def __call__(self, sample):
        try:
            labels = sample['label']
        except:
            raise KeyError('label not in sample')

        # find what labels in this frame & random filter some label
        label_nzz = np.nonzero(np.bincount(labels.flatten()))[0]
        labels_to_be_filtered = np.random.choice(label_nzz, self.random_filtered_label_nums)
        labels_to_be_filtered = self.label + labels_to_be_filtered.tolist()

        # valid means to be holed
        valid_mask = np.zeros_like(labels, bool)

        for l in labels_to_be_filtered:
            mask = labels == l
            valid_mask |= mask

        return valid_mask


class MaskBlacks:
    def __init__(self, vmin=0, vmax=5):
        self.vmax = vmax
        self.vmin = vmin

    def __call__(self, sample):
        rgb = sample['rgb']

        masks = ((rgb >= self.vmin) & (rgb <= self.vmax)).sum(2)
        masks = masks == 3    # rgb channel

        return masks


class SegmentationHighLight:
    def __init__(self, T1=210):
        self.T1 = T1

    def calc_specular_mask(self, cE, cG, cB):
        h, w = cE.shape[:2]
        T1 = self.T1

        p95_cG = cG * 0.95
        p95_cE = cE * 0.95
        p95_cB = cB * 0.95

        rGE = p95_cG / (p95_cE + 1e-8)
        rBE = p95_cB / (p95_cE + 1e-8)
        img_new = np.zeros((h, w, 1), dtype=np.float32)

        mask1, mask2, mask3 = cG > (rGE * T1), cB > (rBE * T1), cE > T1
        mask = mask1 & mask2 & mask3
        img_new[mask] = 255

        return img_new

    def __call__(self, sample):
        rgb = sample['rgb']

        cR = rgb[:, :, 0]
        cG = rgb[:, :, 1]
        cB = rgb[:, :, 2]

        cE = 0.2989 * cR + 0.5870 * cG + 0.1140 * cB

        specular_mask = self.calc_specular_mask(cE, cG, cB)
        specular_mask = specular_mask.squeeze() / 255

        return specular_mask == 1


class Spatter:
    def __init__(self,
                 threshold=True,
                 granularity=8,
                 percentile_void=0.05,
                 percentile_deform=0.02):
        self.threshold = threshold
        self.granularity = granularity
        self.percentile_deform = percentile_deform
        self.percentile_void = percentile_void

    def spatter(self, layer, mask, granularity=10, percentile=0.4):
        holes_mask = self.create_holes_mask(layer, granularity, percentile)

        res = layer.copy().squeeze()
        mask = mask.copy().squeeze()
        res[holes_mask] = 0
        mask[holes_mask] = 0

        return res, holes_mask
        # return res, mask

    def create_holes_mask(self, layer, granularity, percentile):
        gaussian_layer = np.random.uniform(size=layer.shape[1:])
        gaussian_layer = gaussian(gaussian_layer, sigma=granularity)
        threshold = np.percentile(gaussian_layer.reshape([-1]), 100 * (1 - percentile))

        return gaussian_layer > threshold

    def __call__(self, sample):
        if self.threshold:
            pass

        raw_depth = sample['raw_depth'][np.newaxis, ...]
        _, mask = self.spatter(raw_depth,
                               raw_depth > 0,
                               granularity=self.granularity,
                               percentile=self.percentile_void)

        return mask


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
