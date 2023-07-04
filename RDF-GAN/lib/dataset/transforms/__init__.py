import copy

import collections

from .transforms import (HoleImage, RandomFlip, RandomCrop, Normalize,
                         Resize, RandomRescale, MultiScaleLabel)
from .format import ToTensor, TypeCheck


def build_transforms(cfg):
    transform_cfg = copy.deepcopy(cfg)
    name = transform_cfg.pop('type')
    if name == 'HoleImage':
        return HoleImage(**transform_cfg)
    elif name == 'RandomRescale':
        return RandomRescale(**transform_cfg)
    elif name == 'RandomFlip':
        return RandomFlip(**transform_cfg)
    elif name == 'RandomCrop':
        return RandomCrop(**transform_cfg)
    elif name == 'Normalize':
        return Normalize(**transform_cfg)
    elif name == 'ToTensor':
        return ToTensor(**transform_cfg)
    elif name == 'TypeCheck':
        return TypeCheck(**transform_cfg)
    elif name == 'Resize':
        return Resize(**transform_cfg)
    elif name == 'MultiScaleLabel':
        return MultiScaleLabel(**transform_cfg)
    else:
        raise NotImplementedError


class Compose(object):
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            # TODO: fix underlying bug.
            if isinstance(transform, dict):
                transform = build_transforms(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
