import numpy as np
import torch


def to_tensor(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


class ToTensor:
    """Convert some image-like to `torch.Tensor` by given keys."""

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


class TypeCheck:
    """
    Avoid possible errors during type conversion (numpy.ndarry ---> torch.Tensor)
    """
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for k in self.keys:
            v = results[k]
            if v.dtype == np.uint8 or v.dtype == np.uint16:
                v = v.copy().astype(np.float32)
            results[k] = v
        return results
