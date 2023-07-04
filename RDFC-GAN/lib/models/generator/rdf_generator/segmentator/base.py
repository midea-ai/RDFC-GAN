import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod, ABCMeta


class BaseSegmentator(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(BaseSegmentator, self).__init__()

    @abstractmethod
    def forward_net(self, image, **kwargs):
        """perform forward propagation"""
        pass

    def forward(self, data, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(data, **kwargs)
        else:
            return self.forward_test(data, **kwargs)

    def forward_train(self, data, image_type='rgb', device=None):
        if isinstance(data, dict):
            # fetch data from dataloader
            if image_type == 'rgb':
                image = data['image']
            elif image_type == 'depth':
                image = data['depth']
            else:
                raise NotImplementedError
            image = image.to(device)
            pred = self.forward_net(image)
        else:
            pred = self.forward_net(data)

        return pred

    def forward_test(self, data, image_type='rgb', device=None):
        if image_type == 'rgb':
            image = data['image']
        elif image_type == 'depth':
            image = data['depth']
        else:
            raise NotImplementedError(f'The supported image types are RGB and depth, but got {image_type}')

        image = image.to(device)
        pred = self.forward_net(image)

        # Resize to origin
        label = data['label_origin']
        assert isinstance(label, list)
        # TODO: support multi-frame inference
        assert len(label) == 1, f'Only support single frame inference so far'
        image_h, image_w = label[0].shape

        prediction = F.interpolate(pred,
                                   (image_h, image_w),
                                   mode='bilinear',
                                   align_corners=False)
        prediction = torch.argmax(prediction, dim=1)

        return [dict(prediction=prediction.cpu(),
                     label=label[0])]

    def init_weights(self):
        pass
