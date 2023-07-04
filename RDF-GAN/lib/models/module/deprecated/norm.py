import torch.nn as nn
import functools


norm_cfg = {
    'BN': ('bn', nn.BatchNorm2d),
    'IN': {'in', nn.InstanceNorm2d},
    'SyncBN': ('bn', nn.SyncBatchNorm),
    'GN': ('gn', nn.GroupNorm),
    'LN': ('ln', nn.LayerNorm)
}


def build_norm_layer(cfg, num_features, postfix=''):
    """Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to be instantiate a norm layer.
            requires_grad(brequires_gradool): [optional] whether stop gradient updates.
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            crete named layer.

    Returns:

    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    in_discriminator = cfg_.pop('in_discriminator', False)
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type{}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        if name == 'bn':
            if in_discriminator:
                norm_layer = functools.partial(norm_layer, affine=True, track_running_stats=False)
            else:
                norm_layer = functools.partial(norm_layer, affine=True, track_running_stats=True)
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer
