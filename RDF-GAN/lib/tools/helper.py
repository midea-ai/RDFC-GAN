import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
from collections import OrderedDict



def save_samples(sum_iters, fixed_rgb, fixed_raw_depth, fixed_gt_depth, generator, depth_masks=None, sample_dir='./'):
    with torch.no_grad():
        depth1, conf1, depth2, conf2 = generator(fixed_rgb, fixed_raw_depth)
    conf = torch.cat([conf1, conf2], dim=1)
    conf_score = F.softmax(conf, 1)
    pred_depth_map = torch.cat([depth1, depth2], dim=1)
    pred_depth_map = torch.sum(pred_depth_map * conf_score, dim=1, keepdim=True)

    pred_depth = pred_depth_map.cpu().numpy()
    real_depth = fixed_gt_depth.cpu().numpy()
    pred_depth, real_depth = to_data(pred_depth), to_data(real_depth)

    if depth_masks is not None:
        depth_masks = depth_masks.numpy()
        real_depth[depth_masks] = 0

    real_depth = real_depth.astype(np.uint16)
    pred_depth = pred_depth.astype(np.uint16)
    merged = merge_image(real_depth, pred_depth)
    im = Image.fromarray(merged.squeeze(-1))
    im.save(os.path.join(sample_dir, f'sample-{sum_iters:06d}.png'))


def to_data(x, mean=32767.5, std=32767.5):
    x = x.copy()
    x = x * std + mean

    return x


def merge_image(sources, targets):
    b, c, h, w = sources.shape

    row = int(np.sqrt(b))
    merged = np.zeros([c, row * h, row * w * 2], dtype=np.uint16)

    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        if i == row:
            break
        merged[:, i * h:(i + 1) * h, (j * 2) * w:(j * 2 + 1) * w] = s
        merged[:, i * h:(i + 1) * h, (j * 2 + 1) * w:(j * 2 + 2) * w] = t
    merged = merged.transpose([1, 2, 0])

    return merged



# ---------------------------------------logger helper class--------------------------------------- #
import logging
import time
logger_initialized = {}


class Logger:
    def __init__(self, name, local_rank=0, save_dir='./', use_tensorboard=True):
        self.local_rank = local_rank
        logger = logging.getLogger(name)

        if name in logger_initialized:
            return logger
        for logger_name in logger_initialized:
            if name.startswith(logger_name):
                return logger

        stream_handler = logging.StreamHandler()
        handlers = [stream_handler]

        if local_rank == 0:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            log_file = os.path.join(save_dir, f'{timestamp}.log')
            file_handler = logging.FileHandler(log_file, 'w')
            handlers.append(file_handler)
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(
            fmt=fmt, datefmt="%m-%d %H:%M:%S")
        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)

        if local_rank == 0:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)

        logger_initialized[name] = True

        self.logger = logger

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    "the dependencies to use torch.utils.tensorboard "
                    "(applicable to PyTorch 1.1 or higher)"
                )
            self.writer = SummaryWriter(os.path.join(save_dir, 'tf_logs'))
        else:
            self.writer = None

    def log(self, log_msg):
        if self.local_rank < 1:
            self.logger.info(log_msg)

    def scalar_summary(self, tag, value, step):
        if self.local_rank < 1:
            self.writer.add_scalar(tag, value, step)

    def close(self):
        """Close tensorboard connection"""
        if self.local_rank < 1:
            if self.writer is not None:
                self.writer.close()
            else:
                pass


class MovingAverage:
    def __init__(self, val, window_size=50):
        self.window_size = window_size
        self.reset()
        self.push(val)

    def reset(self):
        self.queue = []

    def push(self, val):
        self.queue.append(val)
        if len(self.queue) > self.window_size:
            self.queue.pop(0)

    def avg(self):
        return np.mean(self.queue)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def reduce_loss(loss):
    reduced_loss = dict()
    if dist.is_available() and dist.is_initialized():
        for k, v in loss.items():
            v = v.data.clone()
            dist.all_reduce(v.div_(dist.get_world_size()))
            reduced_loss[k] = v.item()

    else:
        for k, v in loss.items():
            reduced_loss[k] = v.item()
    return reduced_loss


def get_model_state_dict(model):
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    model_state_dict_cpu = weights_to_cpu(model_state_dict)
    return model_state_dict_cpu


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu."""
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def save_checkpoint(module, filename, optimizer, lr_scheduler, meta=None):

    checkpoint = {'meta': meta}

    if isinstance(module, torch.nn.Module):
        checkpoint.update({'model_state_dict': get_model_state_dict(module)})
    elif isinstance(module, dict):
        for k in module:
            # assert isinstance(module[k], torch.nn.Module)
            checkpoint.update({f'{k}_state_dict': get_model_state_dict(module[k])})
    else:
        raise NotImplementedError

    if optimizer is not None:
        if isinstance(optimizer, torch.optim.Optimizer):
            checkpoint.update({'optimizer_state_dict': optimizer.state_dict()})
        elif isinstance(optimizer, dict):
            for name in optimizer:
                checkpoint.update({f'{name}_optimizer_state_dict': optimizer[name].state_dict()})
        else:
            raise NotImplementedError

    if lr_scheduler is not None:
        if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
            checkpoint.update({'lr_scheduler_state_dict': lr_scheduler.state_dict()})
        elif isinstance(lr_scheduler, dict):
            for name in lr_scheduler:
                checkpoint.update({f'{name}_lr_scheduler_state_dict': lr_scheduler[name].state_dict()})
        else:
            raise NotImplementedError

    _dir = os.path.dirname(filename)
    if not os.path.exists(_dir):
        os.mkdir(_dir)

    with open(filename, 'wb') as f:
        torch.save(checkpoint, f)
        f.flush()

def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if hasattr(module, 'module'):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()

    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.log(err_msg)
        else:
            print(err_msg)


def load_checkpoint_vanilla(model: dict, filename, map_location=None, strict=False, logger=None):
    checkpoint = torch.load(filename, map_location=map_location)

    prefix_keys = ['generator', 'disc_rgb', 'disc_depth']

    for i, prefix_key in enumerate(prefix_keys):
        key = f'{prefix_key}_state_dict'
        state_dict = checkpoint[key]
        load_state_dict(model[prefix_key], state_dict, strict, logger)

    return checkpoint


def resume_from_vanilla(models: dict,
                        filename,
                        map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()),
                        logger=None,
                        optimizers=None,
                        lr_schedulers=None):
    checkpoint = load_checkpoint_vanilla(models, filename, map_location=map_location, strict=True, logger=logger)

    assert optimizers is not None and lr_schedulers is not None

    prefix_keys = ['generator', 'disc_rgb', 'disc_depth']
    # resume optimizer & lr_scheduler
    for i, prefix_key in enumerate(prefix_keys):
        optimizer_key = f'{prefix_key}_optimizer_state_dict'
        scheduler_key = f'{prefix_key}_lr_scheduler_state_dict'
        optimizers[prefix_key].load_state_dict(checkpoint[optimizer_key])
        lr_schedulers[prefix_key].load_state_dict(checkpoint[scheduler_key])

    start_epoch = checkpoint['meta']['epoch']

    # iter is not saved
    # start_iter = checkpoint['meta']['iter']

    logger.log(f'resumed from epoch: {start_epoch}')

    return start_epoch


def load_checkpoint(model: dict, filename, map_location=None, strict=False, logger=None):
    checkpoint = torch.load(filename, map_location=map_location)

    prefix_keys = ['generator', 'discriminator']

    for i, prefix_key in enumerate(prefix_keys):
        key = f'{prefix_key}_state_dict'
        state_dict = checkpoint[key]
        load_state_dict(model[prefix_key], state_dict, strict, logger)

    return checkpoint


def resume_from(models: dict,
                filename,
                map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()),
                logger=None,
                optimizers=None,
                lr_schedulers=None):
    checkpoint = load_checkpoint(models, filename, map_location=map_location, strict=True, logger=logger)

    assert optimizers is not None and lr_schedulers is not None

    prefix_keys = ['generator', 'discriminator']
    # resume optimizer & lr_scheduler
    for i, prefix_key in enumerate(prefix_keys):
        optimizer_key = f'{prefix_key}_optimizer_state_dict'
        scheduler_key = f'{prefix_key}_lr_scheduler_state_dict'
        optimizers[prefix_key].load_state_dict(checkpoint[optimizer_key])
        lr_schedulers[prefix_key].load_state_dict(checkpoint[scheduler_key])

    start_epoch = checkpoint['meta']['epoch']

    # iter is not saved
    # start_iter = checkpoint['meta']['iter']

    logger.log(f'resumed from epoch: {start_epoch}')

    return start_epoch


def init_weights(net, init_type='kaiming', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print(f'Apply {init_type} init on {net.__class__.__name__}')
    net.apply(init_func)  # apply the initialization function <init_func>


class LRFactor:
    def __init__(self, decay, gamma):
        assert len(decay) == len(gamma)

        self.decay = decay
        self.gamma = gamma

    def get_factor(self, epoch):
        for (d, g) in zip(self.decay, self.gamma):
            if epoch < d:
                return g
        return self.gamma[-1]


def statistic_params(model: torch.nn.Module):
    """Helper function for counting the number of model parameters."""

    total_num = sum([p.numel() for n, p in model.named_parameters() if "global_guidance_module" not in n and p.requires_grad])

    print(f'Total parameter: {total_num / 1e6} M')

