import os
import torch
from .dist_utils import get_dist_info
from collections import OrderedDict


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


def save_checkpoint(module, filename, optimizer=None, lr_scheduler=None, meta=None):

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


def load_checkpoint(model: dict, filename, map_location=None, strict=False, logger=None):
    assert isinstance(model, dict)
    checkpoint = torch.load(filename, map_location=map_location)

    for prefix_key in model:
        key = f'{prefix_key}_state_dict'
        state_dict = checkpoint[key]
        load_state_dict(model[prefix_key], state_dict, strict, logger)

    return checkpoint

# load model
def load_checkpoint_2(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    print("Normal Load checkpoint from ",fpath," finish")
    return model

def resume_from(models: dict,
                filename,
                map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()),
                logger=None,
                optimizers=None,
                lr_schedulers=None):
    checkpoint = load_checkpoint(models, filename, map_location=map_location, strict=True, logger=logger)

    assert optimizers is not None and lr_schedulers is not None

    if optimizers is not None:
        assert isinstance(optimizers, dict)
        for prefix_key in optimizers:
            optimizer_key = f'{prefix_key}_optimizer_state_dict'
            optimizers[prefix_key].load_state_dict(checkpoint[optimizer_key])

    if lr_schedulers is not None:
        assert isinstance(lr_schedulers, dict)
        for prefix_key in lr_schedulers:
            scheduler_key = f'{prefix_key}_lr_scheduler_state_dict'
            lr_schedulers[prefix_key].load_state_dict(checkpoint[scheduler_key])

    start_epoch = checkpoint['meta']['epoch']
    if logger is not None:
        logger.log(f'resumed from epoch: {start_epoch}')

    return start_epoch
