import os

from config import args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))
import copy

import torch
import yaml
from attrdict import AttrDict

from helper import *
from lib.utils.configurator import cfg, dump_config, load_config
from lib.utils.seed_all import set_random_seed

args_copy = copy.deepcopy(args)
load_config(cfg, args.model_cfg_path)
load_config(cfg, vars(args_copy))
cfg.defrost()        # must be mutable

device = torch.device("cuda", cfg.local_rank)

distributed = cfg.num_gpus > 1


if distributed:
    torch.cuda.set_device(cfg.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

# create working directory
if not os.path.exists(cfg.work_dir):
    os.makedirs(cfg.work_dir, exist_ok=True)

# init logger
logger = Logger(name='Cycle-GAN', local_rank=cfg.local_rank, save_dir=cfg.work_dir)
# logger.log(str(cfg))
logger.log(f'Set random seed to {cfg.seed}')
set_random_seed(cfg.seed)

# create samples directory
if cfg.local_rank == 0:
    cfg.sample_dir = os.path.join(cfg.work_dir, 'samples')
    if not os.path.exists(cfg.sample_dir):
        os.mkdir(cfg.sample_dir)

################################### build dataloader ##################################
train_dataloader, val_dataloader = get_dataloader(cfg)
fixed_training_samples = {}
fixed_testing_samples = {}
for i, data in enumerate(train_dataloader):
    if i >= 2:
        break
    for k, v in data.items():
        if k not in fixed_training_samples:
            fixed_training_samples.setdefault(k, [])
        fixed_training_samples[k].append(v)

for i, data in enumerate(val_dataloader):
    if i >= 2:
        break
    for k, v in data.items():
        if k not in fixed_testing_samples:
            fixed_testing_samples.setdefault(k, [])
        fixed_testing_samples[k].append(v)
for k, v in fixed_training_samples.items():
    if not isinstance(v[0], torch.Tensor):
        continue
    fixed_training_samples[k] = torch.cat(fixed_training_samples[k], dim=0)
for k, v in fixed_testing_samples.items():
    if not isinstance(v[0], torch.Tensor):
        continue
    fixed_testing_samples[k] = torch.cat(fixed_testing_samples[k], dim=0)

########################################################################################



##################################### build model #####################################
from lib.models.build_model import build_model

cfg.steps_per_epoch = len(train_dataloader)
model_cfg = dict(
    type=cfg.model.type,
    device=device,
    distributed=distributed,
    args=cfg,
    is_train=True,
    fixed_testing_samples=fixed_testing_samples,
    fixed_training_samples=fixed_training_samples,
    num_classes = cfg.num_classes,
    out_height = cfg.out_height,
    out_width = cfg.out_width,
    label_wall = cfg.label_wall,
    label_floor = cfg.label_floor,
    label_ceiling = cfg.label_ceiling,
)

model = build_model(model_cfg)

if cfg.load_from is not None:
    logger.log(f'load checkpoint from {cfg.load_from}')
    model.load_from(cfg.load_from)

if cfg.resume_from is not None:
    start_epoch = model.resume_from(cfg.resume_from)
    print(start_epoch)
    assert cfg.epoch == start_epoch

if cfg.local_rank == 0:
    with open(os.path.join(cfg.work_dir, 'config.yaml'), "w", encoding="utf-8") as f:
        dump_config(f)
########################################################################################


##################################### training #####################################

best = float('inf')

if cfg.clip_grad:
    clip_grad = ClipGrads(grad_clip={'max_norm': cfg.max_norm,
                                     'norm_type': cfg.norm_type})
else:
    clip_grad = lambda x : x

from lib.evaluator.evaluator import DistEval, Eval

evaluator = Eval(val_dataloader, device=device) if not distributed else DistEval(val_dataloader, device=device)

num_iters_per_epoch = len(train_dataloader)
_iters = cfg.epoch * num_iters_per_epoch

for epoch in range(cfg.epoch, cfg.n_epochs):
    if distributed:
        train_dataloader.sampler.set_epoch(epoch)
    step_losses = dict()

    for i, data in enumerate(train_dataloader):
        # put data into model
        model.set_input(data)
        # forward , backward, and update model's parameters
        loss_dict = model.optimize_parameters()
        for k in loss_dict:
            if k not in step_losses:
                step_losses[k] = MovingAverage(loss_dict[k],
                                               window_size=cfg.log_interval)
            else:
                step_losses[k].push(loss_dict[k])

        if (i + 1) % cfg.log_interval == 0:
            log_msg = "Train - Epoch [{}][{}/{}]".format(epoch+1,
                                                         i + 1,
                                                         num_iters_per_epoch)
            for name in step_losses:
                val = step_losses[name].avg()
                log_msg += "{}: {:.4f}| ".format(name, val)
                logger.scalar_summary(name, val, _iters + 1)

            logger.log(log_msg)

        if (_iters + 1) % cfg.sample_interval == 0:
            if cfg.local_rank == 0:
                model.save_samples(save_dir=cfg.sample_dir, iters=_iters)

        _iters += 1

    if (epoch + 1) % cfg.save_interval == 0 or (epoch + 1) == cfg.n_epochs:
        if cfg.local_rank == 0:
            model.save_ckpt(filename=os.path.join(cfg.work_dir, f'epoch_{epoch + 1}.pth'),
                            meta=dict(epoch=epoch + 1,
                                      batch_size=cfg.batch_size,
                                      gpus=cfg.num_gpus))

    if (epoch + 1) % cfg.val_interval == 0:
        ret = evaluator.evaluate(model, logger)
        cur_criterion = float(ret[cfg.criterion_to_get_best_ckpt])
        if cur_criterion < best:
            best = cur_criterion
            if cfg.local_rank == 0:
                model.save_ckpt(filename=os.path.join(cfg.work_dir, f'best.pth'),
                                meta=dict(epoch=epoch + 1,
                                          batch_size=cfg.batch_size,
                                          gpus=cfg.num_gpus))
    model.update_learning_rate()
    del step_losses
