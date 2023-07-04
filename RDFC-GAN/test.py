"""
Single GPU testing.
"""
import os

from config import args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))
import imageio
import lib.utils.save_vis as vis
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from helper import  get_dataloader
from lib.evaluator.evaluator import Eval
from lib.models.build_model import build_model
from lib.utils.configurator import cfg, load_config
from lib.utils.seed_all import set_random_seed
from PIL import Image

load_config(cfg, args.model_cfg_path)
load_config(cfg, vars(args))
device = torch.device("cuda", cfg.local_rank)

os.makedirs(cfg.work_dir, exist_ok=True)


print(f'Set random seed to {cfg.seed}')
set_random_seed(cfg.seed)

# build dataloader
_, val_dataloader = get_dataloader(cfg)

# build model
model_cfg = dict(
    type=cfg.model.type,
    device=device,
    distributed=False,
    args=cfg,
    is_train=False,
    num_classes = cfg.num_classes,
    out_height = cfg.out_height,
    out_width = cfg.out_width,
    label_wall = cfg.label_wall,
    label_floor = cfg.label_floor,
    label_ceiling = cfg.label_ceiling,
)
model = build_model(model_cfg)

if cfg.load_from is not None:
    print(f'load checkpoint from {cfg.load_from}')
    model.load_from(cfg.load_from)

mean, std = 5.0, 5.0

filename = os.path.join(args.data_root,'test.txt')
with open(filename, 'r') as f:
    file_list = f.read().splitlines()

assert args.batch_size == 1

# test model's matrics

evaluator = Eval(val_dataloader, device=device)
evaluator.evaluate(model)


# save pred depth

# os.makedirs(cfg.work_dir+"/png_refined_depth",exist_ok=True)
# model.eval()
# for idx, data in enumerate(val_dataloader):

#     with torch.no_grad():
#         ret = model(**data)

#     pred_depth = ret['pred_depth'] if isinstance(ret, dict) else ret 
#     pred_depth = pred_depth * std + mean
#     pred_depth = pred_depth.detach().squeeze().cpu().numpy()
#     pred_depth = pred_depth * 1000.0

#     imageio.imsave(os.path.join(args.work_dir,'png_refined_depth',file_list[idx] + ".png"), pred_depth.astype(np.uint16))



