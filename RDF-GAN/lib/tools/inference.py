
import os
from config import args
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))

import torch
from lib.utils.seed_all import set_random_seed
from lib.tools.helper import load_checkpoint

import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from main import build_model
import cv2


def colored_depth_map(depth, d_min=None, d_max=None, cmap=plt.cm.viridis):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)

    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def save_img(arr, save_path):
    img = Image.fromarray(arr)
    img.save(save_path)


def load_rgb_depth_from_file(rgb_file, depth_file):
    rgb = cv2.imread(rgb_file, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

    return rgb.astype(np.float32), depth.astype(np.float32)


def np2tensor(arr):
    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, 2)
    arr = np.transpose(arr, (2, 0, 1))
    arr = torch.from_numpy(arr)

    return arr.unsqueeze(0)


if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir, exist_ok=True)

set_random_seed(args.seed)

generator, discriminator = build_model(args)
device = torch.device("cuda", args.local_rank)
generator = generator.to(device)
discriminator = discriminator.to(device)

load_checkpoint(model=dict(generator=generator,
                           discriminator=discriminator),
                filename=args.load_from,
                map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))

generator.eval()

depth_file = args.depth_file
rgb_file = args.rgb_file
rgb_mean, rgb_std = np.array([0.485, 0.456, 0.406], dtype=np.float32), np.array([0.229, 0.224, 0.225], dtype=np.float32)
depth_mean, depth_std = np.array([1.5], dtype=np.float32), np.array([1.5], dtype=np.float32)

rgb, depth = load_rgb_depth_from_file(rgb_file, depth_file)

# normalization
rgb = rgb / 255.
rgb = (rgb - rgb_mean) / rgb_std
depth = depth / 1000.
depth = (depth - depth_mean) / depth_std

rgb, depth = np2tensor(rgb), np2tensor(depth)


depth1, _, depth2, _, final_depth = generator(rgb.to(device),
                                              depth.to(device))

pred_depth = final_depth.cpu().squeeze().detach().numpy()
pred_depth = pred_depth * depth_std + depth_mean
depth = depth.cpu().squeeze().detach().numpy()
depth = depth * depth_std + depth_mean
