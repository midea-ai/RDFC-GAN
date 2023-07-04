import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


def colored_depth_map(depth, d_min=None, d_max=None, cmap=plt.cm.viridis):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)

    depth_relative = (depth - d_min) / (d_max - d_min)

    # x = 255 * cmap(depth_relative)[..., :3]

    return 255 * cmap(depth_relative)[..., :3]  # H, W, C


def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    c = x.shape[1]

    x = x.copy()
    if c == 3:
        x = x * 127.5 + 127.5
    elif c == 1:
        x = x.squeeze(1)
        x = colored_depth_map(x).transpose([0, 3, 1, 2])
    else:
        pass

    return x


def merge_images(sources, num_imgs_per_scene, batch_size=16):
    assert isinstance(sources, list)
    _, _, h, w = sources[0].shape
    row = int(np.sqrt(batch_size))
    merged = np.zeros([3, row * h, row * w * num_imgs_per_scene])
    max_num = row * row
    for idx, item in enumerate(zip(*sources)):
        if idx >= max_num:
            break
        i = idx // row
        j = idx % row
        for k, t in enumerate(item):
            merged[:, i * h:(i + 1) * h, (j * num_imgs_per_scene + k) * w:(j * num_imgs_per_scene + k + 1) * w] = t

    merged = merged.transpose(1, 2, 0)

    return merged

def color_label(pred,palette,n):
    if torch.cuda.is_available():
        pred = pred.cpu()

    b,h,w = pred.shape
    seg_color = torch.zeros((b,3,h,w))
    
    for c in range(n):
        seg_color[:,0,:,:] += ((pred == c) * palette[c][0]) 
        seg_color[:,1,:,:] += ((pred == c) * palette[c][1])
        seg_color[:,2,:,:] += ((pred == c) * palette[c][2])

    return seg_color



