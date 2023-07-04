import glob
import os
import random

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image
from torch.nn import init
# import imageio
from torch.nn.utils import clip_grad


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


def set_requires_grad(models, requires_grad=False):
    if not isinstance(models, list):
        models = [models]

    for m in models:
        if m is not None:
            for param in m.parameters():
                param.requires_grad = requires_grad


class ImagePool:
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by latest generators.
    """

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs  = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)

        return return_images

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


import logging
# ---------------------------------------logger helper class--------------------------------------- #
import os
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

def get_dataloader(args):
    from lib.datasets.build_dataloader import build_dataloader

    if args.dataset == 'nyuv2':
        from lib.datasets.nyuv2.nyuv2_dataset_training import \
            NYUV21400Dataset
        DATASET = NYUV21400Dataset
    
        dataset_kwargs = dict(max_depth=10.0,
                            rgb_mean=[0.5, 0.5, 0.5],
                            rgb_std=[0.5, 0.5, 0.5],
                            depth_mean=[5.0],
                            depth_std=[5.0],
                            height = args.resize_height,
                            width = args.resize_width,
                            crop_size_height = args.out_height,
                            crop_size_width = args.out_width,
                              )
        train_dataset = DATASET(data_root=args.data_root,
                                mode='train',
                                **dataset_kwargs)
        val_dataset = DATASET(data_root=args.data_root,
                              mode='test',
                              **dataset_kwargs)
    elif args.dataset == 'sunrgbd':
        from lib.datasets.sunrgbd.sunrgbd_dataset import SUNRGBDPseudoDataset
        DATASET = SUNRGBDPseudoDataset
        dataset_kwargs = dict(max_depth=10.0,
                              rgb_mean=[0.5, 0.5, 0.5],
                              rgb_std=[0.5, 0.5, 0.5],
                              depth_mean=[5.0],
                              depth_std=[5.0],
                              )
        train_dataset = DATASET(data_root=args.data_root,
                                mode='train',
                                **dataset_kwargs)

        val_dataset = DATASET(data_root=args.data_root,
                            mode='test',
                            **dataset_kwargs)


    else:
        raise NotImplementedError

    train_dataloader = build_dataloader(train_dataset,
                                        samples_per_gpu=args.batch_size,
                                        workers_per_gpu=args.num_workers,
                                        num_gpus=args.num_gpus,
                                        dist=args.num_gpus > 1,
                                        pin_memory=False,
                                        drop_last=True)
    val_dataloader = build_dataloader(val_dataset,
                                      samples_per_gpu=args.batch_size,
                                      workers_per_gpu=0,
                                      num_gpus=args.num_gpus,
                                      dist=args.num_gpus > 1,
                                      pin_memory=True,
                                      shuffle=False)

    return train_dataloader, val_dataloader


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


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


class ClipGrads:
    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def __call__(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)


class PointCloudsHandler:
    def __init__(self, intrinsic=None):
        assert isinstance(intrinsic, (list, tuple))
        self.h = intrinsic[0]
        self.w = intrinsic[1]
        self.fx = intrinsic[2]
        self.fy = intrinsic[3]
        self.cx = intrinsic[4]
        self.cy = intrinsic[5]

        self.inv_fx = 1. / self.fx
        self.inv_fy = 1. / self.fy

    def img2points(self, depth, rgb):
        invalid = depth == 0
        x, y = np.meshgrid(np.arange(self.w), np.arange(self.h))

        x3 = (x - self.cx) * depth * self.inv_fx
        y3 = (y - self.cy) * depth * self.inv_fy
        z3 = depth

        points = np.stack([x3.flatten(), y3.flatten(), z3.flatten()], 1)
        points[invalid.flatten(), :] = np.nan
        nan_idx = np.isnan(points[:, 0])
        points = points[~nan_idx]

        rgb = rgb.reshape(-1, 3)
        rgb = rgb[~nan_idx]
        points = np.concatenate([points, rgb], 1)

        return points

    def points2pcd(self, points, save_file):
        """Always save color"""
        assert os.path.splitext(save_file)[-1] == '.pcd'
        f_handle = open(save_file, 'w+')
        points_num = points.shape[0]

        f_handle.write('# .PCD v0.7 - Point Cloud Data file format\nVersion 0.7\nFIELDS x y z rgb\n')
        f_handle.write('SIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH ' + str(points_num) + '\nHEIGHT 1\n')
        f_handle.write('VIEWPOINT 0 0 0 1 0 0 0\nPOINTS ' + str(points_num) + '\nDATA ascii')

        for i in range(points_num):
            # pack rgb
            r, g, b = int(points[i, 3]), int(points[i, 4]), int(points[i, 5])
            rgb = r << 16 | g << 8 | b
            string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2]) + ' ' +str(rgb)
            f_handle.write(string)
        f_handle.close()

        print(f'points information was saved to {save_file}')
    



def addPepperNoise(img , snr = 0.98,p = 0.9): 

    """
        Args:
            img:Tensor
        Returns:
            Tensor img
    """
    assert (isinstance(snr , float) ) or (isinstance(p , float))
    
    if random.uniform(0,1) < p:
        b, c, h, w = img.shape
        noise_pct = 1 - snr
        mask = np.random.choice((0,1,2),size = (b, 1, h, w,),p = [snr, noise_pct/2., noise_pct/2.])
        mask = np.repeat(mask,c,axis = 1)
        mask = torch.from_numpy(mask)
        img[mask == 1] = 1
        img[mask == 2] = -1
    return img        

def norm_normalize(norm_out):
    norm_x, norm_y, norm_z = torch.split(norm_out, 1, dim=1)
    norm = torch.sqrt(norm_x ** 2.0 + norm_y ** 2.0 + norm_z ** 2.0) + 1e-10
    final_out = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm], dim=1)
    return final_out

def visual_depth(path,save_path):
    for pngfile in glob.glob(os.path.join(path,'*.png')):
        #print(pngfile)
        depthmap = Image.open(pngfile)    
        depthmap = np.array(depthmap)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.imshow(depthmap)             
        # plt.colorbar()                   
        plt.savefig(os.path.join(save_path,os.path.basename(pngfile)),bbox_inches='tight',pad_inches = 0)      
        #plt.show()
        plt.close()
        plt.clf()


def label_to_color(path,save_path):
    labels = cv2.imread(path,cv2.IMREAD_UNCHANGED).astype(np.int)
    h,w = labels.shape
    label_rgb = np.zeros((h,w,3),dtype=np.uint8)
    #12 is wall,3 is ceiling,5 is floor
    color = [222,241,23]
    labels_12 = labels == 5
    label_rgb[:,:,0] = labels_12 * color[0]
    label_rgb[:,:,1] = labels_12 * color[1]
    label_rgb[:,:,2] = labels_12 * color[2]

    imageio.imsave(save_path,label_rgb)


def color_label(pred,palette,n):
    if torch.cuda.is_available():
        pred = pred.cpu()

    h,w = pred.shape
    seg_color = torch.zeros((h,w,3))
    
    for c in range(n):
        seg_color[:,:,0] += ((pred == c) * palette[c][0]) 
        seg_color[:,:,1] += ((pred == c) * palette[c][1])
        seg_color[:,:,2] += ((pred == c) * palette[c][2])

    return seg_color

def visual_pcd(color_path,depth_path,intrinsics,s):
    colors = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB).astype(np.float32) /255.0
    depths = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # np.load(depth_path) #

    # convert RGB-D to point cloud
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    # depth factor
    xmap, ymap = np.arange(colors.shape[1]), np.arange(colors.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points.reshape((-1, 3))
    colors = colors.reshape((-1, 3))

    # cv2.imwrite('hw_0907/0001_depth.png', depths.astype(np.uint16)) # convert .npy to .png
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([cloud, coord])
    #o3d.visualization.draw_geometries([cloud],width = 1000,height = 800)
    return cloud
