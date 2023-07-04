from .misc import collect_result_cpu
from lib.utils.progressbar import ProgressBar
from lib.utils.dist_utils import get_dist_info
import torch
import torch.nn.functional as F
from torch import distributed as dist
import time


class Eval:
    def __init__(self, data_loader=None, device=None):
        self.data_loader = data_loader
        self.device = device

    def inference(self, model, tmpdir=None):
        results = []
        dataset = self.data_loader.dataset
        prog_bar = ProgressBar(len(dataset))

        std, mean = dataset.depth_std[0], dataset.depth_mean[0]


        model.eval()
        for idx, data in enumerate(self.data_loader):
            with torch.no_grad():
                depth1, _, depth2, _, final_depth = model(data['rgb'].to(self.device),
                                                          data['raw_depth'].to(self.device))
                pred_depth_map = final_depth
                # pred_depth_map = depth2
                pred_depth_map_reshape = []

                if 'gt_depth_origin' not in data:
                    gt = data['gt_depth'] * std + mean    # (b, 1, h, w)
                else:
                    gt = data['gt_depth_origin']
                    # gt = data['gt_depth'] * std + mean
                for i, depth_origin in enumerate(gt):     # list
                    h, w = depth_origin.shape[-2:]
                    tmp = F.interpolate(pred_depth_map[i:i+1],
                                        (h, w),
                                        mode='bilinear',
                                        align_corners=False)

                    # need un-normalization
                    pred_depth_map_reshape.append(tmp * std + mean)

            tmp = []
            for k in range(len(gt)):
                sample = dict(gt=gt[k].squeeze(),
                              pd=pred_depth_map_reshape[k].cpu().squeeze())
                if 'evaluate_mask' in data:
                    sample.update({'evaluate_mask': data['evaluate_mask'][k].cpu().squeeze()})
                tmp.append(sample)
            results.extend(tmp)

            # results.extend([dict(
            #     gt=gt[k].squeeze(),
            #     # gt=gt[k].cpu().squeeze(),
            #     pd=pred_depth_map_reshape[k].cpu().squeeze(),
            #     evaluate_mask=data['evaluate_mask'][k].cpu().squeeze()
            # ) for k in range(len(gt))])

            # path_save_pred = 'vis_tmp/pred_depth/{:04d}.png'.format(idx)
            # pred = pred_depth_map_reshape[-1].cpu().squeeze().numpy()
            # # pred = (pred * 256.0).astype(np.uint16)
            # pred = (pred * 25.5).astype(np.uint8)
            # pred = Image.fromarray(pred)
            # pred.save(path_save_pred)

            batch_size = len(gt)
            for _ in range(batch_size):
                prog_bar.update()

        return results

    def evaluate(self, model, logger=None):
        results = self.inference(model)
        rank, world_size = get_dist_info()
        if rank == 0:
            self.data_loader.dataset.evaluate(results, logger)


class DistEval(Eval):
    def __init__(self, data_loader=None, device=None, mean=32767.5, std=32767.5):
        super(DistEval, self).__init__(data_loader, device, mean, std)

    def inference(self, model, tmpdir=None, gpu_collect=False):
        results = []
        rank, world_size = get_dist_info()
        dataset = self.data_loader.dataset
        std, mean = dataset.depth_std[0], dataset.depth_mean[0]

        if rank == 0:
            prog_bar = ProgressBar(len(dataset))
        time.sleep(2)
        model.eval()
        for i, data in enumerate(self.data_loader):
            with torch.no_grad():
                depth1, _, depth2, _, final_depth = model(data['rgb'].to(self.device),
                                                          data['raw_depth'].to(self.device))
                pred_depth_map = final_depth
                pred_depth_map_reshape = []
                for i, depth_origin in enumerate(data['gt_depth_origin']):
                    h, w = depth_origin.shape
                    tmp = F.interpolate(pred_depth_map[i:i+1],
                                        (h, w),
                                        mode='bilinear',
                                        align_corners=False)
                    pred_depth_map_reshape.append(tmp * std + mean)
            results.extend([dict(gt=data['gt_depth_origin'][k],
                                 pd=pred_depth_map_reshape[k].squeeze().cpu().numpy()) for k in
                            range(len(data['gt_depth_origin']))])

            if rank == 0:
                batch_size = len(data['gt_depth_origin'])
                for _ in range(batch_size):
                    prog_bar.update()

        # collect results from all ranks
        if gpu_collect:
            raise NotImplementedError('the gpu collect has not been implemented yet.')
        else:
            results = collect_result_cpu(results, len(dataset), tmpdir)

        return results
