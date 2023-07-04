# import torch.nn.functional as F
# from torch import distributed as dist
import time

import torch
from lib.utils.dist_utils import get_dist_info
from lib.utils.progressbar import ProgressBar

from .misc import collect_result_cpu


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
                ret = model(**data)
                fake_depth = ret['pred_depth'] if isinstance(ret, dict) else ret
                fake_depth = fake_depth * std + mean
                real_depth = data['gt_depth'] * std + mean

            results.extend([dict(

                gt=real_depth[k].cpu().squeeze(),
                pd=fake_depth[k].cpu().squeeze(),
                # evaluate_mask=data['evaluate_mask'][k].cpu().squeeze()
                # if isinstance(data['evaluate_mask'], torch.Tensor) else data['evaluate_mask'][k].squeeze()

            ) for k in range(len(fake_depth))])

            batch_size = len(fake_depth)
            for _ in range(batch_size):
                prog_bar.update()
        model.train()

        return results

    def evaluate(self, model, logger=None):
        results = self.inference(model)
        rank, world_size = get_dist_info()
        if rank == 0:
            ret = self.data_loader.dataset.evaluate(results, logger)

            return ret


class DistEval(Eval):
    def __init__(self, data_loader=None, device=None):
        super(DistEval, self).__init__(data_loader, device)

    def inference(self, model, tmpdir=None, gpu_collect=False):
        results = []
        rank, world_size = get_dist_info()
        dataset = self.data_loader.dataset
        std, mean = dataset.depth_std[0], dataset.depth_mean[0]

        if rank == 0:
            prog_bar = ProgressBar(len(dataset))
        time.sleep(2)
        model.eval()
        for idx, data in enumerate(self.data_loader):
            with torch.no_grad():
                ret = model(**data)         # NOTE: Wrap the forward results into dict format
                fake_depth = ret['pred_depth'] if isinstance(ret, dict) else ret
                fake_depth = fake_depth * std + mean
                real_depth = data['gt_depth'] * std + mean

            results.extend([dict(
                gt=real_depth[k].cpu().squeeze(),
                pd=fake_depth[k].cpu().squeeze(),
                evaluate_mask=data['evaluate_mask'][k].cpu().squeeze()
                if isinstance(data['evaluate_mask'], torch.Tensor) else data['evaluate_mask'][k].squeeze()
            ) for k in range(len(fake_depth))])

            if rank == 0:
                batch_size = len(fake_depth)
                for _ in range(batch_size):
                    prog_bar.update()

        model.train()

        # collect results from all ranks
        if gpu_collect:
            raise NotImplementedError('the gpu collect has not been implemented yet.')
        else:
            results = collect_result_cpu(results, len(dataset), tmpdir)

        return results
