import torch
import numpy as np
from .base import BaseMetric


class RDFGANMetric(BaseMetric):
    def __init__(self,
                 t_valid=1e-4):
        super(RDFGANMetric, self).__init__()

        self.t_valid = t_valid

        self.metric_name = ['RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3']

    def evaluate_batch(self, gt, pred):
        if not isinstance(gt, torch.Tensor):
            gt = torch.from_numpy(gt)
        if not isinstance(pred, torch.Tensor):
            pred = torch.from_numpy(pred)

        mask = gt > self.t_valid
        num_valid = mask.sum()

        pred = pred[mask]
        gt = gt[mask]

        # RMSE / MAE
        diff = pred - gt
        diff_abs = torch.abs(diff)
        diff_sqr = torch.pow(diff, 2)

        rmse = diff_sqr.sum() / (num_valid + 1e-8)
        rmse = torch.sqrt(rmse)

        mae = diff_abs.sum() / (num_valid + 1e-8)

        # Rel
        rel = diff_abs / (gt + 1e-8)
        rel = rel.sum() / (num_valid + 1e-8)

        # delta
        r1 = gt / (pred + 1e-8)
        r2 = pred / (gt + 1e-8)
        ratio = torch.max(r1, r2)

        del_1 = (ratio < 1.25).type_as(ratio)
        del_2 = (ratio < 1.25 ** 2).type_as(ratio)
        del_3 = (ratio < 1.25 ** 3).type_as(ratio)
        del_1 = del_1.sum() / (num_valid + 1e-8)
        del_2 = del_2.sum() / (num_valid + 1e-8)
        del_3 = del_3.sum() / (num_valid + 1e-8)

        result = [rmse, mae, rel, del_1, del_2, del_3]
        result = torch.stack(result)
        result = torch.unsqueeze(result, dim=0).detach()

        return result

    def evaluate_all(self, results, logger=None):

        metrics = []

        for i, result in enumerate(results):

            gt = result['gt']
            pred = result['pd']
            if not isinstance(gt, torch.Tensor):
                gt = torch.from_numpy(gt)
            if not isinstance(pred, torch.Tensor):
                pred = torch.from_numpy(pred)

            evaluate_mask = result.get('evaluate_mask', torch.ones_like(gt, dtype=bool))

            pred_inv = 1.0 / (pred + 1e-8)
            gt_inv = 1.0 / (gt + 1e-8)

            # mask = gt > self.t_valid
            mask = (gt > self.t_valid) & evaluate_mask
            num_valid = mask.sum()

            pred = pred[mask]
            gt = gt[mask]

            pred_inv = pred_inv[mask]
            gt_inv = gt_inv[mask]

            pred_inv[pred <= self.t_valid] = 0.0
            gt_inv[gt <= self.t_valid] = 0.0

            # RMSE / MAE
            diff = pred - gt
            diff_abs = torch.abs(diff)
            diff_sqr = torch.pow(diff, 2)

            rmse = diff_sqr.sum() / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = diff_abs.sum() / (num_valid + 1e-8)

            # iRMSE / iMAE
            diff_inv = pred_inv - gt_inv
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = torch.pow(diff_inv, 2)

            irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = diff_inv_abs.sum() / (num_valid + 1e-8)

            # Rel
            rel = diff_abs / (gt + 1e-8)
            rel = rel.sum() / (num_valid + 1e-8)

            # delta
            r1 = gt / (pred + 1e-8)
            r2 = pred / (gt + 1e-8)
            ratio = torch.max(r1, r2)

            del_1 = (ratio < 1.25).type_as(ratio)
            del_2 = (ratio < 1.25 ** 2).type_as(ratio)
            del_3 = (ratio < 1.25 ** 3).type_as(ratio)
            del_1 = del_1.sum() / (num_valid + 1e-8)
            del_2 = del_2.sum() / (num_valid + 1e-8)
            del_3 = del_3.sum() / (num_valid + 1e-8)

            metric = [rmse, mae, irmse, imae, rel, del_1, del_2, del_3]
            metric = torch.stack(metric)
            metric = torch.unsqueeze(metric, dim=0).detach()

            metrics.append(metric)

            # print(f'\r{i + 1}/{len(results)}', end='')


        metrics = np.concatenate(metrics, axis=0)
        metrics = np.mean(metrics, axis=0, keepdims=True)

        ret = {}
        for idx, name in enumerate(self.metric_name):
            ret[name] = metrics[0, idx]

        if logger is not None:
            logger.log(' ')
            for k, v in ret.items():
                logger.log(f'{k}: {v}')

        else:
            for k, v in ret.items():
                print(f'{k}: {v}')
