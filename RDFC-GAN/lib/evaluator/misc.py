import tempfile

import os
import torch
import shutil
from lib.utils.dist_utils import get_dist_info
from lib.utils.path import mkdir_or_exist
import torch.distributed as dist
import pickle


def collect_result_cpu(result_part, size, tmpdir=None):
    """useful when using distributed training"""
    rank, world_size = get_dist_info()
    # create a tmp dir if it not specified
    if tmpdir is None:
        MAX_LEN = 512
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device="cuda")
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mkdir_or_exist(tmpdir)

    # dump the part result to the dir
    with open(os.path.join(tmpdir, f'part_{rank}.pkl'), 'wb') as f:
        pickle.dump(result_part, f)
    dist.barrier()

    # print('hello')

    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = os.path.join(tmpdir, f'part_{i}.pkl')
            with open(part_file, 'rb') as f:
                part_list.append(pickle.load(f))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results
