# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: Vassilis Choutas, vassilis.choutas@tuebingen.mpg.de

from typing import List, Tuple
import sys

import torch
import torch.utils.data as dutils
from .datasets import MeshFolder

from loguru import logger


def build_dataloader(exp_cfg):
    dset_name = exp_cfg.datasets.name
    if dset_name == 'mesh-folder':
        mesh_folder_cfg = exp_cfg.datasets.mesh_folder
        key, *_ = mesh_folder_cfg.keys()
        value = mesh_folder_cfg[key]
        logger.info(f'{key}: {value}\n')
        dataset = MeshFolder(**mesh_folder_cfg)
    else:
        raise ValueError(f'Unknown dataset: {dset_name}')

    batch_size = exp_cfg.batch_size
    num_workers = exp_cfg.datasets.num_workers

    logger.info(
        f'Creating dataloader with B={batch_size}, workers={num_workers}')
    dataloader = dutils.DataLoader(dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=False)

    return {'dataloader': dataloader, 'dataset': dataset}
