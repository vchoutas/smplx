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

from typing import Optional, Tuple

import sys
import os
import os.path as osp

import numpy as np
from psbody.mesh import Mesh
import trimesh

import torch
from torch.utils.data import Dataset
from loguru import logger


class MeshFolder(Dataset):
    def __init__(
        self,
        data_folder: str,
        transforms=None,
        exts: Optional[Tuple] = None
    ) -> None:
        ''' Dataset similar to ImageFolder that reads meshes with the same
            topology
        '''
        if exts is None:
            exts = ['.obj', '.ply']

        self.data_folder = osp.expandvars(data_folder)

        logger.info(
            f'Building mesh folder dataset for folder: {self.data_folder}')

        self.data_paths = np.array([
            osp.join(self.data_folder, fname)
            for fname in os.listdir(self.data_folder)
            if any(fname.endswith(ext) for ext in exts)
        ])
        self.num_items = len(self.data_paths)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index):
        mesh_path = self.data_paths[index]

        # Load the mesh
        mesh = trimesh.load(mesh_path, process=False)

        return {
            'vertices': np.asarray(mesh.vertices, dtype=np.float32),
            'faces': np.asarray(mesh.faces, dtype=np.int32),
            'indices': index,
            'paths': mesh_path,
        }
