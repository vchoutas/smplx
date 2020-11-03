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

import os
import os.path as osp
import pickle

import numpy as np
import torch
from loguru import logger

from .typing import Tensor


def read_deformation_transfer(
    deformation_transfer_path: str,
    device=None,
    use_normal: bool = False,
) -> Tensor:
    ''' Reads a deformation transfer
    '''
    if device is None:
        device = torch.device('cpu')
    assert osp.exists(deformation_transfer_path), (
        'Deformation transfer path does not exist:'
        f' {deformation_transfer_path}')
    logger.info(
        f'Loading deformation transfer from: {deformation_transfer_path}')
    # Read the deformation transfer matrix
    with open(deformation_transfer_path, 'rb') as f:
        def_transfer_setup = pickle.load(f, encoding='latin1')
    if 'mtx' in def_transfer_setup:
        def_matrix = def_transfer_setup['mtx']
        if hasattr(def_matrix, 'todense'):
            def_matrix = def_matrix.todense()
        def_matrix = np.array(def_matrix, dtype=np.float32)
        if not use_normal:
            num_verts = def_matrix.shape[1] // 2
            def_matrix = def_matrix[:, :num_verts]
    elif 'matrix' in def_transfer_setup:
        def_matrix = def_transfer_setup['matrix']
    else:
        valid_keys = ['mtx', 'matrix']
        raise KeyError(f'Deformation transfer setup must contain {valid_keys}')

    def_matrix = torch.tensor(def_matrix, device=device, dtype=torch.float32)
    return def_matrix


def apply_deformation_transfer(
    def_matrix: Tensor,
    vertices: Tensor,
    faces: Tensor,
    use_normals=False
) -> Tensor:
    ''' Applies the deformation transfer on the given meshes
    '''
    if use_normals:
        raise NotImplementedError
    else:
        def_vertices = torch.einsum('mn,bni->bmi', [def_matrix, vertices])
        return def_vertices
