# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import to_tensor, Array, Tensor, LBSOutput


class VertexJointSelector(nn.Module):

    def __init__(
        self,
        vertex_ids: Dict[str, int] = None,
        use_hands=True,
        use_feet_keypoints=True,
        **kwargs
    ) -> None:
        super(VertexJointSelector, self).__init__()

        extra_joints_idxs = []

        face_keyp_idxs = np.array([
            vertex_ids['nose'],
            vertex_ids['reye'],
            vertex_ids['leye'],
            vertex_ids['rear'],
            vertex_ids['lear']], dtype=np.int64)

        extra_joints_idxs = np.concatenate([extra_joints_idxs,
                                            face_keyp_idxs])

        if use_feet_keypoints:
            feet_keyp_idxs = np.array([vertex_ids['LBigToe'],
                                       vertex_ids['LSmallToe'],
                                       vertex_ids['LHeel'],
                                       vertex_ids['RBigToe'],
                                       vertex_ids['RSmallToe'],
                                       vertex_ids['RHeel']], dtype=np.int32)

            extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, feet_keyp_idxs])

        if use_hands:
            self.tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

            tips_idxs = []
            for hand_id in ['l', 'r']:
                for tip_name in self.tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])

            extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, tips_idxs])

        self.register_buffer('extra_joints_idxs',
                             to_tensor(extra_joints_idxs, dtype=torch.long))

    def as_array(self, num_verts: int):
        reg = np.zeros([self.num_joints, num_verts])
        for ii, idx in enumerate(self.extra_joints_idxs):
            reg[ii, idx.item()] = 1.0
        return reg


    @property
    def num_joints(self) -> int:
        return len(self.extra_joints_idxs)

    def forward(
        self,
        vertices: Optional[Tensor] = None,
        joints_transforms: Optional[Tensor] = None,
        skinning_transforms: Optional[Tensor] = None,
        **kwargs,
    ):
        extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs)
        extra_transforms = torch.index_select(
            skinning_transforms, 1, self.extra_joints_idxs).clone()
        extra_transforms[..., :3, 3] = extra_joints

        # joints = torch.cat([joints, extra_joints], dim=1)

        return LBSOutput(
            _joints_transforms=torch.cat(
                [joints_transforms, extra_transforms], dim=1),
        )


class VerticesFromJointsTransforms(nn.Module):
    def __init__(
        self,
        vertex_ids: bool = None,
        skinning_weights: Optional[Tensor] = None,
        v_template: Optional[Tensor] = None,
        joints_template: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        # Convert the arrays to numpy
        if torch.is_tensor(skinning_weights):
            skinning_weights = skinning_weights.detach().cpu().numpy()
        if torch.is_tensor(v_template):
            v_template = v_template.detach().cpu().numpy()
        if torch.is_tensor(joints_template):
            joints_template = joints_template.detach().cpu().numpy()

        extra_joint_parent_indices = []
        extra_joint_transforms = []

        # Iterate over the desired vertex indices.
        for vertex_index in vertex_ids:

            # Find the skinning weights for the vertices
            vertex_skinning_weights = skinning_weights[vertex_index]

            # Select the joint that most influences this vertex
            joint_parent_index = vertex_skinning_weights.argmax(axis=-1)

            # Find the position of the vertex on the template
            vertex_position = v_template[vertex_index]

            transform = np.eye(4, dtype=np.float32)

            rel_pos = vertex_position - joints_template[joint_parent_index]
            transform[:3, 3] = rel_pos

            extra_joint_parent_indices.append(joint_parent_index)
            extra_joint_transforms.append(transform)

        extra_joint_transforms = np.stack(extra_joint_transforms)

        dtype = torch.float32
        extra_joint_transforms = torch.from_numpy(
            extra_joint_transforms).to(dtype=dtype)

        self.register_buffer('extra_joint_transforms', extra_joint_transforms)

        extra_joint_parent_indices = to_tensor(
            extra_joint_parent_indices, dtype=torch.long)
        self.register_buffer(
            'extra_joint_parent_indices', extra_joint_parent_indices)

    @property
    def num_points(self) -> int:
        return len(self.extra_joint_transforms)

    def forward(
        self,
        joints_transforms: Optional[Tensor] = None,
        **kwargs,
    ) -> LBSOutput:
        extra_joints_transforms = torch.matmul(
            joints_transforms[:, self.extra_joint_parent_indices],
            self.extra_joint_transforms,
        )
        return extra_joints_transforms

        return LBSOutput(
            _joints_transforms=torch.cat(
                [joints_transforms, extra_joints_transforms], dim=1),
        )


class JointsFromTransforms(nn.Module):

    def __init__(
        self,
        vertex_ids: bool = None,
        use_hands: bool = True,
        use_feet_keypoints: bool = True,
        skinning_weights: Optional[Tensor] = None,
        v_template: Optional[Tensor] = None,
        joints_template: Optional[Tensor] = None,
        **kwargs
    ) -> None:
        super(JointsFromTransforms, self).__init__()

        extra_joints_idxs = []
        extra_joint_names = []

        face_joint_names = ['nose', 'reye', 'leye', 'rear', 'lear']
        extra_joint_names += ['nose', 'right_eye', 'left_eye', 'right_ear',
                              'left_ear']
        face_keyp_idxs = [vertex_ids[name] for name in face_joint_names]

        extra_joints_idxs += face_keyp_idxs

        if use_feet_keypoints:
            feet_joint_names = [
                'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']
            feet_keyp_idxs = [vertex_ids[name] for name in feet_joint_names]

            extra_joints_idxs += feet_keyp_idxs
            extra_joint_names += [
                'left_big_toe', 'left_small_toe', 'left_heel',
                'right_big_toe', 'right_small_toe', 'right_heel',
            ]

        if use_hands:
            self.tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

            tips_idxs = []
            for hand_id in ['l', 'r']:
                for tip_name in self.tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])

                    extra_joint_names.append(
                        f'{"left" if hand_id == "l" else "right"}_{tip_name}'
                    )

            extra_joints_idxs += tips_idxs

        self.vertex_module = VerticesFromJointsTransforms(
            extra_joints_idxs,
            skinning_weights=skinning_weights,
            v_template=v_template, joints_template=joints_template,
        )
        self.extra_joints_names = extra_joint_names

    @property
    def num_joints(self) -> int:
        from loguru import logger
        logger.info('Hello!')
        return self.vertex_module.num_points

    @property
    def joint_names(self) -> int:
        return len(self.extra_joints_names)


    def forward(
        self,
        # vertices: Tensor,
        # joints: Tensor,
        joints_transforms: Optional[Tensor] = None,
        **kwargs,
    ) -> LBSOutput:
        extra_joints_transforms = self.vertex_module(joints_transforms)
        return LBSOutput(
            _joints_transforms=torch.cat(
                [joints_transforms, extra_joints_transforms], dim=1),
        )


ExtraJointsModule = Union[JointsFromTransforms, VertexJointSelector]


def build_extra_joint_module(
    vertex_ids: Dict[str, int],
    type: str = 'from_vertices',
    use_hands: bool = True,
    use_feet_keypoints: bool = True,
    skinning_weights: Optional[Tensor] = None,
    v_template: Optional[Tensor] = None,
    joints_template: Optional[Tensor] = None,
    **kwargs,
) -> ExtraJointsModule:
    if type == 'from_vertices':
        return VertexJointSelector(
            vertex_ids=vertex_ids,
            use_hands=use_hands,
            use_feet_keypoints=use_feet_keypoints,
            **kwargs,
        )
    elif type == 'from_transforms':
        return JointsFromTransforms(
            vertex_ids=vertex_ids,
            use_hands=use_hands,
            use_feet_keypoints=use_feet_keypoints,
            skinning_weights=skinning_weights,
            v_template=v_template,
            joints_template=joints_template,
            **kwargs,
        )
    else:
        raise ValueError(f'Unknown module for extra joints: {type}')
