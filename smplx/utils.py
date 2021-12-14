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

from typing import NewType, Union, Optional, List, Tuple
from dataclasses import dataclass, asdict, fields

from functools import reduce
import numpy as np
import torch

Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)


@dataclass
class LBSOutput:

    def __init__(
        self,
        # joints: Optional[Tensor] = None,
        vertices: Optional[Tensor] = None,
        joints_transforms: Optional[Tensor] = None,
    ) -> None:
        pass
        self._joints = joints_transforms[..., :3, 3]
        self._vertices = vertices
        self._joints_transforms = joints_transforms

    @property
    def joints(self):
        return self._joints

    @property
    def vertices(self):
        return self._vertices

    @property
    def joints_transforms(self):
        return self._joints_transforms


@dataclass
class ModelOutput:
    vertices: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    joints_transforms: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    transl: Optional[Tensor] = None
    v_shaped: Optional[Tensor] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


@dataclass
class SMPLOutput(ModelOutput):
    betas: Optional[Tensor] = None
    body_pose: Optional[Tensor] = None


@dataclass
class SMPLHOutput(SMPLOutput):
    left_hand_pose: Optional[Tensor] = None
    right_hand_pose: Optional[Tensor] = None
    transl: Optional[Tensor] = None


@dataclass
class SMPLXOutput(SMPLHOutput):
    expression: Optional[Tensor] = None
    jaw_pose: Optional[Tensor] = None


@dataclass
class MANOOutput(ModelOutput):
    betas: Optional[Tensor] = None
    hand_pose: Optional[Tensor] = None


@dataclass
class FLAMEOutput(ModelOutput):
    betas: Optional[Tensor] = None
    expression: Optional[Tensor] = None
    jaw_pose: Optional[Tensor] = None
    neck_pose: Optional[Tensor] = None


def find_joint_kin_chain(joint_id, kinematic_tree):
    kin_chain = []
    curr_idx = joint_id
    while curr_idx != -1:
        kin_chain.append(curr_idx)
        curr_idx = kinematic_tree[curr_idx]
    return kin_chain


def toposort2(data):
    for k, v in data.items():
        v.discard(k)  # Ignore self dependencies
    extra_items_in_deps = reduce(set.union, data.values()) - set(data.keys())
    data.update({item: set() for item in extra_items_in_deps})
    while True:
        ordered = set(item for item, dep in data.items() if not dep)
        if not ordered:
            break
        # yield ' '.join(sorted(ordered))
        yield sorted(ordered)
        data = {item: (dep - ordered) for item, dep in data.items()
                if item not in ordered}
    assert not data, "A cyclic dependency exists amongst %r" % data


def parents_to_top_sort(
    joint_parents
) -> Tuple[List[List[int]], List[List[int]]]:
    if torch.is_tensor(joint_parents):
        joint_parents = joint_parents.detach().cpu().numpy()

    child_to_parent = {}

    for ii, parent in enumerate(joint_parents):
        if ii < 1:
            continue
        child_to_parent[ii] = set([parent.item()])

    parallel_exec = list(toposort2(child_to_parent))
    parallel_exec.pop(0)

    task_group_parents = []
    for task_group in parallel_exec:
        parent_indices = []
        for node in task_group:
            parent_indices.append(joint_parents[node])
        task_group_parents.append(parent_indices)

    return parallel_exec, task_group_parents


def to_tensor(
        array: Union[Array, Tensor], dtype=torch.float32
) -> Tensor:
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)
