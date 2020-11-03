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

from omegaconf import OmegaConf
from loguru import logger
from dataclasses import dataclass
from .utils_cfg import Variable, Pose


@dataclass
class PCA:
    num_comps: int = 12
    flat_hand_mean: bool = False


@dataclass
class PoseWithPCA(Pose):
    pca: PCA = PCA()


@dataclass
class Shape(Variable):
    num: int = 10


@dataclass
class Expression(Variable):
    num: int = 10


@dataclass
class SMPL:
    betas: Shape = Shape()
    global_rot: Pose = Pose()
    body_pose: Pose = Pose()
    translation: Variable = Variable()


@dataclass
class SMPLH(SMPL):
    left_hand_pose: PoseWithPCA = PoseWithPCA()
    right_hand_pose: PoseWithPCA = PoseWithPCA()


@dataclass
class SMPLX(SMPLH):
    expression: Expression = Expression()
    jaw_pose: Pose = Pose()
    leye_pose: Pose = Pose()
    reye_pose: Pose = Pose()


@dataclass
class MANO:
    betas: Shape = Shape()
    wrist_pose: Pose = Pose()
    hand_pose: PoseWithPCA = PoseWithPCA()
    translation: Variable = Variable()


@dataclass
class FLAME:
    betas: Shape = Shape()
    expression: Expression = Expression()
    global_rot: Pose = Pose()
    neck_pose: Pose = Pose()
    jaw_pose: Pose = Pose()
    leye_pose: Pose = Pose()
    reye_pose: Pose = Pose()


@dataclass
class BodyModelConfig:
    model_type: str = 'smplx'
    use_compressed: bool = True
    folder: str = 'models'
    gender: str = 'neutral'
    extra_joint_path: str = ''
    ext: str = 'npz'

    num_expression_coeffs: int = 10

    use_face_contour: bool = True
    joint_regressor_path: str = ''

    smpl: SMPL = SMPL()
    star: SMPL = SMPL()
    smplh: SMPLH = SMPLH()
    smplx: SMPLX = SMPLX()
    mano: MANO = MANO()
    flame: FLAME = FLAME()


conf = OmegaConf.structured(BodyModelConfig)
