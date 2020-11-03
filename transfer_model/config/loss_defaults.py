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
#  from yacs.config import CfgNode as CN

from typing import List, Tuple, Union
from omegaconf import OmegaConf
from loguru import logger
from dataclasses import dataclass, make_dataclass


@dataclass
class LossTemplate:
    type: str = 'l2'
    active: bool = False
    weight: Tuple[float] = (0.0,)
    requires_grad: bool = True
    enable: int = 0


@dataclass
class LossConfig:
    type: str = 'smplify-x'


conf = OmegaConf.structured(LossConfig)
