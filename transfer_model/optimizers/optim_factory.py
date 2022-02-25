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

import sys

from typing import NewType, List, Dict

import torch
import torch.optim as optim
from loguru import logger
from torchtrustncg import TrustRegion

Tensor = NewType('Tensor', torch.Tensor)


def build_optimizer(parameters: List[Tensor],
                    optim_cfg: Dict
                    ) -> Dict:
    ''' Creates the optimizer
    '''
    optim_type = optim_cfg.get('type', 'sgd')
    logger.info(f'Building: {optim_type.title()}')

    num_params = len(parameters)
    parameters = list(filter(lambda x: x.requires_grad, parameters))
    if num_params != len(parameters):
        logger.info(f'Some parameters have requires_grad off')

    if optim_type == 'adam':
        optimizer = optim.Adam(parameters, **optim_cfg.get('adam', {}))
        create_graph = False
    elif optim_type == 'lbfgs' or optim_type == 'lbfgsls':
        optimizer = optim.LBFGS(parameters, **optim_cfg.get('lbfgs', {}))
        create_graph = False
    elif optim_type == 'trust_ncg' or optim_type == 'trust-ncg':
        optimizer = TrustRegion(
            parameters, **optim_cfg.get('trust_ncg', {}))
        create_graph = True
    elif optim_type == 'rmsprop':
        optimizer = optim.RMSprop(parameters, **optim_cfg.get('rmsprop', {}))
        create_graph = False
    elif optim_type == 'sgd':
        optimizer = optim.SGD(parameters, **optim_cfg.get('sgd', {}))
        create_graph = False
    else:
        raise ValueError(f'Optimizer {optim_type} not supported!')
    return {'optimizer': optimizer, 'create_graph': create_graph}


def build_scheduler(optimizer, sched_type='exp',
                    lr_lambda=0.1, **kwargs):
    if lr_lambda <= 0.0:
        return None

    if sched_type == 'exp':
        return optim.lr_scheduler.ExponentialLR(optimizer, lr_lambda)
    else:
        raise ValueError('Unknown learning rate' +
                         ' scheduler: '.format(sched_type))
