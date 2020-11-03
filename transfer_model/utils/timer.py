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

import time
import numpy as np
import torch

from loguru import logger


class Timer(object):
    def __init__(self, name='', sync=False):
        super(Timer, self).__init__()
        self.elapsed = []
        self.name = name
        self.sync = sync

    def __enter__(self):
        if self.sync:
            torch.cuda.synchronize()
        self.start = time.perf_counter()

    def __exit__(self, type, value, traceback):
        if self.sync:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        self.elapsed.append(elapsed)
        logger.info(f'[{self.name}]: {np.mean(self.elapsed):.3f}')


def timer_decorator(sync=False, name=''):
    def wrapper(method):
        elapsed = []

        def timed(*args, **kw):
            if sync:
                torch.cuda.synchronize()
            ts = time.perf_counter()
            result = method(*args, **kw)
            if sync:
                torch.cuda.synchronize()
            te = time.perf_counter()
            elapsed.append(te - ts)
            logger.info(f'[{name}]: {np.mean(elapsed):.3f}')
            return result
        return timed
    return wrapper
