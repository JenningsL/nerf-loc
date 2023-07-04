"""
Author: jenningsliu
Date: 2022-05-25 13:44:53
LastEditors: jenningsliu
LastEditTime: 2022-07-06 16:48:29
FilePath: /nerf-loc/datasets/DSM/multi_scene_dataset.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
import numpy as np

from .dataset import VideoDataset


class MultiSceneDataset(ConcatDataset):
    def __init__(
        self, datasets
    ):
        super().__init__(datasets)
        self.datasets = datasets
        # TODO: scale_factor is different across scene
        scale_factor = np.inf
        for ds in datasets:
            scale_factor = min(scale_factor, ds.scale_factor)
        self.scale_factor = scale_factor

    def set_mode(self, mode):
        for ds in self.datasets:
            ds.set_mode(mode)

    def get_pc_range(self):
        pc_range = np.zeros(6)
        for ds in self.datasets:
            pc_range[:3] = np.minimum(pc_range[:3], ds.pc_range[:3])
            pc_range[3:6] = np.maximum(pc_range[3:6], ds.pc_range[3:6])
        return pc_range