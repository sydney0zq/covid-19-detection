#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019-12-04 19:25 qiang.zhou <theodoruszq@gmail.com>
#
# Distributed under terms of the MIT license.

""" ResNet only. """

import torch
import torch.nn as nn
import sys
sys.path.insert(0, "..")
from model.stem_helper import VideoModelStem
from model.resnet_helper import ResStage
from model.head_helper import ResNetBasicHead

import ops.weight_init_helper as init_helper 

#_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}
_MODEL_STAGE_DEPTH = {50: (1, 1, 1, 1), 101: (3, 4, 23, 3)}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slowonly": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slowonly": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slowonly": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}


class ENModel(nn.Module):
    """
    It builds a ResNet like network backbone without lateral connection.
    Copied from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/model_builder.py
    """
    def __init__(self, arch="i3d",          
                       resnet_depth=50,     # 50/101
                       input_channel=1,
                       num_frames=-1,
                       crop_h=-1,
                       crop_w=-1,
                       num_classes=2,
                       ):
        super(ENModel, self).__init__()
        
        self.num_pathways = 1        # Because it is only slow, so it is 1
        assert arch in _POOL1.keys()
        pool_size = _POOL1[arch]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert resnet_depth in _MODEL_STAGE_DEPTH.keys()
        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[resnet_depth]

        # vanilla params
        num_groups = 1
        width_per_group = 16        # origin: 64
        dim_inner = num_groups * width_per_group
	
        temp_kernel = _TEMPORAL_KERNEL_BASIS[arch]

        self.s1 = VideoModelStem(
                dim_in=[input_channel],
                dim_out=[width_per_group],
                kernel=[temp_kernel[0][0] + [7, 7]],
                stride=[[1, 2, 2]],
                padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
        )

        self.s2 = ResStage(
                dim_in=[width_per_group],
                dim_out=[width_per_group * 4],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[1],
                num_blocks=[d2],
                num_groups=[num_groups],
                num_block_temp_kernel=[d2],
                nonlocal_inds=[[]],
                nonlocal_group=[1],
                instantiation='softmax',
                trans_func_name='bottleneck_transform',
                stride_1x1=False,
                inplace_relu=True,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                    kernel_size=pool_size[pathway],
                    stride=pool_size[pathway],
                    padding=[0, 0, 0]
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = ResStage(
                dim_in=[width_per_group * 4],
                dim_out=[width_per_group * 8],
                dim_inner=[dim_inner * 2],
                temp_kernel_sizes=temp_kernel[2],
                stride=[2],
                num_blocks=[d3],
                num_groups=[num_groups],
                num_block_temp_kernel=[d3],
                nonlocal_inds=[[]],
                nonlocal_group=[1],
                instantiation='softmax',
                trans_func_name='bottleneck_transform',
                stride_1x1=False,
                inplace_relu=True,
        )

        self.s4 = ResStage(
                dim_in=[width_per_group * 8],
                dim_out=[width_per_group * 16],
                dim_inner=[dim_inner * 4],
                temp_kernel_sizes=temp_kernel[3],
                stride=[2],
                num_blocks=[d4],
                num_groups=[num_groups],
                num_block_temp_kernel=[d4],
                nonlocal_inds=[[]],
                nonlocal_group=[1],
                instantiation='softmax',
                trans_func_name='bottleneck_transform',
                stride_1x1=False,
                inplace_relu=True,
        )

        self.s5 = ResStage(
                dim_in=[width_per_group * 16],
                dim_out=[width_per_group * 32],
                dim_inner=[dim_inner * 8],
                temp_kernel_sizes=temp_kernel[4],
                stride=[2],
                num_blocks=[d5],
                num_groups=[num_groups],
                num_block_temp_kernel=[d5],
                nonlocal_inds=[[]],
                nonlocal_group=[1],
                instantiation='softmax',
                trans_func_name='bottleneck_transform',
                stride_1x1=False,
                inplace_relu=True,
        )

        # Classifier
        #self.head = ResNetBasicHead(
        #        dim_in=[width_per_group * 32],
        #        num_classes=num_classes,
        #        pool_size=[
        #            [
        #                num_frames // pool_size[0][0],
        #                crop_h // 32 // pool_size[0][1],
        #                crop_w // 32 // pool_size[0][2],
        #            ]
        #        ],
        #        dropout_rate=0.5,
        #)

        
        self.head = nn.Sequential(
                            nn.AdaptiveMaxPool3d((16, 24, 36)),
                            nn.Conv3d(128, 64, kernel_size = 3, padding = 1),
                            nn.ReLU(inplace=True),
                            nn.AdaptiveMaxPool3d((4, 12, 18)),
                            nn.Conv3d(64, 32, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.AdaptiveMaxPool3d((1, 6, 9)),
                            nn.Dropout3d(p = 0),
                            nn.Conv3d(32, 32, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.AdaptiveMaxPool3d((1, 1, 1)),
                        )
        self.classifier = nn.Sequential(
                            nn.Linear(32, 32),
                            nn.Linear(32, 2)
                        )

        # init weights
        init_helper.init_weights(
            self, fc_init_std=0.01, zero_init_final_bn=True        
        )


    def forward(self, x):
        # for pathway in range(self.num_pathways):
        #    x[pathway] = x[pathway] - self.mean
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.head(x[0])
        n, c = x.size(0), x.size(1)
        x = self.classifier(x.view(n, c))
        #x = self.s4(x)
        #x = self.s5(x)
        #x = self.head(x)
        return x


if __name__ == "__main__":
    model = ENModel()
    aa = torch.ones((1, 1, 10, 128, 128))
    model([aa])
    model_param = sum(x.numel() for x in model.parameters())
    print (model_param)
