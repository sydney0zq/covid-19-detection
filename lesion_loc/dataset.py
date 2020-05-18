#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-03-15 19:00 qiang.zhou <theodoruszq@gmail.com>
#
# Distributed under terms of the CC-NC license.

"""Load npy data of a patient."""

import numpy as np
import torch

def load_processed_ct_images(npy_filepath, clip_range):
    images = np.load(npy_filepath)
    num_frames = len(images)

    left, right = int(num_frames*clip_range[0]), int(num_frames*clip_range[1])
    images = images[left:right]

    num_frames = len(images)
    shape = images.shape

    if False:
        from zqlib import imgs2vid
        imgs2vid(images, "test_image.avi")
    
    images = np.asarray(images, dtype=np.float32)
    images = images / 255.
    # add channel dimension
    images = np.expand_dims(images, axis=1)
    img_info = {"name": npy_filepath, 
                "num_frames": num_frames,
                "clip_range": clip_range,
                "shape": shape}
    th_images = torch.from_numpy(images.copy()).float()
    return th_images, img_info

def combine_ct_images_and_masks(ct_images, ct_masks):
    num_frames = len(ct_images)
    ct_images = np.asarray(ct_images, dtype=np.float32)
    ct_images = ct_images / 255.

    if False:
        from zqlib import imgs2vid
        imgs2vid(np.concatenate([ct_images*255, ct_masks*255], axis=2), "combine_image_mask.avi")

    ct_imasks = np.concatenate(
                    [ct_images[None, ...], ct_masks[None, ...]], 
                    axis=0)
    th_imasks = torch.unsqueeze(
                    torch.from_numpy(ct_imasks.copy()), 0).float()
    return th_imasks



    
if __name__ == "__main__":
    npy_filepath = "../../NpyData/ncov-00001.npy"
    load_processed_ct_images(npy_filepath, clip_range=(0, 1))









