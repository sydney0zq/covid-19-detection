#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019-12-04 14:23 qiang.zhou <theodoruszq@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Load each patient's all/specfied CT images.
"""

from torch.utils import data
from PIL import Image
import os
import torchvision.transforms.functional as TF
import numpy as np
import torch
import random
from scipy.ndimage import zoom

try:
    from ops.dataset_ops import Rand_Affine, Rand_Crop, Rand_Transforms
except:
    #print ("Import external...")
    import sys
    sys.path.insert(0, "..")
    from ops.dataset_ops import Rand_Affine, Rand_Crop, Rand_Transforms

readvdnames = lambda x: open(x).read().rstrip().split('\n')

class CTDataset(data.Dataset):
    def __init__(self, data_home="",
                       split="train",
                       fold_id=None,
                       crop_size=(196, 288),
                       clip_range=(0.2, 0.7),   # useless
                       logger=None):

        _embo_f = os.path.join(data_home, "ImageSets", "ncov_{}.txt".format(split))
        _norm_f = os.path.join(data_home, "ImageSets", "normal_{}.txt".format(split))
        # Build a dictionary to record {path - label} pair
        meta_pos   = [[os.path.join(data_home, "NpyData-size224x336", "{}.npy".format(x)), 1] 
                                for x in readvdnames(_embo_f)]

        meta_neg   = [[os.path.join(data_home, "NpyData-size224x336", "{}.npy".format(x)), 0] 
                                for x in readvdnames(_norm_f)]

        if split == "train":
            if len(meta_pos) > len(meta_neg):
                for i in range(len(meta_pos) - len(meta_neg)):
                    meta_neg.append(random.choice(meta_neg))
            else:
                for i in range(len(meta_neg) - len(meta_pos)):
                    meta_pos.append(random.choice(meta_pos))

        meta = meta_pos + meta_neg
        
        #print (meta)
        self.data_home = data_home
        self.split = split
        self.meta = meta
        self.crop_size = crop_size
        self.clip_range = clip_range
        #print (self.meta)
        self.data_len = len(self.meta)

    def __getitem__(self, index):
        data_path, label = self.meta[index]
        mask_path = data_path.replace('.npy', '-dlmask.npy')

        cta_images = np.load(data_path)
        cta_masks = np.load(mask_path)

        num_frames = len(cta_images)
        shape = cta_images.shape

        # Data augmentation
        if self.split == "train":
            cta_images, cta_masks = Rand_Transforms(cta_images, cta_masks, ANGLE_R=10, TRANS_R=0.1, SCALE_R=0.2, SHEAR_R=10,
                                             BRIGHT_R=0.5, CONTRAST_R=0.3)

        # To Tensor and Resize
        cta_images = np.asarray(cta_images, dtype=np.float32)
        cta_images = cta_images / 255.

        images = np.concatenate([cta_images[None, :, :, :], cta_masks[None, :, :, :]], axis=0) 
        label = np.uint8([label])

        info = {"name": data_path, "num_frames": num_frames, "shape": shape}

        th_img = torch.unsqueeze(torch.from_numpy(images.copy()), 0).float()
        th_label = torch.from_numpy(label.copy()).long()

        return th_img, th_label, info

    def __len__(self):
        return self.data_len

    def debug(self, index):
        import cv2
        from zqlib import assemble_multiple_images
        th_img, th_label, info = self.__getitem__(index)
        # th_img: NxCxTxHxW

        img, label = th_img.numpy()[0, 0, :], th_label.numpy()[0]
        n, h, w = img.shape
        #if n % 2 != 0:
        #    img = np.concatenate([img, np.zeros((1, h, w))], axis=0)
        visual_img = assemble_multiple_images(img*255, number_width=16, pad_index=True)
        os.makedirs("debug", exist_ok=True)
        debug_f = os.path.join("debug/{}.jpg".format(\
                            info["name"].replace('/', '_').replace('.', '')))
        print ("[DEBUG] Writing to {}".format(debug_f))
        cv2.imwrite(debug_f, visual_img)


if __name__ == "__main__":
    # Read valid sliding: 550 seconds
    ctd = CTDataset(data_home="../NCOV-SEG/size192x288-dlmask", split="train", crop_size=(192, 288))
    length = len(ctd)
    ctd[0]
    
    exit()
    ctd.debug(0)
    import time
    s = time.time()
    for i in range(length):
        print (i)
        th_img, th_label, info = ctd[i]
    e = time.time()
    print ("time: ", e-s)

    #images, labels, info = ctd[0]
    #for i in range(10):
    #    ctd.debug(i)
    import pdb
    pdb.set_trace()


