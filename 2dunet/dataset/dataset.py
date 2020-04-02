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
    from ops.dataset_ops import Rand_Transforms
except:
    #print ("Import external...")
    import sys
    sys.path.insert(0, "..")
    from ops.dataset_ops import Rand_Transforms

readvdnames = lambda x: open(x).read().rstrip().split('\n')

class CTDataset(data.Dataset):
    def __init__(self, data_home="",
                       split="train",
                       sample_number=64,
                       clip_range=(0.2, 0.8),
                       logger=None):

        _meta_f = os.path.join(data_home, "ImageSets", "lung_{}.txt".format(split))
        # Build a dictionary to record {path - label} pair
        meta    = [os.path.join(data_home, "NpyData", "{}.npy".format(x)) for x in readvdnames(_meta_f)]

        self.data_home = data_home
        self.split = split
        self.sample_number = sample_number
        self.meta = meta
        self.clip_range = (0.2, 0.85)
        #print (self.meta)
        self.data_len = len(self.meta)
        print ("[WARN] The true clip range is {}".format(self.clip_range))

    def __getitem__(self, index):
        data_path = self.meta[index]

        if self.split == 'test':
            mask_path = data_path
        else:
            mask_path = data_path.replace('.npy', '_lung_mask.npy')

        images = np.load(data_path)
        masks  = np.uint8(np.load(mask_path) > 0)

        # CT clip
        num_frames = len(images)
        left, right = int(num_frames*self.clip_range[0]), int(num_frames*self.clip_range[1])
        images = images[left:right]
        masks = masks[left:right]

        # Random sample
        if self.sample_number > -1:
            num_frames = len(images)
            rand_index = np.random.choice([*range(0, num_frames)], self.sample_number, replace=False)
            images = images[rand_index]
            masks = masks[rand_index]

        num_frames = len(images)
        shape = images.shape
        #h, w = shape[1:]

        # Make it dividable by 16
        #new_h = h + 16 - h % 16
        #new_w = w + 16 - w % 16
        #lh, uh = (new_h-h) / 2, (new_h-h) / 2 + (new_h-h) % 2
        #lw, uw = (new_w-w) / 2, (new_w-w) / 2 + (new_w-w) % 2
        #lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)
        #images = np.pad(images, ((0,0),(lh,uh),(lw,uw)), mode='constant')
        #masks  = np.pad(masks, ((0,0),(lh,uh),(lw,uw)), mode='constant')

        if False:
            from zqlib import imgs2vid
            imgs2vid(np.concatenate([images, masks*255], axis=2), "test.avi")
            import pdb
            pdb.set_trace()

        # Data augmentation
        if self.split == "train":
            images, masks = Rand_Transforms(images, masks, 
                                            #ANGLE_R=10, TRANS_R=0.1, SCALE_R=0.2, SHEAR_R=10,
                                            ANGLE_R=0, TRANS_R=0, SCALE_R=0, SHEAR_R=0,
                                            BRIGHT_R=0.5, CONTRAST_R=0.3)

        # To Tensor and Resize
        images = np.asarray(images, dtype=np.float32)
        images = images / 255.

        images = np.expand_dims(images, axis=1)          # Bx1xHxW, add channel dimension
        masks = masks

        #info = {"name": data_path, "num_frames": num_frames, "shape": shape, "pad": ((lh,uh),(lw,uw))}
        info = {"name": data_path, "num_frames": num_frames, "shape": shape}

        th_img = torch.from_numpy(images.copy()).float()
        th_label = torch.from_numpy(masks.copy()).long()

        if self.split == 'test':
            th_label = torch.zeros_like(th_label)

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
    ctd = CTDataset(data_home="../NCOV-BF/size368x368-trad", split="train", sample_number=4)
    length = len(ctd)
    ctd[10]
    
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


