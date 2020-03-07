#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019-12-04 15:13 qiang.zhou <theodoruszq@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import cv2
from PIL import Image
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF

def Train_Collatefn(data):
    all_F, all_L, all_info = [], [], []

    for i in range(len(data)):
        all_F.append(data[i][0])
        all_L.append(data[i][1])
        all_info.append(data[i][2])
    all_F = torch.cat(all_F, dim=0)
    all_L = torch.cat(all_L, dim=0)
    return all_F, all_L, all_info


# Input is image HxWxC, mask HxW in PIL format
"""Random affine transformations implemented based on torchvision. The input img&mask can be single PIL.Image instance or [PIL.Image] list.
Refer to: https://pytorch.org/docs/stable/torchvision/transforms.html
Params:
    [img] -- [PIL.Image instance or a list of PIL.Image instances]
    [ANGLE_R] -- [Rotation range based on the center point, from -ANGLE_R~ANGLE_R]
    [TRANS_R] -- [Spatial translation, -TRANS_R~TRANS_R, e.g. 0.1 -> 0.1*img_height]
    [SCALE_R] -- [Zoom scale factor, -SCALR_R~SCALE_R, e.g. 0.1 -> 0.1*img_height]
    [SHEAR_R] -- [Shear degree, -SHEAR_R~SHEAR_R]
    [FLIP_B]  -- [Flip uniformly for all PIL.Image instances, default is False]
"""
def Rand_Affine(img, ANGLE_R=10, TRANS_R=0.2, SCALE_R=0.3, SHEAR_R=15, FLIP_B=False):
    assert isinstance(img, Image.Image) or isinstance(img[0], Image.Image) 

    def affop(img, angle, translate, scale, shear, flip):
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        _img = TF.affine(img, angle, translate, scale, shear, resample=Image.BILINEAR)
        return _img
    if isinstance(img, list):
        w, h = img[0].size
    else:
        w, h = img.size
    angle = random.randint(-ANGLE_R, ANGLE_R)
    translate = (random.randint(int(-w*TRANS_R), int(w*TRANS_R)), 
                 random.randint(int(-h*TRANS_R), int(h*TRANS_R)))  # x, y axis
    scale = 1 + round(random.uniform(-SCALE_R, SCALE_R), 1)
    shear = random.randint(-SHEAR_R, SHEAR_R)
    flip = FLIP_B and random.random() >= 0.5
    #print (angle, translate, scale, shear)
    if isinstance(img, list):
        img_L = []
        for i_img in img:
            i_img = affop(i_img, angle, translate, scale, shear, flip)
            img_L.append(i_img)
        return img_L
    else:
        _img = affop(img, angle, translate, scale, shear, flip)
        return _img

# img must be a np.uint8 TxHxW datatype numpy
def Rand_Crop(img, crop_size):
    shape = img.shape[1:]	# h, w
    crop_y = random.randint(0, shape[0] - crop_size[0])
    crop_x = random.randint(0, shape[1] - crop_size[1])
    crop_img = img[:, crop_y:crop_y+crop_size[0], crop_x:crop_x+crop_size[1]]
    return crop_img

""" imgs must be a TxHxW tensor"""
# Affine Transforms
# Color  Transforms


def Rand_Transforms(imgs, masks,
                    ANGLE_R=10, TRANS_R=0.1, 
                    SCALE_R=0.2, SHEAR_R=10,
                    BRIGHT_R=0.5, CONTRAST_R=0.3):
    # To Image.Image instances
    pil_imgs = [Image.fromarray(x) for x in imgs]
    pil_masks = [Image.fromarray(x) for x in masks]
    w, h = pil_imgs[0].size

    # Affine Transforms
    def affop(img, angle, translate, scale, shear):
        _img = TF.affine(img, angle, translate, scale, shear, resample=Image.BILINEAR)
        return _img
    angle = random.randint(-ANGLE_R, ANGLE_R)
    translate = (random.randint(int(-w*TRANS_R), int(w*TRANS_R)), 
                 random.randint(int(-h*TRANS_R), int(h*TRANS_R)))  # x, y axis
    scale = 1 + round(random.uniform(-SCALE_R, SCALE_R), 1)
    shear = random.randint(-SHEAR_R, SHEAR_R)
    pil_imgs = [affop(x, angle, translate, scale, shear) for x in pil_imgs]
    pil_masks = [affop(x, angle, translate, scale, shear) for x in pil_masks]

    # Color Transforms
    def colorop(img, bright, contrast):
        _img = TF.adjust_brightness(img, bright)
        _img = TF.adjust_contrast(_img, contrast)
        return _img
    bright = 1 + round(random.uniform(-BRIGHT_R, BRIGHT_R), 1)
    contrast = 1 + round(random.uniform(-CONTRAST_R, CONTRAST_R), 1)
    pil_imgs = [colorop(x, bright, contrast) for x in pil_imgs]

    imgs = np.asarray([np.asarray(x, dtype=np.uint8) for x in pil_imgs], dtype=np.uint8)
    masks = np.asarray([np.asarray(x, dtype=np.uint8) for x in pil_masks], dtype=np.uint8)
    return imgs, masks


    





if __name__ == "__main__":
    imgs = np.load("../PE-CTA/PE/p0541758.npy")
    Rand_Transforms(imgs)









