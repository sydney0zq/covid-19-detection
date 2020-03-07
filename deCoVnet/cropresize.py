#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-01-09 21:42 qiang.zhou <theodoruszq@gmail.com>
#
# Distributed under terms of the MIT license.

""" 把已经处理好的各向同性的npy文件resize成相同的大小。 """

import os
import numpy as np

from scipy.ndimage import zoom
readvdnames = lambda x: open(x).read().rstrip().split('\n')

src_home = "NpyData-dlmask"
des_home = "NpyData-clip-size192x288"

os.makedirs(des_home, exist_ok=True)

#for d in dirs:
#    os.makedirs(os.path.join(des_home, d), exist_ok=True)

pe_list = readvdnames(f"ImageSets-old/lung_test.txt")[::-1]

new_size = (192, 288)   # 192x288       # average isotropic shape: 193x281 

new_height, new_width = new_size

clip_range = (0.15, 1)

slice_resolution = 1
from zqlib import imgs2vid
import cv2

def resize_cta_images(x):        # dtype is "PE"/"NORMAL"  
    print (x)
    if os.path.isfile(os.path.join(des_home, x+".npy")) is True:
        return
    raw_imgs = np.uint8(np.load(os.path.join(src_home, x+".npy")))
    raw_masks = np.load(os.path.join(src_home, x+"-dlmask.npy"))
    length = len(raw_imgs)

    clip_imgs = raw_imgs[int(length*clip_range[0]):int(length*clip_range[1])]
    clip_masks = raw_masks[int(length*clip_range[0]):int(length*clip_range[1])]

    raw_imgs = clip_imgs
    raw_masks = clip_masks

    #xdata = np.concatenate([raw_imgs[length//2], raw_masks[length//2]*255], axis=1)
    #print (xdata.shape)
    #cv2.imwrite(f"debug/{x}.png", xdata)
    #return

    zz, yy, xx = np.where(raw_masks)
    cropbox = np.array([[np.min(zz), np.max(zz)], [np.min(yy), np.max(yy)], [np.min(xx), np.max(xx)]])
    crop_imgs = raw_imgs[cropbox[0, 0]:cropbox[0, 1],
                         cropbox[1, 0]:cropbox[1, 1],
                         cropbox[2, 0]:cropbox[2, 1]]

    crop_masks = raw_masks[cropbox[0, 0]:cropbox[0, 1],
                          cropbox[1, 0]:cropbox[1, 1],
                          cropbox[2, 0]:cropbox[2, 1]]

    raw_imgs = crop_imgs
    raw_masks = crop_masks

    height, width = raw_imgs.shape[1:3]
    zoomed_imgs = zoom(raw_imgs, (slice_resolution, new_height/height, new_width/width))
    np.save(os.path.join(des_home, x+".npy"), zoomed_imgs)
    zoomed_masks = zoom(raw_masks, (slice_resolution, new_height/height, new_width/width))
    np.save(os.path.join(des_home, x+"-dlmask.npy"), zoomed_masks)

    immasks = np.concatenate([zoomed_imgs, zoomed_masks*255], axis=2)[length//2]
    cv2.imwrite(f"debug/{x}.png", immasks)
    #imgs2vid(immasks, "debug/{}.avi".format(x))


#for x in pe_list:
#    resize_cta_images(x)
#
#exit()

#for x in orcale_normal_list:
#    resize_cta_images(x, "orcale_normal")

from concurrent import futures

num_threads=10

with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
    fs = [executor.submit(resize_cta_images, x, ) for x in pe_list[::-1]]
    for i, f in enumerate(futures.as_completed(fs)):
        print ("{}/{} done...".format(i, len(fs)))
