#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <qiang.zhou@Macbook>
#
# Distributed under terms of the MIT license.

"""
把CT机器输出的DICOM格式的文件进行处理：
1. 规整化HU值，让这些病人的HU值处在一个水平
2. 把这些CT图变成各向同性的(1x1x1)，并保证肺部的比例占据某个阈值以上才开始算为要处理的部分
3. 把肺部区域用最大连通域（形态学）算法分割出来，然后形成一个外接矩形，只保留肺部的区域和分割结果

因为原始的数据有很多混乱的干扰因素，包含了：
1. 托盘的不同可能会导致肺部分割错误/失败
2. 有一些气泡/外部的器械导致肺部和外部空气相连接，导致分割算法失败
3. CT机器扫描是圆形区域，外部的HU值有的会是0，需要进行处理
"""

import numpy as np
import os
from sub_ops import segment_lung_mask, resample, resample_slice
from sub_ops import load_16bit_dicom_images, normalize_16bit_dicom_images
import cv2
#from show_CTA import show_CTA

isotropic_resolution = np.array([1,1,1])
box3d_margin = 5            # pixels
clip_ratio = 0.01
HU_window = np.array([-1200, 600])

# Lambda funcs
listdironly = lambda d: [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
dumpimgs = lambda x, p: [cv2.imwrite("debug/{}_{:05d}.png".format(p, i), xi) for i, xi in enumerate(x)]

# 加一个圆形的mask是因为CT机器一般是圆形扫描的，会导致圆外部的HU值很诡异, 
#这个圆应该比他标准CT的小一些，防止边缘漏掉了
from skimage.draw import circle
circle_mask = np.zeros((512, 512), dtype=np.uint8) 
rr, cc = circle(255, 255, 253)  # the standard is 256, we use 253
circle_mask[rr, cc] = 1
circle_mask_inv = 1-circle_mask

# 最标准的结果，没有任何额外处理
# 1. 把肺部分割出来，失败也不管
# 2. 归一化到1mmx1mmx1mm

def main(path):

    unique_id = path.split('/')[-1]

    if os.path.isfile(os.path.join(Processed_home, f"{unique_id}.npy")):
        return

    print (f"[INFO] Processing {path} - {unique_id}")
    sliceim_pixels, spacing = load_16bit_dicom_images(path)

    # Use a circle to filter out unrelated area
    for i in range(len(sliceim_pixels)):
        sliceim_pixels[i] += -2000 * circle_mask_inv

    # Resample to 1x1x1 and compute the enclosed box
    #sliceim_isotropic = resample(sliceim_pixels, spacing, isotropic_resolution)
    #print (f"\t[SUBINFO] lung images' shape: {sliceim_isotropic.shape}")
    sliceim_isotropic = resample_slice(sliceim_pixels, spacing[0])
    print (f"\t[SUBINFO] lung images' shape: {sliceim_isotropic.shape}")

    # Thresholding specified HU areas and obtained 3D box
    sliceim_mask = normalize_16bit_dicom_images(sliceim_isotropic, HU_window=HU_window)
    np.save(os.path.join(Processed_home, f"{unique_id}.npy"), sliceim_mask)

DEBUG = False

if __name__ == "__main__":
    #CTA_home = "../ncov_clean"
    #Processed_home = "./ncov"
    CTA_home = "../normal_clean"
    Processed_home = "./normal"
    os.makedirs(Processed_home, exist_ok=True)
    CTA_paths = listdironly(CTA_home)

    # Serial
    if DEBUG:
        for path in CTA_paths:
            main(path)
    else:
        # Parallel
        from concurrent import futures
        num_threads=10
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(main, x) for x in CTA_paths]
            for i, f in enumerate(futures.as_completed(fs)):
                print ("{}/{} done...".format(i, len(fs)))
                #print ("{} result is {}".format(i, f.result()))

