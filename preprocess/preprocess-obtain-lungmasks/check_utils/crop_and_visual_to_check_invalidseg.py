
import numpy as np
from zqlib import imgs2vid

readvdnames = lambda x: open(x).read().rstrip().split('\n')

ncov_ids = readvdnames("ncov_all.txt")
normal_ids = readvdnames("normal_all.txt")

import os
#for unique_id in ncov_ids:
#    data_type = "ncov"
#    print (unique_id)
#    imgs = np.load(os.path.join(f"{data_type}", f"{unique_id}.npy"))
#    masks = np.load(os.path.join(f"{data_type}", f"{unique_id}_lung_mask.npy"))
#    
#    zz, yy, xx = np.where(masks)
#    cropbox = np.array([[np.min(zz), np.max(zz)], [np.min(yy), np.max(yy)], [np.min(xx), np.max(xx)]]) 
#    
#    crop_imgs = imgs[cropbox[0, 0]:cropbox[0, 1],
#                     cropbox[1, 0]:cropbox[1, 1],
#                     cropbox[2, 0]:cropbox[2, 1]]
#
#    crop_masks = masks[cropbox[0, 0]:cropbox[0, 1],
#                       cropbox[1, 0]:cropbox[1, 1],
#                       cropbox[2, 0]:cropbox[2, 1]]
#
#    immasks = np.concatenate([crop_imgs, crop_masks*255], axis=2) 
#    imgs2vid(immasks, f"visual/{unique_id}.avi") 



for unique_id in normal_ids:
    data_type = "normal"
    print (unique_id)
    imgs = np.load(os.path.join(f"{data_type}", f"{unique_id}.npy"))
    masks = np.load(os.path.join(f"{data_type}", f"{unique_id}_lung_mask.npy"))
    
    zz, yy, xx = np.where(masks)
    cropbox = np.array([[np.min(zz), np.max(zz)], [np.min(yy), np.max(yy)], [np.min(xx), np.max(xx)]]) 
    
    crop_imgs = imgs[cropbox[0, 0]:cropbox[0, 1],
                     cropbox[1, 0]:cropbox[1, 1],
                     cropbox[2, 0]:cropbox[2, 1]]

    crop_masks = masks[cropbox[0, 0]:cropbox[0, 1],
                       cropbox[1, 0]:cropbox[1, 1],
                       cropbox[2, 0]:cropbox[2, 1]]

    immasks = np.concatenate([crop_imgs, crop_masks*255], axis=2) 
    imgs2vid(immasks, f"visual/{unique_id}.avi") 

