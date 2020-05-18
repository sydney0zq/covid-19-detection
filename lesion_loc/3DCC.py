import numpy as np
import cv2
from zqlib import imgs2vid
from skimage import measure
np.random.seed(0)

def sub_seg(img, win_size):
    h, w = img.shape
    res = np.zeros_like(img)   
    count = 0
    for dh in range(win_size//2, h-win_size//2, win_size):
        for dw in range(win_size//2, w-win_size//2, win_size):
            win_region = img[dh-win_size//2:dh+win_size//2, dw-win_size//2:dw+win_size//2]
            output = cv2.connectedComponents(win_region, connectivity=8, ltype=cv2.CV_32S)
            if output[0] > 3:
                if abs(np.std(win_region)-ref_std) < bias_term :
                    res[dh-win_size//2:dh+win_size//2, dw-win_size//2:dw+win_size//2] = 1
    return res

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def top5_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 5:
        return vals[np.argsort(counts)[-5:]]
    elif len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

list_path = "/your/path/test.txt"
patient_ids = [i_id.strip() for i_id in open(list_path)]
for patient_id in patient_ids:
    # load the npy file of the Lung segmentation
    mask_fn = "/your/mask_npy/path/"+patient_id+".npy"
    mask = np.load(mask_fn)

    T, H, W = mask.shape
    final_mask = np.zeros(mask.shape)
    mask = mask[int(T*0.1):int(T*0.8)]

    win_size = 6
    bias_term = 0.05

    ref_std = 0.5
    res_list = []
    for m in mask:
    	#3DCC
        res = sub_seg(m, win_size)
        res_list.append(res)
    res_list = np.asarray(res_list)
    labels = measure.label(res_list)
    # find the top 5 3D connected components
    l_max = top5_label_volume(labels, bg=res_list[0, 0, 0])
    select_ = np.zeros(labels.shape)
    # save the top 5 3D connected components
    for l_i in [l_max[-1]]:
        select_ = select_ + (labels==l_i)
    final_mask[int(T*0.1):int(T*0.8)] = select_
    final_mask = final_mask.astype(np.uint8)
    np.save("binary-seg/"+patient_id+".npy",final_mask)
    # transform images to video
    imgs2vid(np.concatenate([mask*255, select_*255], axis=2), "./results/"+patient_id.split('.')[0]+".avi")

