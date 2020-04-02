#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <qiang.zhou@Macbook>
#
# Distributed under terms of the MIT license.

"""
Helper functions to load and process CTA images.
"""

import numpy as np
from skimage import measure
import pydicom
import cv2
import os
import scipy

# 2020-01-02: Dump images of TxHxW
dumpimgs = lambda x, p: [cv2.imwrite("debug/{}_{:05d}.png".format(p, i), xi) for i, xi in enumerate(x)]

################### LOAD CTA ###################

""" Return a int32 dtype numpy image of a directory. """
def load_16bit_dicom_images(path, verbose=True):
    slices = [pydicom.read_file(os.path.join(path, s), force=True)
                        for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    #import pdb
    #pdb.set_trace()
    total_cnt = len(slices)
    
    slice_thickness = np.abs(slices[0].ImagePositionPatient[2]-
                             slices[1].ImagePositionPatient[2])
    # Remove duplicated slices
    ignore_cnt = 0
    if slice_thickness == 0:
        unique_slices = [slices[0]]
        start_pos = slices[0].ImagePositionPatient[2]

        for s in slices[1:]:
            if s.ImagePositionPatient[2] != start_pos:
                unique_slices.append(s)
                start_pos = s.ImagePositionPatient[2]
            else:
                ignore_cnt += 1
        slices = unique_slices
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2]-
                                 slices[1].ImagePositionPatient[2])

    if verbose:
        print ("[INFO] Total/Ignore/Reserved {}/{}/{} CTAs, slice_thickess {}...\
               ".format(total_cnt, ignore_cnt, len(slices), slice_thickness))

    for s in slices:
        s.SliceThickness = slice_thickness

    # Fix HU
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image <= -2000] = 0
    image[image >= 3000] = 3000

    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept   # -1024
        slope = slices[slice_number].RescaleSlope           # 1

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)

    numpySpacing = [float(slice_thickness)] + [float(x) for x in slices[0].PixelSpacing]

    numpyImage = np.array(image, dtype=np.int16)            # SxHxW
    numpySpacing = np.array(numpySpacing, dtype=np.float)

    return numpyImage, numpySpacing

""" Convert a TxHxW image array to video """
def imgs2vid(imgs, output_fn="test.avi", fps=5):
    import cv2
    import numpy as np
    _, height, width = imgs.shape
    video_handler = cv2.VideoWriter(output_fn, cv2.VideoWriter_fourcc(*"MJPG"), \
										fps, (width, height), isColor=False)
    for img in imgs:
        img = np.uint8(img)
        video_handler.write(img)
    cv2.destroyAllWindows()
    video_handler.release()

def show_8bit_norm_images(dicom_dir, video_fn=None):
    dicom_images, spacing = load_16bit_dicom_images(dicom_dir)
    min_value, max_value = np.min(dicom_images), np.max(dicom_images)
    norm_images = (dicom_images - min_value) / (max_value - min_value)
    norm_images = np.uint8(norm_images * 255)
    if video_fn is not None:
        print ("Writing to file {}...".format(video_fn))
        imgs2vid(norm_images, video_fn)
        
    return norm_images


################### Obtain lung mask  ###################

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


"""
一些意料之外的情况说明
1. 病人底部的托盘，如果是一个闭合的托盘，容易导致这个托盘被视为肺部
2. 外部有气孔进入病人肺部，有部分病人需要用外部的呼吸设备，这样会导致背景和肺部连接在一起了
"""
def segment_lung_mask(cta_images, fill_lung_structures=True):
    #binary_image_u = np.array(cta_images > -320, dtype=np.uint8)
    #binary_image = np.uint8(binary_image_u & binary_image_b) + 1
    # > -320 is all background
    binary_image = np.array(cta_images > -320, dtype=np.uint8) + 1  # >-320: 2
    # https://scikit-image.org/docs/0.12.x/api/skimage.measure.html#skimage.measure.label
    # Label connected regions of an integer array.
    labels = measure.label(binary_image)

    # Pick the pixel in the left-top corner to determine which label
    # is air.
    #cv2.imwrite("bimg_1.png", 255*(binary_image[0] == 1))
    #cv2.imwrite("bimg_2.png", 255*(binary_image[0] == 2))
    zz, yy, xx = labels.shape
    bg_label_list = [labels[0, 0, 0], labels[0, -1, 0], labels[0, -1, -1], 
                     labels[0, -1, xx//2], labels[0, -10, xx//2],
                     labels[0, -55, xx//2]]
    for bg_label in bg_label_list:
        #print (bg_label, np.sum(labels == bg_label))
        binary_image[labels == bg_label] = 2
    #cv2.imwrite("bimg_1_new.png", 255*(binary_image[0] == 1))
    #cv2.imwrite("bimg_2_new.png", 255*(binary_image[0] == 2))
    #import pdb
    #pdb.set_trace()

    # To this step: binary_image contains the segments of lung vessels
    # https://sm.ms/image/3PICrcky4Y5gd9z
    if fill_lung_structures:
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:
                binary_image[i][labeling != l_max] = 1
    # After filling, the vessels are included
    # https://sm.ms/image/wELgb8zIajSuO15
    #cv2.imwrite("b0_fill.png", binary_image[80]*127)
    binary_image -= 1   # Make the image actual binary
    binary_image = 1-binary_image # Lungs are now 1
    #import pdb
    #pdb.set_trace()
    #import pdb
    #pdb.set_trace()

    # Remove other air pockets inside body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets 
        binary_image[labels != l_max] = 0
    
    # For lung nodule problem, the segment area is the exact region of its task
    # but for lung PE problem, the segment area is far from it because we need the
    # artery region for discrimination, finally we will use a enclosed box to 
    # save them
    #import pdb
    #pdb.set_trace()

    return binary_image


# 所有CT的spacing统一为1mm
def resample(image, spacing, new_spacing=[1,1,1]):
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, 
                                             real_resize_factor,
                                             mode="nearest")
    return image

def resample_slice(image, spacing):
    image = scipy.ndimage.interpolation.zoom(image, 
                                             (spacing, 1, 1),
                                             mode="nearest")
    return image

def normalize_16bit_dicom_images(cta_image, HU_window=np.array([-1000., 400.]), bound_values=[0, 1]):
    # Outlier
    #mid = (0-HU_window[0])/(HU_window[1] - HU_window[0])
    #cta_image[cta_image == 0] = HU_window[0]

    th_cta_image = (cta_image-HU_window[0])/(HU_window[1] - HU_window[0])
    th_cta_image[th_cta_image < 0] = bound_values[0]
    th_cta_image[th_cta_image >= 1] = bound_values[1]
    th_cta_image_mask = (th_cta_image*255).astype('uint8')
    return th_cta_image_mask


################### Visualization###################

""" image have to be TxHxW numpy array """
def plot_3d(image, save_fn="3d.png", threshold=-300):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    image = image.transpose(2, 1, 0)    # TxHxW -> WxHxT
    verts, faces, norm, val = measure.marching_cubes_lewiner(image, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])

    plt.savefig(save_fn)



if __name__ == "__main__":

    # UNITTEST
    #cta_images = np.load("sample.npy")
    #segment_lung_mask(cta_images[60:], False)
    # Right
    #cta_images, spacing = load_16bit_dicom_images("../normal_clean/normal-00015")

    i = 4
    pid = "{:05d}".format(i)
    cta_images, spacing = load_16bit_dicom_images(f"../ncov_clean/ncov-{pid}")
    dicom_images = cta_images
    min_value, max_value = np.min(dicom_images), min(3000, np.max(dicom_images))

    print (min_value, max_value)
    norm_images = (dicom_images - min_value) / (max_value - min_value)
    norm_images = np.uint8(norm_images * 255)

    s, h, w = norm_images.shape

    crop_images = norm_images[:, int(0.25*h):int(0.75*h), int(0.15*w):int(0.85*h)]
    print (crop_images.shape)
    imgs2vid(crop_images, f"ncov-{pid}-crop1.avi")
    import pdb
    pdb.set_trace()
    #imgs2vid(norm_images, "normal-00001.avi")

    #segment_lung_mask(cta_images, False)


if __name__ == "__main2__":
    # :TEST: Load CTA
    numpyImage, numpySpacing = load_CTA_images_raw("../PE_clean/p3254665")
    segment_artery_mask(numpyImage, (100, 680))
    import pdb
    pdb.set_trace()
    lung_mask = segment_lung_mask(numpyImage, fill_lung_structures=False)

    slices, height, width = lung_mask.shape
    #lung_mask_ratio = np.sum(lung_mask.reshape((slices, height*width)), axis=-1) / (height*width)
    #left, right = np.min(np.where(lung_mask_ratio > clip_ratio)), \
    #              np.max(np.where(lung_mask_ratio > clip_ratio))
    #print ("\t[SUBINFO] CTA range bounds in 0 < {}~{} > {}".format(left, right, slices))
    plot_3d(numpyImage, save_fn="test.png", threshold=100)


    #image = resample(numpyImage, numpySpacing)


    import pdb
    pdb.set_trace()
    plot_3d(numpyImage)
    #segment_lung_mask(numpyImage)

