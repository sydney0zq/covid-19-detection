import numpy as np
import sys
import cv2
import os
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from skimage import measure

if __name__ == "__main__":
    list_path = "/your/path/test.txt"
    img_ids = [i_id.strip() for i_id in open(list_path)]
    correct = []
    for img_id in img_ids:
        mask_dir = "your/path/cam-mask/"+img_id+".npy"
        mask = np.load(mask_dir)
        ANNPATH = "/your/path/Annotations/"
        gt = np.zeros(mask.shape)
        anno_files = os.listdir(ANNPATH+img_id+"/")
        anno_files.sort()
        for anno_file in anno_files:
            anno = ET.parse(ANNPATH+img_id+"/"+anno_file)
            boxes = []
            TO_REMOVE = 1
            for obj in anno.iter("object"):
                name = obj.find("name").text.lower().strip()
                bb = obj.find("bndbox")
                box = [
                    bb.find("xmin").text,
                    bb.find("ymin").text,
                    bb.find("xmax").text,
                    bb.find("ymax").text,
                ]
                bndbox = tuple(
                    map(lambda x: x - TO_REMOVE, list(map(int, box)))
                )
                if name in boxes:
                    boxes = np.concatenate((boxes[name],np.array(bndbox)[np.newaxis,:]),axis=0)
                else:   
                    boxes = np.array(bndbox)[np.newaxis,:]
                for i in range(boxes.shape[0]):
                    box = boxes[i]
                    gt[int(anno_file.split('.')[0]),box[1]:box[3],box[0]:box[2]] = 1
        import cc3d
        labels_out = cc3d.connected_components(gt.astype(np.uint16), out_dtype=np.uint16)
        N = np.max(labels_out)
        zz, yy, xx = np.where(mask > 0)
        for segid in range(1,N+1):
            if mask.sum()==0:
                find = False
                break
            gt_covid = labels_out * (labels_out == segid)
            # centroid
            centroid = [zz[int(len(zz)/2)],yy[int(len(zz)/2)],xx[int(len(zz)/2)]]
            if gt_covid[centroid[0],centroid[1],centroid[2]]!=0:
                find = True
                break
            else:
                find = False
        if find :
            correct.append(1)
        else:
            correct.append(0)

    correct = np.array(correct)
    print(correct)
    print(np.mean(correct))
    # import pdb; pdb.set_trace()
