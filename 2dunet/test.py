#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019-12-04 19:00 qiang.zhou <theodoruszq@gmail.com>
#
# Distributed under terms of the MIT license.

from importlib import import_module
import random, sys, yaml, os, json, time
import numpy as np

import torch
import torch.nn.functional as F

from dataset.dataset_test import CTDataset
from ops.dataset_ops import Train_Collatefn
from ops.log_ops import setup_logger
from ops.stat_ops import ScalarContainer, count_consective_num
from ops.acc_ops import topks_correct, topk_errors, topk_accuracies, CalcIoU
from sklearn.metrics import average_precision_score

random.seed(0); torch.manual_seed(0); np.random.seed(0)

CFG_FILE = "cfgs/test.yaml"

############### Set up Variables ###############
with open(CFG_FILE, "r") as f: cfg = yaml.safe_load(f)

DATA_ROOT = cfg["DATASETS"]["DATA_ROOT"]
MODEL_UID = cfg["MODEL"]["MODEL_UID"]
PRETRAINED_MODEL_PATH = cfg["MODEL"]["PRETRAINED_MODEL_PATH"]
NUM_CLASSES = cfg["MODEL"]["NUM_CLASSES"]
SAMPLE_NUMBER = int(cfg["DATALOADER"]["SAMPLE_NUMBER"])
NUM_WORKERS = int(cfg["DATALOADER"]["NUM_WORKERS"])
RESULE_HOME = cfg["TEST"]["RESULE_HOME"]
LOG_FILE = cfg["TEST"]["LOG_FILE"]

model = import_module(f"model.{MODEL_UID}")
UNet = getattr(model, "UNet")

############### Set up Dataloaders ###############
Validset = CTDataset(data_home=DATA_ROOT,
                               split='test',
                               sample_number=SAMPLE_NUMBER)

model = UNet(n_channels=1, n_classes=NUM_CLASSES)
model = torch.nn.DataParallel(model).cuda()

model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, \
             map_location=f'cuda:{0}'), strict=True)

############### Logging out some info ###############
logger = setup_logger(logfile=LOG_FILE)
logger.info("Config {}...".format(CFG_FILE))
logger.info("{}".format(json.dumps(cfg, indent=1)))
logger.warning(f"Loading init model path {PRETRAINED_MODEL_PATH}")

############### Testing ###############
ValidLoader = torch.utils.data.DataLoader(Validset,
                                    batch_size=1,
                                    num_workers=NUM_WORKERS,
                                    collate_fn=Train_Collatefn,
                                    shuffle=False,)

logger.info("Do evaluation...")

os.makedirs(RESULE_HOME, exist_ok=True)
os.makedirs("visual", exist_ok=True)

with torch.no_grad():
    for i, (all_F, all_M, all_info) in enumerate(ValidLoader):
        logger.info (all_info)
        all_E = []
        images = all_F.cuda()
        #(lh, uh), (lw, uw) = all_info[0]["pad"]
        num = len(images)

        for ii in range(num):
            image = images[ii:ii+1]
            pred = model(image)
            pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
            all_E.append(pred)

        all_E = torch.cat(all_E, dim=0).cpu().numpy().astype('uint8')
        all_OF = np.uint8(all_F[:, 0, :, :].cpu().numpy().astype('float32') * 255)

        unique_id = all_info[0]["name"].split('/')[-1].replace('.npy', '')
        np.save("{}/{}.npy".format(RESULE_HOME, unique_id), all_OF)
        np.save("{}/{}-dlmask.npy".format(RESULE_HOME, unique_id), all_E)

        if False:
            from zqlib import imgs2vid
            imgs2vid(np.concatenate([all_OF, all_E*255], axis=2), "visual/{}.avi".format(unique_id))
        #import pdb
        #pdb.set_trace()


