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

from dataset.dataset import CTDataset
from ops.dataset_ops import Train_Collatefn
from ops.log_ops import setup_logger
from ops.stat_ops import ScalarContainer, count_consective_num
from ops.acc_ops import topks_correct, topk_errors, topk_accuracies, CalcIoU
from sklearn.metrics import average_precision_score

random.seed(0); torch.manual_seed(0); np.random.seed(0)


CFG_FILE = sys.argv[1]

############### Set up Variables ###############
with open(CFG_FILE, "r") as f: cfg = yaml.safe_load(f)

DATA_ROOT = cfg["DATASETS"]["DATA_ROOT"]
MODEL_UID = cfg["MODEL"]["MODEL_UID"]
NUM_CLASSES = cfg["MODEL"]["NUM_CLASSES"]
TRAIN_CROP_SIZE = tuple(cfg["DATALOADER"]["TRAIN_CROP_SIZE"])
SAMPLE_NUMBER = int(cfg["DATALOADER"]["SAMPLE_NUMBER"])
LEARNING_RATE = float(cfg["SOLVER"]["LEARNING_RATE"])
WEIGHT_DECAY = float(cfg["SOLVER"]["WEIGHT_DECAY"])
LR_DECAY = float(cfg["SOLVER"]["LR_DECAY"])
TRAIN_EPOCH = int(cfg["SOLVER"]["TRAIN_EPOCH"])
BATCH_SIZE_PER_GPU = int(cfg["DATALOADER"]["BATCH_SIZE_PER_GPU"])
SNAPSHOT_FREQ = int(cfg["SOLVER"]["SNAPSHOT_FREQ"])
LOG_FILE = cfg["SOLVER"]["LOG_FILE"]
NUM_WORKERS = int(cfg["DATALOADER"]["NUM_WORKERS"])
SNAPSHOT_HOME = cfg["SOLVER"]["SNAPSHOT_HOME"]
RESUME_EPOCH = int(cfg["SOLVER"]["RESUME_EPOCH"])
INIT_MODEL_PATH = cfg["SOLVER"]["INIT_MODEL_PATH"]
INIT_MODEL_STRICT = eval(cfg["SOLVER"]["INIT_MODEL_STRICT"])
SNAPSHOT_MODEL_TPL = cfg["SOLVER"]["SNAPSHOT_MODEL_TPL"]
RESUME_BOOL = bool(INIT_MODEL_PATH != "")

model = import_module(f"model.{MODEL_UID}")
UNet = getattr(model, "UNet")

############### Set up Dataloaders ###############
Trainset = CTDataset(data_home=DATA_ROOT,
                     split='train',
                     sample_number=SAMPLE_NUMBER)
Validset = CTDataset(data_home=DATA_ROOT,
                     split='valid',
                     sample_number=SAMPLE_NUMBER)

model = UNet(n_channels=1, n_classes=2)
model = torch.nn.DataParallel(model).cuda()
TrainLoader = torch.utils.data.DataLoader(Trainset,
                                    batch_size=BATCH_SIZE_PER_GPU,
                                    num_workers=NUM_WORKERS,
                                    collate_fn=Train_Collatefn,
                                    shuffle=True,
                                    pin_memory=True)

############### Set up Optimization ###############
optimizer = torch.optim.Adam([{"params": model.parameters(), "initial_lr": LEARNING_RATE}], 
                                                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
lr_scher = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY, last_epoch=RESUME_EPOCH)
criterion = torch.nn.CrossEntropyLoss(reduction="mean")

if INIT_MODEL_PATH != "":
    model.load_state_dict(torch.load(INIT_MODEL_PATH, \
                 map_location=f'cuda:{0}'), strict=INIT_MODEL_STRICT)

############### Logging out some training info ###############
os.makedirs(SNAPSHOT_HOME, exist_ok=True)
Epoch_CE, Epoch_mIOU = [ScalarContainer() for _ in range(2)]
logger = setup_logger(logfile=LOG_FILE)

logger.info("Config {}...".format(CFG_FILE))
logger.info("{}".format(json.dumps(cfg, indent=1)))
if INIT_MODEL_PATH != "":
    logger.warning(f"Loading init model path {INIT_MODEL_PATH}")

dset_len, loader_len = len(Trainset), len(TrainLoader)
logger.info("Setting Dataloader | dset: {} / loader: {}, | N: {}".format(\
                            dset_len, loader_len, -1))


############### Training ###############
for e in range(RESUME_EPOCH, TRAIN_EPOCH):
    # Update learning rate
    rT, all_tik = 0, time.time()        # rT -> run training time

    for i, (all_F, all_M, all_info) in enumerate(TrainLoader):
        optimizer.zero_grad()

        tik = time.time()
        preds = model(all_F.cuda())
        labels = all_M.cuda()
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(F.softmax(preds, dim=1), dim=1)
        miou = torch.mean(CalcIoU(preds, labels)).item()

        rT += time.time()-tik
        Epoch_CE.write(loss); Epoch_mIOU.write(miou);
        #break

    dT = (time.time()-all_tik) - rT     # dT -> data loading time

    Ece, Emiou = Epoch_CE.read(), Epoch_mIOU.read()
    logger.info("EN | E-I [{}-{}] | CE: {:1.5f} | TrainmIOU: {:1.3f} | dT/rT: {:.3f} / {:.3f}".format(e, loader_len, Ece, Emiou, dT, rT))

    if e % SNAPSHOT_FREQ == 0 or e >= TRAIN_EPOCH-1:
        model.eval()

        model_save_path = os.path.join(SNAPSHOT_HOME, SNAPSHOT_MODEL_TPL.format(e))
        logger.info (f"Dump weights {model_save_path} to disk...")
        torch.save(model.state_dict(), model_save_path)

        ValidLoader = torch.utils.data.DataLoader(Validset,
                                            batch_size=1, 
                                            num_workers=NUM_WORKERS,
                                            collate_fn=Train_Collatefn,
                                            shuffle=True,)

        Val_CE, Val_mIOU = [ScalarContainer() for _ in range(2)]
        logger.info("Do evaluation...")
        with torch.no_grad():
            for i, (all_F, all_M, all_info) in enumerate(ValidLoader):
                labels = all_M.cuda()
                preds = model(all_F.cuda())
                val_loss = criterion(preds, labels)

                preds = torch.argmax(F.softmax(preds, dim=1), dim=1)
                val_miou = torch.mean(CalcIoU(preds, labels)).item()

                Val_CE.write(val_loss); Val_mIOU.write(val_miou)
        Ece, Emiou = Val_CE.read(), Val_mIOU.read()
        logger.info("VALIDATION | E [{}] | CE: {:1.5f} | ValmIOU: {:1.3f}".format(e, Ece, Emiou)) 
    
        model.train()

    if LR_DECAY != 1:
        lr_scher.step()
        logger.info("Setting LR: {}".format(optimizer.param_groups[0]["lr"]))


