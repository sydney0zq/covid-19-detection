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
import pprint

import torch

from dataset.dataset import CTDataset
from ops.dataset_ops import Train_Collatefn
from ops.dist_ops import synchronize, all_reduce
from ops.log_ops import setup_logger
from ops.stat_ops import ScalarContainer, count_consective_num
from ops.acc_ops import topks_correct, topk_errors, topk_accuracies
from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve
from sklearn.metrics import precision_recall_curve
import torch.nn.functional as F
from metrics import sensitivity_specificity

random.seed(0); torch.manual_seed(0); np.random.seed(0)
DIST_FLAG = torch.cuda.device_count() > 1

local_rank = 0

CFG_FILE = sys.argv[1]

#FOLD_ID = int(sys.argv[2])

############### Set up Variables ###############
with open(CFG_FILE, "r") as f: cfg = yaml.safe_load(f)

DATA_ROOT = cfg["DATASETS"]["DATA_ROOT"]
MODEL_UID = cfg["MODEL"]["MODEL_UID"]
ARCH = cfg["MODEL"]["ARCH"]
DEPTH = cfg["MODEL"]["DEPTH"]
NUM_CLASSES = cfg["MODEL"]["NUM_CLASSES"]
TRAIN_CROP_SIZE = tuple(cfg["DATALOADER"]["TRAIN_CROP_SIZE"])
CLIP_RANGE = [float(x) for x in cfg["DATALOADER"]["CLIP_RANGE"]]
LEARNING_RATE = float(cfg["SOLVER"]["LEARNING_RATE"])
WEIGHT_DECAY = float(cfg["SOLVER"]["WEIGHT_DECAY"])
LR_DECAY = float(cfg["SOLVER"]["LR_DECAY"])
TRAIN_EPOCH = int(cfg["SOLVER"]["TRAIN_EPOCH"])
BATCH_SIZE_PER_GPU = int(cfg["DATALOADER"]["BATCH_SIZE_PER_GPU"])
SNAPSHOT_FREQ = int(cfg["SOLVER"]["SNAPSHOT_FREQ"])
#LOG_FILE = cfg["SOLVER"]["LOG_FILE"].replace("train", "train-{:02d}".format(FOLD_ID))
LOG_FILE = cfg["SOLVER"]["LOG_FILE"]
NUM_WORKERS = int(cfg["DATALOADER"]["NUM_WORKERS"])
SNAPSHOT_HOME = cfg["SOLVER"]["SNAPSHOT_HOME"]
RESUME_EPOCH = int(cfg["SOLVER"]["RESUME_EPOCH"])
INIT_MODEL_PATH = cfg["SOLVER"]["INIT_MODEL_PATH"]
INIT_MODEL_STRICT = eval(cfg["SOLVER"]["INIT_MODEL_STRICT"])
#SNAPSHOT_MODEL_TPL = "{:02d}-".format(FOLD_ID) + cfg["SOLVER"]["SNAPSHOT_MODEL_TPL"]
SNAPSHOT_MODEL_TPL = cfg["SOLVER"]["SNAPSHOT_MODEL_TPL"]
RESUME_BOOL = bool(INIT_MODEL_PATH != "")

model = import_module(f"model.{MODEL_UID}")
ENModel = getattr(model, "ENModel")

############### Set up Dataloaders ###############
Trainset = CTDataset(data_home=DATA_ROOT,
                     split='train',
                     #fold_id=FOLD_ID,
                     crop_size=TRAIN_CROP_SIZE,
                     clip_range=CLIP_RANGE)
Validset = CTDataset(data_home=DATA_ROOT,
                     split='valid',
                     #fold_id=FOLD_ID,
                     crop_size=TRAIN_CROP_SIZE,
                     clip_range=CLIP_RANGE)

if DIST_FLAG:
    assert False, "Multi-GPU training is not supported for now..."
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()
    world_size = torch.distributed.get_world_size()
    Trainsampler = torch.utils.data.distributed.DistributedSampler(Trainset)
    model = ENModel(arch=ARCH, resnet_depth=DEPTH, 
                    num_frames=SAMPLE_NUM, crop_h=TRAIN_CROP_SIZE[0],
                    crop_w=TRAIN_CROP_SIZE[1], num_classes=NUM_CLASSES)
    model = torch.nn.parallel.DistributedDataParallel(
                                        model.to(local_rank),
                                        device_ids=[local_rank],
                                        output_device=local_rank,
                                        broadcast_buffers=False,)
    TrainLoader = torch.utils.data.DataLoader(Trainset,
                                        batch_size=BATCH_SIZE_PER_GPU, 
                                        num_workers=NUM_WORKERS,
                                        collate_fn=Train_Collatefn,
                                        shuffle=False,
                                        sampler=Trainsampler)
else:
    model = ENModel(arch=ARCH, resnet_depth=DEPTH, 
                    input_channel=2,
                    crop_h=TRAIN_CROP_SIZE[0],
                    crop_w=TRAIN_CROP_SIZE[1], num_classes=NUM_CLASSES)
    model = torch.nn.DataParallel(model).cuda()
    TrainLoader = torch.utils.data.DataLoader(Trainset,
                                        batch_size=BATCH_SIZE_PER_GPU,
                                        num_workers=NUM_WORKERS,
                                        collate_fn=Train_Collatefn,
                                        shuffle=True,
                                        pin_memory=True)


#model.eval()

############### Set up Optimization ###############
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
lr_scher = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY, last_epoch=-1)
criterion = torch.nn.CrossEntropyLoss(reduction="mean")

if INIT_MODEL_PATH != "":
    model.load_state_dict(torch.load(INIT_MODEL_PATH, \
                 map_location=f'cuda:{local_rank}'), strict=INIT_MODEL_STRICT)

############### Logging out some training info ###############
os.makedirs(SNAPSHOT_HOME, exist_ok=True)
Epoch_CE, Epoch_Acc = [ScalarContainer() for _ in range(2)]
logger = setup_logger(logfile=LOG_FILE)
if local_rank == 0:
    logger.info("Config {}...".format(CFG_FILE))
    logger.info("{}".format(json.dumps(cfg, indent=1)))
    if INIT_MODEL_PATH != "":
        logger.warning(f"Loading init model path {INIT_MODEL_PATH}")

dset_len, loader_len = len(Trainset), len(TrainLoader)
if local_rank == 0:
    logger.info("Setting Dataloader | dset: {} / loader: {}, | N: {}".format(\
                                dset_len, loader_len, -1))

############### Training ###############
for e in range(RESUME_EPOCH, TRAIN_EPOCH):
    # Update learning rate
    rT, all_tik = 0, time.time()        # rT -> run training time

    for i, (all_F, all_L, all_info) in enumerate(TrainLoader):
        optimizer.zero_grad()

        tik = time.time()
        preds = model([all_F.cuda(non_blocking=True)])   # I3D
        labels = all_L.cuda(non_blocking=True)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        acc = topk_accuracies(preds, labels, [1])[0]
        rT += time.time()-tik
        Epoch_CE.write(loss); Epoch_Acc.write(acc);
        #break

    dT = (time.time()-all_tik) - rT     # dT -> data loading time

    Ece, Eacc = Epoch_CE.read(), Epoch_Acc.read()
    logger.info("EN | E-R [{}-{}] | I [{}] | CE: {:1.5f} | TrainAcc: {:1.3f} | dT/rT: {:.3f} / {:.3f}".format(e, local_rank, loader_len, Ece, Eacc, dT, rT))

    if local_rank == 0:
        if e % SNAPSHOT_FREQ == 0 or e >= TRAIN_EPOCH-1:
            #model.eval()
            model_save_path = os.path.join(SNAPSHOT_HOME, SNAPSHOT_MODEL_TPL.format(e))
            logger.info (f"Dump weights {model_save_path} to disk...")
            torch.save(model.state_dict(), model_save_path)

            ValidLoader = torch.utils.data.DataLoader(Validset,
                                                batch_size=1, 
                                                num_workers=NUM_WORKERS,
                                                collate_fn=Train_Collatefn,
                                                shuffle=True,)

            Val_CE, Val_Acc = [ScalarContainer() for _ in range(2)]
            logger.info("Do evaluation...")
            with torch.no_grad():
                gts = []
                pcovs = []
                for i, (all_F, all_L, all_info) in enumerate(ValidLoader):
                    labels = all_L.cuda(non_blocking=True)
                    preds = model([all_F.cuda(non_blocking=True)])
                    val_loss = criterion(preds, labels)
                    val_acc = topk_accuracies(preds, labels, [1])[0]

                    prob_preds = F.softmax(preds, dim=1)
                    prob_normal = prob_preds[0, 0].item()
                    prob_ncov = prob_preds[0, 1].item()
                    gt = labels.item()

                    gts.append(gt)
                    pcovs.append(prob_ncov)

                    Val_CE.write(val_loss); Val_Acc.write(val_acc)

                #Eap = average_precision_score(gts, pcovs)
                gts, pcovs = np.asarray(gts), np.asarray(pcovs)
                _, _, Eauc = sensitivity_specificity(gts, pcovs)

                Ece, Eacc = Val_CE.read(), Val_Acc.read()
                logger.info("VALIDATION | E [{}] | CE: {:1.5f} | ValAcc: {:1.3f} | ValAUC: {:1.3f}".format(e, Ece, Eacc, Eauc))

    if LR_DECAY != 1:
        lr_scher.step()
        if local_rank == 0: 
            logger.info("Setting LR: {}".format(optimizer.param_groups[0]["lr"]))

    if DIST_FLAG: torch.distributed.barrier()


