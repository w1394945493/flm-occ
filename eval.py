import os
import time
from tqdm import tqdm
from collections import Counter, defaultdict
import math
import numpy as np

import torch
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from datetime import timedelta

from utils.training import \
    create_lr_scheduler, seed_everything, backup_codes, get_total_norm, check_grad_nan_inf
from utils.metrics import \
    pre_recall_f1score_iou, stat_pre_recall_f1score_iou, MeanIoUAcc, generate_eval_tables

from model.flm_occ import FLMOcc
from datasets_.scannet import ScannetDataset as UsedDataset
import configs.scannet as UsedConfig
DC, MC, TC = UsedConfig.Dataset, UsedConfig.Model, UsedConfig.Train
PATH_CKPT = '/path/to/your/checkpoint.pth'




def accumulate_confusion_matrix(iou_cal, miou_cal, outputs, batch, include_empty: bool=True, threshold=0.5):
    (
        B_Pdensity,
        B_Plogit,
        list_map_out__in,
        list_mask_input,
    ) = (
        outputs['densities_dense'],
        outputs['logits_dense'],
        outputs['map_out__in'],
        outputs['list_mask_input'],
    )
    
    for i in range(len(B_Pdensity)):
        Pdensity = B_Pdensity[i]
        Plogit   = B_Plogit[i]
        map_out__in = list_map_out__in[i]
        mask_input  = list_mask_input[i]
        labels = batch['labels_query'][i] + int(not include_empty)

        # label of empty is -1 if not include_empty
        gt = labels[map_out__in][mask_input]
        pred_geo = Pdensity > threshold # non-empty is 1, empty is 0
        
        # points in frustum but not covered by GS are predicted as empty
        label_not_covered = labels[map_out__in][~mask_input]
        
        gt = torch.cat([gt, label_not_covered])
        pred_geo = torch.cat([pred_geo.long(), torch.zeros_like(label_not_covered)])
        
        pred_sem = torch.softmax(Plogit, dim=-1).argmax(dim=-1).long() + int(not include_empty)
        pred_sem = torch.cat([pred_sem, torch.zeros_like(label_not_covered)])
        pred_sem = pred_sem * pred_geo
        
        if iou_cal is not None:
            iou_cal.update(pred_geo, gt.bool())
        if miou_cal is not None:
            miou_cal.update(pred_sem, gt)
    
    return iou_cal, miou_cal


def validate_model(model, dataloader, fabric, threshold=0.5):
    iou_cal = MeanIoUAcc(2, ignore_index=-1, device=fabric.device)
    miou_cal = MeanIoUAcc(DC.num_semantics, ignore_index=-1, device=fabric.device)
    
    num_batches = len(dataloader)
    loop = tqdm(enumerate(dataloader), total=num_batches, ncols=120)
    

    for i, batch in loop:
        outputs = model(batch, do_local_agg=True, levels_out=TC.levels_out)

        accumulate_confusion_matrix(iou_cal, miou_cal, outputs, batch, DC.include_empty, threshold)
        
    iou_bin = iou_cal.sync().compute()[1]
    miou, ious = miou_cal.sync().compute()
    
    return iou_bin, miou, ious




if __name__ == '__main__':

    
    
    torch.set_float32_matmul_precision('high')
    if TC.mixed_precision:
    # if False:
        # Note that parameters are still stored in float32
        # Fabric will take care of autocasting for bf16-mixed
        fabric = Fabric(
            accelerator="cuda",
            devices=1,
            precision="bf16-mixed",
            strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=100)),
        )
    else:
        fabric = Fabric(
            accelerator="cuda",
            devices=1,
            strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=100)),
        )
    
    fabric.launch()

    model = FLMOcc(MC)
    model = fabric.setup(model)
    print('Model created.')
    
    # ckpt = torch.load(path_ckpt, map_location='cpu', weights_only=False)
    ckpt = fabric.load(PATH_CKPT, weights_only=False)
    print('Loading checkpoint from', PATH_CKPT)
    
    model_state_dict = ckpt["model_state_dict"]
    model.load_state_dict(model_state_dict, strict=True)

    model = model.eval()
    # model = model.cuda().eval();
    print('Model loaded to GPU.')

    dataset = UsedDataset(DC, 'val', fabric.global_rank)
    print(f'Dataset size: {len(dataset)}')
    sampler_val = torch.utils.data.DistributedSampler(dataset, shuffle=False) # 自动设置world_size和rank
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler_val,
        num_workers=12,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=1,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )
    dataloader = fabric.setup_dataloaders(dataloader, use_distributed_sampler=False) # False to use my own sampler
    
    # ----------------- validation -----------------
    print('Starting validation...')
    list_ious = []
    list_mious = []
    # for th in np.linspace(0.1, 1.3, 7):
    for th in [0.5]:
        with torch.no_grad():
            iou_bin, miou, ious = validate_model(model, dataloader, fabric, th)
        print("IoU/Geo", iou_bin[1].item())
        print("IoU/nempty_mIoU", ious[1:].nanmean().item())
        list_ious.append(iou_bin[1].item())
        list_mious.append(ious[1:].nanmean().item())
    print(list_ious)
    print(list_mious)
    