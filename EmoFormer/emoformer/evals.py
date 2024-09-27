import logging
import collections
import time
import numpy as np
import cv2
import ast
import operator
import csv
import pandas as pd
from tqdm import tqdm
import os
import glob
import torch
from torch.utils.data import DataLoader

import emoformer.utils as utils
import emoformer.utils.misc
from emoformer.datasets.dsec_mos_val_data import DsecMosValDataset

from emoformer.datasets import path_config as dataset_path_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_eval_save_dir_name_from_args(_args):
    _dir_name = 'infer_%s%3d%slpp_mode%d_sc%0.2f_%d' % (
        _args.dataset,
        _args.val_size,
        'msc' if _args.msc else 'ssc',
        _args.lprop_mode,
        _args.lprop_scale,
        int(time.time()))
    return _dir_name


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_predictions_flip_ms(model, samples, targets, gt_shape, ms=True, ms_gather='mean', flip=True,
                                flip_gather='mean', scales=None, sigmoid=True):
    # import ipdb;ipdb.set_trace()
    outputs = compute_predictions_ms(model, samples, targets, gt_shape, ms=ms, ms_gather=ms_gather,
                                     scales=scales, sigmoid=sigmoid)
    outputs['pred_masks'] = utils.misc.interpolate(outputs['pred_masks'], size=gt_shape, mode="bilinear",
                                                   align_corners=False)
    if flip:
        samples_flipped, targets_flipped = augment_flip(samples, targets)

        outputs_flipped = compute_predictions_ms(model, samples_flipped, targets_flipped, gt_shape, ms=ms,
                                                 ms_gather=ms_gather, scales=scales)
        outputs_flipped['pred_masks'] = utils.misc.interpolate(outputs_flipped['pred_masks'], size=gt_shape,
                                                               mode="bilinear", align_corners=False)
        if flip_gather == 'max':
            outputs['pred_masks'] = torch.max(outputs_flipped['pred_masks'].flip(-1), outputs['pred_masks'])
        else:
            outputs['pred_masks'] = (outputs_flipped['pred_masks'].flip(-1) + outputs['pred_masks']) / 2.0
    return outputs


def compute_predictions_ms(model, samples, targets, gt_shape, ms=True, ms_gather='mean',
                           scales=None, sigmoid=True):
    if scales is None:
        scales = [1]
    mask_list = []
    org_shape = samples.tensors.shape[-2:]	## torch.Size([473, 630])
    
    ## import ipdb;ipdb.set_trace()
    
    for scale in scales:
        size = [int(val * scale) for val in org_shape]
        tensors = samples.tensors
        mask = samples.mask
        if scale != 1:
            tensors = utils.misc.interpolate(tensors, size=size, mode="bilinear", align_corners=False)
            mask = utils.misc.interpolate(mask.unsqueeze(1).long().float(), size=size, mode="nearest").squeeze(1)
            mask[mask > 0.5] = True
            mask[mask <= 0.5] = False
            mask = mask.bool()
        ms_sample = utils.misc.NestedTensor(tensors, mask)
        model_output = model(ms_sample)
        pred = utils.misc.interpolate(model_output['pred_masks'], size=gt_shape, mode="bilinear", align_corners=False)
        if sigmoid:
            pred = pred.sigmoid()
        mask_list.append(pred)
    if ms:
        if ms_gather == 'max':
            ms_pred = torch.max(torch.stack(mask_list, dim=0), dim=0)
            output_result = {'pred_masks': ms_pred}
        else:
            ms_pred = torch.mean(torch.stack(mask_list, dim=0), dim=0)
            output_result = {'pred_masks': ms_pred}
    else:
        output_result = {'pred_masks': mask_list[0]}
    return output_result


def augment_flip(samples, targets, dim=-1):
    samples.tensors = samples.tensors.flip(dim)
    samples.mask = samples.mask.flip(dim)
    # import ipdb;ipdb.set_trace()
    if 'flows' in targets[0]:
        targets[0]['flows'] = targets[0]['flows'].flip(dim)
    return samples, targets

def eval_iou(annotation, segmentation):
    """
    Collected from https://github.com/fperazzi/davis/blob/main/python/lib/davis/measures/jaccard.py
    Compute region similarity as the Jaccard Index.
         Arguments:
             annotation   (ndarray): binary annotation   map.
             segmentation (ndarray): binary segmentation map.
         Return:
             jaccard (float): region similarity
    """
    annotation = annotation.astype(np.bool_)
    segmentation = segmentation.astype(np.bool_)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
            np.sum((annotation | segmentation), dtype=np.float32)


@torch.no_grad()
def infer(model, data_loader, device, msc=False, flip=False, save_pred=False, out_dir='./results/dsec_mos/', msc_scales=None):
    
    if msc and msc_scales is not None:
        _scales = msc_scales
    elif msc and msc_scales is None:
        _scales = [0.75, 0.8, 0.9, 1, 1.1, 1.2, 1.3]
    else:
        _scales = [1]
    model.eval()
    i_iter = 0
    iou_list = []
    vid_iou_dict = {}
    running_video_name = None
    for samples, targets in tqdm(data_loader):
        i_iter = i_iter + 1
        video_name = targets[0]['video_name']
        frame_ids = targets[0]['frame_ids']
        center_frame_name = targets[0]['center_frame']
        center_frame_index = frame_ids.index(center_frame_name)
        samples = samples.to(device)
        targets = [{k: v.to(device) if k in ['masks'] else v for k, v in t.items()} for t in targets]
        # ###############################################
        center_gt_path = targets[0]['mask_paths'][center_frame_index]
        center_gt = cv2.imread(center_gt_path, cv2.IMREAD_GRAYSCALE)
        center_gt[center_gt > 0] = 1
        gt_shape = center_gt.shape
        if running_video_name is not None and video_name != running_video_name:
            video_iou = np.mean(list(vid_iou_dict[running_video_name].values()))
            logger.debug('video_name:%s iou:%0.3f' % (running_video_name, video_iou))
        running_video_name = video_name
        outputs = compute_predictions_flip_ms(model, samples, targets, gt_shape, ms=msc, ms_gather='mean',
                                              flip=flip, flip_gather='mean', scales=_scales)
        # import ipdb; ipdb.set_trace()
        src_masks = outputs["pred_masks"]
        yc_logits = src_masks.squeeze(0).cpu().detach().numpy()[center_frame_index, :, :].copy()
        yc_binmask = yc_logits.copy()
        yc_binmask[yc_binmask > 0.5] = 1
        yc_binmask[yc_binmask <= 0.5] = 0
        out = yc_binmask.astype(center_gt.dtype)
        # ########################################
        iou = eval_iou(center_gt.copy(), out.copy())
        iou_list.append(iou)
        if video_name not in vid_iou_dict:
            vid_iou_dict[video_name] = {}
        vid_iou_dict[video_name][center_frame_name] = iou
        if save_pred:
            logits_out_dir = os.path.join(out_dir, 'logits', video_name)
            if not os.path.exists(logits_out_dir):
                os.makedirs(logits_out_dir)
            cv2.imwrite(os.path.join(logits_out_dir, '%s.png' % center_frame_name),
                        (yc_logits.astype(np.float32) * 255).astype(np.uint8))
            bm_out_dir = os.path.join(out_dir, 'bin_mask', video_name)
            if not os.path.exists(bm_out_dir):
                os.makedirs(bm_out_dir)
            cv2.imwrite(os.path.join(bm_out_dir, '%s.png' % center_frame_name),
                        (out.astype(np.float32) * 255).astype(np.uint8))  # it is 0, 1
    video_iou = np.mean(list(vid_iou_dict[running_video_name].values()))
    logger.debug('video_name:%s iou:%0.3f' % (running_video_name, video_iou))
    video_mean_iou = np.mean([np.mean(list(vid_iou_f.values())) for _, vid_iou_f in vid_iou_dict.items()])
    # ### ### Write the results to CSV ### ###
    csv_file_name = 'dsec_mos_results.csv'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    video_names = []
    video_ious = []
    for k, v in vid_iou_dict.items():
        vid_iou = np.mean(list(v.values()))
        video_names.append(k)
        video_ious.append('%0.3f' % vid_iou)
        logger.debug('video_name:%s iou:%0.3f' % (k, vid_iou))
    video_names.append('Video Mean')
    video_ious.append('%0.3f' % video_mean_iou)
    with open(os.path.join(out_dir, csv_file_name), 'w') as f:
        cf = csv.writer(f)
        cf.writerow(video_names)
        cf.writerow(video_ious)
    logger.debug('Videos Mean IOU: %0.3f' % video_mean_iou)
    return video_mean_iou


def run_inference(args, device, model, load_state_dict=True, out_dir=None, videos=None):
    if out_dir is None:
        out_dir = args.output_dir
    if out_dir is None or len(out_dir) == 0:
        out_dir = './results'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # ### Data Loader #########
    if not hasattr(args, 'save_pred'):
        args.save_pred = False
    if not hasattr(args, 'msc'):
        args.msc = False
    if not hasattr(args, 'flip'):
        args.flip = False
    
    dataset_val = DsecMosValDataset(num_frames=args.num_frames, val_size=args.val_size, sequence_names=args.sequence_names)
    
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_val = torch.utils.data.BatchSampler(sampler_val, args.batch_size, drop_last=False)
    data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val, collate_fn=utils.misc.collate_fn,
                                 num_workers=args.num_workers)
    with torch.no_grad():
        if load_state_dict:
            state_dict = torch.load(args.model_path)['model']
            model.load_state_dict(state_dict, strict=True)
        model.eval()
        # import ipdb;ipdb.set_trace()
        
        infer(model, data_loader_val, device, msc=args.msc, flip=args.flip,
                           save_pred=args.save_pred, out_dir=out_dir)
        
