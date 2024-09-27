"""
Dataloader for inference.
"""

import math
import torch
import torch.utils.data
import os
from PIL import Image
import cv2
import numpy as np
import glob
import logging

from emoformer.datasets import path_config as dataset_path_config
import emoformer.datasets.transforms as T

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DsecMosValDataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=6, val_size=473, sequence_names=None, max_sc=None):
        super(DsecMosValDataset, self).__init__()
        
        self.num_frames = num_frames
        self.split = 'val'
        
        self.im_size = val_size
        self._transforms = make_validation_transforms(min_size=val_size, max_sc=max_sc)

        self.dsec_mos_val_seqs_file = dataset_path_config.dsec_mos_val_seqs_file
        self.dsec_mos_rgb_path = dataset_path_config.dsec_mos_rgb_path
        self.dsec_mos_gt_path = dataset_path_config.dsec_mos_gt_path
        
        self.frames_info = {'dsec_mos': {},}
        
        self.img_ids = []
        
        logger.debug('loading dsec_mos val seqs...')
        
        if sequence_names is None or len(sequence_names) == 0:
            with open(self.dsec_mos_val_seqs_file, 'r') as f:
                video_names = f.readlines()
                video_names = [name.strip() for name in video_names]
        else:
            video_names = sequence_names
        logger.debug('dsec_mos-val num of videos: {}'.format(len(video_names)))
        for video_name in video_names:
            frames = sorted(glob.glob(os.path.join(self.dsec_mos_gt_path, video_name, '*.png')))
            self.frames_info['dsec_mos'][video_name] = [frame_path.split('/')[-1][:-4] for frame_path in frames]
            self.img_ids.extend([('dsec_mos', video_name, frame_index) for frame_index in range(len(frames))])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_ids_i = self.img_ids[idx]
        dataset, video_name, frame_index = img_ids_i
        vid_len = len(self.frames_info[dataset][video_name])
        center_frame_name = self.frames_info[dataset][video_name][frame_index]
        frame_indices = [(x + vid_len) % vid_len for x in range(frame_index - math.floor(float(self.num_frames) / 2),
                                                                frame_index + math.ceil(float(self.num_frames) / 2), 1)]
        assert len(frame_indices) == self.num_frames
        frame_ids = []
        img = []
        
        masks = []
        mask_paths = []
        
        for frame_id in frame_indices:
            frame_name = self.frames_info[dataset][video_name][frame_id]
            frame_ids.append(frame_name)
            
            img_path = os.path.join(self.dsec_mos_rgb_path, video_name, frame_name + '.png')
            
            gt_path = os.path.join(self.dsec_mos_gt_path, video_name, frame_name + '.png')
            
            img_i = Image.open(img_path).convert('RGB')
            img.append(img_i)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt[gt > 0] = 255
            masks.append(torch.Tensor(np.expand_dims(np.asarray(gt.copy()), axis=0)))
            mask_paths.append(gt_path)
            
        masks = torch.cat(masks, dim=0)
        target = {'dataset': dataset, 'video_name': video_name, 'center_frame': center_frame_name,
                  'frame_ids': frame_ids, 'masks': masks, 'vid_len': vid_len, 'mask_paths': mask_paths}
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        return torch.cat(img, dim=0), target


def make_validation_transforms(min_size=360, max_sc=None):
    if max_sc is None:
        max_sc = 1.8
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
        T.RandomResize([min_size], max_size=int(max_sc * min_size)),
        normalize,
    ])

