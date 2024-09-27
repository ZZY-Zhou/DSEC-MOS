"""
Dataloader with train data.
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


class DsecMosTrainDataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=6, train_size=300):
        self.num_frames = num_frames
        
        self.split = 'train'
        
        self._transforms = make_train_transform(train_size=train_size)

        self.dsec_mos_train_seqs_file = dataset_path_config.dsec_mos_train_seqs_file
        self.dsec_mos_rgb_path = dataset_path_config.dsec_mos_rgb_path
        self.dsec_mos_gt_path = dataset_path_config.dsec_mos_gt_path
        
        self.dsec_mos_prior_path = dataset_path_config.dsec_mos_prior_path
        
        self.frames_info = {'dsec_mos': {},}
        
        self.img_ids = []
        logger.debug('loading dsec_mos train seqs...')
        with open(self.dsec_mos_train_seqs_file, 'r') as f:
            video_names = f.readlines()
            video_names = [name.strip() for name in video_names]
            logger.debug('dsec_mos-train num of videos: {}'.format(len(video_names)))
            for video_name in video_names:
            
                frames = sorted(glob.glob(os.path.join(self.dsec_mos_gt_path, video_name, '*.png')))
                
                priors = sorted(glob.glob(os.path.join(self.dsec_mos_prior_path, video_name, '*.png')))
                
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
        
        priors = []
        
        # import ipdb;ipdb.set_trace()
        for frame_id in frame_indices:
            frame_name = self.frames_info[dataset][video_name][frame_id]
            frame_ids.append(frame_name)
            
            img_path = os.path.join(self.dsec_mos_rgb_path, video_name, frame_name + '.png')
            gt_path = os.path.join(self.dsec_mos_gt_path, video_name, frame_name + '.png')
            prior_path = os.path.join(self.dsec_mos_prior_path, video_name, frame_name + '.png')
            
            # import ipdb;ipdb.set_trace()
            img_i = Image.open(img_path).convert('RGB')
            img.append(img_i)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt[gt > 0] = 255
            masks.append(torch.Tensor(np.expand_dims(np.asarray(gt.copy()), axis=0)))
            
            prior = cv2.imread(prior_path, cv2.IMREAD_GRAYSCALE)
            prior[prior > 0] = 255
            priors.append(torch.Tensor(np.expand_dims(np.asarray(prior.copy()), axis=0)))
            
        # import ipdb;ipdb.set_trace()
        masks = torch.cat(masks, dim=0)
        
        priors = torch.cat(priors, dim=0)
        
        target = {'dataset': dataset, 'video_name': video_name, 'center_frame': center_frame_name,
                  'frame_ids': frame_ids, 'masks': masks, 'vid_len': vid_len, 'priors': priors}
        
        # import ipdb;ipdb.set_trace()
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # import ipdb;
        # ipdb.set_trace()
        
        return torch.cat(img, dim=0), target


def make_train_transform(train_size=None):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResize(scales, max_size=800),
        T.PhotometricDistort(),
        T.Compose([
            T.RandomResize([500, 600, 700]),
            T.RandomSizeCrop(473, 750),
            T.RandomResize([train_size], max_size=int(1.8 * train_size)),
        ]),
        normalize,
    ])
