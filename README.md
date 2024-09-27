# Event-Free Moving Object Segmentation from Moving Ego Vehicle

This repository is for the paper **Event-Free Moving Object Segmentation from Moving Ego Vehicle**, by
[Zhuyun Zhou](https://scholar.google.com/citations?user=sXolUXMAAAAJ&hl=en&oi=ao),
[Zongwei Wu](https://scholar.google.com/citations?user=3QSALjX498QC&hl=en&oi=ao),
[Danda Pani Paudel](https://scholar.google.com/citations?user=W43pvPkAAAAJ&hl=en&oi=ao),
[Rémi Boutteau](https://scholar.google.com/citations?user=U-SrcPkAAAAJ&hl=en&oi=ao),
[Fan Yang](https://scholar.google.com/citations?user=GNQHje8AAAAJ&hl=en&oi=ao),
[Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en&oi=ao),
[Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en&oi=ao),
[Dominique Ginhac](https://scholar.google.com/citations?user=fkdCT5kAAAAJ&hl=en&oi=ao).

PDF version of the paper is available [here](https://arxiv.org/abs/2305.00126).

Dataset ***DSEC-MOS*** **: ***DSEC*** - ***M***oving ***O***bject ***S***egmentation** can be found [here](#dataset).


## Contents

1. [Abstract](#abstract)
2. [News](#news)
3. [Citation](#citation)
4. [Dataset](#dataset)
5. [Pre-trained Weights](#pre-trained-weights)
6. [Installation](#installation)


## Abstract

Moving object segmentation (MOS) in dynamic scenes is an important, challenging, but under-explored research topic for autonomous driving, especially for sequences obtained from moving ego vehicles. 
Most segmentation methods leverage motion cues obtained from optical flow maps. However, since these methods are often based on optical flows that are pre-computed from successive RGB frames, this neglects the temporal consideration of events occurring within the inter-frame, consequently constraining its ability to discern objects exhibiting relative staticity but genuinely in motion. To address these limitations, we propose to exploit event cameras for better video understanding, which provide rich motion cues without relying on optical flow. To foster research in this area, we first introduce a novel large-scale dataset called DSEC-MOS for moving object segmentation from moving ego vehicles, which is the first of its kind. For benchmarking, we select various mainstream methods and rigorously evaluate them on our dataset. Subsequently, we devise EmoFormer, a novel network able to exploit the event data. For this purpose, we fuse the event temporal prior with spatial semantic maps to distinguish genuinely moving objects from the static background, adding another level of dense supervision around our object of interest. Our proposed network relies only on event data for training but does not require event input during inference, making it directly comparable to frame-only methods in terms of efficiency and more widely usable in many application cases.
The exhaustive comparison highlights a significant performance improvement of our method over all other methods.


## News

* Sep. 27, 2024: Code of our ***EmoFormer*** is released.


## Citation

```BibTeX
@inproceedings{zhou2024event,
  title={Event-Free Moving Object Segmentation from Moving Ego Vehicle},
  author={Zhou, Zhuyun and Wu, Zongwei and Pani Paudel, Danda and Boutteau, R{\'e}mi and Yang, Fan and Van Gool, Luc and Timofte, Radu and Ginhac, Dominique},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2024},
  organization={IEEE}
}
```


## Dataset

***DSEC-MOS*** **: ***DSEC*** - ***M***oving ***O***bject ***S***egmentation** can be downloaded here: [Training]() and [Testing]().

In total, our DSEC-MOS dataset contains 16 sequences (13314 frames), with 11 sequences (10495 frames) for training and 5 other sequences (2819 frames) for testing.

In each sequence:
* `image_calib`: RGB frames calibrated to the event-based coordinates, so that RGB and event maps have the same field of view and the same resolution;
* `gt_mask`: ground truth masks of moving objects;
* `prior`: event frame of 50ms from left sensor, as suggested in the dataset [DSEC-MOD](#sister-dataset-dsec-mod) in paper [ RGB-Event Fusion for Moving Object Detection in Autonomous Driving](https://github.com/ZZY-Zhou/RENet), with PDF of paper [here](https://arxiv.org/abs/2209.08323).

The format should be:
```
└── DSEC_MOS
    ├── training
    │   ├── zurich_city_00_a
    │   │   ├── image_calib
    │   │   │   ├── 000001.png
    │   │   │   └── ...
    │   │   ├── gt_mask
    │   │   │   ├── 000001.png
    │   │   │   └── ...
    │   │   └── prior  
    │   │   │   ├── 000001.png
    │   │   │   └── ...
    │   └── ...
    └── testing
        ├── zurich_city_13_a
        │   └── ...
        └── ... 
```


### Parent Dataset: DSEC

DSEC is available here: [ https://dsec.ifi.uzh.ch](https://dsec.ifi.uzh.ch).

Details can be found in the paper [ DSEC: A Stereo Event Camera Dataset for Driving Scenarios](https://rpg.ifi.uzh.ch/docs/RAL21_DSEC.pdf).


### Sister Dataset: DSEC-MOD

DSEC-MOD is available here: [ https://github.com/ZZY-Zhou/RENet](https://github.com/ZZY-Zhou/RENet).

Details can be found in the paper [ RGB-Event Fusion for Moving Object Detection in Autonomous Driving](https://arxiv.org/abs/2209.08323).


## Pre-trained Weights

Our pre-trained weights for our RENet can be downloaded [here]().


## Installation

1. Clone

```
git clone https://github.com/ZZY-Zhou/DSEC-MOS
cd DSEC-MOS
```

2. Create and activate conda environment

```
conda create -n ENV_NAME
conda activate ENV_NAME
```
