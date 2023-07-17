# DSEC-MOS: Segment Any Moving Object with Moving Ego Vehicle

This repository is for the paper **DSEC-MOS: Segment Any Moving Object with Moving Ego Vehicle**, by
[Zhuyun Zhou](https://scholar.google.com/citations?user=sXolUXMAAAAJ&hl=en&oi=ao),
[Zongwei Wu](https://scholar.google.com/citations?user=3QSALjX498QC&hl=en&oi=ao),
[Rémi Boutteau](https://scholar.google.com/citations?user=U-SrcPkAAAAJ&hl=en&oi=ao),
[Fan Yang](https://scholar.google.com/citations?user=GNQHje8AAAAJ&hl=en&oi=ao),
[Dominique Ginhac](https://scholar.google.com/citations?user=fkdCT5kAAAAJ&hl=en&oi=ao).

PDF version of the paper is available [here](https://arxiv.org/abs/2305.00126).

## Abstract

Moving Object Segmentation (MOS), a crucial task in computer vision, has numerous applications such as surveillance, autonomous driving, and video analytics. Existing datasets for moving object segmentation mainly focus on RGB or Lidar videos, but lack additional event information that can enhance the understanding of dynamic scenes. To address this limitation, we propose a novel dataset, called DSEC-MOS. Our dataset includes frames captured by RGB cameras embedded on moving vehicules and incorporates event data, which provide high temporal resolution and low-latency information about changes in the scenes. To generate accurate segmentation mask annotations for moving objects, we apply the recently emerged large model SAM - Segment Anything Model - with moving object bounding boxes from DSEC-MOD serving as prompts and calibrated RGB frames, then further revise the results. Our DSEC-MOS dataset contains in total 16 sequences (13314 images). To the best of our knowledge, DSEC-MOS is also the first moving object segmentation dataset that includes event camera in autonomous driving.

## Citation

```BibTeX
@article{zhou2023dsec,
  title={DSEC-MOS: Segment Any Moving Object with Moving Ego Vehicle},
  author={Zhou, Zhuyun and Wu, Zongwei and Boutteau, R{\'e}mi and Yang, Fan and Ginhac, Dominique},
  journal={arXiv preprint arXiv:2305.00126},
  year={2023}
}
```
