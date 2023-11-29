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

## Abstract

Moving object segmentation (MOS) in dynamic scenes is challenging for autonomous driving, especially for sequences obtained from moving ego vehicles. 
Most state-of-the-art methods leverage motion cues obtained from optical flow maps. 
However, since these methods are often based on optical flows that are pre-computed from successive RGB frames, this neglects the temporal consideration of events occurring within inter-frame and limits the practicality of these methods in real-life situations.
To address these limitations, we propose to exploit event cameras for better video understanding, which provide rich motion cues without relying on optical flow. To foster research in this area, we first introduce a novel large-scale dataset called DSEC-MOS for moving object segmentation from moving ego vehicles. Subsequently, we devise EmoFormer, a novel network able to exploit the event data. For this purpose, we fuse the event prior with spatial semantic maps to distinguish moving objects from the static background, adding another level of dense supervision around our object of interest - moving ones. Our proposed network relies only on event data for training but does not require event input during inference, making it directly comparable to frame-only methods in terms of efficiency and more widely usable in many application cases.
An exhaustive comparison with 8 state-of-the-art video object segmentation methods highlights a significant performance improvement of our method over all other methods.

## Citation

```BibTeX
@article{zhou2023event,
  title={Event-Free Moving Object Segmentation from Moving Ego Vehicle},
  author={Zhou, Zhuyun and Wu, Zongwei and Pani Paudel, Danda and Boutteau, R{\'e}mi and Yang, Fan and Van Gool, Luc and Timofte, Radu and Ginhac, Dominique},
  journal={arXiv preprint arXiv:2305.00126},
  year={2023}
}
```
