# SFC-Net
## [Paper] SFC-Net: A Spatial-Frequency Collaborative Alignment and Fusion Network for Weakly Aligned RGB-T Salient Object Detection
These are the official code releases.

## Results
<img src="https://github.com/jubo-neu/SFC-Net/blob/main/vis.png" alt="Image">

## Highlights
- Proposes SFC-Net for weakly aligned RGB-T salient object detection.
- Features MFFM for robust cross-modal semantic alignment.
- Introduces SBIM and FDCM for spatial-frequency dual-domain decoupling.
- Employs CSAM to calibrate features across spatial and frequency domains.

## TODO
- [x] Val and Test codes.
- [x] Train code and Models.
- [ ] Checkpoints.

### News
- 2026-4-27: Pretrained weights will be released after the paper is officially accepted.
- 2026-4-27: Train code and Models have been released.
- 2026-4-26: Val and Test codes have been released.

## Preparation

To start, we prefer creating the environment using conda:

```
conda env create -f environment.yml
conda activate SFCNet
```

## Getting the data

- Download datasets [Unaligned-VT](https://github.com/lz118/Deep-Correlation-Network).


## Training
1. Download the pretrained UniRepLKNet model [here](https://github.com/AILab-CVC/UniRepLKNet).

2. Start training:
```bash
python /train.py
```

## Test
```bash
python /test.py
```

## Citation
If you find this repository useful in your project, please cite the following work:
```bash

```

## Contact us
If you have any questions, please contact us [jbchen@stumail.neu.edu.cn](jbchen@stumail.neu.edu.cn).
