# TKRL_DCMIS

The official implementation of our paper
["TKRL: Targeted Knowledge Rectification Learning against Teacher-Originated Defects in Domain Continual Segmentation"](https://ieeexplore.ieee.org/document/11359675),
published in **IEEE Journal of Biomedical and Health Informatics (JBHI), 2026**.

TKRL is the third work in our domain continual medical image segmentation series.
Building upon TED and CauAug, TKRL further studies a deeper challenge in continual
segmentation:

> What if the old teacher model itself already contains defective knowledge?

To address this problem, TKRL introduces **Targeted Knowledge Rectification Learning**
for rectifying:

- knowledge gaps;
- knowledge biases;
- teacher-originated defects.

before defective knowledge is propagated to future continual models.

## 📖 Table of Contents

- [TKRL\_DCMIS](#tkrl_dcmis)
  - [📖 Table of Contents](#-table-of-contents)
  - [Overview](#overview)
  - [What's New](#whats-new)
  - [Research Motivation](#research-motivation)
  - [Method Evolution](#method-evolution)
  - [Key Idea](#key-idea)
  - [Framework](#framework)
  - [Requirements](#requirements)
  - [Project Structure](#project-structure)
  - [Data Preparation](#data-preparation)
  - [Run](#run)
  - [Analysis](#analysis)
  - [Ablation Study](#ablation-study)
  - [Acknowledgement](#acknowledgement)
  - [Citation](#citation)

## Overview

Domain continual medical image segmentation aims to continuously adapt segmentation
models to sequentially arriving medical domains while preserving previously learned
knowledge.

Knowledge distillation is widely used in continual segmentation because it transfers old
knowledge from a frozen teacher model to a new student model.

However, existing methods generally assume that the old teacher model provides:

- complete knowledge;
- unbiased semantic representations;
- reliable supervision.

This assumption is often unrealistic in medical image segmentation.

Due to:

- limited training data;
- annotation inaccuracies;
- domain heterogeneity;
- imperfect model capacity;

older teacher models may inherently contain defective knowledge.

TKRL reveals a new challenge in continual segmentation:

> Teacher-Originated Defects (TOD)

including:

- knowledge gaps;
- knowledge biases.

These defects are progressively propagated through continual distillation and eventually
exacerbate catastrophic forgetting.

To address this issue, TKRL introduces targeted knowledge rectification mechanisms for
both probing hidden knowledge and correcting inherited bias.

## What's New

TKRL further extends our previous research line on domain continual medical image
segmentation.

Previous works:

- **TED_DCMIS**
  - improves old knowledge retention through tri-enhanced distillation.
  - 📄 [Paper (MedIA)](https://www.sciencedirect.com/science/article/abs/pii/S1361841524000379)
  - 💻 [Code (GitHub)](https://github.com/PerceptionComputingLab/TED_DCMIS)

- **CauAug_DCMIS**
  - studies continual segmentation from a causal learning perspective.
  - 📄 [Paper (IEEE JBHI)](https://ieeexplore.ieee.org/document/11054328)
  - 💻 [Code (GitHub)](https://github.com/PerceptionComputingLab/CauAug_DCMIS)

TKRL further asks:

> Even if knowledge distillation is well-designed, what if the teacher model itself is
> already defective?

## Research Motivation

TED improves old knowledge retention.

CauAug further mitigates causal bias during old-new knowledge learning.

However, both approaches still implicitly assume that the teacher model itself is
reliable.

In practice, teacher models in medical image segmentation may inherently contain:

1. **Knowledge Gaps**
   - important anatomical patterns may not be fully learned;
   - rare structures may be underrepresented;
   - incomplete feature coverage may accumulate over continual learning.

2. **Knowledge Biases**
   - ambiguous annotations may introduce biased representations;
   - inaccurate boundaries may propagate through distillation;
   - teacher errors may continuously accumulate.

Existing continual learning methods mainly focus on preserving teacher knowledge, but
rarely ask whether the teacher knowledge itself should first be rectified.

Therefore, TKRL introduces targeted knowledge rectification learning to:

- probe hidden knowledge gaps;
- correct inherited semantic bias;
- prevent defective knowledge propagation.

## Method Evolution

TKRL is the third stage of our research line on domain continual medical image
segmentation.

```text
TED
└── How to better retain old knowledge?
    ├── diversity enhancement
    ├── transfer accuracy enhancement
    └── fusion stability enhancement

CauAug
└── How to causally optimize both old and new knowledge?
    ├── causal intervention
    ├── causal augmentation
    └── confounder disentanglement

TKRL
└── How to rectify defective teacher knowledge?
    ├── knowledge gap probing
    ├── knowledge bias correction
    └── teacher-originated defect rectification
```

The overall evolution of this research series is:

```text
TED: Old knowledge retention
  ↓
CauAug: Causal learning of both old and new knowledge
  ↓
TKRL: Rectification of teacher-originated defects
```

Compared with previous works, TKRL shifts the research focus from:

- knowledge retention

to:

- knowledge rectification.

## Key Idea

The core idea of TKRL is:

> Continual learning should not only preserve old knowledge, but also identify and
> rectify defective teacher knowledge before distillation.

To achieve this goal, TKRL introduces two complementary mechanisms:

- probing hidden knowledge gaps;
- correcting biased semantic representations.

Specifically, TKRL introduces:

- **Probe-augmented Class Distillation (PCD)**
  - generates boundary-directed knowledge probes;
  - uncovers underrepresented teacher knowledge;
  - bridges hidden knowledge gaps.

- **Variance-guided Masked Autoencoder (VMA)**
  - identifies high-uncertainty semantic regions;
  - reconstructs biased representations;
  - corrects teacher-originated semantic bias.

## Framework

TKRL follows a distillation-based domain continual learning pipeline.

At each continual stage:

- the previous model is frozen as the teacher model;
- the current model learns the new domain;
- teacher knowledge is first rectified before distillation.

The overall optimization objective jointly combines:

- segmentation learning for the current domain;
- standard continual distillation;
- knowledge gap probing;
- teacher bias correction.

<p align="center">
  <img src="figures/framework.png" width="95%">
</p>

## Requirements

- Python 3.8.15
- PyTorch
- CUDA

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
--ablation/
--analysis/
--data_prep/
--mp/
--storage/
--README.md
--requirements.txt
--main.py
--get.py
--args.py
--command
```

## Data Preparation

Please refer to the data preparation instructions:

```bash
cat data_prep/readme.md
python data_prep/prostate_prepare.py
python data_prep/hippocampus_prepare.py
python data_prep/polyp_prepare.py
```

## Run

Please check the example commands:

```bash
cat command
```

Example for prostate continual segmentation:

```bash
python main.py   --approach tkrl --epochs 30 --experiment-name  polyp-tkrl   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-tkrl.log
```

## Analysis

```bash
python analysis/eval_dataset.py   # evaluate each dataset and each approach
python analysis/table_figure.py   # generate tables and figures in the paper
python analysis/save_images.py    # save segmentation results
python analysis/visual_pcd.py      # visualize knowledge probes
python analysis/visual_vma.py # visualize variance-guided masks
```

## Ablation Study

```bash
# Ablation study of Probe-augmented Class Distillation (PCD)
python main.py   --approach pcd --epochs 30 --experiment-name  polyp-pcd   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-pcd.log

# Ablation study of Variance-guided Masked Autoencoder (VMA)
python main.py   --approach vma --epochs 30 --experiment-name  polyp-vma   --batch-size 16 --device-ids 0 --dataset polyp --resume-from polyp-seq > log/polyp-vma.log
```

## Acknowledgement

Our code is inspired by
<a href="https://github.com/MECLabTUDA/ACS">ACS</a>
and our previous works:

- <a href="https://github.com/PerceptionComputingLab/TED_DCMIS">TED</a>
- <a href="https://github.com/PerceptionComputingLab/CauAug_DCMIS">CauAug</a>

## Citation

```bibtex
@ARTICLE{11359675,
  author={Zhu, Zhanshi and Gu, Wenjian and Li, Xiangyu and Li, Qince and Yuan, Yongfeng and Wang, Wei and Wang, Kuanquan and Dong, Suyu and Li, Shuo},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={TKRL: Targeted Knowledge Rectification Learning Against Teacher-Originated Defects in Domain Continual Segmentation}, 
  year={2026},
  volume={},
  number={},
  pages={1-14},
  keywords={Image segmentation;Data models;Biomedical imaging;Autoencoders;Probes;Adaptation models;Training;Biological system modeling;Bioinformatics;Annotations;Domain continual segmentation;knowledge distillation;masked autoencoder;teacher-originated defects},
  doi={10.1109/JBHI.2026.3656447}}

```