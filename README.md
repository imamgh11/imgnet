# A New Paradigm of Face Verification: Identity Through Relational Patterns, Not Absolute Values

> *In Javanese, one expresses gratitude as "matur suwun"; in Sundanese, the same sentiment is conveyed as "hatur nuhun". Despite different surface structures, both phrases encode identical meaning through internally consistent relational patterns. This linguistic observation from two major languages of Indonesia inspired the central hypothesis of this work: that identity can be encoded through consistent relational patterns rather than absolute values.*

**Author:** Imam Ghozali — Independent Researcher  
**Contact:** imam.gh98@gmail.com

---

## Overview

IMGNet is a lightweight face verification model built around a novel **Sliding Window (SW) Block** that replaces the first convolutional layer with a multi-scale relational operation. Instead of dot products on absolute pixel values, SW Block computes learnable functions of pixel *differences* at three prime scales (3×3, 5×5, 7×7), making it illumination-robust by design.

Three novel similarity metrics are introduced:

| Metric | Description | Range |
|--------|-------------|-------|
| **IMG Sign Score** | Sliding window sign pattern matching (no amplitude) | 0–1 |
| **AMP IMG Score** | Sign pattern × amplitude consistency | 0–1 |
| **Chain Score** | Quality of contiguous match runs (reward/punish) | 0–1 |

All three metrics share a single threshold derived from IMG Sign Score sweep — AMP and Chain use the same threshold value.

---

## Architecture — SW357 + Conv10 (1SW + 10Conv)

```
SW1    : 112×112 → 56×56   (SW Block, windows [3,5,7])
Conv2  : 56×56  → 56×56   (stride=1)
Conv3  : 56×56  → 28×28   (stride=2)
Conv4  : 28×28  → 28×28   (stride=1)
Conv5  : 28×28  → 28×28   (stride=1)
Conv6  : 28×28  → 14×14   (stride=2)
Conv7  : 14×14  → 14×14   (stride=1)
Conv8  : 14×14  → 14×14   (stride=1)
Conv9  : 14×14  → 7×7     (stride=2)
Conv10 : 7×7   → 7×7     (stride=1)
GAP → FC(256→1024) → BN
```

**Parameters:** 2,774,176 (~10.58 MB FP32, ~5.29 MB FP16)

---

## Results

### Benchmark (LFW pre-aligned 112×112, epoch 39 plateau)

| Dataset | IMG Sign | AMP | Chain | Vote 1/3 | Vote 2/3 | Cosine |
|---------|----------|-----|-------|----------|----------|--------|
| LFW | **96.27%** | 90.45% | 95.12% | **96.27%** | 95.13% | 95.53% |
| AgeDB-30 | 78.80% | 74.22% | 72.87% | 78.80% | 74.73% | 77.22% |
| CALFW | 78.73% | 74.92% | 76.87% | 78.73% | 77.15% | 78.32% |
| CPLFW | 76.85% | 68.88% | 75.23% | 76.85% | 75.25% | 74.62% |
| Combined | 81.02% | 77.41% | 79.30% | 81.02% | 79.47% | 79.49% |

*Trained on CASIA-WebFace (490k images, 10,572 identities)*  
*Threshold from IMG Sign Score sweep — AMP and Chain use the same threshold*

### Comparison with Pretrained Models

| Model | Dataset | LFW | Params |
|-------|---------|-----|--------|
| MobileNetV2 | MS1MV2 | 99.55% | 2.29M |
| **IMGNet Conv10** | **CASIA 490k** | **96.27%** | **2.77M** |
| MobileNetV1_0.25 | MS1MV2 | 98.76% | 0.36M |

---

## Datasets

| Dataset | Link | Description |
|---------|------|-------------|
| CASIA-WebFace aligned | [Kaggle](https://www.kaggle.com/datasets/luongkhang04/aligned-casia) | Training dataset, aligned & cropped, 490k images, 10,572 identities |
| Benchmark (LFW/AgeDB/CALFW/CPLFW) | [Kaggle](https://www.kaggle.com/datasets/yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112) | Validation datasets, pre-aligned 112×112 |

```
train/
  train_sw357_conv10_imgsign_a100.py  — Training on A100/Colab
  train_eval_sw357_conv10_gtx.py      — 1-epoch test on GTX
  train_eval_sw357_conv13_gtx.py      — Conv13 variant test
  precrop_casia.py                    — Pre-crop CASIA with MTCNN

eval/
  eval_lfw_gtx_chain_conv10.py        — Eval Conv10 + Chain Score (GTX)
  eval_lfw_gtx_imgsign_conv10.py      — Eval Conv10 IMG Sign (GTX)
  eval_benchmarks_a100.py             — Multi-dataset benchmark (A100)
  eval_metric_comparison_a100.py      — FaceNet/ArcFace metric test

app/
  face_compare_conv10.py              — Desktop UI comparison app (tkinter)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install torch torchvision facenet-pytorch insightface Pillow numpy scikit-learn
```

### 2. Download checkpoint

Place `best_model_epoch39_plateau.pth` in your working directory.

### 3. Eval on LFW

```bash
# Edit CKPT_PATH and LFW_DIR in the script first
python eval_lfw_gtx_imgsign_conv10.py
```

### 4. Run comparison app

```bash
python face_compare_conv10.py
```

---

## Voting System

Three metrics, one threshold (from IMG Sign sweep):

```
2/3 or 3/3 pass → ✅ MATCH
1/3 pass        → ⚠️  UNCERTAIN
0/3 pass        → ❌ DIFFERENT
```

---


---

## Citation

```
Ghozali, I. (2026). IMG: Index-Based Match Scoring with Grade — 
A Novel Similarity Metric for Face Verification. 
Zenodo. https://doi.org/10.5281/zenodo.20748457
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20748457.svg)](https://zenodo.org/records/20748457)

---

## License

MIT License
