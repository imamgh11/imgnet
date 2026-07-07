# IMG — Relational Pattern-Based Similarity Metric

**A Universal Similarity Metric for Computer Vision**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20748457-blue)](https://doi.org/10.5281/zenodo.20748457)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** Imam Ghozali — Independent Researcher  
📧 imam.gh98@gmail.com

---

## Overview

Traditional similarity metrics such as cosine similarity compare embedding vectors through **global angular relationships**.

**IMG** introduces a different paradigm: instead of comparing absolute vector values, IMG compares **local relational patterns** inside the embedding.

The proposed framework consists of three complementary metrics:

1. **IMG Sign Score**
2. **AMP IMG Score**
3. **Chain Score**

> **Note:** This work does **not** propose replacing cosine similarity. Instead, IMG is proposed as an *alternative* similarity metric. Experimental results suggest that the optimal similarity metric depends on how the embedding itself is learned.

---

# Relational Learning Hypothesis

In Javanese, one expresses gratitude as **"matur suwun"**; in Sundanese, the same sentiment is conveyed as **"hatur nuhun"**. Despite different surface structures, both phrases encode identical meaning through internally consistent relational patterns.

This linguistic observation inspired the central hypothesis of this work:

> **Identity can be encoded through consistent relational patterns rather than absolute values.**

Instead of forcing embeddings to occupy a specific angular position, the proposed method trains the network to preserve **local relational consistency**.

Consequently, similarity is evaluated by comparing relational patterns rather than absolute vector orientation.

---

# Relational Training Objective

Unlike ArcFace, which explicitly optimizes cosine similarity using Angular Margin Loss,

```math
L_{ArcFace}
=
-\log
\frac
{e^{s\cos(\theta_y+m)}}
{e^{s\cos(\theta_y+m)}+\sum_j e^{s\cos\theta_j}}
```

the proposed method directly optimizes the desired similarity metric itself.

For two embeddings

```math
E_1,E_2\in\mathbb{R}^{1024}
```

the objective is to maximize their **local sign agreement**.

## Soft Sign Agreement

For each embedding dimension,

```math
a_i=
\frac{\tanh(\beta E_{1,i}E_{2,i})+1}{2}
```

where

- positive product → agreement
- negative product → disagreement

Unlike a hard sign comparison, the hyperbolic tangent provides a smooth differentiable approximation.

## Sliding Window Aggregation

For each sliding window,

```math
S_k=\sum_{i=k}^{k+W-1}a_i
```

where

- Window size **W = 11**
- Threshold **T = 8**

## Differentiable Matching Gate

```math
M_k=\sigma\left(50(S_k-T+0.5)\right)
```

which approximates

```math
M_k \approx
\begin{cases}
1 & \text{if } S_k \ge T \\
0 & \text{if } S_k < T
\end{cases}
```

while remaining differentiable.

## IMG Sign Score

```math
IMG(E_1,E_2)=\frac1N\sum_{k=1}^{N}M_k
```

where

```math
N=d-W+1
```

## Relational Loss

Positive pairs:

```math
L_{same}=(1-IMG)^2
```

Negative pairs:

```math
L_{diff}=IMG^2
```

Final objective:

```math
L=L_{same}+L_{diff}
```

This is exactly the objective used during training.

---

# Training Pipeline

| Hyperparameter | Value |
|---|---:|
| Dataset | CASIA-WebFace |
| Identities | 10,572 |
| Images | ~490k aligned faces |
| Embedding Dimension | 1024 |
| Batch Size | 16 |
| Optimizer | Adam |
| Learning Rate | 1×10⁻⁴ |
| Epochs | 50 |
| Warm-up | 5 |
| Scheduler | Cosine Annealing |
| Weight Decay | 1×10⁻⁵ |

Positive pairs consist of two images belonging to the same identity, while negative pairs are randomly sampled from different identities.

Unlike ArcFace, no angular-margin loss, cosine loss, or triplet loss is used. The network is optimized **entirely using the proposed relational objective**.

---

# Why does this matter?

Traditional face-recognition losses optimize embeddings for cosine similarity.

The proposed approach instead optimizes embeddings directly for the intended inference metric.

Consequently,

- Embeddings trained with **Angular Margin Loss** naturally favor **cosine similarity**.
- Embeddings trained with the proposed **relational loss** naturally favor **IMG Sign**.

This suggests that **the similarity metric and the embedding loss should be designed together rather than independently.**

---

--- ## Key Idea | Metric | What it measures | |---|---| | **Cosine Similarity** | Global vector direction | | **IMG Sign** | Local relational sign patterns | | **AMP IMG** | Relational patterns + local amplitude consistency | | **Chain Score** | Continuity of matching relational patterns | --- ## Architecture Used in This Paper **SW357 Block**
Conv2 → Conv3 → Conv4 → Conv5 → Conv6 → Conv7 → Conv8 → Conv9 → Conv10
     → Global Average Pooling → FC → BatchNorm
| Property | Value | |---|---| | Parameters | 2,774,176 | | Model Size (FP32) | 10.58 MB | | Training Dataset | CASIA-WebFace (490k aligned images, 10,572 identities) | --- ## Benchmark ### SW357 Embedding (native) | Dataset | IMG Sign | AMP | Chain | Cosine | |----------|---------:|-------:|-------:|-------:| | LFW | 96.27% | 90.45% | 95.12% | 95.53% | | AgeDB-30 | 78.80% | 74.22% | 72.87% | 77.22% | | CALFW | 78.73% | 74.92% | 76.87% | 78.32% | | CPLFW | 76.85% | 68.88% | 75.23% | 74.62% | | **Combined** | **81.02%** | **77.41%** | **79.30%** | **79.49%** | ### ArcFace Evaluation (relational metric tested on external embedding) | Dataset | IMG Sign | AMP | Chain | Cosine | |----------|---------:|-------:|-------:|-------:| | LFW | 99.58% | 99.48% | 97.02% | 99.82% | | AgeDB-30 | 96.85% | 93.92% | 73.62% | 98.07% | | CALFW | 95.62% | 94.52% | 84.18% | 96.10% | | CPLFW | 93.22% | 91.33% | 77.13% | 94.45% | **Observation:** Cosine remains the best metric for ArcFace because ArcFace is explicitly optimized using Angular Margin Loss. However, IMG Sign remains highly competitive despite never being used during ArcFace training. --- ## Main Finding Results suggest that **Similarity Metric** and **Embedding Loss Function** should be considered together: - Embeddings trained with **Angular Margin Loss** naturally favor **cosine similarity**. - Embeddings trained with the proposed **relational loss** naturally favor **IMG Sign**. **Therefore, there is no universally best similarity metric.** The optimal metric depends on how the embedding space is learned. --- ## Metric Definitions ### IMG Sign Score
python
def img_sign_score_np(e1, e2):
    n = len(e1) - WINDOW_SIZE + 1
    mc = 0
    for i in range(n):
        s1 = np.where(e1[i:i+WINDOW_SIZE] >= 0, 1, -1)
        s2 = np.where(e2[i:i+WINDOW_SIZE] >= 0, 1, -1)
        if np.sum(s1 == s2) >= THRESHOLD:
            mc += 1
    return mc / n
### AMP IMG Score
python
def amp_img_score_np(e1, e2):
    n = len(e1) - WINDOW_SIZE + 1
    total = 0
    for i in range(n):
        w1 = e1[i:i+WINDOW_SIZE]
        w2 = e2[i:i+WINDOW_SIZE]
        s1 = np.where(w1 >= 0, 1, -1)
        s2 = np.where(w2 >= 0, 1, -1)
        if np.sum(s1 == s2) >= THRESHOLD:
            a1 = np.mean(np.abs(w1))
            a2 = np.mean(np.abs(w2))
            total += max(0, 1 - abs(a1 - a2) / max(a1, a2, 1e-6))
    return total / n
### Chain Score
python
def chain_score_np(e1, e2):
    n = len(e1) - WINDOW_SIZE + 1
    flags = []
    for i in range(n):
        ...
    total = sum(flags)
    img_sign = total / n
    ...
    avg_chain = total / n_chains
    diff = avg_chain - NEUTRAL_LEN
    score = img_sign + (
        REWARD_RATE * diff
        if diff >= 0
        else PUNISH_RATE * diff
    ) / 100
    return np.clip(score, 0, 1)
--- ## Conclusion IMG is proposed as an alternative similarity metric rather than a replacement for cosine similarity. Experiments indicate that cosine similarity performs best for embeddings trained with angular-margin objectives, while IMG Sign performs best for embeddings trained with the proposed relational objective. The framework is model-agnostic and can be applied to embeddings generated by different architectures. --- ## Citation If you use this work, please cite via: - **Zenodo (DOI):** https://doi.org/10.5281/zenodo.20748457 - **GitHub:** https://github.com/imamgh11/imgnet --- ## License This project is licensed under the **MIT License**.
