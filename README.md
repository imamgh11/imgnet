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

> **Keep the existing Architecture, Benchmark, Metric Definitions, Citation, and License sections from the original README below this section.**
