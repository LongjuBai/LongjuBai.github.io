---
layout: post
title: A Brief Summary on Positional Encoding Methods
date: 2024-08-01
description: An overview of different positional encoding techniques used in transformers.
tags: transformers positional-encoding machine-learning
categories: research
---

### Preliminary

Unlike recurrent architectures, transformers process inputs in parallel all at once. However, self-attention does not consider the position of a word in sequence, making sentences like "I think therefore I am" and "I am therefore I think" appear the same.

The notations used are as follows:

- \( S_N = \{ w_i \}_{i=1}^N \): a sequence of \( N \) input tokens with \( w_i \) being the \( i \)-th element.
- \( E_N = \{ x_i \}_{i=1}^N \): corresponding word embeddings of \( S_N \), where \( x_i \in \mathbb{R}^d \) is the \( d \)-dimensional word embedding vector of token \( w_i \) without positional information.

The positional information is incorporated into \( q_m \), \( k_n \), and \( v_n \) for the \( m \)-th and \( n \)-th positions through functions \( f_q \), \( f_k \), and \( f_v \), respectively:

\[
q_m = f_q(x_m, m), \quad k_n = f_k(x_n, n), \quad v_n = f_v(x_n, n)
\]

The query and key values are then used to compute the attention weights, and the output is computed as the weighted sum over the value representation:

\[
a_{m,n} = \frac{\exp\left(\frac{q_m^T k_n}{\sqrt{d}}\right)}{\sum_{j=1}^{N} \exp\left(\frac{q_m^T k_j}{\sqrt{d}}\right)}, \quad o_m = \sum_{n=1}^{N} a_{m,n} v_n
\]

Existing transformer-based positional encoding methods primarily focus on selecting a suitable function to form the equation above.

---

### Absolute Positional Encoding (APE)

There are typically two forms of Absolute Positional Encoding (APE):

1. **Trainable Position Vectors** \( \boldsymbol{p}_{i} \in\left\{\boldsymbol{p}_{t}\right\}_{t=1}^{L} \):
    \[
    f_{t: t \in\{q, k, v\}}\left(\boldsymbol{x}_{i}, i\right):=\boldsymbol{W}_{t: t \in\{q, k, v\}}\left(\boldsymbol{x}_{i}+\boldsymbol{p}_{i}\right) 
    \]
   where \( L \) is the maximum sequence length. This form is used in the GPT-2 model, which means there won’t be a position embedding for unseen input lengths after the maximum length during training.

2. **Sinusoidal Positional Encoding**:
    \[
    \begin{cases} 
    \boldsymbol{p}_{i, 2 t} & =\sin \left(k / 10000^{2 t / d}\right) \\
    \boldsymbol{p}_{i, 2 t+1} & =\cos \left(k / 10000^{2 t / d}\right) 
    \end{cases}
    \]

---

### Relative Positional Encoding (RePE)

Relative Positional Encoding (RePE) was first proposed by Shaw et al. in 2018, and a simplified version is used in the T5 model:

- **Shaw's Version**:
    \[
    \begin{array}{r}
    f_{q}\left(\boldsymbol{x}_{m}\right):=\boldsymbol{W}_{q} \boldsymbol{x}_{m} \\
    f_{k}\left(\boldsymbol{x}_{n}, n\right):=\boldsymbol{W}_{k}\left(\boldsymbol{x}_{n}+\tilde{\boldsymbol{p}}_{r}^{k}\right)\\
    f_{v}\left(\boldsymbol{x}_{n}, n\right):=\boldsymbol{W}_{v}\left(\boldsymbol{x}_{n}+\tilde{\boldsymbol{p}}_{r}^{v}\right)
    \end{array}
    \]
   where \( \tilde{\boldsymbol{p}}_{r}^{k}, \tilde{\boldsymbol{p}}_{r}^{v} \in \mathbb{R}^{d} \) are trainable relative position embeddings, with \( r = \operatorname{clip}\left(m-n, r_{\min }, r_{\max }\right) \) representing the relative distance between positions \( m \) and \( n \).

- **T5 Simplified Version**: 
    The relative position encoding is applied directly as a scalar bias to the logits of the attention mechanism. The attention logit for token \( m \) attending to token \( n \) is computed as:
    \[
    \text{Logit}(m, n) = \frac{\boldsymbol{q}_m \cdot \boldsymbol{k}_n}{\sqrt{d}} + b_r
    \]

---

### Rotary Positional Encoding (RoPE)

Instead of adding positional embedding vectors, RoPE applies a rotation to the representation vectors before computing their dot products, integrating both absolute and relative positional information.

#### 2D Form

In a 2-dimensional space, RoPE applies the following transformation:

\[
f_{\{q, k\}}\left(\boldsymbol{x}_{m}, m\right) = \left(\begin{array}{cc}
\cos m \theta & -\sin m \theta \\
\sin m \theta & \cos m \theta
\end{array}\right) \boldsymbol{W}_{\{q, k\}} \boldsymbol{x}_m
\]

where:
- \( \boldsymbol{W}_{\{q, k\}} \) are the weight matrices for queries and keys.
- \( m \) is the current position index.
- \( \theta \) is a constant that determines the rate of rotation per position step, encoding the absolute position.

The attention mechanism then calculates the dot product between these rotated embeddings.

#### General Form

For higher dimensions (where the embedding dimension \( d \) is even), RoPE generalizes this 2D rotation across all pairs of dimensions in the embedding space by dividing it into \( d/2 \) sub-blocks. Each sub-block undergoes a 2D rotation. The generalized form of RoPE is:

\[
f_{\{q, k\}}\left(\boldsymbol{x}_{m}, m\right) = \boldsymbol{R}_{\Theta, m}^{d} \boldsymbol{W}_{\{q, k\}} \boldsymbol{x}_m
\]

where \( \boldsymbol{R}_{\Theta, m}^{d} \) is the block-diagonal rotation matrix for position \( m \), constructed as:

\[
\boldsymbol{R}_{\Theta, m}^{d} = \left(\begin{array}{cccc}
\boldsymbol{R}_{\theta_1, m} & 0 & \cdots & 0 \\
0 & \boldsymbol{R}_{\theta_2, m} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \boldsymbol{R}_{\theta_{d/2}, m}
\end{array}\right)
\]

Each block \( \boldsymbol{R}_{\theta_i, m} \) is a 2D rotation matrix:

\[
\boldsymbol{R}_{\theta_i, m} = \left(\begin{array}{cc}
\cos m \theta_i & -\sin m \theta_i \\
\sin m \theta_i & \cos m \theta_i
\end{array}\right)
\]

with \( \theta_i = 10000^{-2(i-1)/d} \) representing the rotation angles for each sub-block, encoding positions across different dimensions.

---

This blog post provides a high-level overview of the primary positional encoding methods in transformers, including Absolute Positional Encoding, Relative Positional Encoding, and Rotary Positional Encoding, each with unique mechanisms for encoding positional information in transformer architectures.
