---
layout: post
title: A Brief Summary on Positional Encoding Methods
date: 2024-08-22
description: A summary of positional encoding methods in transformers
tags: transformers positional-encoding
categories: Research NLP-Foundations
---

## Preliminary

Unlike recurrent architectures, transformers process inputs in parallel all at once. However, self-attention doesn't consider the position of a word in a sequence; thus, sentences like "I think therefore I am" and "I am therefore I think" are treated the same.

The notations are shown below:

- $$ S_N = \{ w_i \}_{i=1}^N $$: a sequence of $$ N $$ input tokens with $$ w_i $$ being the $$ i $$-th element.
- $$ E_N = \{ x_i \}_{i=1}^N $$: corresponding word embeddings of $$ S_N $$; $$ x_i \in \mathbb{R}^d $$ is the $$ d $$-dimensional word embedding vector of token $$ w_i $$ without position information.

$$ q_m $$, $$ k_n $$, and $$ v_n $$ incorporate the $$ m $$-th and $$ n $$-th positions through $$ f_q $$, $$ f_k $$, and $$ f_v $$, respectively.

$$
q_m = f_q(x_m, m), \quad
k_n = f_k(x_n, n), \quad
v_n = f_v(x_n, n)
\tag{1}
$$

The query and key values are then used to compute the attention weights, while the output is computed as the weighted sum over the value representations.

$$
a_{m,n} = \frac{\exp\left( \frac{ q_m^\top k_n }{ \sqrt{d} } \right)}{ \sum_{j=1}^{N} \exp\left( \frac{ q_m^\top k_j }{ \sqrt{d} } \right)}, \quad
o_m = \sum_{n=1}^{N} a_{m,n} v_n
\tag{2}
$$

The existing approaches of transformer-based positional encoding mainly focus on choosing a suitable function to form Equation (1).

---

## Absolute Positional Encoding (APE)

There are usually two forms of APE:

- **Trainable position vectors** $$ \boldsymbol{p}_{i} \in \{ \boldsymbol{p}_{t} \}_{t=1}^{L} $$, where $$ L $$ is the maximum sequence length.

  $$
  f_{t \in \{ q, k, v \} }\left( \boldsymbol{x}_{i}, i \right) := \boldsymbol{W}_{t \in \{ q, k, v \} } \left( \boldsymbol{x}_{i} + \boldsymbol{p}_{i} \right)
  \tag{3}
  $$

- **Generate $$ \boldsymbol{p}_{i} $$ using the sinusoidal function:**

  $$
  \begin{cases}
  \boldsymbol{p}_{i, 2t} = \sin \left( \frac{i}{10000^{2t / d}} \right), \\
  \boldsymbol{p}_{i, 2t+1} = \cos \left( \frac{i}{10000^{2t / d}} \right)
  \end{cases}
  \tag{4}
  $$

The first form is used in the GPT-2 model; thus, when facing unseen input lengths, there will be no position embeddings beyond the maximum length encountered during training.

---

## Relative Positional Encoding (RePE)

We introduce the original RePE proposed by Shaw et al. in 2018 and a simplified version used in the T5 model:

- **Shaw's Version:**

  $$
  \begin{aligned}
  f_{q}\left( \boldsymbol{x}_{m} \right) &= \boldsymbol{W}_{q} \boldsymbol{x}_{m}, \\
  f_{k}\left( \boldsymbol{x}_{n}, n \right) &= \boldsymbol{W}_{k} \left( \boldsymbol{x}_{n} + \tilde{\boldsymbol{p}}_{r}^{k} \right), \\
  f_{v}\left( \boldsymbol{x}_{n}, n \right) &= \boldsymbol{W}_{v} \left( \boldsymbol{x}_{n} + \tilde{\boldsymbol{p}}_{r}^{v} \right)
  \end{aligned}
  \tag{5}
  $$

  where $$ \tilde{\boldsymbol{p}}_{r}^{k}, \tilde{\boldsymbol{p}}_{r}^{v} \in \mathbb{R}^{d} $$ are trainable relative position embeddings. Note that $$ r = \operatorname{clip}\left( m - n, r_{\min}, r_{\max} \right) $$ represents the relative distance between positions $$ m $$ and $$ n $$.

  For example, if we set the maximum relative length to be 4, then a sequence of 5 words will have a total of 9 embeddings to be learned (1 embedding for the current word, 4 embeddings for the words to the left, and 4 embeddings for the words to the right of the current word).

- **Simplified Version Used in T5 Model:**

  The relative position encoding is applied directly as a scalar bias to the logits of the attention mechanism.

  Specifically, the scalar bias for each relative position $$ r = m - n $$ is added to the dot product of the query and key vectors before the softmax operation.

  The attention logit for token $$ m $$ attending to token $$ n $$ is computed as:

  $$
  \text{Logit}(m, n) = \frac{ \boldsymbol{q}_m \cdot \boldsymbol{k}_n }{ \sqrt{d} } + b_r
  $$

---

## Rotary Positional Encoding (RoPE)

Instead of simply adding positional embedding vectors, RoPE applies a rotation to the representation vectors before computing their dot products, integrating both absolute and relative position information.

### 2D Form

For a 2-dimensional space, the transformation can be described as:

$$
f_{\{ q, k \} }\left( \boldsymbol{x}_{m}, m \right) =
\begin{pmatrix}
\cos m \theta & -\sin m \theta \\
\sin m \theta & \cos m \theta
\end{pmatrix}
\boldsymbol{W}_{ \{ q, k \} } \boldsymbol{x}_m
\tag{6}
$$

where:

- $$ \boldsymbol{W}_{\{ q, k \} } $$ are the weight matrices for queries and keys.
- $$ m $$ is the current position index.
- $$ \theta $$ is a constant that determines the rate of rotation per position step, effectively encoding the absolute position.

The attention mechanism then calculates the dot product between these rotated embeddings.

### General Form

In higher dimensions (where the embedding dimension $$ d $$ is even), RoPE generalizes the 2D rotation across all pairs of dimensions in the embedding space.

This is achieved by dividing the $$ d $$-dimensional space into $$ d/2 $$ sub-blocks and applying a 2D rotation to each sub-block. The generalized form of RoPE is given by:

$$
f_{\{ q, k \} }\left( \boldsymbol{x}_{m}, m \right) = \boldsymbol{R}_{\Theta, m}^{d} \boldsymbol{W}_{ \{ q, k \} } \boldsymbol{x}_m
\tag{7}
$$

where $$ \boldsymbol{R}_{\Theta, m}^{d} $$ is the block-diagonal rotation matrix for position $$ m $$, constructed as:

$$
\boldsymbol{R}_{\Theta, m}^{d} =
\begin{pmatrix}
\boldsymbol{R}_{\theta_1, m} & 0 & \cdots & 0 \\
0 & \boldsymbol{R}_{\theta_2, m} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \boldsymbol{R}_{\theta_{d/2}, m}
\end{pmatrix}
$$

Each block $$ \boldsymbol{R}_{\theta_i, m} $$ is a 2D rotation matrix:

$$
\boldsymbol{R}_{\theta_i, m} =
\begin{pmatrix}
\cos m \theta_i & -\sin m \theta_i \\
\sin m \theta_i & \cos m \theta_i
\end{pmatrix}
$$

with $$ \theta_i = 10000^{-2(i-1)/d} $$ representing the rotation angles for each sub-block, ensuring a diverse encoding of positions across different dimensions.

---

By applying these positional encoding methods, transformers can effectively incorporate sequence order information into the model, enhancing their ability to understand and generate language with respect to word positions.
