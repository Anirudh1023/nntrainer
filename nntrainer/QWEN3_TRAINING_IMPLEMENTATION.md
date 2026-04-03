# Qwen3 Training Implementation Requirements

## Overview

This document lists all layers used by Qwen3 that require training implementation (backward pass).

## Implementation Requirements

| # | Layer | File | calcDerivative | calcGradient | Notes |
|---|-------|------|---------------|--------------|-------|
| 1 | SwiGLU | swiglu.cpp | ✅ YES | N/A | Activation function, no weights |
| 2 | RMSNorm | rms_norm.cpp | ✅ YES | N/A | Normalization layer, no weights |
| 3 | ReshapedRMSNorm | reshaped_rms_norm.cpp | ✅ YES | N/A | Qwen3-specific Q/K normalization, no weights |
| 4 | EmbeddingLayer | embedding_layer.cpp | N/A | ✅ YES | Token embeddings, discrete input |
| 5 | TieWordEmbedding | tie_word_embedding.cpp | N/A | ✅ YES | Shared embeddings, discrete input |
| 6 | LMHead | lm_head.cpp | ✅ YES | ✅ YES | Output projection to vocabulary |
| 7 | MHACore | mha_core.cpp | ✅ YES | N/A | Multi-head attention, no weights |

## Summary

- **Total layers needing implementation**: 7
- **Layers needing calcDerivative only**: 5 (SwiGLU, RMSNorm, ReshapedRMSNorm, LMHead, MHACore)
- **Layers needing calcGradient only**: 2 (EmbeddingLayer, TieWordEmbedding)
- **Layers needing both calcDerivative and calcGradient**: 1 (LMHead)

## Implementation Details

### calcDerivative Only (5 layers)
Compute gradient w.r.t. input activations to pass to previous layer.

### calcGradient Only (2 layers)
Compute gradient w.r.t. learnable weights for optimizer updates.
- EmbeddingLayer and TieWordEmbedding: No calcDerivative needed because input is discrete token IDs.

### Both Methods (1 layer)
LMHead: Has both learnable weights and continuous input gradient flow.

## Current Status

| Layer | Current calcDerivative | Current calcGradient |
|-------|----------------------|---------------------|
| SwiGLU | Empty implementation | N/A |
| RMSNorm | Throws exception | N/A |
| ReshapedRMSNorm | Throws exception | N/A |
| EmbeddingLayer | Throws exception | Empty implementation |
| TieWordEmbedding | Throws exception | Empty implementation |
| LMHead | Throws exception | Throws exception |
| MHACore | Empty implementation | Empty implementation |

## Layer Usage in Qwen3

| Layer | Usage Frequency | Where Used |
|-------|----------------|------------|
| ReshapedRMSNorm | 2 × NUM_LAYERS | Q and K normalization in each decoder block |
| RMSNorm | 3 × NUM_LAYERS + 1 | Pre-attention, pre-MLP (each block), final normalization |
| MHACore | NUM_LAYERS | Attention mechanism in each decoder block |
| SwiGLU | NUM_LAYERS | MLP activation in each decoder block |
| EmbeddingLayer | 1 | Token embeddings at model input (if not tied) |
| TieWordEmbedding | 0 or 2 | Shared embeddings at input and output (if tied) |
| LMHead | 1 | Output projection (if embeddings not tied) |

## Method Explanations

### calcDerivative
Computes the gradient of the loss with respect to the layer's input (dL/dx).
- Used for backpropagation to previous layer
- Required for all layers with continuous inputs/outputs
- Not needed for layers with discrete inputs (token IDs)

### calcGradient
Computes the gradient of the loss with respect to the layer's learnable weights (dL/dW).
- Used for optimizer to update weights
- Only needed for layers with learnable parameters
- Not needed for layers without weights (activations, normalizations)

## Implementation Notes

### Discrete Input Layers (EmbeddingLayer, TieWordEmbedding)
- No calcDerivative needed because input is discrete token IDs
- calcGradient accumulates gradients for each token position in the embedding matrix

### Activation/Normalization Layers (SwiGLU, RMSNorm, ReshapedRMSNorm)
- Only calcDerivative needed
- No learnable weights, so no calcGradient needed

### Projection Layers (LMHead)
- Both calcDerivative and calcGradient needed
- Has learnable weight matrix
- Receives continuous input gradients from previous layer

### Attention Layer (MHACore)
- Only calcDerivative needed
- Most complex implementation
- Computes gradients for Q, K, V to pass to FullyConnected layers
- No learnable weights in MHACore itself

## Layer Stack (Per Qwen3 Decoder Block)

```
Input (Built-in)
  ↓
EmbeddingLayer OR TieWordEmbedding
  ↓
[Decoder Block × NUM_LAYERS]
  ├─ RMSNorm
  ├─ FullyConnected (Q) → ReshapedRMSNorm
  ├─ FullyConnected (K) → ReshapedRMSNorm
  ├─ FullyConnected (V)
  ├─ MHACore
  ├─ FullyConnected (O)
  ├─ Addition (Residual)
  ├─ RMSNorm
  ├─ FullyConnected (Up)
  ├─ FullyConnected (Gate)
  ├─ SwiGLU
  ├─ FullyConnected (Down)
  └─ Addition (Residual)
  ↓
RMSNorm (Final)
  ↓
LMHead OR TieWordEmbedding
  ↓
Output (Logits)
```

## Built-in Layers (Already Implemented)

- Input
- FullyConnected
- Addition

These layers already have complete backward pass implementations in NNTrainer core.
