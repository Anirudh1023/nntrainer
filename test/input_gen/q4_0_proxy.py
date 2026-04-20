#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import numpy as np

def constrain_to_q4_0(weights_array):
    """
    Takes an FP32 numpy array for Conv2D (H, W, in_c, out_c)
    Mathematically simulates the GGML Q4_0 block quantization and dequantization,
    so the resulting output exactly matches the W4 execution loss natively!
    """
    orig_shape = weights_array.shape
    h, w, in_c, out_c = orig_shape
    
    K = h * w * in_c
    N = out_c
    
    padded_K = ((K + 31) // 32) * 32
    padded_N = ((N + 31) // 32) * 32
    
    k_2d = weights_array.reshape((K, N))
    
    # NNTrainer GGML transpose format (N, K)
    w_t = k_2d.T 
    w_padded = np.pad(w_t, ((0, padded_N - N), (0, padded_K - K)), mode='constant')
    
    for n in range(padded_N):
        for k_idx in range(0, padded_K, 32):
            block = w_padded[n, k_idx:k_idx + 32]
            
            # Find the actual signed value bearing the maximum absolute magnitude
            max_idx = np.argmax(np.abs(block))
            max_val = block[max_idx]
            
            d = max_val / -8.0
            if d == 0:
                w_padded[n, k_idx:k_idx + 32] = 0.0
                continue
                
            # GGML Q4_0 Quantization math (as mirrored in quantizer.cpp)
            x0 = block / d
            xi = np.minimum(15, np.floor(x0 + 8.5).astype(np.int8))
            
            # GGML Dequantization restitution
            w_padded[n, k_idx:k_idx + 32] = (xi - 8) * d
            
    w_unpadded = w_padded[:N, :K]
    w_restored = w_unpadded.T.reshape(orig_shape)
    
    return w_restored

def record_single_q4_0(layer, input_shape, test_name, input_type='int'):
    """ Custom recorder wrapper fixing Q4 precision """
    # 1. We must mock the layer build to initialize FP32 variables first
    # (assuming standard tensorflow.keras setup)
    layer.build(input_shape)
    
    # 2. Extract FP32 weights, crush them through the GGML Q4_0 mathematical constraint, and inject them back
    weights = layer.get_weights()
    if len(weights) > 0:
        weights[0] = constrain_to_q4_0(weights[0])
        layer.set_weights(weights)
        
    # 3. Call standard record_single
    from recorder import record_single
    record_single(layer, input_shape, test_name, input_type=input_type)
