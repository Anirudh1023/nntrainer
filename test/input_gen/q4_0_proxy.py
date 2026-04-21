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

def record_single_q4_0(layer, input_shape, test_name, call_args=None, input_type='int'):
    """ Custom recorder wrapper fixing Q4 precision """
    if call_args is None:
        call_args = {}
        
    from recorder import attach_trans_layer, _rand_like, _get_writer
    import tensorflow as tf
    
    # 1. We must mock the layer build securely via the standard layer translation struct
    layer = attach_trans_layer(layer)
    layer.build(input_shape)
    
    # 2. Extract FP32 weights, crush them through the GGML Q4_0 mathematical constraint, and precisely inject them back!
    # Target the inner Keras layer directly because the TransLayer wrapper doesn't support set_weights.
    weights = layer.tf_layer.get_weights()
    if len(weights) > 0:
        weights[0] = constrain_to_q4_0(weights[0])
        layer.tf_layer.set_weights(weights)
        
    # 3. Simulate identical record execution organically bypassing standard wrappers
    if isinstance(input_shape, list):
        inputs = [_rand_like(in_shape, 1, input_type) for in_shape in input_shape]
    else:
        inputs = _rand_like(input_shape, 1, input_type)

    initial_weights = [tf.Variable(i) for i in layer.weights]

    for _ in range(4):
        layer.call(inputs, **call_args) # warm layer multiple times

    with tf.GradientTape(persistent=True) as tape:
        if isinstance(inputs, list):
            list([tape.watch(inp) for inp in inputs])
        else:
            tape.watch(inputs)
        outputs = layer.call(inputs, **call_args)
        dy_constant = outputs * 2  # set incoming derivative to 2 instead of 1

    weights_out = layer.weights.copy()
    # Execute gradient explicitly on the base Keras Variables directly! 
    # (Bypasses TF 2.16 Keras 3 `EagerTensor` missing `.trainable` attribute crash in wrapper `.trainable_weights`)
    gradients = tape.gradient(dy_constant, layer.tf_layer.trainable_weights)
    derivatives = tape.gradient(dy_constant, inputs)

    try:
        gradients = layer.to_nntr_trainable_weights(gradients)


    def pad_layer_tensors(tensors):
        out = []
        for tensor in tensors:
            v_np = tensor.numpy() if hasattr(tensor, 'numpy') else np.array(tensor)
            if len(v_np.shape) == 4:
                out_c, in_c, h, w_dim = v_np.shape
                N = out_c
                K = h * w_dim * in_c
                padded_K = ((K + 31) // 32) * 32
                padded_N = ((N + 31) // 32) * 32
                if padded_K != K or padded_N != N:
                    v_flat = np.reshape(v_np, (N, K))
                    v_padded = np.pad(v_flat, ((0, padded_N - N), (0, padded_K - K)), mode='constant')
                    # GGML naturally crushes memory bounds recursively iterating col-wise across K. 
                    # NNTrainer's TensorDim reads sequentially contiguously over width natively.
                    v_out = v_padded.T
                    out.append(tf.convert_to_tensor(v_out))
                    continue
            out.append(tensor)
        return out

    initial_weights = pad_layer_tensors(initial_weights)
    weights_out = pad_layer_tensors(weights_out)
    gradients = pad_layer_tensors(gradients)

    with open(test_name + ".nnlayergolden", "wb") as f:
        writer = _get_writer(f)

        def write_tensor(tensors):
            if not isinstance(tensors, list):
                tensors = [tensors]
            for tensor in tensors:
                writer(tf.size(tensor), tensor)

        write_tensor(initial_weights)
        write_tensor(inputs)
        write_tensor(outputs)
        write_tensor(gradients)
        write_tensor(weights_out)
        write_tensor(derivatives)
