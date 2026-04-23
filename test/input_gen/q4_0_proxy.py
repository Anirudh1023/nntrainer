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
            
            max_idx = np.argmax(np.abs(block))
            max_val = block[max_idx]
            
            d = max_val / -8.0
            if d == 0:
                w_padded[n, k_idx:k_idx + 32] = 0.0
                continue
                
            x0 = block / d
            xi = np.minimum(15, np.floor(x0 + 8.5).astype(np.int8))
            w_padded[n, k_idx:k_idx + 32] = (xi - 8) * d
            
    w_unpadded = w_padded[:N, :K]
    w_restored = w_unpadded.T.reshape(orig_shape)
    
    return w_restored

def _pad_weight_for_q4(weight_np, filter_size, in_channels, kH, kW):
    """
    Takes an NNTR-format weight (out_c, in_c, kH, kW) and produces
    a padded+transposed FP32 tensor of shape (padded_K, padded_N)
    ready for direct consumption by GgmlQuantizer::quantize().
    
    The quantizer expects input (H=padded_K, W=padded_N), transposes it
    internally to (padded_N, padded_K), then quantizes row-by-row.
    So we store the data pre-transposed: padded[k][n] = weight[n][k].
    """
    N = filter_size
    K = in_channels * kH * kW
    padded_N = ((N + 31) // 32) * 32
    padded_K = ((K + 31) // 32) * 32
    
    # Flatten to (N, K) matrix — each row is one filter
    w_2d = weight_np.reshape(N, K)
    
    # Create padded output (padded_K, padded_N) — transposed layout
    padded = np.zeros((padded_K, padded_N), dtype=np.float32)
    for n in range(N):
        for k in range(K):
            padded[k][n] = w_2d[n][k]
    
    return padded

def record_single_q4_0(layer, input_shape, test_name, call_args=None, input_type='int'):
    """ Custom recorder that applies Q4_0 quantization loss to weights,
        then records golden data. Weight tensors are padded+transposed
        to match the layout expected by GgmlQuantizer::quantize(). """
    if call_args is None:
        call_args = {}
        
    from recorder import attach_trans_layer, _rand_like, _get_writer
    import tensorflow as tf
    
    layer = attach_trans_layer(layer)
    layer.build(input_shape)
    
    # Get Conv2D properties for padding computation
    tf_layer = layer.tf_layer
    filters = tf_layer.filters
    kernel_h, kernel_w = tf_layer.kernel_size
    # input_shape is (batch, channels, H, W) in NNTR format
    if isinstance(input_shape, (list, tuple)):
        in_channels = input_shape[1] if not isinstance(input_shape[0], (list, tuple)) else input_shape[0][1]
    else:
        in_channels = input_shape[1]
    
    # Apply Q4_0 quantization loss to weights
    weights = tf_layer.get_weights()
    if len(weights) > 0:
        weights[0] = constrain_to_q4_0(weights[0])
        tf_layer.set_weights(weights)
        
    # Generate inputs
    if isinstance(input_shape, list):
        inputs = [_rand_like(in_shape, 1, input_type) for in_shape in input_shape]
    else:
        inputs = _rand_like(input_shape, 1, input_type)

    initial_weights = [tf.Variable(i) for i in layer.weights]

    for _ in range(4):
        layer.call(inputs, **call_args)

    with tf.GradientTape(persistent=True) as tape:
        if isinstance(inputs, list):
            list([tape.watch(inp) for inp in inputs])
        else:
            tape.watch(inputs)
        outputs = layer.call(inputs, **call_args)
        dy_constant = outputs * 2

    weights_out = layer.weights.copy()
    gradients = tape.gradient(dy_constant, tf_layer.trainable_weights)
    derivatives = tape.gradient(dy_constant, inputs)

    try:
        gradients = layer.to_nntr_trainable_weights(gradients)
    except AttributeError:
        pass

    # Pad weight tensors (kernel only, skip bias) for Q4_0 alignment.
    # The C++ sizeCheckedReadTensor will read these directly and quantize.
    def pad_weights_list(tensor_list):
        out = []
        for tensor in tensor_list:
            arr = tensor.numpy() if hasattr(tensor, 'numpy') else np.array(tensor)
            if len(arr.shape) == 4 and arr.shape[0] == filters:
                # This is the kernel tensor in NNTR format (out_c, in_c, kH, kW)
                padded = _pad_weight_for_q4(arr, filters, in_channels, kernel_h, kernel_w)
                out.append(tf.constant(padded))
            else:
                out.append(tensor)
        return out

    initial_weights = pad_weights_list(initial_weights)
    weights_out = pad_weights_list(weights_out)
    gradients = pad_weights_list(gradients)

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
