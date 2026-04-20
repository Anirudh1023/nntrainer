#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import warnings
import os
import sys

# Ensure recorder is imported successfully
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from recorder import record_single
from q4_0_proxy import record_single_q4_0

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import numpy as np
    import tensorflow.keras as K

if __name__ == "__main__":
    print("Generating isolated Conv2D nnlayergolden tests...")
    
    conv = K.layers.Conv2D(3, 2)
    record_single_q4_0(conv, (1, 1, 4, 4), "conv2d_sb_minimum_q4_0")
    record_single_q4_0(conv, (3, 1, 4, 4), "conv2d_mb_minimum_q4_0")

    conv = K.layers.Conv2D(2, 3, padding="same")
    record_single_q4_0(conv, (1, 1, 4, 4), "conv2d_sb_same_remain_q4_0")
    record_single_q4_0(conv, (3, 1, 4, 4), "conv2d_mb_same_remain_q4_0", input_type="float")

    conv = K.layers.Conv2D(2, 3, strides=2, padding="same")
    record_single_q4_0(conv, (1, 3, 4, 4), "conv2d_sb_same_uneven_remain_q4_0")
    record_single_q4_0(conv, (3, 3, 4, 4), "conv2d_mb_same_uneven_remain_q4_0")

    conv = K.layers.Conv2D(2, 3, strides=2, padding="valid")
    record_single_q4_0(conv, (1, 3, 7, 7), "conv2d_sb_valid_drop_last_q4_0")
    record_single_q4_0(conv, (3, 3, 7, 7), "conv2d_mb_valid_drop_last_q4_0")

    conv = K.layers.Conv2D(3, 2, strides=3)
    record_single_q4_0(conv, (1, 2, 5, 5), "conv2d_sb_no_overlap_q4_0")
    record_single_q4_0(conv, (3, 2, 5, 5), "conv2d_mb_no_overlap_q4_0")

    conv = K.layers.Conv2D(3, 1, strides=2)
    record_single_q4_0(conv, (1, 2, 5, 5), "conv2d_sb_1x1_kernel_q4_0")
    record_single_q4_0(conv, (3, 2, 5, 5), "conv2d_mb_1x1_kernel_q4_0")

    conv = K.layers.Conv2D(2, 3, dilation_rate=(2, 2))
    record_single_q4_0(conv, (1, 3, 11, 11), "conv2d_sb_dilation_q4_0")
    record_single_q4_0(conv, (3, 3, 11, 11), "conv2d_mb_dilation_q4_0")

    conv = K.layers.Conv2D(2, 3, padding="same", dilation_rate=(2, 2))
    record_single_q4_0(conv, (1, 3, 11, 11), "conv2d_sb_same_dilation_q4_0")
    record_single_q4_0(conv, (3, 3, 11, 11), "conv2d_mb_same_dilation_q4_0")
    
    print("Done generating subset!")
