## LoRA + Quantization — Summary

1. **QLoRA** — fine-tune a quantized (INT4) base model by attaching small FP16 LoRA adapters. Only A and B are trained, base model is frozen.

2. **Forward pass** — two parallel paths:
    - Base: INT4 weights → dequantize → FP16 matmul
    - LoRA: X → A → B → ΔY
    - Both outputs added together

3. **No integer matmul on base path** — because weights are dequantized to FP16 before matmul, and X is FP16. You get memory savings but not compute savings.

4. **If activations are also quantized** — base path can do INT8 matmul. But LoRA path still runs in FP16 because A and B are FP16.

5. **Backward pass** — gradients flow through both paths. LoRA A and B get updated. Base model gradients flow through but no weight update. Activations must be stored from all layers for backprop — even frozen ones.

6. **LoRA weights stay FP16 during training** — gradient updates are tiny (e.g. 0.00003). INT8 step size is too coarse to capture them.

7. **At inference — three options:**
    - Dequantize → merge → requantize back to INT8 (single matmul, slight accuracy loss)
    - Dequantize → merge → keep FP16 (accurate but loses quantization benefit)
    - Keep separate (QLoRA default — INT8 base + FP16 LoRA path)

8. **Merging and requantizing** uses fresh S and Z — the original ones are thrown away since the weight distribution shifts after adding ΔW. Error is small for low ranks (r=4,8) and grows with rank.