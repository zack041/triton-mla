# Triton MLA Decode Kernel

> Dedicated Triton MLA decode kernel (`mla.py`).
>
> High performance in long context/large batch, under the constraint of fixed kv splits (essential for CUDA graph).

---

### Benchmark

Hardware: **NVIDIA H100**

**DeepSeek V2 Lite**
![DeepSeek V2 Lite benchmark](benchmark/deepseek%20v2%20lite.png)

**Standard DeepSeek**
![Standard benchmark](benchmark/standard%20deepseek.png)

---
*Note: currently only tuned for h100.*
