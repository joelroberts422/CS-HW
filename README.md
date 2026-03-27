# CS HW 3 — GPU Compilation & Profiling

## Requirements

- Python 3.12
- NVIDIA GPU with CUDA 12.8+ (tested on RTX 4090)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Notebooks

### `jax_jit_analysis.ipynb`

Compares JAX eager vs. JIT execution of a trig-powers kernel on a 5000x5000 matrix. Profiles kernel launch counts, total CUDA execution time, and memory throughput using `torch.profiler` (via CUPTI).

Run all cells top-to-bottom. The first JIT call triggers XLA compilation and is excluded from timing.

### `torch_compile_analysis.ipynb`

Benchmarks a 3-layer `SimpleNN` across four `torch.compile` backends: `eager`, `aot_eager`, `inductor`, and `cudagraphs`. Measures forward and backward pass times with `torch.cuda.synchronize()` for accurate GPU timing. The first iteration is discarded to exclude compilation overhead.

Run all cells top-to-bottom. Cell 3 compiles the model under each backend; Cell 4 runs the timed loops; Cell 5 plots the results.
