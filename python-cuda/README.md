# grace-runtime-cuda12

NVIDIA CUDA 12 backend for [grace-runtime](https://github.com/ICAMS/grace-runtime), using the shared GRACE-1L/2L/3L kernels.

**Current PyPI packages are placeholders.** Once runtime wheels are published:

```bash
pip install "grace-runtime-cuda12[cublas]"
```

The matching CPU package is installed automatically. Requires Linux x86-64 and a compatible NVIDIA driver.
See the [GPU manual](https://github.com/ICAMS/grace-runtime/blob/master/docs/backends.md#nvidia-cuda)
for supported cards, driver requirements, source builds and the `[cublas]` extra (GRACE-3L ~1.4x faster, UQ slightly faster).
