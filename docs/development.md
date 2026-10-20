# Development: build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # GRACE_ARCH_FLAGS defaults to -march=skylake-avx512 -mprefer-vector-width=512
cmake --build build -j 8 && (cd build && ctest --output-on-failure)
pip install -e . && pytest python/tests
scripts/build_wheels.sh dist/wheelhouse          # manylinux2014 wheels without Docker (micromamba, conda-forge gcc)
scripts/test_wheels.sh dist/wheelhouse           # each wheel in clean venvs, CPython 3.10 - 3.13
```

GitHub Actions build the wheels (Actions tab, "Run workflow") and keep them as artifacts of the run:
- `CPU wheels` (`.github/workflows/cpu-wheels.yml`): the libgrace C++ tests, the `grace-runtime` sdist and wheels,
  then `scripts/test_wheels.sh` with each wheel;
- `CUDA wheels` (`.github/workflows/cuda-wheels.yml`): Kokkos per GPU architecture, the `grace-runtime-cuda12` sdist
  and wheels, then `scripts/check_cuda_wheel.py` and a check on a machine without GPU (`device="auto"` runs on the CPU).
  The runners have no GPU; the GPU tests are `scripts/test_cuda_wheels.sh`, run on a machine with a GPU.

The tests use registry models from the cache: GRACE-2L-SMAX-OMAT-medium, GRACE-1L-OMAT-large-base,
GRACE-2L-OMAT-medium-ft-E, GRACE-3L-OMAT-large and GRACE-2L-SMAX-OMAT-large. The C++ tests skip when a model is
missing; `grace-runtime models download` fetches them. GRACE-2L-SMAX-OMAT-large is checked against TensorFlow fp64
references (`tests/fixtures/ref_tf`), on CPU (`python/tests/test_ref_tf.py`) and GPU
(`python/tests/test_cuda_parity.py`). Paths can be overridden:
- C++ and Python tests: `GRACE_TEST_M1` / `GRACE_TEST_M2` / `GRACE_TEST_M3` / `GRACE_TEST_UQ1` / `GRACE_TEST_UQ2`;
- Python tests: `GRACE_TEST_M2L` (GRACE-2L-SMAX-OMAT-large);
- C++ tests: `GRACE_TEST_NOUQ`, a model without a UQ head;
- `GRACE_TEST_OLD_UQ`: a model with a UQ schema other than 6 (optional).

Other test files:
- `python/tests/test_vs_tp.py` (parity with TPCalculator) needs tensorpotential and the `-fp64` SavedModels (GRACE-3L:
  its fp32 SavedModel); run it on CPU.
- `python/tests/test_runtimes.py` checks that grace-runtime shares a process with scikit-learn, an MKL numpy or
  TensorFlow (OpenMP runtimes).

## A CUDA wheel for other GPU architectures

The CUDA backend is a separate package, `grace-runtime-cuda12`; it requires the matching version of `grace-runtime`.
Building a wheel creates a file; installing it makes the backend available in your Python environment.

The [CUDA wheel script](../scripts/build_cuda_wheels.sh) requires `micromamba` on `PATH`, network access, and Linux
x86-64. It creates build environments with CUDA 12.9 and the host compiler, builds the pinned Kokkos release, and
repairs the wheels for manylinux2014. You do not need a system CUDA toolkit for this workflow.

For an RTX 3060, build `sm86` (Ampere):

```bash
# From the repository root, with your target Python environment activated:
export PYTHONS="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export GRACE_KK_ARCHS='sm86'
export GRACE_KOKKOS_CACHE="$PWD/.kokkos-wheel"
export NPROC=8

bash scripts/build_cuda_wheels.sh "$PWD/dist/cuda-wheelhouse"

# Main package first, then the locally built CUDA backend:
python -m pip install .
python -m pip install --no-index --find-links="$PWD/dist/cuda-wheelhouse" grace-runtime-cuda12
```

Skip the main-package installation if the matching version is already installed. Pip selects the CUDA wheel
compatible with the active Python. Use a fresh output directory when switching builds to avoid selecting an older
wheel of the same version. Lower `NPROC` if compilation exhausts host memory.

Other supported build targets are `sm70`, `sm75`, `sm80`, `sm86`, `sm89`, `sm90`, `sm100`, and `sm120`. For the default multi-architecture
wheel, use `export GRACE_KK_ARCHS='sm70;sm80;sm90;sm100;sm120'`. The loader also supports compatible architectures within
these GPU generations; see the [CUDA package README](../python-cuda/README.md).

The installed wheel needs an NVIDIA driver compatible with CUDA 12; it does not need a CUDA toolkit. Optional
cuBLAS acceleration for UQ projections: the `cublas` extra (see [Optional dependencies](../README.md#optional-dependencies)).
