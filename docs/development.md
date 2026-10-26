# Development: build and test

[Documentation map](../README.md#guides) · [Installation](installation.md)

## CPU build and tests

Run from the repository root. Use a fresh environment and choose the target CPU explicitly; the top-level CMake
project otherwise defaults to AVX-512 flags, which are unsuitable for many CPUs. `-march=native` is for the build
machine; for portable Linux x86-64-v3 binaries use `-march=x86-64-v3` with a compiler that supports it.

```bash
micromamba create -n grace-runtime-dev -c conda-forge python=3.12 pip cxx-compiler cmake
micromamba activate grace-runtime-dev
python -m pip install ".[test]" --config-settings=cmake.define.GRACE_RUNTIME_ISAS=native
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGRACE_ARCH_FLAGS=-march=native
cmake --build build -j 4
(cd build && ctest --output-on-failure)
python -m pytest python/tests
```

For an editable installation, replace `".[test]"` with `-e ".[test]"`. Some tests need optional packages or GPU
hardware and skip when unavailable; report passed, skipped and failed counts. The Python suite normally keeps
`device="auto"` on the CPU; GPU tests explicitly request their GPU backend. Run those tests only on a machine
with the corresponding backend built and a usable GPU.

Known test limitation: on an AVX-512 machine, `test_dispatch.py::test_v3_and_v4_agree` requires both ISA modules
even after a native-only installation. To run that comparison, build the default dual-ISA Python package with
`python -m pip install ".[test]"` and a compiler supporting `x86-64-v3`/`x86-64-v4`. A missing-module failure in
that test does not test the accuracy of the native module; do not report it as a successful ISA comparison.

### Test models

Pre-download the five registry models before testing, especially on machines without network access:

```bash
grace-runtime models download GRACE-2L-SMAX-OMAT-medium GRACE-1L-OMAT-large-base \
    GRACE-2L-OMAT-medium-ft-E GRACE-3L-OMAT-large GRACE-2L-SMAX-OMAT-large
```

C++ tests skip when their models are missing (exit 77); Python tests may resolve and download models through the
registry. A successful run with skipped model tests is not model validation. GRACE-2L-SMAX-OMAT-large is checked
against frozen TensorFlow fp64 references (`tests/fixtures/ref_tf`) on CPU (`python/tests/test_ref_tf.py`) and GPU
(`python/tests/test_gpu_parity.py`), without needing TensorFlow for those reference tests.

Overrides:

| Variable | Use |
| --- | --- |
| `GRACE_TEST_M1`, `GRACE_TEST_M2`, `GRACE_TEST_M3` | 1L, 2L, 3L model files |
| `GRACE_TEST_UQ1`, `GRACE_TEST_UQ2` | UQ model files |
| `GRACE_TEST_M2L` | Python large-2L model file |
| `GRACE_TEST_NOUQ` | Model without a UQ head, for C++ tests |
| `GRACE_TEST_OLD_UQ` | Model with a UQ schema other than 6 (optional) |

`python/tests/test_vs_tp.py` additionally needs tensorpotential and its fp64 SavedModels (3L: its fp32 SavedModel).
`python/tests/test_runtimes.py` checks OpenMP coexistence with optional scikit-learn, MKL numpy or TensorFlow.

## Release wheel builds

For maintainers building Linux x86-64 CPU wheels, with `micromamba` and network access:

```bash
bash scripts/build_wheels.sh dist/wheelhouse
bash scripts/test_wheels.sh dist/wheelhouse
```

The scripts create the compiler/Python environments, build manylinux2014 wheels without Docker, and test wheels
in clean environments. The CPU wheel targets CPython 3.10, 3.11 and an abi3 wheel for >= 3.12.

GitHub Actions workflows build wheels and retain run artifacts:

- `CPU wheels`: CPU library tests, the source distribution and CPU wheels, then wheel tests.
- `CUDA wheels`: Kokkos per architecture, the CUDA package, wheel-content checks and CPU fallback checks on runners
  without GPUs. These checks do not establish GPU correctness; run `scripts/test_cuda_wheels.sh` on a GPU machine.

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
export NPROC=8

bash scripts/build_cuda_wheels.sh "$PWD/dist/cuda-wheelhouse"

# Main package first, then the locally built CUDA backend:
python -m pip install . --config-settings=cmake.define.GRACE_RUNTIME_ISAS=native
python -m pip install --no-index --find-links="$PWD/dist/cuda-wheelhouse" grace-runtime-cuda12
```

Skip the main-package installation if the matching version is already installed. Pip selects the CUDA wheel
compatible with the active Python. Use a fresh output directory when switching builds to avoid selecting an older
wheel of the same version. Lower `NPROC` if compilation exhausts host memory.

Other supported build targets are `sm70`, `sm75`, `sm80`, `sm86`, `sm89`, `sm90`, `sm100`, and `sm120`. For the default multi-architecture
wheel, use `export GRACE_KK_ARCHS='sm70;sm80;sm90;sm100;sm120'`. The loader also supports compatible architectures within
these GPU generations; see the [CUDA package README](../python-cuda/README.md).

The installed wheel needs an NVIDIA driver compatible with CUDA 12; it does not need a CUDA toolkit. Optional
cuBLAS acceleration for UQ projections: the `cublas` extra (see [optional dependencies](installation.md#optional-dependencies)).

## GPU engine tests and tuning

Kernels whose launch bounds do not fit the GPU (registers and shared memory of several teams per SM, e.g. the GRACE-1L
bond backward on RTX 20xx / T4) launch without them; `GRACE_LB_FALLBACK=1` takes that fallback on any NVIDIA GPU
(debugging; the CUDA tests `*_lb_fallback` use it).

Kokkos with HIP for your GPU (here MI250X; MI300X: `Kokkos_ARCH_AMD_GFX942`, MI300A: `Kokkos_ARCH_AMD_GFX942_APU`,
RX 7900: `Kokkos_ARCH_AMD_GFX1100`), then the engine tests of this repository against it:

```bash
cmake -S kokkos-5.2.2 -B kokkos-build -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_CXX_STANDARD=20 \
      -DKokkos_ENABLE_HIP=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_AMD_GFX90A=ON \
      -DCMAKE_INSTALL_PREFIX="$PWD/kokkos-hip"
cmake --build kokkos-build -j 16 && cmake --install kokkos-build
cmake -S . -B build-hip -DCMAKE_CXX_COMPILER=hipcc -DGRACE_WITH_KOKKOS=ON -DGRACE_ARCH_FLAGS=-march=native -DKokkos_ROOT="$PWD/kokkos-hip"
cmake --build build-hip -j 16 && (cd build-hip && ctest)
```

- Floating-point atomics: `-DGRACE_HIP_UNSAFE_FP_ATOMICS=AUTO|ON|OFF` (default `AUTO`) compiles the engines with
  `-munsafe-fp-atomics`, the hardware atomics (5-16 % faster on MI250X). They are only correct on device memory
  from `hipMalloc`, so `AUTO` turns them on for the discrete CDNA GPUs (gfx908, gfx90a, gfx940-942) and off for the
  MI300A APU and RDNA. A LAMMPS build for a discrete CDNA GPU gets the same by `-DCMAKE_CXX_FLAGS=-munsafe-fp-atomics`
  (never on MI300A).
- GEMMs (GRACE-3L): rocBLAS of the system when present (`librocblas.so.5` or `.so.4`, loaded at run time, never
  linked or shipped), otherwise built-in GEMM kernels (on CDNA on the matrix cores). `GRACE_RUNTIME_NO_ROCBLAS=1`
  selects the built-in ones.
- Matrix cores (GRACE-3L): on gfx90a the product basis runs on the fp32 matrix cores (MFMA; 1.24-1.32x faster on
  MI250X, about 0.5 GB more GPU memory for 31k atoms); `GRACE3L_SPBF_MFMA=0` turns it off, on gfx94x (MI300) it is
  opt-in (`GRACE3L_SPBF_MFMA=1`) until tested there. A line on stderr says which kernels run.
- Matrix cores (GRACE-1L, -2L): on gfx90a the bond kernels of the basis and of layer 2 and the dense FC layers run on
  the fp32 matrix cores (MFMA); `GRACE_FUSED_MFMA=0` turns them off, on gfx94x (MI300) they are opt-in
  (`GRACE_FUSED_MFMA=1`) until tested there. A line on stderr says which kernels run.
- GRACE-1L, -2L layer-1 bond backward: one wavefront per atom on every HIP target; `GRACE_FUSED_WAVE_BWD=0` selects
  the previous kernels (one team of 256 per atom). On CUDA the same 64-thread kernel is the default for GRACE-2L only
  (`GRACE_FUSED_WAVE_BWD=1` / `=0` force it on / off for both).
- GRACE-2L layer 2 on GPUs: `GRACE_RED_BLK=0` turns off the blocked forward reducers of I1 / rho and I2 (another fp32
  summation order: per-atom energies differ by ~1e-5 eV). On HIP `GRACE_L2_ROWS=0` and `GRACE_L2_BWD_SHFL=0` select
  the previous layouts of the matrix-core bond kernels and the shared-memory force sums of the layer-2 backward.
