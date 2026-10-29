# Development: build and test

[Documentation map](index.md#guides) · [Installation](installation.md)

## CPU build and tests

From the repository root, create a development environment and build for the current CPU:

```bash
micromamba create -n grace-runtime-dev -c conda-forge python=3.12 pip cxx-compiler cmake
micromamba activate grace-runtime-dev
pip install ".[test]"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build --output-on-failure
pytest python/tests
```

On macOS, install [OpenMP](installation.md#macos) first. For an editable install, use `pip install -e ".[test]"`.
The C++ project builds for the machine it runs on (`-march=native`), so a build directory runs only on CPUs like the
one that built it. Set `GRACE_ARCH_FLAGS` for another target, for example `-DGRACE_ARCH_FLAGS=-march=x86-64-v3`, or
`"-DGRACE_ARCH_FLAGS=-march=native;-mprefer-vector-width=512"` for 512-bit vectors on AVX-512 CPUs. An existing build
directory keeps its flags (configure prints them); `cmake -U GRACE_ARCH_FLAGS build` takes the default again.
For a native-only Python build, add `--config-settings=cmake.define.GRACE_RUNTIME_ISAS=native` to `pip install`.
The Linux v3/v4 comparison test needs both ISA modules.

### Test models

Download test models before running offline:

```bash
grace-runtime models download GRACE-2L-SMAX-OMAT-medium GRACE-1L-OMAT-large-base \
    GRACE-2L-OMAT-medium-ft-E GRACE-3L-OMAT-large GRACE-2L-SMAX-OMAT-large
```

`GRACE_TEST_M1`, `GRACE_TEST_M2` and `GRACE_TEST_M3` select local 1L, 2L and 3L files.
C++ tests skip missing models; Python tests may download them. GPU tests need a built backend and usable GPU.
Report passed, skipped and failed counts: skipped tests do not validate a backend.

Optional tests compare against tensorpotential or check coexistence with other OpenMP runtimes.
These need the corresponding packages and models.

## Documentation site

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Run `mkdocs build --strict` to check the site before publishing. The API page reads Python syntax trees and the
native Model constructor binding; it never imports the runtime. Edit public docstrings to update its descriptions.

## Release wheel builds

On Linux x86-64 or macOS arm64, with micromamba and network access (the wheels of that platform):

```bash
bash scripts/build_wheels.sh dist/wheelhouse
bash scripts/test_wheels.sh dist/wheelhouse
```

The scripts build portable CPU wheels and test them in clean environments.
GPU wheel-content and CPU-fallback checks do not validate GPU execution; run
`scripts/test_cuda_wheels.sh` on an NVIDIA GPU machine and `scripts/test_rocm_wheels.sh` on an AMD GPU machine.

## GPU backend builds

The [GPU manual](backends.md) is the source for supported hardware, runtime requirements and Python backend builds.
The engine tests below are for kernel development.

## GPU engine tests and tuning

Build Kokkos for your GPU, then configure the engine tests against it. For example, with HIP on MI250X:

```bash
cmake -S kokkos-5.2.2 -B kokkos-build -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_CXX_STANDARD=20 \
    -DKokkos_ENABLE_HIP=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_AMD_GFX90A=ON \
    -DCMAKE_INSTALL_PREFIX="$PWD/kokkos-hip"
cmake --build kokkos-build -j 4
cmake --install kokkos-build
cmake -S . -B build-hip -DCMAKE_CXX_COMPILER=hipcc -DGRACE_WITH_KOKKOS=ON -DKokkos_ROOT="$PWD/kokkos-hip"
cmake --build build-hip -j 4
ctest --test-dir build-hip --output-on-failure
```

Keep automatic kernel selection for normal use. Useful debugging overrides:

| Setting | Effect |
| --- | --- |
| `GRACE_LB_FALLBACK=1` | CUDA kernels without launch bounds |
| `GRACE_RUNTIME_NO_ROCBLAS=1` | Built-in GRACE-3L GEMMs instead of rocBLAS |
| `GRACE3L_SPBF_MFMA=0` | Disable GRACE-3L product-basis matrix kernels |
| `GRACE_FUSED_MFMA=0` | Disable GRACE-1L/2L matrix kernels |

`-DGRACE_HIP_UNSAFE_FP_ATOMICS=AUTO` chooses hardware atomics only on supported discrete AMD GPUs.
Do not force them on MI300A. For AMD Python wheels, see the [ROCm build guide](backends.md#build-the-amd-backend).
