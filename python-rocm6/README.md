# grace-runtime-rocm6

AMD GPU backend (ROCm 6) of [grace-runtime](../README.md): the fused GRACE-1L / 2L / 3L Kokkos kernels (as LAMMPS
`grace/{1l,2l,3l}/fused/kk`) for `GRACERuntimeCalculator` and `grace_runtime.compute_batch`.

The Python ROCm package is not released yet; the current PyPI package is a placeholder with no runtime code.
Install the [CPU package from source](../docs/installation.md#cpu-from-source), then build this backend from
the same source revision as described below.

`device="auto"` (the default) uses the GPU when the backend and model are compatible, otherwise the CPU;
`device="rocm"` (or `"rocm:N"`, `"hip"` is an alias) requires it
explicitly. Modules: gfx90a (MI210, MI250, MI250X) and gfx942 (MI300A, MI300X, MI325X; built, not yet run on such a
GPU). AMD GPU code runs only on its own architecture: other GPUs fall back to the CPU (`device="auto"`, with a warning)
or raise (`device="rocm"`); build a wheel for them from source (below).

Needs ROCm 6 (6.2 or newer) on the machine: the wheel links the HIP runtime of the system (`libamdhip64.so.6`) and
bundles no ROCm library. The dynamic loader must find it (a ROCm installation registers its `lib` directory with
`ldconfig`; on clusters, load the ROCm module). rocBLAS of the ROCm installation is used for the GRACE-3L GEMMs when
present (loaded at run time); otherwise built-in kernels run. `grace-runtime info` names the GPU, its architecture,
the HIP runtime and whether rocBLAS was found. One GPU per process; on a cluster run one process per GPU (or GCD).
Linux x86-64 only.

## Source build for another AMD GPU

With ROCm 6 installed (here in `/opt/rocm`) and `g++`, for an RX 7900 (`gfx1100`); other architectures: gfx906,
gfx908, gfx90a, gfx940, gfx942, gfx1030, gfx1100, gfx1103, gfx1201:

```bash
export ROCM_PATH=/opt/rocm GRACE_KK_ARCHS=gfx1100 NPROC=8
export PYTHONS="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
bash scripts/build_rocm_wheels.sh "$PWD/dist/rocm-wheelhouse"
python -m pip install --no-index --find-links="$PWD/dist/rocm-wheelhouse" grace-runtime-rocm6
grace-runtime info
```

Run from the repository root with your Python environment active and the matching CPU package already installed.
Use a fresh output directory when rebuilding a wheel of the same version. `--no-index` does not fetch dependencies.

Without `ROCM_PATH`, the script unpacks the compiler and headers of ROCm 6.2.4 from AMD's RHEL 8 packages into
`~/.cache/grace-runtime/rocm` (about 300 MB download, no root needed); run it in the image
`quay.io/pypa/manylinux_2_28_x86_64` for portable (manylinux_2_28) wheels, as the release builds do.
