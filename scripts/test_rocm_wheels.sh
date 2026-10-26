#!/bin/bash
# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
# test_rocm_wheels.sh <cpu wheelhouse> <rocm wheelhouse> [gpu index]: the GPU tests on an AMD GPU host with ROCm 6
# (libamdhip64.so.6 found by the dynamic loader: ld.so.conf of the ROCm installation, or a loaded rocm module). A
# fresh venv of PYTHON (default python3) with the grace-runtime and grace-runtime-rocm6 wheels of the wheelhouses that
# match it (numpy, ase, pytest from PyPI), the tests copied out of the repository; then
#   1. import, plug-in, architecture, device="auto" backend, rocBLAS, grace-runtime info;
#   2. the GPU test files (parity with the CPU calculator, batches, auto device, plug-in logic, memory, allocation
#      failures, chunks, several calculators; TorchSim and device batches only with torch for ROCm in the environment),
#      a release gate: only the skips GRACE_TEST_ALLOW_SKIP matches pass (card sizes, timings, the CUDA-only PTX path,
#      the unsupported-model check without that model, torch), so a missing model or package is a failure;
#   3. an architecture without a module (GRACE_RUNTIME_ROCM_ARCH=gfx000): auto -> CPU with a warning, rocm ->
#      ImportError.
# [gpu index] sets HIP_VISIBLE_DEVICES; without it the visible devices of the environment are used (Slurm sets
# ROCR_VISIBLE_DEVICES). Run it on a GPU nobody else uses. GRACE_CACHE: the test models.
set -uo pipefail
CPU_WH=$(realpath "$1"); GPU_WH=$(realpath "$2")
[ -n "${3:-}" ] && export HIP_VISIBLE_DEVICES=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export GRACE_CACHE=${GRACE_CACHE:-$HOME/.cache/grace-runtime-test}
unset GRACE_RUNTIME_DEVICE GRACE_RUNTIME_ROCM_ARCH GRACE_RUNTIME_NO_ROCBLAS PYTHONPATH
fail=0
check() { if [ "$1" = 0 ]; then echo "PASS $2"; else echo "FAIL $2"; fail=1; fi; }
W=$(mktemp -d "${TMPDIR:-/tmp}/gn-rocm-test.XXXX")
"${PYTHON:-python3}" -m venv "$W/venv" || { echo "FAIL venv"; exit 1; }
P="$W/venv/bin/python"
# the two packages from the wheelhouses only (on dev the CPU package has the version of a release on PyPI), then
# their dependencies and pytest from PyPI
"$P" -m pip install -q --no-index --no-deps --find-links "$CPU_WH" --find-links "$GPU_WH" grace-runtime \
  grace-runtime-rocm6 && "$P" -m pip install -q --find-links "$CPU_WH" --find-links "$GPU_WH" grace-runtime-rocm6 pytest
check $? "install wheels"
"$P" -m pip list 2>/dev/null | grep -E "^grace-runtime"
mkdir -p "$W/python" "$W/tests"
cp -r "$ROOT/python/tests" "$W/python/"
cp -r "$ROOT/tests/fixtures" "$W/tests/"
cd "$W" || exit 1

"$P" - <<EOF
import grace_runtime as g, grace_runtime._rocm as r, grace_runtime._gpu as gpu
print("grace-runtime", g.__version__, "abi", g.abi, "modules", r.available())
p = r.probe(0)
print("rocm:0", p["arch"], p["name"], f"{p['total_mem'] / 2**30:.1f} GiB", "HIP", r.version_str(p["runtime"]), p["lib"])
calc = g.GRACERuntimeCalculator("$GRACE_CACHE/GRACE-2L-OMAT-medium-ft-E/kokkos.npz")
m = r.ensure_init("rocm")
print("auto backend:", calc.backend, " module:", m.__name__, " BLAS:", m.blas() or "built-in kernels")
assert calc.backend.startswith("rocm:") and m.__name__.startswith("grace_runtime_rocm6."), (calc.backend, m.__name__)
EOF
check $? "import / auto device / plug-in module"
"$W/venv/bin/grace-runtime" info
check $? "grace-runtime info"

ALLOW="for .*cards|timing test|older GPU family|PTX|unsupported test model|torch|CUDA|NVIDIA|cuBLAS"
GRACE_TEST_ALLOW_SKIP=${GRACE_TEST_ALLOW_SKIP-$ALLOW} "$P" -m pytest -q -p no:cacheprovider -rs \
  python/tests/test_gpu_*.py python/tests/test_cuda_plugin.py python/tests/test_rocm_plugin.py \
  python/tests/test_batch.py python/tests/test_auto_device.py python/tests/test_torchsim.py 2>&1 | tail -40
check "${PIPESTATUS[0]}" "GPU tests"

GRACE_RUNTIME_ROCM_ARCH=gfx000 "$P" - <<EOF
import warnings, grace_runtime as g
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    calc = g.GRACERuntimeCalculator("$GRACE_CACHE/GRACE-2L-OMAT-medium-ft-E/kokkos.npz")
assert calc.backend == "cpu" and any("gfx000" in str(x.message) for x in w), (calc.backend, [str(x.message) for x in w])
try:
    g.GRACERuntimeCalculator("$GRACE_CACHE/GRACE-2L-OMAT-medium-ft-E/kokkos.npz", device="rocm")
    raise SystemExit("no ImportError")
except ImportError as e:
    import grace_runtime._rocm as r
    assert all(a in str(e) for a in r.available()), e
    print("unsupported arch:", e)
EOF
check $? "architecture without a module"
cd / && rm -rf "$W"
exit $fail
