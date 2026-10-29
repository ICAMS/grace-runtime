#!/bin/bash
# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
# test_cuda_wheels.sh <cpu wheelhouse> <cuda wheelhouse> [gpu index]: the GPU tests on a GPU host. A clean micromamba
# env (CPython 3.12, numpy / ase / pytest from conda-forge) with the grace-runtime and grace-runtime-cuda12 abi3 wheels
# (and nvidia-cublas-cu12 with WITH_CUBLAS=1, the default), the tests copied out of the repository; then
#   1. import, plug-in, architecture, device="auto" backend, the cuBLAS in use: WITH_CUBLAS=1 the wheel's
#      (site-packages); WITH_CUBLAS=system the host's CUDA 12 toolkit (no extra; the host must have one in the loader
#      path); WITH_CUBLAS=0 none (steps 1 and 2 set GRACE_RUNTIME_NO_CUBLAS=1, since the host may have a toolkit);
#   2. the GPU test files (parity with the CPU calculator, batches, auto device, plug-in logic,
#      memory, allocation failures, chunks, several calculators; TorchSim and device batches with WITH_TORCH=1, which
#      installs the [torchsim] extra),
#      a release gate: only the skips GRACE_TEST_ALLOW_SKIP matches pass (card sizes, timings, the PTX path without an
#      older module, the unsupported-model check without that model), so a missing model or package is a failure;
#   3. an architecture without a module (GRACE_RUNTIME_CUDA_ARCH=sm61): auto -> CPU with a warning, cuda -> ImportError;
#   4. WITH_TF=1: TensorFlow on the GPU and grace-runtime CUDA in one process (env gn-cuda-tf, tensorflow[and-cuda]).
# Stops unless the card is idle (no compute apps, utilization < 5 %). GRACE_CACHE: the test models.
set -uo pipefail
CPU_WH=$(realpath "$1"); GPU_WH=$(realpath "$2")
export CUDA_DEVICE_ORDER=PCI_BUS_ID    # CUDA index = nvidia-smi index (the idle check looks at the card in use)
export CUDA_VISIBLE_DEVICES=${3:-${CUDA_VISIBLE_DEVICES:-0}}
D=${CUDA_VISIBLE_DEVICES%%,*}
for t in $(seq 15); do    # idle: no compute apps on the card, utilization < 5 % (15 s to settle)
  U=$(nvidia-smi -i "$D" --query-gpu=uuid,utilization.gpu --format=csv,noheader,nounits)
  A=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "${U%%,*}")
  [ "${U##*, }" -lt 5 ] && [ "$A" = 0 ] && break
  [ "$t" = 15 ] && { echo "GPU $D not idle (util ${U##*, } %, $A apps): stop"; exit 1; }
  sleep 1
done
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
MM=${MICROMAMBA:-${MAMBA_EXE:-$(command -v micromamba)}}
export GRACE_CACHE=${GRACE_CACHE:-$HOME/.cache/grace-runtime-test}
unset GRACE_RUNTIME_DEVICE GRACE_RUNTIME_CUDA_ARCH GRACE_RUNTIME_NO_CUBLAS PYTHONPATH
case ${WITH_CUBLAS:=1} in 0|1|system) ;; *) echo "WITH_CUBLAS must be 0, 1 or system"; exit 1;; esac
# torch from PyPI brings nvidia-cublas-cu12, which comes before a system cuBLAS
[ "$WITH_CUBLAS" = system ] && [ "${WITH_TORCH:-0}" = 1 ] && { echo "WITH_CUBLAS=system needs WITH_TORCH=0"; exit 1; }
NOCB=(); [ "$WITH_CUBLAS" = 0 ] && NOCB=(GRACE_RUNTIME_NO_CUBLAS=1)    # the built-in GEMMs even with a system cuBLAS
fail=0
check() { if [ "$1" = 0 ]; then echo "PASS $2"; else echo "FAIL $2"; fail=1; fi; }
E=gn-cuda-test-$(hostname -s)
"$MM" env remove -y -q -n "$E" > /dev/null 2>&1
"$MM" create -y -q -n "$E" -c conda-forge python=3.12 numpy ase pytest pyyaml > /dev/null || { echo "FAIL env"; exit 1; }
P="$MAMBA_ROOT_PREFIX/envs/$E/bin/python"
CPU_WHL=$(ls "$CPU_WH"/grace_runtime-*-cp312-abi3-*.whl | head -1)
"$P" -m pip install -q "$CPU_WHL" "$GPU_WH"/grace_runtime_cuda12-*-cp312-abi3-*.whl
check $? "install wheels"
if [ "$WITH_CUBLAS" = 1 ]; then
  "$P" -m pip install -q "nvidia-cublas-cu12>=12.4,<13"; check $? "install nvidia-cublas-cu12"
fi
if [ "${WITH_TORCH:-0}" = 1 ]; then "$P" -m pip install -q "$CPU_WHL[torchsim]"; check $? "install [torchsim]"; fi
W=$(mktemp -d /tmp/gn-cuda-test.XXXX)
mkdir -p "$W/python" "$W/tests"
cp -r "$ROOT/python/tests" "$W/python/"
cp -r "$ROOT/tests/fixtures" "$W/tests/"
cd "$W"
echo "== $(hostname -s) $(nvidia-smi -i ${CUDA_VISIBLE_DEVICES%%,*} --query-gpu=name,driver_version,compute_cap --format=csv,noheader)"

env "${NOCB[@]}" "$P" - <<EOF
import grace_runtime as g, grace_runtime._cuda as c
print("grace-runtime", g.__version__, "abi", g.abi, "modules", c.available())
calc = g.GRACERuntimeCalculator("$GRACE_CACHE/GRACE-2L-OMAT-medium-ft-E/kokkos.npz")
m = c.ensure_init("cuda")
lib = c.cublas_library()
print("auto backend:", calc.backend, " module:", m.__name__, " cuBLAS:", m.cublas(), lib)
assert m.cublas() == ("$WITH_CUBLAS" != "0"), "cuBLAS in use: %s, WITH_CUBLAS=$WITH_CUBLAS" % m.cublas()
want = {"1": "wheel", "system": "system", "0": None}["$WITH_CUBLAS"]
if want:    # the wheel's copy in site-packages, or the host toolkit's outside it
    assert lib.get("source") == want and ("/site-packages/" in lib["path"]) == (want == "wheel"), lib
else:
    assert "why" in lib, lib
assert calc.backend.startswith("cuda:") and m.__name__.startswith("grace_runtime_cuda12.")
EOF
check $? "import / auto device / plug-in module"

ALLOW="for .*cards|timing test|older GPU family|unsupported test model"
[ "${WITH_TORCH:-0}" = 1 ] || ALLOW="$ALLOW|torch"
[ "$WITH_CUBLAS" = 1 ] || ALLOW="$ALLOW|nvidia.cublas"    # the wheel's cuBLAS
env "${NOCB[@]}" GRACE_TEST_ALLOW_SKIP="${GRACE_TEST_ALLOW_SKIP-$ALLOW}" \
  "$P" -m pytest -q -p no:cacheprovider python/tests/test_gpu_*.py \
  python/tests/test_cuda_plugin.py python/tests/test_rocm_plugin.py python/tests/test_batch.py \
  python/tests/test_auto_device.py python/tests/test_torchsim.py 2>&1 | tail -15
check ${PIPESTATUS[0]} "GPU tests"

GRACE_RUNTIME_CUDA_ARCH=sm61 "$P" - <<EOF
import warnings, grace_runtime as g
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    calc = g.GRACERuntimeCalculator("$GRACE_CACHE/GRACE-2L-OMAT-medium-ft-E/kokkos.npz")
assert calc.backend == "cpu" and any("sm61" in str(x.message) for x in w), (calc.backend, [str(x.message) for x in w])
try:
    g.GRACERuntimeCalculator("$GRACE_CACHE/GRACE-2L-OMAT-medium-ft-E/kokkos.npz", device="cuda")
    raise SystemExit("no ImportError")
except ImportError as e:
    import grace_runtime._cuda as c
    assert all(a in str(e) for a in c.available()), e
    print("unsupported arch:", e)
EOF
check $? "architecture without a module"

if [ "${WITH_TF:-0}" = 1 ]; then
  T=gn-cuda-tf
  TP="$MAMBA_ROOT_PREFIX/envs/$T/bin/python"
  [ -x "$TP" ] || { "$MM" create -y -q -n "$T" -c conda-forge python=3.12 pip > /dev/null &&
                    "$TP" -m pip install -q "tensorflow[and-cuda]" ase; }
  "$TP" -m pip install -q --force-reinstall --no-deps "$CPU_WH"/grace_runtime-*-cp312-abi3-*.whl \
    "$GPU_WH"/grace_runtime_cuda12-*-cp312-abi3-*.whl
  "$TP" - <<EOF
import numpy as np, tensorflow as tf
from ase.io import read
gpus = tf.config.list_physical_devices("GPU")
for d in gpus:
    tf.config.experimental.set_memory_growth(d, True)
with tf.device("/GPU:0"):
    x = tf.random.normal((2048, 2048))
    print("TF matmul on", gpus[0].name, float(tf.reduce_sum(tf.matmul(x, x))))
from grace_runtime import GRACERuntimeCalculator
at = read("tests/fixtures/WMo54.data", format="lammps-data", atom_style="atomic", Z_of_type={1: 74, 2: 42})
m = "$GRACE_CACHE/GRACE-2L-OMAT-medium-ft-E/kokkos.npz"
g = at.copy(); g.calc = GRACERuntimeCalculator(m, device="cuda")
c = at.copy(); c.calc = GRACERuntimeCalculator(m, device="cpu", num_threads=4)
d = abs(g.get_potential_energy() - c.get_potential_energy()) / len(at)
print("TF + grace-runtime CUDA in one process: |dE|/atom", d)
assert d <= 1e-5
EOF
  check $? "TensorFlow + grace-runtime CUDA in one process"
fi
cd /; rm -rf "$W"
exit $fail
