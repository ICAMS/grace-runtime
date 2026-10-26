#!/bin/bash
# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
# test_wheels.sh [wheelhouse]: For each CPython (PYTHONS, default 3.10 3.11 3.12 3.13): a venv on a
# micromamba env that holds only the test dependencies from conda-forge (numpy, ase, ... built for glibc 2.17, so
# old hosts work too), pip install of the matching wheel (abi3 for >= 3.12; newest version in the wheelhouse), then
# the python test suite (no TensorFlow) with the module the CPU picks and with GRACE_RUNTIME_CPU=v3 where the CPU has
# AVX2. A run fails unless pytest exits 0 and reports passed tests (a crash such as SIGILL is a failure).
# Test models: $GRACE_CACHE (grace-runtime models download GRACE-2L-SMAX-OMAT-medium GRACE-1L-OMAT-large-base
# GRACE-2L-OMAT-medium-ft-E). A release gate: only the skips GRACE_TEST_ALLOW_SKIP matches pass (python/tests/conftest.py;
# here the GPU tests, card sizes and timings included, TorchSim, AVX-512-only, MKL / scikit-learn, TensorFlow and the
# example notebook in an env made before it needed matplotlib; without phonopy only its phonon section is left out), so
# a missing model is a failure.
set -uo pipefail
ALLOW="CUDA|GPU|cards|timing test|torch|AVX-512|sklearn|MKL|tensorpotential|SavedModel|unsupported test model"
ALLOW+="|example notebook"
export GRACE_TEST_ALLOW_SKIP=${GRACE_TEST_ALLOW_SKIP-$ALLOW}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WH=${1:-$ROOT/dist/wheelhouse}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
MM=${MICROMAMBA:-${MAMBA_EXE:-$(command -v micromamba)}}
DEPS="numpy scipy ase pytest pyyaml threadpoolctl pip phonopy matplotlib-base"    # phonons_batch, the notebook
TMP=${TMPDIR:-/tmp}/gn-wheeltest-$$
rc=0
for py in ${PYTHONS:-3.10 3.11 3.12 3.13}; do
  e="gn-test-py${py/./}"
  if [ ! -x "$MAMBA_ROOT_PREFIX/envs/$e/bin/python" ]; then
    out=$("$MM" create -y -q -n "$e" -c conda-forge "python=$py" $DEPS 2>&1) ||
      { echo "CPython $py: env FAILED"; echo "$out" | tail -15; rc=1; continue; }
  fi
  tag="cp${py/./}-cp${py/./}"; [ "${py#3.}" -ge 12 ] && tag="cp312-abi3"
  whl=$(ls -t "$WH"/grace_runtime-*-"$tag"-manylinux*.whl 2>/dev/null | head -1)
  [ -n "$whl" ] || { echo "CPython $py: no $tag wheel in $WH"; rc=1; continue; }
  "$MAMBA_ROOT_PREFIX/envs/$e/bin/python" -m venv --system-site-packages "$TMP/$e"
  V="$TMP/$e/bin/python"
  $V -m pip install -q "$whl" || { echo "CPython $py: install FAILED"; rc=1; continue; }
  $V -c "import grace_runtime as g, sys; print(f'CPython {sys.version.split()[0]}: {g.__version__} module {g.cpu_module}')" \
    || { echo "CPython $py: import FAILED"; rc=1; continue; }
  $V -c "import sys, grace_runtime; assert not any(m.startswith(('tensorflow', 'tensorpotential')) for m in sys.modules)" || rc=1
  for cpu in "" v3; do
    log="$TMP/$e-pytest${cpu}.log"
    (cd "$TMP" && GRACE_RUNTIME_CPU=$cpu $V -m pytest -q -p no:cacheprovider "$ROOT/python/tests" \
        --ignore="$ROOT/python/tests/test_vs_tp.py" > "$log" 2>&1)
    st=$?
    last=$(tail -1 "$log")
    echo "  pytest ${cpu:-auto}: exit $st: $last"
    if [ $st -ne 0 ] || ! grep -q -E "[0-9]+ passed" <<< "$last"; then rc=1; tail -30 "$log"; fi
  done
done
rm -rf "$TMP"
exit $rc
