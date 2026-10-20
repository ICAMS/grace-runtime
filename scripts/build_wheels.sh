#!/bin/bash
# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
# build_wheels.sh [outdir]: the sdist and manylinux_2_17 x86_64 wheels of grace-runtime (CPython 3.10, 3.11 and one
# abi3 wheel for >= 3.12), without Docker: conda-forge compilers with the glibc 2.17 sysroot in micromamba envs
# (created when missing), libstdc++ linked statically (python/CMakeLists.txt), auditwheel vendors libgomp.
#   MAMBA_ROOT_PREFIX (default ~/micromamba), PYTHONS (default "3.10 3.11 3.12")
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$ROOT/dist/wheelhouse}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
MM=${MICROMAMBA:-${MAMBA_EXE:-$(command -v micromamba)}}    # the binary (micromamba is often a shell function)
PKGS="gxx_linux-64 sysroot_linux-64=2.17 cmake ninja pip python-build auditwheel patchelf"
env_for() { echo "gn-build-py${1/./}"; }
ensure_env() {
  local e; e=$(env_for "$1")
  [ -x "$MAMBA_ROOT_PREFIX/envs/$e/bin/python" ] || "$MM" create -y -q -n "$e" -c conda-forge "python=$1" $PKGS
  # the build requirements of pyproject.toml from PyPI (conda-forge lags for older CPythons)
  "$MM" run -n "$e" python -m pip install -q "scikit-build-core>=1.1" "nanobind>=3.1" auditwheel patchelf
}
mkdir -p "$OUT/raw"
rm -f "$OUT"/raw/*.whl
for py in ${PYTHONS:-3.10 3.11 3.12}; do
  ensure_env "$py"
  e=$(env_for "$py")
  echo "== CPython $py ($e)"
  "$MM" run -n "$e" python -m build --no-isolation --wheel -o "$OUT/raw" "$ROOT" > "$OUT/build-$py.log" 2>&1 \
    || { tail -30 "$OUT/build-$py.log"; exit 1; }
done
e=$(env_for "$(echo ${PYTHONS:-3.10 3.11 3.12} | awk '{print $1}')")
"$MM" run -n "$e" python -m build --no-isolation --sdist -o "$OUT" "$ROOT" > "$OUT/build-sdist.log" 2>&1
for w in "$OUT"/raw/*.whl; do
  "$MM" run -n "$e" auditwheel repair --plat manylinux_2_17_x86_64 -w "$OUT" "$w"
done
ls -la "$OUT"
