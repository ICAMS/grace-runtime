#!/bin/bash
# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
# build_wheels.sh [outdir]: the sdist and the wheels of grace-runtime for this platform (CPython 3.10, 3.11 and one
# abi3 wheel for >= 3.12), without Docker, in micromamba envs (created when missing):
#   Linux x86-64 / aarch64: manylinux_2_17, conda-forge compilers with the glibc 2.17 sysroot, auditwheel;
#   macOS arm64: macosx_11_0_arm64, Apple clang with conda-forge's llvm-openmp, shipped in the wheel.
#   MAMBA_ROOT_PREFIX (default ~/micromamba), PYTHONS (default "3.10 3.11 3.12")
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$ROOT/dist/wheelhouse}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
MM=${MICROMAMBA:-${MAMBA_EXE:-$(command -v micromamba)}}    # the binary (micromamba is often a shell function)
BUILD_ARGS=    # the ARM wheels: the portable aarch64 module (python/CMakeLists.txt)
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64|Linux-aarch64)
    ARCH=$(uname -m)
    COMPILERS="gxx_linux-${ARCH/x86_64/64} sysroot_linux-${ARCH/x86_64/64}=2.17"
    TOOLS="auditwheel patchelf"
    if [ "$ARCH" = aarch64 ]; then BUILD_ARGS="-C cmake.define.GRACE_RUNTIME_ISAS=aarch64"; fi
    repair() { "$MM" run -n "$1" auditwheel repair --plat manylinux_2_17_$ARCH -w "$OUT" "$2"; } ;;
  Darwin-arm64)
    # llvm-openmp in the env of the building Python: python/CMakeLists.txt takes it from $CONDA_PREFIX
    COMPILERS="llvm-openmp"
    TOOLS="delocate"
    BUILD_ARGS="-C cmake.define.GRACE_RUNTIME_ISAS=aarch64 -C cmake.define.GRACE_RUNTIME_BUNDLE_OPENMP=ON"
    export MACOSX_DEPLOYMENT_TARGET=11.0
    # --exclude: libomp is in the wheel already and must stay @rpath/libomp.dylib
    repair() { "$MM" run -n "$1" delocate-wheel --require-archs arm64 --exclude libomp -w "$OUT" "$2"; } ;;
  *) echo "build_wheels.sh: no wheels for $(uname -s) $(uname -m)" >&2; exit 1 ;;
esac
PKGS="$COMPILERS cmake ninja pip python-build $TOOLS"
env_for() { echo "gn-build-py${1/./}"; }
ensure_env() {
  local e; e=$(env_for "$1")
  [ -x "$MAMBA_ROOT_PREFIX/envs/$e/bin/python" ] || "$MM" create -y -q -n "$e" -c conda-forge "python=$1" $PKGS
  # the build requirements of pyproject.toml from PyPI (conda-forge lags for older CPythons)
  "$MM" run -n "$e" python -m pip install -q "scikit-build-core>=1.1" "nanobind>=3.1" $TOOLS
}
mkdir -p "$OUT/raw"
rm -f "$OUT"/raw/*.whl
for py in ${PYTHONS:-3.10 3.11 3.12}; do
  ensure_env "$py"
  e=$(env_for "$py")
  echo "== CPython $py ($e)"
  "$MM" run -n "$e" python -m build --no-isolation --wheel $BUILD_ARGS -o "$OUT/raw" "$ROOT" \
    > "$OUT/build-$py.log" 2>&1 || { tail -30 "$OUT/build-$py.log"; exit 1; }
done
e=$(env_for "$(echo ${PYTHONS:-3.10 3.11 3.12} | awk '{print $1}')")
"$MM" run -n "$e" python -m build --no-isolation --sdist -o "$OUT" "$ROOT" > "$OUT/build-sdist.log" 2>&1
for w in "$OUT"/raw/*.whl; do
  repair "$e" "$w"
done
ls -la "$OUT"
