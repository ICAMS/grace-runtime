#!/bin/bash
# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
# build_cuda_wheels.sh [outdir]: grace-runtime-cuda12 sdist and manylinux_2_17 x86_64 wheels (CPython 3.10, 3.11 and one
# abi3 wheel for >= 3.12), each holding one module per GPU architecture, without Docker: conda-forge gcc with the glibc
# 2.17 sysroot and cuda-nvcc 12.9 in micromamba envs; Kokkos (pinned in python-cuda/CMakeLists.txt) built once per
# architecture into GRACE_KOKKOS_CACHE; auditwheel vendors libgomp and the redistributable libcudart, excludes the
# driver (libcuda.so.1). The sources are staged with the repository layout (python/kk, python/src, libgrace, cmake).
#   MAMBA_ROOT_PREFIX (default ~/micromamba), PYTHONS (default "3.10 3.11 3.12"), GRACE_KK_ARCHS (default
#   "sm70;sm80;sm90;sm100;sm120"), GRACE_KOKKOS_CACHE (default ~/.cache/grace-runtime/kokkos-wheel, CI: $CI_PROJECT_DIR/.kokkos), NPROC
set -eEuo pipefail
trap 'echo "build_cuda_wheels.sh: line $LINENO failed: $BASH_COMMAND" >&2' ERR
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$ROOT/dist/cuda-wheelhouse}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
MM=${MICROMAMBA:-${MAMBA_EXE:-$(command -v micromamba)}}
ARCHS=${GRACE_KK_ARCHS:-sm70;sm80;sm90;sm100;sm120}
KC=${GRACE_KOKKOS_CACHE:-$HOME/.cache/grace-runtime/kokkos-wheel}
NPROC=${NPROC:-16}
CUDA=12.9
PKGS="gxx_linux-64=13 sysroot_linux-64=2.17 cuda-nvcc=$CUDA cuda-cudart-dev=$CUDA cuda-cudart-static=$CUDA \
cuda-version=$CUDA cmake ninja pip python-build auditwheel patchelf"
env_for() { echo "gn-cuda-build-py${1/./}"; }
ensure_env() {
  local e; e=$(env_for "$1")
  echo "== env $e"
  [ -x "$MAMBA_ROOT_PREFIX/envs/$e/bin/python" ] || "$MM" create -y -n "$e" -c conda-forge "python=$1" $PKGS | tail -5
  # cuBLAS headers only (the modules dlopen libcublas at run time): libcublas-dev of CUDA >= 12.8 declares glibc >= 2.28,
  # which hosts with an older glibc (the CI runner) cannot solve; nothing of it runs at build time
  [ -f "$MAMBA_ROOT_PREFIX/envs/$e/targets/x86_64-linux/include/cublas_v2.h" ] ||
    CONDA_OVERRIDE_GLIBC=2.28 "$MM" install -y -n "$e" --no-deps -c conda-forge "libcublas-dev=${CUDA}.*" | tail -3
  "$MM" run -n "$e" python -m pip install -q "scikit-build-core>=1.1" "nanobind>=3.1" auditwheel patchelf
}
# compilers of env $1 (conda-forge names the host compiler x86_64-conda-linux-gnu-g++)
compilers() {
  local p="$MAMBA_ROOT_PREFIX/envs/$1/bin"
  export CXX="$p/x86_64-conda-linux-gnu-g++" CC="$p/x86_64-conda-linux-gnu-gcc" CUDACXX="$p/nvcc"
  export NVCC_WRAPPER_DEFAULT_COMPILER="$CXX" PATH="$p:$PATH"
}

# Kokkos per architecture, once (pinned release of python-cuda/CMakeLists.txt)
KV=$(sed -n 's/^set(GRACE_KOKKOS_VERSION "\(.*\)")/\1/p' "$ROOT/python-cuda/CMakeLists.txt")
KS=$(sed -n 's/^set(GRACE_KOKKOS_SHA256 "\(.*\)")/\1/p' "$ROOT/python-cuda/CMakeLists.txt")
first=$(echo ${PYTHONS:-3.10 3.11 3.12} | awk '{print $1}')
ensure_env "$first"
E0=$(env_for "$first")
mkdir -p "$KC"
echo "== Kokkos $KV in $KC"
if [ ! -d "$KC/kokkos-$KV" ]; then
  curl -sSL -o "$KC/kokkos-$KV.tar.gz" "https://github.com/kokkos/kokkos/releases/download/$KV/kokkos-$KV.tar.gz"
  echo "$KS  $KC/kokkos-$KV.tar.gz" | sha256sum -c -
  tar -xzf "$KC/kokkos-$KV.tar.gz" -C "$KC"
fi
declare -A KARCH=([sm70]=VOLTA70 [sm75]=TURING75 [sm80]=AMPERE80 [sm86]=AMPERE86 [sm89]=ADA89 [sm90]=HOPPER90 [sm100]=BLACKWELL100 [sm120]=BLACKWELL120)
for a in ${ARCHS//;/ }; do
  P="$KC/install-$KV-$a"
  if [ ! -f "$P/lib64/cmake/Kokkos/KokkosConfig.cmake" ] && [ ! -f "$P/lib/cmake/Kokkos/KokkosConfig.cmake" ]; then
    ( compilers "$E0"
      "$MM" run -n "$E0" cmake -S "$KC/kokkos-$KV" -B "$KC/build-$KV-$a" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$P" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_CUDA_COMPILER="$CUDACXX" \
        -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_${KARCH[$a]}=ON -DCMAKE_CXX_STANDARD=20 \
        -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON > "$KC/kokkos-$a.log" 2>&1 &&
      "$MM" run -n "$E0" cmake --build "$KC/build-$KV-$a" -j "$NPROC" >> "$KC/kokkos-$a.log" 2>&1 &&
      "$MM" run -n "$E0" cmake --install "$KC/build-$KV-$a" >> "$KC/kokkos-$a.log" 2>&1 ) \
      || { grep -a -A12 "FAILED:" "$KC/kokkos-$a.log" | grep -v "nvcc warning" | head -60
           grep -a -m 10 -E "error:|Killed|signal|internal compiler error" "$KC/kokkos-$a.log"; free -g; nproc
           rm -rf "$KC/build-$KV-$a"; exit 1; }
  fi
  export "GRACE_KOKKOS_ROOT_$(echo $a | tr a-z A-Z)=$P"
done

# staged sources, versions from the repository pyproject
STAGE="$OUT/stage/grace-runtime-cuda12"
rm -rf "$STAGE"; mkdir -p "$STAGE/python"
cp -r "$ROOT/python-cuda/." "$STAGE/"
cp -r "$ROOT/python/kk" "$ROOT/python/src" "$STAGE/python/"
cp -r "$ROOT/libgrace" "$ROOT/cmake" "$STAGE/"
cp "$ROOT/LICENSE.md" "$ROOT/THIRD_PARTY_NOTICES.md" "$STAGE/"
rm -f "$STAGE"/python/grace_runtime/_kk_*.so "$STAGE"/grace_runtime_cuda12/_kk_*.so
V=$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml")
sed -i -e "s/^version = \".*\"/version = \"$V\"/" -e "s/\"grace-runtime==[^\"]*\"/\"grace-runtime==$V\"/" "$STAGE/pyproject.toml"

mkdir -p "$OUT/raw"
rm -f "$OUT"/raw/*.whl
for py in ${PYTHONS:-3.10 3.11 3.12}; do
  ensure_env "$py"
  e=$(env_for "$py")
  echo "== CPython $py ($e)"
  ( compilers "$e"
    export GRACE_KK_ARCHS="${ARCHS//;/,}" CMAKE_BUILD_PARALLEL_LEVEL="$NPROC"
    "$MM" run -n "$e" python -m build --no-isolation --wheel -o "$OUT/raw" "$STAGE" ) > "$OUT/build-$py.log" 2>&1 \
    || { grep -a -A12 "FAILED:" "$OUT/build-$py.log" | grep -v "nvcc warning" | head -60
         grep -a -m 10 -E "error:|Killed|signal|internal compiler error" "$OUT/build-$py.log"; tail -8 "$OUT/build-$py.log"
         free -g; exit 1; }
done
"$MM" run -n "$E0" python -m build --no-isolation --sdist -o "$OUT" "$STAGE" > "$OUT/build-sdist.log" 2>&1
for w in "$OUT"/raw/*.whl; do
  "$MM" run -n "$E0" auditwheel repair --plat manylinux_2_17_x86_64 --exclude libcuda.so.1 -w "$OUT" "$w"
done
ls -la "$OUT"
