#!/bin/bash
# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
# build_rocm_wheels.sh [outdir]: grace-runtime-rocm6 sdist and wheels (CPython 3.10, 3.11 and one abi3 wheel for
# >= 3.12), each holding one module per AMD GPU architecture. Two uses:
#   * portable manylinux_2_28 x86_64 wheels (the GitHub workflow): run it inside the image
#     quay.io/pypa/manylinux_2_28_x86_64 (docker run, apptainer exec). CPythons from /opt/python, the image's g++
#     compiles the C++ code and links (static libstdc++, libgomp); ROCm $ROCM_VERSION (clang++ for the Kokkos code, the
#     HIP headers, libamdhip64 to link against) is unpacked from AMD's RHEL 8 packages into ROCM_CACHE (no root, no
#     package manager); auditwheel vendors libgomp and leaves the HIP runtime out (libamdhip64.so.6 and what it loads
#     come from the ROCm installed on the user's machine).
#   * a wheel for your own GPU with the ROCm installed on your machine, no container:
#     ROCM_PATH=/opt/rocm GRACE_KK_ARCHS=gfx1100 PYTHONS=3.12 bash scripts/build_rocm_wheels.sh (python3.12 on PATH;
#     the wheel gets the platform tag of your machine).
# Never in the wheels and never linked: rocBLAS (loaded at run time when the system has it), hipBLAS, rocSOLVER.
# Kokkos (pinned in python-rocm6/CMakeLists.txt) is built once per architecture and ROCm version into
# GRACE_KOKKOS_CACHE. The sources are staged with the repository layout (python/kk, python/src, libgrace, cmake).
#   ROCM_PATH (a ROCm 6 installation; default: unpack ROCm ROCM_VERSION, default 6.2.4, into ROCM_CACHE, default
#   ~/.cache/grace-runtime/rocm), PYTHONS (default "3.10 3.11 3.12"), GRACE_KK_ARCHS (default "gfx90a;gfx942"),
#   GRACE_KOKKOS_CACHE (default ~/.cache/grace-runtime/kokkos-rocm-wheel), NPROC (default 8), CXX (default g++)
# shellcheck disable=SC2030,SC2031  # PATH set in subshells on purpose (one venv per CPython)
set -eEuo pipefail
trap 'echo "build_rocm_wheels.sh: line $LINENO failed: $BASH_COMMAND" >&2' ERR
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:-$ROOT/dist/rocm-wheelhouse}" && cd "${1:-$ROOT/dist/rocm-wheelhouse}" && pwd)
ARCHS=${GRACE_KK_ARCHS:-gfx90a;gfx942}
PYTHONS=${PYTHONS:-3.10 3.11 3.12}
KC=${GRACE_KOKKOS_CACHE:-$HOME/.cache/grace-runtime/kokkos-rocm-wheel}
NPROC=${NPROC:-8}
CXX=$(command -v "${CXX:-g++}")
export CXX

# a venv per CPython with the build tools: /opt/python/cpXY-cpXY in the manylinux image, else pythonX.Y from PATH
W="$OUT/venvs"
venv_for() { echo "$W/py${1/./}"; }
ensure_venv() {
  local v; v=$(venv_for "$1")
  if [ ! -x "$v/bin/python" ]; then
    local p="/opt/python/cp${1/./}-cp${1/./}/bin/python"
    [ -x "$p" ] || p=$(command -v "python$1")
    "$p" -m venv "$v"
    "$v/bin/python" -m pip install -q -U pip
    "$v/bin/python" -m pip install -q "scikit-build-core>=1.1" "nanobind>=3.1" build "cmake>=3.24" ninja auditwheel \
      patchelf
  fi
}
first=$(echo "$PYTHONS" | awk '{print $1}')
ensure_venv "$first"
V0=$(venv_for "$first")

# cpio (newc) archive on stdin -> files under $1 (the manylinux image has rpm2cpio, no cpio)
UNCPIO='
import os, sys
dest, f, links = sys.argv[1], sys.stdin.buffer, {}
pad = lambda n: f.read(-n % 4)
while True:
    h = f.read(110)
    mode, nlink, size, namesize = (int(h[o:o + 8], 16) for o in (14, 38, 54, 94))
    ino = int(h[6:14], 16)
    name = f.read(namesize)[:-1].decode(); pad(110 + namesize)
    data = f.read(size); pad(size)
    if name == "TRAILER!!!":
        break
    path = os.path.join(dest, os.path.normpath(name[2:] if name.startswith("./") else name.lstrip("/")))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kind = mode & 0o170000
    if kind == 0o040000:
        os.makedirs(path, exist_ok=True)
    elif kind == 0o120000:
        os.path.lexists(path) and os.remove(path); os.symlink(data.decode(), path)
    elif kind == 0o100000:
        links.setdefault(ino, []).append(path)
        if size or nlink == 1:    # hard links: the data comes with the last name
            for p in links.pop(ino):
                with open(p, "wb") as out:
                    out.write(data)
                os.chmod(p, mode & 0o7777)
'
# ROCm: the given installation, else AMD's RHEL 8 packages of ROCM_VERSION unpacked (the compiler and the files needed
# to compile and link HIP code: about 300 MB to download, 1.1 GB unpacked)
if [ -z "${ROCM_PATH:-}" ]; then
  RV=${ROCM_VERSION:-6.2.4}
  RC=${ROCM_CACHE:-$HOME/.cache/grace-runtime/rocm}/$RV
  ROCM_PATH=$RC/opt/rocm-$RV
  PKGS="rocm-core rocm-llvm rocm-device-libs hip-devel hipcc hip-runtime-amd hsa-rocr hsa-rocr-devel hsakmt-roct-devel
        comgr rocprofiler-register rocprim-devel rocthrust-devel"
  MARK="$RV: $(tr -s ' \n' ' ' <<< "$PKGS")"
  if [ "$(cat "$RC/.done" 2>/dev/null)" != "$MARK" ]; then
    echo "== ROCm $RV (RHEL 8 packages) into $RC"
    rm -rf "$RC"; mkdir -p "$RC/rpm"
    # shellcheck disable=SC2086  # one argument per package
    "$V0/bin/python" - "https://repo.radeon.com/rocm/rhel8/$RV/main/" "$RC/rpm" $PKGS <<'PY'
import gzip, hashlib, re, sys, urllib.request
base, dest, want = sys.argv[1], sys.argv[2], set(sys.argv[3:])
md = urllib.request.urlopen(base + "repodata/repomd.xml").read().decode()
primary = re.search(r'<data type="primary">.*?<location href="([^"]+)"', md, re.S).group(1)
xml = gzip.decompress(urllib.request.urlopen(base + primary).read()).decode()
found = {}    # name -> (location, sha256 of the package from the repository metadata)
for pkg in re.findall(r'<package type="rpm">(.*?)</package>', xml, re.S):
    name = re.search(r"<name>([^<]+)</name>", pkg).group(1)
    if name in want:
        sha = re.search(r'<checksum type="sha256"[^>]*>([0-9a-f]{64})</checksum>', pkg).group(1)
        found[name] = (re.search(r'<location href="([^"]+)"', pkg).group(1), sha)
if set(found) != want:
    sys.exit(f"not in {base}: {sorted(want - set(found))}")
for name, (loc, sha) in sorted(found.items()):
    path = f"{dest}/{loc.rsplit('/', 1)[-1]}"
    urllib.request.urlretrieve(base + loc, path)
    with open(path, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != sha:
            sys.exit(f"sha256 mismatch: {loc}")
    print("  ", loc.rsplit("/", 1)[-1])
PY
    for r in "$RC"/rpm/*.rpm; do rpm2cpio "$r" | "$V0/bin/python" -c "$UNCPIO" "$RC"; done
    rm -rf "$RC/rpm"
    # absolute links of the packages (/opt/rocm-X/...) -> relative, so the unpacked tree works where it is
    find "$RC" -type l -lname '/*' | while read -r l; do
      ln -sfn "$(realpath -m --relative-to="$(dirname "$l")" "$RC$(readlink "$l")")" "$l"
    done
    echo "$MARK" > "$RC/.done"
  fi
fi
export ROCM_PATH
HIPC=${HIPCXX:-$ROCM_PATH/llvm/bin/clang++}
[ -x "$HIPC" ] || { echo "no ROCm clang++ at $HIPC (ROCM_PATH, HIPCXX)"; exit 1; }
RVER=$(cut -d- -f1 "$ROCM_PATH/.info/version" 2>/dev/null || echo unknown)
# grace-runtime-rocm6: the modules need libamdhip64.so.6 (ROCM_MAJOR = 6 of the plug-in)
case $RVER in 6.*) ;; *) echo "ROCm $RVER at $ROCM_PATH: grace-runtime-rocm6 needs ROCm 6 (set ROCM_PATH)"; exit 1 ;; esac
echo "== ROCm $RVER ($ROCM_PATH), $("$HIPC" --version | head -1); $("$CXX" --version | head -1)"

# Kokkos per architecture, once per ROCm version (the pinned release of python-rocm6/CMakeLists.txt, its options)
KV=$(sed -n 's/^set(GRACE_KOKKOS_VERSION "\(.*\)")/\1/p' "$ROOT/python-rocm6/CMakeLists.txt")
KS=$(sed -n 's/^set(GRACE_KOKKOS_SHA256 "\(.*\)")/\1/p' "$ROOT/python-rocm6/CMakeLists.txt")
mkdir -p "$KC"
echo "== Kokkos $KV in $KC"
if [ ! -d "$KC/kokkos-$KV" ]; then
  curl -sSL -o "$KC/kokkos-$KV.tar.gz" "https://github.com/kokkos/kokkos/releases/download/$KV/kokkos-$KV.tar.gz"
  echo "$KS  $KC/kokkos-$KV.tar.gz" | sha256sum -c -
  tar -xzf "$KC/kokkos-$KV.tar.gz" -C "$KC"
fi
GCCDIR=$(dirname "$("$CXX" -print-libgcc-file-name)")    # clang++ takes the libstdc++ of CXX
for a in ${ARCHS//[;,]/ }; do
  P="$KC/install-$KV-rocm$RVER-$a"
  if [ ! -f "$P/lib64/cmake/Kokkos/KokkosConfig.cmake" ] && [ ! -f "$P/lib/cmake/Kokkos/KokkosConfig.cmake" ]; then
    B="$KC/build-$KV-rocm$RVER-$a"
    KARCH=AMD_${a^^}
    ( export PATH="$V0/bin:$PATH"
      cmake -S "$KC/kokkos-$KV" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$P" \
        -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_HIP_COMPILER="$HIPC" \
        "-DCMAKE_HIP_FLAGS=${HIPFLAGS:-} --gcc-install-dir=$GCCDIR" -DCMAKE_HIP_ARCHITECTURES="$a" \
        -DCMAKE_PREFIX_PATH="$ROCM_PATH" -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON -DKokkos_ENABLE_HIP=ON \
        -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ARCH_"$KARCH"=ON \
        -DKokkos_ENABLE_IMPL_HIP_MALLOC_ASYNC=OFF -DKokkos_ENABLE_TESTS=OFF -DCMAKE_CXX_STANDARD=20 \
        -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON > "$KC/kokkos-$a.log" 2>&1 &&
      cmake --build "$B" -j "$NPROC" >> "$KC/kokkos-$a.log" 2>&1 &&
      cmake --install "$B" >> "$KC/kokkos-$a.log" 2>&1 ) \
      || { set +e; grep -a -A12 "FAILED:" "$KC/kokkos-$a.log" | head -60
           grep -a -m 10 -E "error:|Error|Killed|signal" "$KC/kokkos-$a.log"; tail -5 "$KC/kokkos-$a.log"; free -g; nproc
           rm -rf "$B"; exit 1; }
    rm -rf "$B"
  fi
  export "GRACE_KOKKOS_ROOT_${a^^}=$P"
done

# staged sources, versions from the repository pyproject
STAGE="$OUT/stage/grace-runtime-rocm6"
rm -rf "$STAGE"; mkdir -p "$STAGE/python"
cp -r "$ROOT/python-rocm6/." "$STAGE/"
cp -r "$ROOT/python/kk" "$ROOT/python/src" "$STAGE/python/"
cp -r "$ROOT/libgrace" "$ROOT/cmake" "$STAGE/"
cp "$ROOT/LICENSE.md" "$ROOT/THIRD_PARTY_NOTICES.md" "$STAGE/"
rm -f "$STAGE"/grace_runtime_rocm6/_kk_*.so
V=$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml")
sed -i -e "s/^version = \".*\"/version = \"$V\"/" -e "s/\"grace-runtime==[^\"]*\"/\"grace-runtime==$V\"/" "$STAGE/pyproject.toml"

mkdir -p "$OUT/raw"
rm -f "$OUT"/raw/*.whl
for py in $PYTHONS; do
  ensure_venv "$py"
  v=$(venv_for "$py")
  echo "== CPython $py"
  ( export PATH="$v/bin:$PATH" GRACE_KK_ARCHS="${ARCHS//;/,}" CMAKE_BUILD_PARALLEL_LEVEL="$NPROC"
    python -m build --no-isolation --wheel -o "$OUT/raw" "$STAGE" ) > "$OUT/build-$py.log" 2>&1 \
    || { set +e; grep -a -A12 "FAILED:" "$OUT/build-$py.log" | head -60
         grep -a -m 10 -E "error:|Killed|signal" "$OUT/build-$py.log"; tail -8 "$OUT/build-$py.log"; free -g; exit 1; }
done
"$V0/bin/python" -m build --no-isolation --sdist -o "$OUT" "$STAGE" > "$OUT/build-sdist.log" 2>&1
# the HIP runtime and the libraries it loads stay out of the wheel: they come with the user's ROCm
for w in "$OUT"/raw/*.whl; do
  "$V0/bin/auditwheel" repair --plat "${AUDITWHEEL_PLAT:-auto}" --exclude libamdhip64.so.6 \
    --exclude libhsa-runtime64.so.1 --exclude libamd_comgr.so.2 --exclude librocprofiler-register.so.0 -w "$OUT" "$w"
done
# every wheel: one module per architecture, no bundled or linked rocBLAS / hipBLAS / rocSOLVER / Tensile files, ...
# (scripts/check_rocm_wheel.py; the manylinux platform tag only for the container builds)
for w in "$OUT"/*.whl; do
  "$V0/bin/python" "$ROOT/scripts/check_rocm_wheel.py" "$w" "${ARCHS//;/,}" "${AUDITWHEEL_PLAT:--}"
done
ls -la "$OUT"
