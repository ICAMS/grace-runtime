# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""merge_cuda_wheels.py <outdir> <wheel>...: one grace-runtime-cuda12 wheel per CPython tag from wheels built for one GPU
architecture each (GitHub workflow: one job per architecture). Wheels of the same file name are merged: the union of
their files (every file they share must be identical; the _kk_<arch> modules differ), a new RECORD. Exit status 1 on a
conflict."""
import base64
import hashlib
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
groups = defaultdict(list)
for w in map(Path, sys.argv[2:]):
    groups[w.name].append(w)

rc = 0
for name, wheels in sorted(groups.items()):
    files = {}      # path -> (ZipInfo, bytes, source wheel)
    record = None
    for w in wheels:
        with zipfile.ZipFile(w) as z:
            for info in z.infolist():
                if info.filename.endswith(".dist-info/RECORD"):
                    record = info.filename
                    continue
                if info.is_dir():       # auditwheel's repair adds some; files imply their directories
                    continue
                data = z.read(info)
                if info.filename in files and files[info.filename][1] != data:
                    print(f"FAIL {name}: {info.filename} differs between {files[info.filename][2]} and {w}")
                    rc = 1
                files.setdefault(info.filename, (info, data, w))
    if record is None:
        print(f"FAIL {name}: no RECORD")
        rc = 1
        continue
    lines = []
    with zipfile.ZipFile(out / name, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files, key=lambda p: (".dist-info/" in p, p)):     # dist-info last, as wheels have it
            info, data, _ = files[path]
            z.writestr(info, data, zipfile.ZIP_DEFLATED)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            lines.append(f"{path},sha256={digest},{len(data)}")
        lines.append(f"{record},,")
        z.writestr(record, "\n".join(lines) + "\n")
    mods = sorted(p.rsplit("/", 1)[-1].split(".")[0] for p in files if "/_kk_" in p and p.endswith(".so"))
    print(f"{name}: {len(wheels)} wheels -> {', '.join(mods)}")
sys.exit(rc)
