# Models and the model cache

[Documentation map](index.md#guides) · [Usage](usage.md)

## Supported models

GRACE-1L, GRACE-2L and GRACE-3L foundation models are available by registry name:

```bash
grace-runtime models list
grace-runtime models download GRACE-2L-OMAT-medium-ft-E
```

OMAT models use a 6 Å cutoff; SMAX variants use element-dependent cutoffs of 5–7.5 Å.
Older registry aliases are accepted. Legacy GRACE-FS and early OAM/OMAT models without a Kokkos export require
tensorpotential's `TPCalculator` instead. PACE (ACE) models fitted with pacemaker run from their `.yaml`: see
[PACE models](pace.md).

To use your own model, export it with tensorpotential and pass the resulting file:

```bash
grace_utils -p model.yaml -c CHECKPOINT export_kokkos -o model.npz
```

After exporting `model.npz`, load it with:

```python
from grace_runtime import Model

model = Model("model.npz")
```

Model files are validated on load; malformed arrays raise `ValueError`. If writing an export yourself,
use `np.savez`, not `np.savez_compressed`.

## Downloads and cache

Named models download on first use and are reused from `$GRACE_CACHE` or `~/.cache/grace`,
at `<name>/kokkos.npz`. The cache is shared with tensorpotential; `X` and `X-fp64` use the same file.
Downloads are checked against the registry checksum when provided. To download again:

```bash
grace-runtime models download --force GRACE-2L-OMAT-medium-ft-E
```

For silent downloads, import `registry` from `grace_runtime` and call `registry.resolve(name, quiet=True)`.
Custom entries in `<cache>/models_registry.yaml` need `kokkos_url` and `pip install PyYAML`;
entries without a checksum are accepted with a warning.

## Offline use

Download models before the job. When transferring a cache, copy each model directory with its
`.grace_origin.kokkos.json` record, then set `GRACE_CACHE` to the cache root.
Alternatively, pass an existing local file to avoid registry access (replace the path below):

```python
from grace_runtime import Model

model = Model("/path/to/kokkos.npz")
```

Reuse the loaded model in calculators and batch functions. Keep its file unchanged while in use;
start a new process after replacing weights so cached GPU models are reloaded.

See [Model](api.md#model) for its constructor and metadata.
