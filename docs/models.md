# Models and the model cache

[Documentation map](index.md#guides) · [Usage](usage.md)

## Supported models

GRACE-1L, GRACE-2L and GRACE-3L foundation models are available by registry name:

```bash
grace-runtime models list
grace-runtime models download GRACE-2L-OMAT-medium-ft-E
```

OMAT models use a 6 Å cutoff; SMAX variants use element-dependent cutoffs of 5–7.5 Å.
Older registry aliases are accepted. The early models GRACE-1L-OAM, GRACE-1L-OMAT, GRACE-2L-OAM and GRACE-2L-OMAT
run as well, without an uncertainty head. Legacy GRACE-FS models require tensorpotential's `TPCalculator`. PACE (ACE) models fitted
with pacemaker run from their `.yaml`: see [PACE models](pace.md).

To use your own model, export it with tensorpotential and pass the resulting file:

```bash
grace_utils -p model.yaml -c CHECKPOINT export_kokkos -o model.npz
```

Fits of the early GRACE-1L/2L presets export the same way. Their files are format 2; an older grace-runtime refuses
them and asks for an update.

GRACE-1L and GRACE-2L fits with `max_order` 3 (no AAAA / BBBB products; for example the small GRACE-OFF models) run as
well as those with `max_order` 4, with the same results; the exported file has no AAAA / BBBB arrays and grace-runtime
adds an inactive stage when it loads it. Such models have no uncertainty head. Other values of `max_order` are refused
by `export_kokkos`. GRACE-2L fits with `max_order` 3 need a tensorpotential newer than 0.6.2.1 to export.

A model published as `model.yaml` and `saved_model/` without its training checkpoints needs a checkpoint first (with a
tensorpotential newer than 0.6.2.1):

```bash
grace_utils -p model.yaml saved_model_to_checkpoint -s saved_model -o checkpoint
grace_utils -p model.yaml -c checkpoint export_kokkos -o model.npz
```

(`export_kokkos` refuses a checkpoint that does not hold the model's weights, such as the files in
`saved_model/variables/`.)

On a GPU, GRACE-2L fits up to the widths of the registry models run on the fast kernels, narrower layers zero-padded
with the same results and a warning (lmax at most 4; not models with an uncertainty head): a fit of the current presets
with at most 42 layer-1 and 32 layer-2 radial channels, 64 layer-2 product functions and `n_mlp_dens` at most 16, and
any fit with at most 32 layer-1 and 48 layer-2 radial channels, 48 layer-2 product functions and `n_mlp_dens` at most
12. GRACE-1L fits with up to 64 radial channels and FC outputs run on the fast kernels too: FC widths 20, 32, 48 and 64
and radial widths up to 32, 48 and 64 on kernels of their own, other widths zero-padded the same way. Wider GRACE-2L
fits stop with an error that names their widths, also with `device="auto"`; with `GRACE_GENERIC_KERNELS=1` set in the
environment they run on generic GPU kernels, several times slower, with a warning. Please open an
[issue](https://github.com/ICAMS/grace-runtime/issues) with the widths the message names so that fast kernels can be
added.

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
