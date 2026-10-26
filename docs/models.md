# Models and the model cache

[Documentation map](../README.md#guides) · [Usage](usage.md)

## Supported models

| Architecture | Foundation models (`grace-runtime models list`) |
| --- | --- |
| GRACE-1L | GRACE-1L-OMAT-medium-{base,ft-E,ft-AM}, GRACE-1L-OMAT-large-{base,ft-E,ft-AM}, GRACE-1L-SMAX-OMAT-large, GRACE-1L-SMAX-large |
| GRACE-2L | GRACE-2L-OMAT-medium-{base,ft-E,ft-AM}, GRACE-2L-OMAT-large-{base,ft-E,ft-AM}, GRACE-2L-SMAX-OMAT-medium, GRACE-2L-SMAX-OMAT-large, GRACE-2L-SMAX-medium, GRACE-2L-SMAX-large |
| GRACE-3L | GRACE-3L-OMAT-large, GRACE-3L-OMAT-large-ft-AM |

OMAT: trained on OMat24 (`-base`), fine-tuned with more weight on energies (`-ft-E`) or on sAlex + MPtrj (`-ft-AM`);
6 Å cutoff. SMAX and SMAX-OMAT: trained on SMAX (and OMat24), element-dependent cutoffs between 5.0 and 7.5 Å. The
older names (e.g. `GRACE-2L-OAM-L`, `GRACE-2L-OMAT-M`) are aliases. All of them have a UQ head. Your own GRACE-1L / 2L / 3L models run
too: export them with tensorpotential's `grace_utils -p model.yaml -c <checkpoint> export_kokkos -o model.npz` and
pass the file. Legacy models (GRACE-FS, the first OAM / OMAT releases) have no such export; use tensorpotential's
`TPCalculator` for them. See the cache and offline instructions below.

## Downloads and cache

```bash
grace-runtime models list                     # the foundation 1L / 2L / 3L models, aliases, cached or not
grace-runtime models download GRACE-2L-OMAT-medium-ft-E
```

`models.json` lists the foundation models of tensorpotential's registry; its sha256 values come from the Hugging Face
API.

Files live in `$GRACE_CACHE` or `~/.cache/grace`, at `<name>/kokkos.npz`. This is the cache tensorpotential uses, so
each file is downloaded once. `X` and `X-fp64` share one file.

Downloads show a progress bar with transferred bytes, speed, and ETA when the server provides the file size.
This applies both to `grace-runtime models download` and to automatic downloads when creating a calculator or calling
`compute_batch`. Cached models do not display download progress. For silent downloads from Python, use
`registry.resolve(name, quiet=True)` after `from grace_runtime import registry`.

Next to each file, `.grace_origin.kokkos.json` records its URL and sha256:

- A record with the expected URL or sha256: the file is used.
- A record grace-runtime wrote with neither the expected URL nor sha256: the file is downloaded again.
- A file tensorpotential downloaded from its own registry snapshot is used with a warning, so two packages of
  different releases do not re-download in turn.
- A file without a record is used if its sha256 matches.

New downloads are checked against the registry sha256 when one is provided. An accepted origin record avoids
rehashing the cached file on every use; it is not a check for subsequent local edits to that file.

`grace-runtime models download --force NAME` downloads again in any case.

Legacy models (FS, the first OAM/OMAT) give the reason they cannot run. User entries in
`<cache>/models_registry.yaml` that have a `kokkos_url` are read too, when `PyYAML` is installed
(`python -m pip install PyYAML`). Entries without a known sha256 are accepted with a warning, not verified.

A local `kokkos.npz` (a fine-tuned or reduced-element export) is checked when it is loaded: every array the model
reads must have its numpy dtype (float arrays as float32 / float64, index arrays as integers; integer-valued floats
such as `output_scale = 1` are converted by value), no NaN / Inf, and the shape the model's own sizes give; index
arrays must point inside what they index. A malformed export is a `ValueError` naming the array. Save it with
`np.savez` (not `np.savez_compressed`).

## Offline use

Download the model on a machine with network access before starting an offline job:

```bash
grace-runtime models download GRACE-2L-OMAT-medium-ft-E
```

Keep the model's directory, including `.grace_origin.kokkos.json`, when transferring the cache. Point `GRACE_CACHE`
to the cache root in the job environment, or pass the `kokkos.npz` file directly:

```python
from grace_runtime import Model

model = Model("/path/to/kokkos.npz")  # a file path does not contact the model registry
```

Use this loaded `model` in calculators and batch functions to avoid rereading it. Keep the file unchanged while
it is in use: GPU model caches are keyed by its path, so replacing that file does not update an already loaded
GPU model. Start a new process after replacing weights.
