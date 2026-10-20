# Models and the model cache

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
- A record grace-runtime wrote with another URL means a newer release, and the file is downloaded again.
- A file tensorpotential downloaded from its own registry snapshot is used with a warning, so two packages of
  different releases do not re-download in turn.
- A file without a record is used if its sha256 matches.

`grace-runtime models download --force NAME` downloads again in any case.

Legacy models (FS, the first OAM/OMAT) give the reason they cannot run. User entries in
`<cache>/models_registry.yaml` that have a `kokkos_url` are read too.
