# Equation of state, elastic constants and phonons

[Documentation map](index.md#guides) · [Relaxation](relax.md)

These functions batch the scaled, strained or displaced cells of multiple structures. Reuse a `Model` and relax
reference structures before calculating properties:

```python
from ase.build import bulk
from grace_runtime import Model, elastic_batch, eos_batch, relax_batch

model = Model("GRACE-2L-OMAT-medium-ft-E")
crystals = [bulk("Al", "fcc", a=4.04, cubic=True), bulk("Ti", "hcp", a=2.95, c=4.68)]
results = relax_batch(model, crystals, fmax=1e-3, cell_filter="frechet")
assert all(r["converged"] for r in results)
relaxed = [r["atoms"] for r in results]

for r in eos_batch(model, relaxed):
    print(r["equilibrium_volume"], r["equilibrium_bulk_modulus"])
for r in elastic_batch(model, relaxed):
    print(r["C"].round(1), r["BH"], r["GH"])
```

Device and thread selection are automatic. Pass `device="cpu"`, `"cuda"` or `"rocm"` to require a backend.

## eos_batch

Fits an energy–volume curve. Defaults: 11 points over ±10% volume and a fifth-order polynomial fit.
Set `fit="birchmurnaghan"`, `"murnaghan"`, `"birch"`, `"vinet"` or `"pouriertarantola"` for an ASE fit.

| Option | Use |
| --- | --- |
| `num_of_point`, `volume_range` | Sampling count and fractional range, or a pair of absolute volumes in Å³ |
| `relax="none"` | Keep scaled atoms fixed; `"atoms"` relaxes positions, `"atoms+shape"` also relaxes cell shape at fixed volume |
| `fmax=0.005`, `max_steps=500` | Relaxation limits |

Results include the sampled curve, equilibrium values and fit error; see [result keys and units](api.md#eos-and-elastic-results).
No minimum in the sampled range gives `None` equilibrium values and a warning.
Check convergence for relaxed points. Very narrow volume ranges make B′ sensitive to fp32 round-off.

## elastic_batch

Fits the full 6×6 elastic matrix from stress versus strain in Voigt order (xx, yy, zz, yz, xz, xy),
with engineering shear strains. Defaults: five points per strain over ±0.005 and internal atomic relaxation.

Use `relax_atoms=False` for clamped atoms. Otherwise, `fmax=1e-3` and `max_steps=500` control internal relaxation.
Tight force convergence matters at small strains. If using `fix_symmetry=True`, keep `symprec` well below the
smallest strain-induced displacement so the relaxation is not suppressed.

Results include the stiffness and compliance matrices and Voigt/Reuss/Hill moduli; see the
[result reference](api.md#eos-and-elastic-results). Large `asymmetry` suggests poor convergence or excessive strain;
singular matrices yield `None` Reuss/Hill results. Check convergence for every relaxed strain.

## phonons_batch, phonopy_forces_batch

Install `pip install "grace-runtime[phonons]"`. `phonons_batch` returns one phonopy object per periodic structure:

```python
from ase.build import bulk
from grace_runtime import Model, phonons_batch, relax_batch

model = Model("GRACE-2L-OMAT-medium-ft-E")
crystals = [bulk("Al", "fcc", a=4.04), bulk("Si", "diamond", a=5.43)]
results = relax_batch(model, crystals, fmax=1e-4, cell_filter="frechet")
assert all(r["converged"] for r in results)
relaxed = [r["atoms"] for r in results]

for ph in phonons_batch(model, relaxed):
    ph.run_mesh([20, 20, 20], is_gamma_center=True)
    ph.run_total_dos()
    ph.run_thermal_properties(temperatures=[300])
    print(ph.force_constants.shape)
```

| Option | Use |
| --- | --- |
| `supercell` | Repeat count, three repeats or 3×3 matrix; by default choose a diagonal cell with plane spacings ≥ `min_length=10` Å |
| `distance=0.01` | Displacement amplitude in Å |

Forces from the undisplaced cell are subtracted, and force constants are symmetrized to enforce the acoustic
sum rule. Relax the reference cell first; residual stress is not corrected. Input isotope masses are preserved.

For custom displacement schemes, create phonopy objects and call `phonopy_forces_batch(model, phonons)`,
then `ph.produce_force_constants()`. See the [phonon example](https://github.com/ICAMS/grace-runtime/blob/master/python/examples/phonons/phonons_batch.py).
Automatic band paths additionally need `pip install seekpath`.
Use [phonopy's API](https://phonopy.github.io/phonopy/phonopy-module.html) for bands, DOS and thermal properties.

See the [API reference](api.md#properties) for signatures, parameter descriptions and result keys.
