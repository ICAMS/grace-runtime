# Dispersion correction (DFT-D3)

[Documentation map](index.md#guides) · [Usage example](usage.md#dispersion-correction)

Optional DFT-D3 with Becke-Johnson damping adds two-body dispersion to the model.
The Axilrod–Teller–Muto (ATM) three-body term is available separately. Both are off by default; D4 is unsupported.

## Turning it on

```python
from ase.build import bulk
from grace_runtime import D3, GRACERuntimeCalculator, compute_batch

model = "GRACE-2L-OMAT-medium-ft-E"
structures = [bulk("W", "bcc", a=a, cubic=True) for a in (3.1, 3.2)]
calc = GRACERuntimeCalculator(model, dispersion="d3-pbe")
structures[0].calc = calc
print(structures[0].get_potential_energy())
results = compute_batch(model, structures, dispersion="d3-pbe")
print([r["energy"] for r in results])
```

Option variations, continuing the example above:

```python
calc = GRACERuntimeCalculator(model, dispersion=D3("pbe", cutoff=20.0))
calc = GRACERuntimeCalculator(model, dispersion="d3-pbe-atm")
```

The same `dispersion` argument works with relaxation, properties and the TorchSim adapter.

## Which parameters for which model

Use the functional of the training data:

| Model training data | Setting |
| --- | --- |
| Registry foundation models, PBE/PBE+U without dispersion | `"d3-pbe"` |
| r2SCAN without dispersion | `"d3-r2scan"` |
| Data already containing dispersion | Do not add D3 again |

Use `D3(a1=..., a2=..., s8=...)` for custom parameters; `a2` is in Bohr.
Bare functional names such as `"pbe"` are not accepted as `dispersion` strings.

## Results

Energy, forces, stress and atomic energies include the correction. `dispersion_energy` gives its contribution;
`dispersion_parts=True` adds force, stress and atomic-energy parts to batch/TorchSim results.
For a calculator, request these with `calc.get_property(...)`.
See the [result reference](api.md#calculator-and-batch-results). UQ remains the model's uncertainty.

## Cutoffs and accuracy

| Parameter | Default | Meaning |
| --- | --- | --- |
| `cutoff` | 15 Å | Pair cutoff |
| `switch_width` | 3 Å | Smooth switch to zero before the pair cutoff |
| `cn_cutoff` | 10 Å | Coordination-number cutoff |

The model cutoff is unchanged. Increase the pair cutoff and check convergence for volumes, surfaces and adsorption
energies. Coordination-dependent coefficients can also be sensitive to `cn_cutoff`, especially for carbon.
`switch_width=0` uses a hard pair cutoff.

## Three-body term (ATM)

Use `"d3-pbe-atm"` or `D3("pbe", atm=True)`. Defaults: `atm_cutoff=10.0` Å and `atm_switch_width=1.0` Å.
ATM is off by default and can be substantially more expensive than two-body D3.
Its cutoff may need increasing for converged solid-state energies. GPU atomic sums can vary slightly between runs.

## Cost

Longer cutoffs increase cost: roughly with the cube of the pair cutoff and the sixth power of the ATM cutoff.
Check accuracy and runtime on representative structures before large calculations.

## Limits

- Elements H–Lr (1–103), Becke-Johnson damping only.
- Coordination numbers use a hard cutoff, with very small energy discontinuities at crossings.
- Dispersion is not yet available in the LAMMPS pair styles.

The method follows Grimme et al., J. Chem. Phys. 132, 154104 (2010), and Grimme, Ehrlich and Goerigk,
J. Comput. Chem. 32, 1456 (2011). Reference data and third-party notices are in
[THIRD_PARTY_NOTICES.md](https://github.com/ICAMS/grace-runtime/blob/master/THIRD_PARTY_NOTICES.md).

See the [API reference](api.md#dispersion) for signatures, parameter descriptions and result keys.
