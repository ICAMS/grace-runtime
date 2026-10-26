# Dispersion correction (DFT-D3)

[Documentation map](../README.md#guides) · [Usage example](usage.md#dispersion-correction)

grace-runtime can add the DFT-D3 dispersion correction with Becke-Johnson (rational) damping to a GRACE model:
the two-body terms C6 and C8 with coordination-number dependent C6 coefficients (S. Grimme, J. Antony, S. Ehrlich,
H. Krieg, J. Chem. Phys. 132, 154104 (2010); S. Grimme, S. Ehrlich, L. Goerigk, J. Comput. Chem. 32, 1456 (2011)).
Optionally with the Axilrod-Teller-Muto three-body term (ATM, off by default). It reproduces simple-dftd3 for the
same cutoffs (`s9 = 0`, or `s9 = 1` with the three-body term). The reference data come from tad-dftd3 and tad-mctc
(Apache-2.0, see `THIRD_PARTY_NOTICES.md`). D4 is not available.

The correction is off by default.

## Turning it on

```python
from grace_runtime import D3, GRACERuntimeCalculator, compute_batch, relax_batch

calc = GRACERuntimeCalculator(model, dispersion="d3-pbe")                         # PBE parameters, default cutoffs
calc = GRACERuntimeCalculator(model, dispersion=D3("r2scan", cutoff=20.0))         # r2SCAN, longer cutoff
calc = GRACERuntimeCalculator(model, dispersion=D3(a1=0.4289, a2=4.4407, s8=0.7875))  # own parameters, a2 in Bohr
calc = GRACERuntimeCalculator(model, dispersion="d3-pbe-atm")                     # with the three-body term
calc = GRACERuntimeCalculator(model, dispersion=D3("pbe", atm=True, atm_cutoff=12.0))
results = compute_batch(model, structures, dispersion="d3-pbe")
relaxed = relax_batch(model, structures, cell_filter="frechet", dispersion="d3-pbe")    # also eos_batch, elastic_batch

from grace_runtime.torchsim import GRACERuntimeModel
ts_model = GRACERuntimeModel(model, device="cuda", dispersion="d3-pbe")
```

Short names are `"d3-<functional>"` and `"d3-<functional>-atm"`; a bare functional name such as `"pbe"` is refused,
so that a name never changes its method. Presets:

| Name | a1 | a2 (Bohr) | s8 | s6 | Reference |
|---|---|---|---|---|---|
| `d3-pbe` | 0.4289 | 4.4407 | 0.7875 | 1 | Grimme, Ehrlich, Goerigk, J. Comput. Chem. 32, 1456 (2011) |
| `d3-r2scan` | 0.49484001 | 5.73083694 | 0.78981345 | 1 | Ehlert et al., J. Chem. Phys. 154, 061101 (2021) |

## Which parameters for which model

Use the functional of the model's training data, and only for training data without dispersion:

- All GRACE foundation models of the registry (OMat24, sAlex, MPtrj; PBE / PBE+U): `"d3-pbe"`. The +U does not change
  the D3 parameters.
- A model fine-tuned on r2SCAN data without dispersion: `"d3-r2scan"`.
- Do not add D3 to a model trained on data that already contains dispersion (for example PBE+D3 or wB97M-V data):
  that counts it twice.

## Results

`energy`, `forces`, `stress` and `energies` are the sums of model and correction. The correction alone:

| Key | Calculator | `compute_batch` / TorchSim |
|---|---|---|
| `dispersion_energy` | always | always |
| `dispersion_forces`, `dispersion_stress`, `dispersion_energies` | `calc.get_property(...)` (stored with every evaluation of the totals; the forces and stress need one with forces) | `dispersion_parts=True` |

UQ results (`gamma`, `atomic_sigma`, ...) are those of the model; the correction has no uncertainty of its own.

## Cutoffs and accuracy

`D3(..., cutoff=15.0, switch_width=3.0, cn_cutoff=10.0)` (Angstrom) are the defaults: pair terms within 15 Angstrom,
switched off smoothly between 12 and 15 Angstrom (energy and forces stay continuous for molecular dynamics), and
coordination numbers within 10 Angstrom. `switch_width=0` gives a hard cutoff. The model's own cutoff does not change.

The original D3 code and VASP use about 50 Angstrom for the pair terms and 21 Angstrom for the coordination numbers.
Difference to simple-dftd3 at its defaults (60 / 40 Bohr = 32 / 21 Angstrom), measured with a 3 Angstrom switch and
`cn_cutoff=10`:

| `cutoff` (Angstrom) | 12 | 15 | 20 | 30 |
|---|---|---|---|---|
| graphite: energy (meV/atom), stress (GPa) | 2.8, 0.044 | 1.1, 0.047 | 0.18, 0.048 | -0.22, 0.049 |
| copper | 17, 0.24 | 7.5, 0.09 | 2.5, 0.026 | 0.23, 0.005 |
| MgO | 8.0, 0.13 | 3.5, 0.053 | 1.1, 0.016 | 0.10, 0.003 |
| water | 0.53, 0.008 | 0.24, 0.003 | 0.08, 0.001 | 0.004, 0.0003 |

The dispersion energy is less negative than the converged one. At equal density the energy difference is close to a
constant per atom, so energy differences between similar structures keep most of it; volumes, surface and adsorption
energies are more sensitive: raise `cutoff` when you need them converged. The forces of these lattices vanish by
symmetry; with the atoms displaced (0.05 Angstrom, random), the largest force difference at the defaults is
2.5 meV/Angstrom in graphite and below 0.1 meV/Angstrom in copper, MgO and water.

The counting function of the coordination numbers never reaches zero (about 1e-7 per neighbour far away), so
`cn_cutoff` is part of the definition rather than a convergence parameter (simple-dftd3: 40 Bohr = 21 Angstrom). It
matters only for carbon-like elements, whose C6 coefficients change fast with the coordination number. With the atoms
displaced as above, `cutoff=15`:

| `cn_cutoff` (Angstrom) | 10 (default) | 15 |
|---|---|---|
| graphite: energy (meV/atom), max force (meV/Angstrom), stress (GPa) | 1.04, 2.5, 0.050 | 1.17, 2.0, 0.029 |
| copper | 7.5, 0.03, 0.091 | 7.5, 0.03, 0.091 |
| MgO | 3.5, 0.02, 0.053 | 3.5, 0.02, 0.053 |
| water | 0.24, 0.06, 0.003 | 0.24, 0.08, 0.003 |
| cost per atom, GPU (RTX 3090) | 0.55-1.1 us | 0.85-1.5 us |
| cost per atom, CPU (16 cores) | 3-6 us | 6-14 us |

## Three-body term (ATM)

`D3(..., atm=True, atm_cutoff=10.0, atm_switch_width=1.0)` adds the Axilrod-Teller-Muto term of simple-dftd3 (`s9 =
1`): for every triangle of atoms (and periodic images) with all three sides below `atm_cutoff`,
`sqrt(C6_ij C6_ik C6_jk) (3 cos a cos b cos c + 1) / (r_ij r_ik r_jk)^3`, with the coordination-number dependent C6 of
the two-body term and zero damping with the D3 van der Waals radii. Each side is switched off smoothly over the last
`atm_switch_width` Angstrom. It is off by default because the published D3(BJ) parameters were fitted without it
(simple-dftd3 and the other MLIP packages also leave it off).

The term is repulsive in dense solids and converges slowly with the cutoff (as the cube of 1 / cutoff). Truncation error
against simple-dftd3 at its 21 Angstrom, meV/atom (the whole term in brackets), switch 1 Angstrom:

| `atm_cutoff` | Cu (+71) | MgO (+25) | graphite (+9.3) | water (+1.6) |
|---|---|---|---|---|
| 8 Angstrom | -25 | -9.7 | -1.8 | -0.09 |
| 10 Angstrom (default) | -7.3 | -2.7 | -0.54 | -0.04 |
| 12 Angstrom | -3.5 | -1.3 | -0.25 | -0.02 |
| 15 Angstrom | -1.4 | -0.5 | -0.09 | -0.01 |

Its cost grows with the sixth power of the cutoff (the number of triangles); time of the three-body term alone per
force call, 4 000 atoms (Cu, water, graphite, MgO), relative to GRACE-1L (measured):

| `atm_cutoff` | GPU (RTX 3090) | CPU (16 cores, AVX-512) |
|---|---|---|
| 8 Angstrom | 0.2-0.6 (0.6-1.5 us/atom) | 1.7-2.7 (16-27 us/atom) |
| 10 Angstrom | 1.0-1.7 (2.6-4.5 us/atom) | 7-10 (69-98 us/atom) |
| 12 Angstrom | 2.3-4.7 (6.2-12.5 us/atom) | |

On the GPU the three-body sums are added atomically in varying order: with `atm=True`, energies, forces and stress of
the same structure may differ in the last bits between runs.

## Cost

The correction visits all atom pairs within the cutoff (about 1500 neighbours per atom at 15 Angstrom in a dense
solid) without storing a neighbour list. On the GPU the pair terms are evaluated in single precision with double
precision sums, as the GRACE network; on the CPU in double precision.

Time per force call at the default cutoffs, copper and water, 1 000 to 32 000 atoms (measured):

| | per atom | relative to GRACE-1L | GRACE-2L | GRACE-3L |
|---|---|---|---|---|
| GPU (NVIDIA RTX 3090) | 0.55-1.1 us | 0.2-0.45 | 0.12-0.22 | below 0.1 |
| CPU (16 cores, AVX-512) | 3-6 us | 0.3-0.5 | 0.13-0.22 | |

The cost grows with the cube of the cutoff: about 1.8 times at 20 Angstrom and 4 times at 30 Angstrom of the cost at
15 Angstrom.

## Limits

- Elements 1 to 103 (H to Lr); any other element raises `ValueError`.
- Becke-Johnson damping only; the three-body term only with it (as simple-dftd3's rational damping).
- The coordination numbers use a hard cutoff, as simple-dftd3: the energy jumps by about 1e-8 eV when a pair crosses
  it, negligible for molecular dynamics.
- Not in the LAMMPS pair styles yet.
