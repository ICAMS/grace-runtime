// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Loader of the PACE npz arrays (of a file, or of a .yaml converted by pace_convert.h), with the checks before any
// value sizes or indexes.
#include "grace/pace_model.h"
#include "grace/error.h"

#include "npz3.h"
#include "pace_convert.h"

namespace grace {

using namespace npz;

namespace {

double req_double(const cnpy::npz_t &z, const std::string &k)
{
  detail::require_scalar(z, k);
  return npz_require_double(z, k)[0];
}

void check(bool ok, const std::string &key, const std::string &what)
{
  if (!ok) throw bad_array(key, what);
}

}    // namespace

bool is_pace_yaml(const std::string &path)
{
  const size_t dot = path.find_last_of('.');
  const std::string ext = dot == std::string::npos ? "" : path.substr(dot);
  return ext == ".yaml" || ext == ".yml";
}

void PaceModel::load(const std::string &path, const std::string &asi)
{
  *this = PaceModel();
  const bool yaml = is_pace_yaml(path);
  if (!yaml && !asi.empty())
    throw Error("an active set (.asi) goes with a PACE .yaml: " + path);
  const cnpy::npz_t z = yaml ? pace::npz_from_yaml(path, asi) : npz_open(path);
  const int ver = npz_get_int_scalar(z, "pace_format_version", -1);
  if (ver != kPaceFormatVersion)
    throw Error("not a PACE model of format " + std::to_string(kPaceFormatVersion) + " (pace_format_version " +
                std::to_string(ver) + "): " + path);
  element_names = npz_names(z, "element_names");
  n_elements = (int) element_names.size();
  check(n_elements >= 1 && n_elements <= kPaceMaxElements, "element_names",
        "has " + std::to_string(n_elements) + " elements, expected 1 .. " + std::to_string(kPaceMaxElements));
  element_z = npz_require_int(z, "element_z");
  require_size("element_z", element_z, n_elements);
  inner_cutoff_type = npz_require_int_in(z, "inner_cutoff_type", kInnerDensity, kInnerZbl);
  const auto defined = npz_require_int(z, "bond_defined");
  require_size("bond_defined", defined, (size_t) n_elements * n_elements);

  elements.resize(n_elements);
  for (int mu = 0; mu < n_elements; mu++) {
    const std::string p = "e" + std::to_string(mu) + "_";
    PaceElement &e = elements[mu];
    e.ndensity = npz_require_int_in(z, p + "ndensity", 1, kPaceMaxDensity);
    e.npoti = npz_require_int_in(z, p + "npoti", kFinnisSinclair, kFinnisSinclairShiftedScaled);
    e.fs_parameters = npz_require_double(z, p + "fs_parameters");
    require_size(p + "fs_parameters", e.fs_parameters, 2 * e.ndensity);
    e.rho_core_cut = req_double(z, p + "rho_core_cut");
    e.drho_core_cut = req_double(z, p + "drho_core_cut");
    e.rank = npz_require_int(z, p + "rank");
    e.nrows = (int) e.rank.size();
    const auto sh = npz_shape(z, p + "ms");    // [rows, rankmax]; rows may be 0
    check(sh.size() == 2 && (int) sh[0] == e.nrows && sh[1] >= 1 && sh[1] <= (size_t) kPaceMaxRank, p + "ms",
          "has shape " + detail::dims_str(sh) + ", expected [" + std::to_string(e.nrows) + ", 1 .. " +
              std::to_string(kPaceMaxRank) + "]");
    e.rankmax = (int) sh[1];
    const std::pair<const char *, std::vector<int> *> idx[] = {{"mus", &e.mus}, {"ns", &e.ns}, {"ls", &e.ls},
                                                                {"ms", &e.ms}};
    for (const auto &[k, dst] : idx) {
      *dst = npz_require_int(z, p + k);
      require_size(p + k, *dst, (size_t) e.nrows * e.rankmax);
    }
    require_index(p + "mus", e.mus, n_elements);
    e.ctildes = npz_require_double(z, p + "ctildes");
    require_size(p + "ctildes", e.ctildes, (size_t) e.nrows * e.ndensity);
    for (int r = 0; r < e.nrows; r++) {
      check(e.rank[r] >= 1 && e.rank[r] <= e.rankmax, p + "rank",
            "has " + std::to_string(e.rank[r]) + " at " + std::to_string(r) + ", outside 1 .. rankmax");
      for (int t = 0; t < e.rank[r]; t++) {
        const size_t q = (size_t) r * e.rankmax + t;
        check(e.ls[q] >= 0 && e.ls[q] <= kPaceMaxLmax, p + "ls",
              "has " + std::to_string(e.ls[q]) + ", outside 0 .. " + std::to_string(kPaceMaxLmax));
        check(e.ms[q] >= -e.ls[q] && e.ms[q] <= e.ls[q], p + "ms", "has |m| > l");
        check(e.ns[q] >= 1, p + "ns", "has n < 1");
      }
    }
  }

  bonds.resize((size_t) n_elements * n_elements);
  for (int i = 0; i < n_elements; i++)
    for (int j = 0; j < n_elements; j++) {
      PaceBond &b = bonds[(size_t) i * n_elements + j];
      if (!defined[(size_t) i * n_elements + j]) continue;
      const std::string p = "b" + std::to_string(i) + "_" + std::to_string(j) + "_";
      b.defined = true;
      b.radbase = npz_require_int_in(z, p + "radbase", kChebExpCos, kSBessel);
      b.nradmax = npz_require_int_in(z, p + "nradmax", 0, kPaceMaxNradbase);
      b.lmax = npz_require_int_in(z, p + "lmax", 0, kPaceMaxLmax);
      b.nradbase = npz_require_int_in(z, p + "nradbase", 1, kPaceMaxNradbase);
      const auto rp = npz_get_double(z, p + "radparameters");
      check(!rp.empty() || (b.radbase != kChebExpCos && b.radbase != kChebPow), p + "radparameters",
            "is empty; ChebExpCos and ChebPow need lambda");
      b.lambda = rp.empty() ? 0.0 : rp[0];
      b.crad = npz_get_double(z, p + "crad");
      require_size(p + "crad", b.crad, (size_t) b.nradmax * (b.lmax + 1) * b.nradbase);
      b.rcut = req_double(z, p + "rcut");
      check(b.rcut > 0.0, p + "rcut", "must be positive");
      b.dcut = req_double(z, p + "dcut");
      b.r_in = req_double(z, p + "r_in");
      b.delta_in = req_double(z, p + "delta_in");
      b.prehc = req_double(z, p + "prehc");
      b.lambdahc = req_double(z, p + "lambdahc");
      b.zi = element_z[i];
      b.zj = element_z[j];
      if (inner_cutoff_type == kInnerZbl) check(b.zi > 0 && b.zj > 0, "element_z", "ZBL needs atomic numbers");
    }
  if (z.count("uq_grade")) {    // the active set of the extrapolation grade
    has_uq = true;
    uq_grade = npz_require_int_in(z, "uq_grade", 0, 1);
    for (int mu = 0; mu < n_elements; mu++) {
      const std::string p = "e" + std::to_string(mu) + "_";
      PaceElement &e = elements[mu];
      e.uq_nfunc = npz_get_int_scalar(z, p + "uq_nfunc", -1);
      check(e.uq_nfunc >= 0, p + "uq_nfunc", "is missing or negative");
      e.uq_off = npz_require_int(z, p + "uq_off");
      require_size(p + "uq_off", e.uq_off, (size_t) e.nrows + 1);
      e.uq_func = npz_require_int(z, p + "uq_func");
      e.uq_cg = npz_require_double(z, p + "uq_cg");
      require_size(p + "uq_cg", e.uq_cg, e.uq_func.size());
      check(e.uq_off[0] == 0 && e.uq_off[e.nrows] == (int) e.uq_func.size(), p + "uq_off", "does not span uq_func");
      for (int r = 0; r < e.nrows; r++) check(e.uq_off[r] <= e.uq_off[r + 1], p + "uq_off", "is not ascending");
      for (int f : e.uq_func) check(f >= 0 && f < e.uq_nfunc, p + "uq_func", "has a function outside 0 .. uq_nfunc");
      const size_t n = (size_t) e.uq_nfunc * (uq_grade ? e.ndensity : 1);
      e.asi = npz_require_double(z, p + "asi");
      require_size(p + "asi", e.asi, n * n);
    }
  }
  for (int mu = 0; mu < n_elements; mu++) {    // every product's factors have a bond within its basis
    const PaceElement &e = elements[mu];
    for (int r = 0; r < e.nrows; r++)
      for (int t = 0; t < e.rank[r]; t++) {
        const size_t q = (size_t) r * e.rankmax + t;
        const PaceBond &b = bond(mu, e.mus[q]);
        const std::string p = "e" + std::to_string(mu) + "_";
        check(b.defined, p + "mus", "uses a pair without a radial basis");
        check(e.ls[q] <= b.lmax && e.ns[q] <= (e.rank[r] == 1 ? b.nradbase : b.nradmax), p + "ns",
              "exceeds the radial basis of its pair");
      }
  }
}

}    // namespace grace
