// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The GRACE-3L Kokkos engine builds without LAMMPS (<float,float>, the base of the fused engine of grace/3l/fused/kk;
// <float,double>, fp64 geometry, the reference of the fp32 tests below), sets a model up on the default device, reports the message widths of the two
// layered exchanges as LAMMPS used them (comm_forward / comm_reverse) and refuses run() before setup(). Smoke run on
// the W54 / WMo54 fixture cells (CSR bonds with ghost images, a layered owner <-> image exchange): finite energy,
// the CPU grace/3l/fused reference (tests/fixtures/ref) within fp32 parity, chunksize 16 == one chunk.
//   env GRACE_TEST_FIXTURES (default tests/fixtures), GRACE_TEST_REF_DIR (default tests/fixtures/ref)
#include "grace/kk/csr_images.h"
#include "grace/kk/grace_3l_kk_impl.h"
#include "grace/kk/traits.h"
#include "grace_cnpy/cnpy.h"
#include "../test_util.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using Dev = Kokkos::DefaultExecutionSpace;    // as LAMMPS LMPDeviceType: an execution space (Kokkos::Cuda on GPU)
using T = grace::kk::DefaultTraits<Dev>;
using E3 = grace::kk::Grace3LKK<Dev, float, double, T>;
using E3F = grace::kk::Grace3LKK<Dev, float, float, T>;
template class grace::kk::Grace3LKK<Dev, float, double, T>;
template class grace::kk::Grace3LKK<Dev, float, float, T>;

namespace {
template<class E> void check_engine(const std::shared_ptr<grace::GRACE3LModel> &m)
{
  E e;
  std::vector<int> type2mu = {-1, 0};
  for (int k = 0; k < 2; k++) {    // run() before setup(): refused, also after set_model() without setup()
    typename E::RunInput3 in;
    bool thrown = false;
    try {
      e.run(in);
    } catch (const grace::Error &) {
      thrown = true;
    }
    CHECK(thrown);
    if (k == 0) e.set_model(m, type2mu);
  }
  e.setup();
  CHECK(e.cutoff(0, 0) > 4.0);
  // GRACE-3L-OMAT-large: eq1_norm 64 x 25 (layer 0), eq2_norm 64 x 50 (layer 1)
  CHECK(e.comm_width(0) == 1600);
  CHECK(e.comm_width(1) == 3200);
  CHECK(e.uq_available() == m->has_uq);
}

// owner <-> image message rows of both layers with the engine's pack / unpack kernels
template<class Eng, class Images> struct ImageExchange3 : Eng::Exchange3 {
  Eng *eng = nullptr;
  const Images *img = nullptr;
  typename T::t_double_1d buf;
  int ensure(int layer)
  {
    const int ng = img->nall - img->nlocal;
    const size_t need = (size_t) ng * eng->comm_width(layer);
    if (buf.extent(0) < need) buf = typename T::t_double_1d("xbuf", need);
    return ng;
  }
  void forward(int layer) override
  {
    if (const int ng = ensure(layer)) {
      eng->pack_forward(layer, ng, img->owner, buf);
      eng->unpack_forward(layer, ng, img->nlocal, buf);
    }
  }
  void reverse(int layer) override
  {
    if (const int ng = ensure(layer)) {
      eng->pack_reverse(layer, ng, img->nlocal, buf);
      eng->unpack_reverse(layer, ng, img->owner, buf);
    }
  }
};

struct Result {
  double e = 0;
  std::vector<double> f;
};

// a cell as CSR bonds (offset, j, rij per bond) through the engine with the layered owner <-> image exchange
template<class Eng> Result run_csr(Eng &eng, int n, std::vector<double> pos, const std::vector<int> &type, int ntypes,
                                   const double *cell, const std::vector<int64_t> &offset, const int *j,
                                   const double *rij, double shift = 0.0, bool energy_only = false)
{
  grace::kk::CsrImagesKK<Dev, T> img;
  for (auto &v : pos) v += shift;    // the whole cell translated: bond vectors unchanged
  img.build(n, pos.data(), type.data(), cell, offset.data(), j, rij);
  typename T::t_kkacc_1d_3 f("f", img.nall);
  typename Eng::RunInput3 in;
  in.x = img.x; in.f = f; in.type = img.type; in.ilist = img.ilist; in.numneigh = img.numneigh;
  in.neighbors = img.neighbors; in.inum = n; in.nlocal = n; in.nall = img.nall; in.ntypes = ntypes;
  in.eflag_global = 1; in.vflag_global = 1; in.energy_only = energy_only;
  ImageExchange3<Eng, decltype(img)> ex;
  ex.eng = &eng; ex.img = &img;
  in.ex = &ex;
  auto out = eng.run(in);
  img.fold(f);
  Result r;
  r.e = out.eng_vdwl;
  auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
  for (int a = 0; a < n; a++)
    for (int k = 0; k < 3; k++) r.f.push_back(hf(a, k));
  return r;
}

template<class Eng> Result run_cell(Eng &eng, const std::string &npz_path, double shift = 0.0, bool energy_only = false)
{
  auto npz = grace::cnpy::npz_load(npz_path);
  const int n = (int) npz["numbers"].shape[0];
  std::vector<int> type(n);
  for (int a = 0; a < n; a++) type[a] = npz["numbers"].data<int>()[a] == 74 ? 1 : 2;    // W = 1, Mo = 2
  const auto &off = npz["offset"];
  std::vector<int64_t> offset(off.data<int64_t>(), off.data<int64_t>() + off.shape[0]);
  std::vector<double> pos(npz["positions"].data<double>(), npz["positions"].data<double>() + 3 * n);
  return run_csr(eng, n, pos, type, 2, npz["cell"].data<double>(), offset, npz["j"].data<int>(),
                 npz["rij"].data<double>(), shift, energy_only);
}

// A periodic orthorhombic cell with its CSR bonds (brute force over the images within the cutoff)
struct Cell {
  int n = 0, ntypes = 0;
  std::vector<double> pos, cell = std::vector<double>(9, 0.0), rij;
  std::vector<int> type, j;
  std::vector<int64_t> offset;
  std::vector<std::string> species;    // LAMMPS type t -> species[t - 1]
};

void add_bonds(Cell &c, double rcut)
{
  const double L[3] = {c.cell[0], c.cell[4], c.cell[8]};
  int nim[3];
  for (int k = 0; k < 3; k++) nim[k] = (int) std::ceil(rcut / L[k]);
  c.offset.assign(1, 0);
  for (int a = 0; a < c.n; a++) {
    for (int b = 0; b < c.n; b++)
      for (int sx = -nim[0]; sx <= nim[0]; sx++)
        for (int sy = -nim[1]; sy <= nim[1]; sy++)
          for (int sz = -nim[2]; sz <= nim[2]; sz++) {
            const double d[3] = {c.pos[3 * b] - c.pos[3 * a] + sx * L[0], c.pos[3 * b + 1] - c.pos[3 * a + 1] + sy * L[1],
                                 c.pos[3 * b + 2] - c.pos[3 * a + 2] + sz * L[2]};
            const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if (r2 < 1e-8 || r2 >= rcut * rcut) continue;
            c.j.push_back(b);
            c.rij.insert(c.rij.end(), d, d + 3);
          }
    c.offset.push_back((int64_t) c.j.size());
  }
}

// lattice sites (fractional basis per conventional cell) repeated rep^3 times, displaced by a fixed pseudo-random
// amount up to `noise` A per component
Cell lattice_cell(double a, const std::vector<std::array<double, 3>> &basis, const std::vector<int> &basis_type,
                  int rep, double noise, const std::vector<std::string> &species, int type_stride = 0)
{
  Cell c;
  uint64_t s = 12345;
  auto rnd = [&s]() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double) (s >> 11) / 9007199254740992.0 * 2.0 - 1.0;
  };
  for (int x = 0; x < rep; x++)
    for (int y = 0; y < rep; y++)
      for (int z = 0; z < rep; z++)
        for (size_t q = 0; q < basis.size(); q++) {
          c.pos.push_back((x + basis[q][0]) * a + noise * rnd());
          c.pos.push_back((y + basis[q][1]) * a + noise * rnd());
          c.pos.push_back((z + basis[q][2]) * a + noise * rnd());
          c.type.push_back(type_stride ? 1 + (c.n * type_stride) % (int) species.size() : basis_type[q]);
          c.n++;
        }
  c.cell[0] = c.cell[4] = c.cell[8] = rep * a;
  c.ntypes = (int) species.size();
  c.species = species;
  add_bonds(c, 6.5);    // >= every model cutoff; the kernel applies per-pair cutoffs
  return c;
}

void smoke(const std::shared_ptr<grace::GRACE3LModel> &m)
{
  const std::string fx = env_or("GRACE_TEST_FIXTURES", "tests/fixtures");
  const std::string rd = env_or("GRACE_TEST_REF_DIR", "tests/fixtures/ref");
  std::vector<int> t2m(3, -1);
  for (int e = 0; e < (int) m->element_names.size(); e++) {
    if (m->element_names[e] == "W") t2m[1] = e;
    if (m->element_names[e] == "Mo") t2m[2] = e;
  }
  for (const char *cell : {"W54", "WMo54"}) {
    std::ifstream d(rd + "/" + cell + "_3l.dump"), t(rd + "/" + cell + "_3l.thermo");
    CHECK(d.good() && t.good());
    std::string line;
    for (int k = 0; k < 9; k++) std::getline(d, line);
    std::vector<double> rf(3 * 54);
    for (int a = 0; a < 54; a++) {
      int id; double fx_, fy, fz, pe;
      d >> id >> fx_ >> fy >> fz >> pe;
      rf[3 * (id - 1)] = fx_; rf[3 * (id - 1) + 1] = fy; rf[3 * (id - 1) + 2] = fz;
    }
    double step, etot;
    t >> step >> etot;
    // test_fp32_engine_matches_cpu: <float,float>, fp64 only in r_ij, |r|, rhat and the energy)
    // within the 1L/2L fused GPU parity of the CPU fused style
    {
      E3F e;
      e.set_model(m, t2m);
      e.setup();
      e.chunksize = 0;
      const Result one = run_cell(e, fx + "/" + cell + ".npz");
      double df = 0;
      for (size_t q = 0; q < one.f.size(); q++) df = std::max(df, std::fabs(one.f[q] - rf[q]));
      std::printf("%-6s fp32 E %.10f (CPU fused %.10f) dE/n %.3e  max|dF| %.3e\n", cell, one.e, etot,
                  std::fabs(one.e - etot) / 54, df);
      CHECK(std::fabs(one.e - etot) / 54 <= 1e-6);
      CHECK(df <= 1e-5);
    }
    E3 e;
    e.set_model(m, t2m);
    e.setup();
    e.chunksize = 0;
    Result one = run_cell(e, fx + "/" + cell + ".npz");
    e.chunksize = 16;
    Result chunked = run_cell(e, fx + "/" + cell + ".npz");
    const int n = (int) one.f.size() / 3;
    double df = 0, dfc = 0, fsum[3] = {0, 0, 0};
    for (int q = 0; q < 3 * n; q++) {
      df = std::max(df, std::fabs(one.f[q] - rf[q]));
      dfc = std::max(dfc, std::fabs(one.f[q] - chunked.f[q]));
      fsum[q % 3] += one.f[q];
    }
    std::printf("%-6s E %.10f (CPU fused %.10f) dE/n %.3e  max|dF| %.3e  chunk16: dE/n %.3e max|dF| %.3e  "
                "|sum F| %.2e\n", cell, one.e, etot, std::fabs(one.e - etot) / n, df,
                std::fabs(one.e - chunked.e) / n, dfc, std::fabs(fsum[0]) + std::fabs(fsum[1]) + std::fabs(fsum[2]));
    CHECK(std::isfinite(one.e));
    CHECK(std::fabs(one.e - etot) / n <= 2e-6);    // the GPU (mixed) vs the CPU fused (fp32) style
    CHECK(df <= 5e-5);
    CHECK(std::fabs(one.e - chunked.e) / n <= 1e-6);
    CHECK(dfc <= 1e-5);
  }
  const std::string w54 = fx + "/W54.npz";
  auto maxdf = [](const Result &a, const Result &b) {
    double d = 0;
    for (size_t q = 0; q < a.f.size(); q++) d = std::max(d, std::fabs(a.f[q] - b.f[q]));
    return d;
  };
  // test_resetup_spline_to_mlp: setup() again on a used engine after spline_mb 0 (the radial tables -> the GEMM MLP,
  // whose scratch was 1 row) gives the fresh MLP engine's result
  {
    E3 e;
    e.set_model(m, t2m);
    e.setup();
    e.chunksize = 0;
    run_cell(e, w54);
    e.spline_mb = 0;
    e.setup();
    const Result a = run_cell(e, w54);
    E3 f;
    f.spline_mb = 0;
    f.set_model(m, t2m);
    f.setup();
    f.chunksize = 0;
    const Result b = run_cell(f, w54);
    std::printf("re-setup tables -> MLP vs fresh MLP engine: dE/n %.3e  max|dF| %.3e\n", std::fabs(a.e - b.e) / 54,
                maxdf(a, b));
    CHECK(std::fabs(a.e - b.e) / 54 <= 1e-7 && maxdf(a, b) <= 1e-5);
  }
  // test_recompute_backward_and_shared_gpu_store: chunks of 16 without the forward store (the backward recomputes
  // L1/L2) == one chunk; store_fwd auto with the GPU shared by 2^20 ranks declines the store; energy_only == full E
  {
    E3 e;
    e.set_model(m, t2m);
    e.setup();
    e.chunksize = 0;
    const Result one = run_cell(e, w54);
    e.chunksize = 16;
    e.store_fwd = 0;
    const Result rec = run_cell(e, w54);
    CHECK(e.fwd_stored == 0);
    e.store_fwd = -1;
    e.ranks_per_device = 1 << 20;
    const Result shared = run_cell(e, w54);
    const int stored_shared = e.fwd_stored;
    const Result eo = run_cell(e, w54, 0.0, true);
    std::printf("chunk 16 recompute: dE/n %.3e max|dF| %.3e; 2^20 ranks: stored %d dE/n %.3e max|dF| %.3e; "
                "energy_only dE/n %.3e\n", std::fabs(one.e - rec.e) / 54, maxdf(one, rec), stored_shared,
                std::fabs(one.e - shared.e) / 54, maxdf(one, shared), std::fabs(one.e - eo.e) / 54);
    CHECK(std::fabs(one.e - rec.e) / 54 <= 1e-6 && maxdf(one, rec) <= 1e-5);
    CHECK(stored_shared == 0);
    CHECK(std::fabs(one.e - shared.e) / 54 <= 1e-6 && maxdf(one, shared) <= 1e-5);
    CHECK(std::fabs(one.e - eo.e) / 54 <= 1e-6);
  }
  // test_fp32_geometry_origin: <float,float> takes bond vectors as differences of the positions in
  // their own precision: the W54 cell translated by 500 A gives the same forces (x_i rounded to float first: ~1e-3)
  {
    E3F e;
    e.set_model(m, t2m);
    e.setup();
    const Result a = run_cell(e, fx + "/W54.npz"), b = run_cell(e, fx + "/W54.npz", 500.0);
    double df = 0;
    for (size_t q = 0; q < a.f.size(); q++) df = std::max(df, std::fabs(a.f[q] - b.f[q]));
    std::printf("fp32 W54 translated by 500 A: dE/n %.3e  max|dF| %.3e\n", std::fabs(a.e - b.e) / 54, df);
    CHECK(std::fabs(a.e - b.e) / 54 <= 1e-6 && df <= 5e-6);
  }
  // test_fp32_matches_mixed_multi_element: <float,float> within 1e-5 eV/A, 1e-6 eV/atom of <float,double> on
  // multi-element cells with short bonds and large forces (perovskite SrTiO3; a 25-element bcc solid solution)
  {
    const std::vector<std::string> hea = {"Al", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y",  "Zr",
                                          "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Hf", "Ta", "W",  "Re", "Os", "Ir"};
    std::vector<std::pair<std::string, Cell>> cells;
    cells.emplace_back("SrTiO3", lattice_cell(3.905, {{0, 0, 0}, {.5, .5, .5}, {.5, .5, 0}, {.5, 0, .5}, {0, .5, .5}},
                                              {1, 2, 3, 3, 3}, 3, 0.08, {"Sr", "Ti", "O"}));
    cells.emplace_back("HEA25", lattice_cell(3.165, {{0, 0, 0}, {.5, .5, .5}}, {0, 0}, 4, 0.08, hea, 7));
    for (auto &[name, c] : cells) {
      std::vector<int> map(c.ntypes + 1, -1);
      for (int t = 1; t <= c.ntypes; t++)
        for (int e = 0; e < (int) m->element_names.size(); e++)
          if (m->element_names[e] == c.species[t - 1]) map[t] = e;
      CHECK(std::count(map.begin() + 1, map.end(), -1) == 0);
      for (const bool tables : {true, false}) {    // radial cubic tables (default) and the radial MLP (spline_mb 0)
        E3 mixed;
        E3F fp32;
        if (!tables) mixed.spline_mb = fp32.spline_mb = 0;
        mixed.set_model(m, map);
        mixed.setup();
        fp32.set_model(m, map);
        fp32.setup();
        const Result a = run_csr(mixed, c.n, c.pos, c.type, c.ntypes, c.cell.data(), c.offset, c.j.data(), c.rij.data());
        const Result b = run_csr(fp32, c.n, c.pos, c.type, c.ntypes, c.cell.data(), c.offset, c.j.data(), c.rij.data());
        double fmax = 0;
        for (double v : a.f) fmax = std::max(fmax, std::fabs(v));
        std::printf("%-6s %4d atoms %7zu bonds %-6s fp32 vs mixed: dE/n %.3e  max|dF| %.3e  (max|F| %.2f)\n",
                    name.c_str(), c.n, c.j.size(), tables ? "tables" : "MLP", std::fabs(a.e - b.e) / c.n, maxdf(a, b),
                    fmax);
        CHECK(std::fabs(a.e - b.e) / c.n <= 1e-6);
        CHECK(maxdf(a, b) <= 1e-5);
      }
      // test_fused_stages: the FC folded into the cp_l projections and the layer-1 radial from the tables inside
      // ComputeA1 / ForceL1F (where the tables exist) == the plain fp32 engine, one chunk, chunks of 16 with and
      // without the forward store, energy only
      {
        E3F plain, fold;
        fold.fc_fold = 1;
        fold.fuse_radial = 1;
        double dev_gb = 0.0;
#ifdef KOKKOS_ENABLE_CUDA
        size_t fr = 0, tot = 0;
        if (cudaMemGetInfo(&fr, &tot) == cudaSuccess) dev_gb = (double) tot / 1e9;
#endif
        // tables for all 625 HEA25 pairs (9.4 GB) where the device has room; auto otherwise
        const bool want_tables = c.ntypes <= 3 || dev_gb >= 20.0;
        fold.spline_mb = (c.ntypes > 3 && want_tables) ? 12000.0 : -1.0;
        plain.set_model(m, map);
        plain.setup();
        fold.set_model(m, map);
        fold.setup();
        CHECK(fold.fold_on[0] && fold.fold_on[1] && fold.fold_on[2]);
        if (want_tables) CHECK(fold.spline_on && fold.l1_radial_fused());
        auto run = [&](E3F &e) {
          return run_csr(e, c.n, c.pos, c.type, c.ntypes, c.cell.data(), c.offset, c.j.data(), c.rij.data());
        };
        plain.chunksize = fold.chunksize = 0;
        const Result a = run(plain), b = run(fold);
        fold.chunksize = 16;
        fold.store_fwd = 1;
        const Result s = run(fold);
        const int stored = fold.fwd_stored;
        fold.store_fwd = 0;
        const Result r = run(fold);
        const Result eo = run_csr(fold, c.n, c.pos, c.type, c.ntypes, c.cell.data(), c.offset, c.j.data(),
                                  c.rij.data(), 0.0, true);
        std::printf("%-6s fused (radial from tables %d) vs plain: dE/n %.3e max|dF| %.3e; chunk 16 store %d: dE/n "
                    "%.3e max|dF| %.3e; recompute: dE/n %.3e max|dF| %.3e; energy_only dE/n %.3e\n", name.c_str(),
                    (int) fold.l1_radial_fused(),
                    std::fabs(a.e - b.e) / c.n, maxdf(a, b), stored, std::fabs(a.e - s.e) / c.n, maxdf(a, s),
                    std::fabs(a.e - r.e) / c.n, maxdf(a, r), std::fabs(a.e - eo.e) / c.n);
        CHECK(stored == 1);
        for (const Result *x : {&b, &s, &r}) CHECK(std::fabs(a.e - x->e) / c.n <= 1e-6 && maxdf(a, *x) <= 1e-5);
        CHECK(std::fabs(a.e - eo.e) / c.n <= 1e-6);
      }
    }
  }
}
}    // namespace

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  auto m = std::make_shared<grace::GRACE3LModel>();
  m->load(test_model("GRACE_TEST_M3", "GRACE-3L-OMAT-large"));
  check_engine<E3>(m);
  // a second engine on the same model: the reduce inputs are grouped once (the shared model is not permuted again)
  std::vector<std::vector<int>> ci;
  for (const auto &r : m->reduce)
    for (const auto &ins : r.instr) ci.push_back(ins.collect_ind);
  CHECK(m->reduce_inputs_grouped);
  check_engine<E3F>(m);
  size_t k = 0;
  for (const auto &r : m->reduce)
    for (const auto &ins : r.instr) CHECK(ins.collect_ind == ci[k++]);
  std::printf("kk 3l setup ok\n");
  smoke(m);
  std::printf("kk 3l smoke ok\n");
  return 0;
}
