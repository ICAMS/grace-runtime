// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The Kokkos engines on CSR bonds (grace/kk/csr_images.h: ghost images, image exchange, batch sums) against the
// LAMMPS GPU references of the same styles (tests/fixtures/ref_gpu from ref_lammps_gpu.sh): energy, per-atom energy,
// forces, pressure within fp32 parity (GPU runs are not reproducible bit for bit); the per-atom virial sums to the
// global one; two cells in one batch give each cell's result.
//   env GRACE_TEST_FIXTURES (default tests/fixtures), GRACE_TEST_REF_DIR (default tests/fixtures/ref_gpu)
#include "grace/kk/csr_images.h"
#include "grace/kk/grace_1l_fused_kk_impl.h"
#include "grace/kk/grace_1l_kk_impl.h"
#include "grace/kk/grace_2l_fused_kk_impl.h"
#include "grace/kk/grace_2l_kk_impl.h"
#include "grace/kk/traits.h"
#include "grace_cnpy/cnpy.h"
#include "../test_util.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

using Dev = Kokkos::DefaultExecutionSpace;
using T = grace::kk::DefaultTraits<Dev>;
#include "kk_csr_util.h"
using E1 = grace::kk::Grace1LKK<Dev, double, double, T>;
using E1F = grace::kk::Grace1LFusedKK<Dev, T>;
using E2 = grace::kk::Grace2LKK<Dev, double, double, T>;
using E2F = grace::kk::Grace2LFusedKK<Dev, T>;
template class grace::kk::Grace1LKK<Dev, double, double, T>;
template class grace::kk::Grace1LFusedKK<Dev, T>;
template class grace::kk::Grace2LKK<Dev, double, double, T>;
template class grace::kk::Grace2LFusedKK<Dev, T>;

namespace {
using namespace kkt;

// the 2L fused engine with rho of every chunk's forward (stage_reduce_I1) kept on the host and compared with d_rho when
// the layer-1 backward of the chunk starts (stage_zero_L1_adj: after the layer-1 recompute of a multi-chunk run)
struct E2FRho : E2F {
  std::vector<float> fwd;    // [inum][rho_n_out]
  double maxdiff = 0;
  int checked = 0;
  size_t row0() const { return (size_t) this->chunk_offset * this->d_rho.extent(1); }
  std::vector<float> rows()    // the chunk's rho
  {
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), this->d_rho);
    std::vector<float> r;
    for (int i = 0; i < this->chunk_size; i++)
      for (int k = 0; k < (int) h.extent(1); k++) r.push_back(h(i, k));
    return r;
  }
  void stage_reduce_I1() override
  {
    E2F::stage_reduce_I1();
    const std::vector<float> r = rows();
    fwd.resize(std::max(fwd.size(), (size_t) this->inum * this->d_rho.extent(1)));
    std::copy(r.begin(), r.end(), fwd.begin() + row0());
  }
  void stage_zero_L1_adj() override
  {
    const std::vector<float> r = rows();
    for (size_t q = 0; q < r.size(); q++)
      maxdiff = std::max(maxdiff, (double) std::fabs(r[q] - fwd[row0() + q]));
    checked++;
    E2F::stage_zero_L1_adj();
  }
};

// engine guards for callers without LAMMPS: run() before setup() and nall < nlocal are errors; an engine
// copy outlives the original (shared cuBLAS handle); a second model on one engine with several chunks gives the
// result of a fresh engine; the layer-1 recompute of several chunks without the layer-1 store gives the forward's rho
void check_guards(const std::shared_ptr<grace::GRACE2LModel> &m2, const std::shared_ptr<grace::GRACE2LModel> &m2b,
                  const std::string &fx, const std::string &rd)
{
  const Cell c = load_cell(fx + "/WMo54.npz");
  const Ref ref = load_ref(rd + "/WMo54_2l_fused", c.n);
  {
    E2F e;
    e.set_model(m2, type2mu_for(m2->element_names));
    typename E2F::RunInput in;
    bool threw = false;
    try { e.run(in); } catch (const grace::Error &) { threw = true; }
    std::printf("guard run before setup: %s\n", threw ? "throws" : "NO ERROR");
    CHECK(threw);
  }
  {
    E2F e;
    e.set_model(m2, type2mu_for(m2->element_names));
    e.setup();
    grace::kk::CsrImagesKK<Dev, T> img;
    img.build(c.n, c.pos.data(), c.type.data(), c.cell.data(), c.offset.data(), c.j.data(), c.rij.data());
    typename T::t_kkacc_1d_3 f("f", img.nall);
    typename E2F::RunInput in;
    in.x = img.x; in.f = f; in.type = img.type; in.ilist = img.ilist; in.numneigh = img.numneigh;
    in.neighbors = img.neighbors; in.inum = c.n; in.nlocal = c.n; in.ntypes = 2; in.eflag_global = 1;
    in.nall = c.n - 1;
    bool threw = false;
    try { e.run(in); } catch (const grace::Error &) { threw = true; }
    std::printf("guard nall < nlocal: %s\n", threw ? "throws" : "NO ERROR");
    CHECK(threw);
  }
  {
    std::optional<E2F> orig;
    orig.emplace();
    orig->set_model(m2, type2mu_for(m2->element_names));
    orig->setup();
    E2F copy(*orig);
    orig.reset();    // the original goes first; the copy keeps a valid cuBLAS handle
    compare("WMo54 2L copy", run(copy, c), ref, c, tol_fp32_2l);
  }
  {
    E2F a, b;
    a.chunksize = b.chunksize = 16;    // 4 chunks of WMo54: the layer-1 store per chunk
    a.set_model(m2b, type2mu_for(m2b->element_names));
    a.setup();
    run(a, c);
    a.set_model(m2, type2mu_for(m2->element_names));
    a.setup();
    b.set_model(m2, type2mu_for(m2->element_names));
    b.setup();
    const Result ra = run(a, c), rb = run(b, c);
    double df = 0;
    for (int q = 0; q < 3 * c.n; q++) df = std::max(df, std::fabs(ra.f[q] - rb.f[q]));
    std::printf("second model, 4 chunks vs fresh engine: dE/n %.3e  max|dF| %.3e\n", std::fabs(ra.e - rb.e) / c.n, df);
    CHECK(std::fabs(ra.e - rb.e) / c.n <= tol_fp32.e && df <= tol_fp32.f);
    compare("WMo54 2L 4 chunks", rb, ref, c, tol_fp32_2l);
  }
  {   // 4 chunks without the layer-1 store: the recomputed rho of each chunk is the one of its forward, bit for bit
    E2FRho a;
    a.chunksize = 16;
    a.store_l1 = 0;
    a.set_model(m2, type2mu_for(m2->element_names));
    a.setup();
    run(a, c);
    std::printf("4 chunks, layer 1 recomputed: %d chunks, max|rho - forward rho| %.3e\n", a.checked, a.maxdiff);
    CHECK(a.checked == 4 && a.maxdiff == 0.0);
  }
}
}    // namespace

// GraceFusedKK::compact_bonds with an unmapped (type2mu -1) neighbour and an unmapped central atom: no bonds to or from
// them. The cutoff table is a window of a larger one with long cutoffs around it, so that a read at element -1 (before:
// bc(mui, map(type(j))) ahead of any check) keeps the bond instead of reading outside the allocation.
void check_compact_bonds_unmapped()
{
  Kokkos::View<double *[3], Kokkos::LayoutRight, Dev> x("x", 3);
  Kokkos::View<int *, Dev> ty("type", 3), map("map", 3), num("num", 3), ilist("ilist", 2), mask("mask", 3), cnum;
  Kokkos::View<int **, Dev> neigh("neigh", 3, 2), cnb;
  Kokkos::View<double **, Kokkos::LayoutRight, Dev> big("cutoffs", 3, 3);
  Kokkos::deep_copy(big, 100.0);
  const auto bc = Kokkos::subview(big, std::make_pair(1, 2), std::make_pair(1, 2));    // model element 0: 5 A
  Kokkos::deep_copy(bc, 5.0);
  auto hx = Kokkos::create_mirror_view(x);
  auto hty = Kokkos::create_mirror_view(ty), hmap = Kokkos::create_mirror_view(map), hnum = Kokkos::create_mirror_view(num);
  auto hil = Kokkos::create_mirror_view(ilist);
  auto hn = Kokkos::create_mirror_view(neigh);
  // atoms 0 and 2 of type 1 (element 0), atom 1 of type 2 (unmapped), 1 A apart; central atoms 0 and 1
  const double pos[3][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  for (int a = 0; a < 3; a++)
    for (int k = 0; k < 3; k++) hx(a, k) = pos[a][k];
  hty(0) = 1; hty(1) = 2; hty(2) = 1;
  hmap(0) = -1; hmap(1) = 0; hmap(2) = -1;
  hil(0) = 0; hil(1) = 1;
  hnum(0) = 2; hn(0, 0) = 1; hn(0, 1) = 2;
  hnum(1) = 2; hn(1, 0) = 0; hn(1, 1) = 2;
  hnum(2) = 0;
  Kokkos::deep_copy(x, hx); Kokkos::deep_copy(ty, hty); Kokkos::deep_copy(map, hmap); Kokkos::deep_copy(num, hnum);
  Kokkos::deep_copy(ilist, hil); Kokkos::deep_copy(neigh, hn);
  grace::kk::GraceFusedKK::compact_bonds<Dev, T>(2, 2, x, ty, map, neigh, num, ilist, bc, 0, 1, 2, 0.0, mask, cnb, cnum);
  const auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cnum);
  const auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cnb);
  std::printf("compact_bonds, unmapped neighbour / central atom: bonds %d (want 1, to atom 2: %d), %d (want 0)\n",
              hc(0), hb(0, 0), hc(1));
  CHECK(hc(0) == 1 && hb(0, 0) == 2 && hc(1) == 0);
}

// a fused engine with the number of bonds its last run compacted (GPU; the host backends keep no compact list: 0)
template<class Eng> struct WithBonds : Eng {
  long bonds() const
  {
    if (!this->d_cnum.data()) return 0;
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), this->d_cnum);
    long s = 0;
    for (int ii = 0; ii < this->inum; ii++) s += h(ii);
    return s;
  }
};

// LAMMPS pair_style hybrid with a NULL type: type 2 (Mo) unmapped (type2mu -1), the type-1 (W) atoms central. No bonds
// to or from an unmapped atom (grace/type_map.h): the W atoms see what they see in the cell without the Mo atoms, and
// the Mo atoms get no force, and the GPU bond compaction keeps the bonds of W alone (it read the cutoff table at element
// -1 before and kept bonds to Mo, which the bond kernels then skipped).
template<class Eng, class M> void check_unmapped_neighbours(const std::shared_ptr<M> &m, const char *name,
                                                            const std::string &fx)
{
  const Cell c = load_cell(fx + "/WMo54.npz");
  const Cell a = type1_first(c, true), w = type1_first(c, false);
  CHECK(w.n > 0 && w.n < a.n);
  std::vector<int> t2m = type2mu_for(m->element_names);
  t2m[2] = -1;
  WithBonds<Eng> ea, ew;
  ea.set_model(m, t2m);
  ea.setup();
  ew.set_model(m, type2mu_for(m->element_names));
  ew.setup();
  const Result ra = run(ea, a, 1, 0, true, nullptr, nullptr, w.n), rw = run(ew, w);
  double df = 0, fmo = 0;
  for (int q = 0; q < 3 * w.n; q++) df = std::max(df, std::fabs(ra.f[q] - rw.f[q]));
  for (int q = 3 * w.n; q < 3 * a.n; q++) fmo = std::max(fmo, std::fabs(ra.f[q]));
  std::printf("%s Mo unmapped (%d W central, %d Mo): vs W alone dE/n %.3e max|dF| %.3e, max|F(Mo)| %.3e, compact "
              "bonds %ld (W alone %ld)\n", name, w.n, a.n - w.n, std::fabs(ra.e - rw.e) / w.n, df, fmo, ea.bonds(),
              ew.bonds());
  CHECK(std::fabs(ra.e - rw.e) / w.n <= 2e-6 && df <= 2e-5 && fmo == 0.0 && ea.bonds() == ew.bonds());
}

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  const std::string fx = env_or("GRACE_TEST_FIXTURES", "tests/fixtures");
  const std::string rd = env_or("GRACE_TEST_REF_DIR", "tests/fixtures/ref_gpu");
  auto m1 = std::make_shared<grace::GRACE1LModel>();
  m1->load(test_model("GRACE_TEST_M1", "GRACE-1L-OMAT-large-base"));
  auto m2 = std::make_shared<grace::GRACE2LModel>();
  m2->load(test_model("GRACE_TEST_M2", "GRACE-2L-SMAX-OMAT-medium"));
  {
    E1 e; e.set_model(m1, type2mu_for(m1->element_names)); e.setup();
    check_engine(e, "1L kk fp64", "1l_kk", tol_fp64, fx, rd);
  }
  {
    E1F e; e.set_model(m1, type2mu_for(m1->element_names)); e.setup();
    check_engine(e, "1L fused", "1l_fused", tol_fp32, fx, rd);
    check_vs_fp64(e, "1L fused", "1l_kk", fx, rd);
    check_energy_sum(e, "1L fused", fx);
  }
  {
    E2 e; e.set_model(m2, type2mu_for(m2->element_names)); e.setup();
    check_engine(e, "2L kk fp64", "2l_kk", tol_fp64, fx, rd);
  }
  {
    E2F e; e.set_model(m2, type2mu_for(m2->element_names)); e.setup();
    check_engine(e, "2L fused", "2l_fused", tol_fp32_2l, fx, rd);
    check_vs_fp64(e, "2L fused", "2l_kk", fx, rd);
    check_energy_sum(e, "2L fused", fx);
  }
  auto m2b = std::make_shared<grace::GRACE2LModel>();
  m2b->load(test_model("GRACE_TEST_UQ2", "GRACE-2L-OMAT-medium-ft-E"));
  check_guards(m2, m2b, fx, rd);
  check_empty_list<E1F>(m1, "1L fused", fx);
  check_empty_list<E2F>(m2, "2L fused", fx);
  check_compact_bonds_unmapped();
  check_unmapped_neighbours<E1F>(m1, "1L fused", fx);
  check_unmapped_neighbours<E2F>(m2, "2L fused", fx);
  std::printf("kk csr ok\n");
  return 0;
}
