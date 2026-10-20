// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Helpers of the CSR-image engine tests (test_kk_csr, test_kk_csr_3l): fixture cells, LAMMPS GPU references, one
// engine run on CSR bonds with ghost images (and the engine's exchange: 1L none, 2L ImageExchangeKK, 3L the layered
// ImageExchange3KK), comparison with tolerances. Expects Dev and T (traits) defined by the including test.
#pragma once

#include "grace/kk/csr_images.h"
#include "grace_cnpy/cnpy.h"
#include "../test_util.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace kkt {
struct Cell {
  int n = 0;
  std::vector<int> type, sys;    // LAMMPS types: W = 1, Mo = 2 (fixtures' specorder)
  std::vector<double> pos, cell, rij;
  std::vector<int64_t> offset;
  std::vector<int> j;
  double volume = 0;
};
Cell load_cell(const std::string &path)
{
  auto npz = grace::cnpy::npz_load(path);
  Cell c;
  const auto &num = npz["numbers"];
  c.n = (int) num.shape[0];
  for (int a = 0; a < c.n; a++) c.type.push_back(num.data<int>()[a] == 74 ? 1 : 2);
  c.sys.assign(c.n, 0);
  c.pos.assign(npz["positions"].data<double>(), npz["positions"].data<double>() + 3 * c.n);
  c.cell.assign(npz["cell"].data<double>(), npz["cell"].data<double>() + 9);
  const auto &off = npz["offset"];
  c.offset.assign(off.data<int64_t>(), off.data<int64_t>() + off.shape[0]);
  const auto &jj = npz["j"];
  c.j.assign(jj.data<int>(), jj.data<int>() + jj.shape[0]);
  const auto &r = npz["rij"];
  c.rij.assign(r.data<double>(), r.data<double>() + r.num_vals);
  const double *h = c.cell.data();
  c.volume = std::fabs(h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                       h[2] * (h[3] * h[7] - h[4] * h[6]));
  return c;
}
Cell concat(const Cell &a, const Cell &b)
{
  Cell c = a;
  c.n = a.n + b.n;
  c.type.insert(c.type.end(), b.type.begin(), b.type.end());
  for (int i = 0; i < b.n; i++) c.sys.push_back(1);
  c.pos.insert(c.pos.end(), b.pos.begin(), b.pos.end());
  c.cell.insert(c.cell.end(), b.cell.begin(), b.cell.end());
  for (int i = 1; i <= b.n; i++) c.offset.push_back(a.offset.back() + b.offset[i]);
  for (int x : b.j) c.j.push_back(x + a.n);
  c.rij.insert(c.rij.end(), b.rij.begin(), b.rij.end());
  return c;
}
struct Ref {
  std::vector<double> f, pe;
  double etot = 0, p[6] = {0, 0, 0, 0, 0, 0};
};
Ref load_ref(const std::string &stem, int n)
{
  Ref r;
  r.f.assign(3 * n, 0); r.pe.assign(n, 0);
  std::ifstream d(stem + ".dump");
  CHECK(d.good());
  std::string line;
  for (int k = 0; k < 9; k++) std::getline(d, line);
  for (int a = 0; a < n; a++) {
    int id; double fx, fy, fz, pe;
    d >> id >> fx >> fy >> fz >> pe;
    r.f[3 * (id - 1)] = fx; r.f[3 * (id - 1) + 1] = fy; r.f[3 * (id - 1) + 2] = fz; r.pe[id - 1] = pe;
  }
  std::ifstream t(stem + ".thermo");
  CHECK(t.good());
  double step;
  t >> step >> r.etot >> r.p[0] >> r.p[1] >> r.p[2] >> r.p[3] >> r.p[4] >> r.p[5];
  return r;
}
std::vector<int> type2mu_for(const std::vector<std::string> &names)
{
  std::vector<int> t2m(3, -1);
  for (int e = 0; e < (int) names.size(); e++) {
    if (names[e] == "W") t2m[1] = e;
    if (names[e] == "Mo") t2m[2] = e;
  }
  CHECK(t2m[1] >= 0 && t2m[2] >= 0);
  return t2m;
}

struct Result {
  double e = 0, vir[6] = {0, 0, 0, 0, 0, 0}, vir_atom[6] = {0, 0, 0, 0, 0, 0}, dedl = 0;
  std::vector<double> f, eatom;
  std::vector<double> sys_e, sys_vir;    // batch: per system energy, virial [nsys][6]
  int nimages = 0;
};

// two-group coupling (grace/coupling.h): equation, lambda, group bits (1, 2) per local atom
struct Couple {
  int eq = 0;
  double lambda = 0.0;
  std::vector<int> mask;
};
// xblock > 0: images per block of the owner <-> image exchange (default: the exchange's own)
// vglobal false: no global virial (vflag_global 0; per-atom virials still on)
// sec: when given, the seconds of the engine run alone (fenced)
template<class Eng> Result run(Eng &eng, const Cell &c, int nsys = 1, int xblock = 0, bool vglobal = true,
                               const Couple *cp = nullptr, double *sec = nullptr)
{
  grace::kk::CsrImagesKK<Dev, T> img;
  img.build(c.n, c.pos.data(), c.type.data(), c.cell.data(), c.offset.data(), c.j.data(), c.rij.data(),
            nsys > 1 ? c.sys.data() : nullptr);
  typename T::t_kkacc_1d_3 f("f", img.nall);
  typename T::t_kkacc_1d ea("ea", img.nall);
  typename T::t_kkacc_1d_6 va("va", img.nall);
  typename Eng::RunInput in;
  in.x = img.x; in.f = f; in.type = img.type; in.ilist = img.ilist; in.numneigh = img.numneigh;
  in.neighbors = img.neighbors; in.inum = c.n; in.nlocal = c.n; in.ntypes = 2;
  in.eflag_global = 1; in.eflag_atom = 1; in.vflag_global = vglobal ? 1 : 0; in.vflag_atom = 1;
  in.eatom = ea; in.vatom = va;
  typename T::t_int_1d mask;
  if (cp) {    // images carry their owner's groups
    mask = typename T::t_int_1d("mask", img.nall);
    auto hm = Kokkos::create_mirror_view(mask);
    auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), img.owner);
    for (int a = 0; a < img.nall; a++) hm(a) = cp->mask[a < c.n ? a : ho(a - c.n)];
    Kokkos::deep_copy(mask, hm);
    in.couple_eq = cp->eq; in.couple_lambda = cp->lambda; in.couple_bit1 = 1; in.couple_bit2 = 2; in.mask = mask;
  }
  typename Eng::RunOutput out;
  Kokkos::fence();
  const auto t0 = std::chrono::steady_clock::now();
  if constexpr (requires { typename Eng::Exchange3; }) {
    grace::kk::ImageExchange3KK<Eng, decltype(img)> ex;
    ex.eng = &eng; ex.img = &img;
    if (xblock > 0) ex.block = xblock;
    in.nall = img.nall; in.ex = &ex;
    out = eng.run(in);
  } else if constexpr (requires { typename Eng::Exchange; }) {
    grace::kk::ImageExchangeKK<Eng, decltype(img)> ex;
    ex.eng = &eng; ex.img = &img;
    if (xblock > 0) ex.block = xblock;
    in.nall = img.nall; in.exchange = &ex;
    out = eng.run(in);
  } else {
    out = eng.run(in);
  }
  Kokkos::fence();
  if (sec) *sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  img.fold(f);
  img.fold(va);
  Result r;
  r.nimages = img.nall - img.nlocal;
  r.e = out.eng_vdwl;
  r.dedl = out.dedl;
  for (int k = 0; k < 6; k++) r.vir[k] = out.virial[k];
  auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
  auto he = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ea);
  auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), va);
  for (int a = 0; a < c.n; a++) {
    for (int k = 0; k < 3; k++) r.f.push_back(hf(a, k));
    r.eatom.push_back(he(a));
    for (int k = 0; k < 6; k++) r.vir_atom[k] += hv(a, k);
  }
  if (nsys > 1) {
    Kokkos::View<int *, Dev> sid("sid", c.n);
    auto hs = Kokkos::create_mirror_view(sid);
    for (int a = 0; a < c.n; a++) hs(a) = c.sys[a];
    Kokkos::deep_copy(sid, hs);
    grace::kk::BatchAccKK<Dev> acc;
    acc.sum(c.n, nsys, ea, va, sid);
    auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), acc.acc);
    for (int s = 0; s < nsys; s++) {
      r.sys_e.push_back(ha(s, 0));
      for (int k = 0; k < 6; k++) r.sys_vir.push_back(ha(s, 1 + k));
    }
  }
  return r;
}

// Tol: energy per atom, per-atom energy, force, pressure (relative). fp64 engines agree with the LAMMPS styles to
// ~1e-14 (same kernels, summation order only); the fused (fp32) ones differ from run to run at the fp32 level (the
// per-atom energies; their sum is fp64).
struct Tol {
  double e, ei, f, p;
};
constexpr Tol tol_fp64 = {1e-10, 1e-10, 1e-10, 1e-10}, tol_fp32 = {2e-6, 1e-5, 2e-5, 1e-5};
// fp32 against fp64: per-atom energies to 1e-5 (2L), so the energy per atom of a 2-atom cell (W2tri) as well; the
// pressure of W2tri (2 atoms, 9e4 bar) to 1.2e-5 (W54, WMo54: < 8e-6), at fp32 rounding of the bond forces
constexpr Tol tol_fp32_vs_fp64 = {2e-5, 2e-5, 2e-5, 2e-5};
void compare(const std::string &what, const Result &r, const Ref &ref, const Cell &c, const Tol &t)
{
  const double nktv2p = 1.6021765e6;    // eV/A^3 -> bar (LAMMPS units metal)
  double df = 0, dpe = 0, dp = 0, pscale = 1, dva = 0, vscale = 1;
  for (int q = 0; q < 3 * c.n; q++) df = std::max(df, std::fabs(r.f[q] - ref.f[q]));
  for (int a = 0; a < c.n; a++) dpe = std::max(dpe, std::fabs(r.eatom[a] - ref.pe[a]));
  for (int k = 0; k < 6; k++) {
    dp = std::max(dp, std::fabs(r.vir[k] / c.volume * nktv2p - ref.p[k]));
    pscale = std::max(pscale, std::fabs(ref.p[k]));
    dva = std::max(dva, std::fabs(r.vir_atom[k] - r.vir[k]));
    vscale = std::max(vscale, std::fabs(r.vir[k]));
  }
  std::printf("%-16s images %5d  dE/n %.3e  max|de_i| %.3e  max|dF| %.3e  max|dP|/max|P| %.3e  "
              "|sum vatom - vir|/max|vir| %.3e\n", what.c_str(), r.nimages, std::fabs(r.e - ref.etot) / c.n, dpe, df,
              dp / pscale, dva / vscale);
  CHECK(std::fabs(r.e - ref.etot) / c.n <= t.e);
  CHECK(dpe <= t.ei);
  CHECK(df <= t.f);
  CHECK(dp / pscale <= t.p);
  CHECK(dva / vscale <= 1e-9);
}
template<class Eng> void check_engine(Eng &eng, const char *tag, const char *ref, const Tol &t, const std::string &fx,
                                      const std::string &rd)
{
  for (const char *cell : {"W54", "WMo54", "W2tri"}) {
    const Cell c = load_cell(fx + "/" + cell + ".npz");
    compare(std::string(cell) + " " + tag, run(eng, c), load_ref(rd + "/" + cell + "_" + ref, c.n), c, t);
  }
  // the owner <-> image exchange in blocks of 7 images (many blocks, the last one partial) == the reference
  if constexpr (requires { typename Eng::Exchange3; } || requires { typename Eng::Exchange; }) {
    const Cell c = load_cell(fx + "/WMo54.npz");
    compare(std::string("WMo54 ") + tag + " xblk", run(eng, c, 1, 7), load_ref(rd + "/WMo54_" + ref, c.n), c, t);
  }
  // batch: W54 (cubic) and W2tri (triclinic, other cell) in one run == each alone: each system's own cell
  const Cell a = load_cell(fx + "/W54.npz"), b = load_cell(fx + "/W2tri.npz");
  const Result ra = run(eng, a), rb = run(eng, b), rab = run(eng, concat(a, b), 2);
  double de = std::max(std::fabs(rab.sys_e[0] - ra.e) / a.n, std::fabs(rab.sys_e[1] - rb.e) / b.n), df = 0, dv = 0;
  for (int q = 0; q < 3 * a.n; q++) df = std::max(df, std::fabs(rab.f[q] - ra.f[q]));
  for (int q = 0; q < 3 * b.n; q++) df = std::max(df, std::fabs(rab.f[3 * a.n + q] - rb.f[q]));
  for (int k = 0; k < 6; k++)
    dv = std::max({dv, std::fabs(rab.sys_vir[k] - ra.vir[k]) / a.n, std::fabs(rab.sys_vir[6 + k] - rb.vir[k]) / b.n});
  std::printf("%-16s batch W54+W2tri vs alone: dE/n %.3e  max|dF| %.3e  max|dvir|/n %.3e\n", tag, de, df, dv);
  CHECK(de <= t.e && df <= t.f && dv <= 10 * t.e);
  // isolated atom: no bonds, finite energy, zero force
  Cell iso;
  iso.n = 1; iso.type = {1}; iso.sys = {0}; iso.pos = {0, 0, 0}; iso.cell = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  iso.offset = {0, 0}; iso.volume = 8000;
  const Result ri = run(eng, iso);
  CHECK(std::isfinite(ri.e) && ri.f[0] == 0.0 && ri.f[1] == 0.0 && ri.f[2] == 0.0);
}
// the fused (fp32) engine against the frozen fp64 references of the same model (refs <cell>_<ref>)
template<class Eng> void check_vs_fp64(Eng &eng, const char *tag, const char *ref, const std::string &fx,
                                       const std::string &rd)
{
  for (const char *cell : {"W54", "WMo54", "W2tri"}) {
    const Cell c = load_cell(fx + "/" + cell + ".npz");
    compare(std::string(cell) + " " + tag + " vs fp64", run(eng, c), load_ref(rd + "/" + cell + "_" + ref, c.n), c,
            tol_fp32_vs_fp64);
  }
}
// the total energy is the fp64 sum of the per-atom energies, also in one large run: W54 doubled to 27648 atoms (an
// fp32 sum is off by ~1e-3 eV there, as large as the energy change of a Monte Carlo swap)
template<class Eng> void check_energy_sum(Eng &eng, const char *tag, const std::string &fx)
{
  Cell big = load_cell(fx + "/W54.npz");
  for (int k = 0; k < 9; k++) big = concat(big, big);
  const Result r = run(eng, big);
  double s = 0;
  for (double e : r.eatom) s += e;
  std::printf("%-16s %d atoms: E %.6f  |E - sum e_i| %.3e eV\n", tag, big.n, r.e, std::fabs(r.e - s));
  CHECK(std::fabs(r.e - s) <= 1e-6);
}
}    // namespace kkt
