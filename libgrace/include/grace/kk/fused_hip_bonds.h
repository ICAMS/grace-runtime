// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// GPU helpers of the fused GRACE-1L / GRACE-2L bond kernels (grace/1l/fused/kk, grace/2l/fused/kk), developed on HIP
// and used on CUDA as well; the matrix-core kernels are CDNA only.
//
// compact_bonds: per run, the neighbours of each local atom that are bonds (within the bond cutoff; for a coupled
// pair within the shifted cutoff), in neighbour-list order: cnb(ii, k), k < cnum(ii), ii = position in the ilist.
// The neighbour list carries the skin (bcc W, 6 A cutoff, 1 A skin: 88 entries for 58 bonds); the bond kernels
// iterate over the bonds only. The list is exact (the kernels' fp64 test), so the GPU-only kernels below and the
// matrix-core layer-2 kernels take every listed bond without testing the cutoff again.
// BondList: the bonds of an atom as the shared (CUDA / HIP / host) bond kernels read them: on a GPU the compact list,
// on the host backends the neighbour list (those kernels keep their cutoff test).
//
// bond_bwd_wave: the bond backward of the radial-angular basis A[i, n, lm] = sum_j z[mu_j, n] R_nl(x_ij) Y_lm(r_ij)
// (GRACE-1L g1lf:bond_bwd, GRACE-2L g2lf:L1_bwd), lmax 4: one team of 64 threads per atom, thread = bond (rounds of
// 64 bonds). The thread keeps its bond's Y and dY / dr in registers and loops over all radial channels; the force of
// the bond is summed over the channels in registers. Compared with the (bond, channel) threads of the team-of-256
// kernels: no team barrier and no shared-memory sum over the channels per round of TEAM / NR bonds, and the
// geometry of a bond is computed once by its own thread. Shared memory: only the atom's dE/dA rows.
//   per (bond, channel): S_l = sum_{lm in l} a_lm Y_lm, T_l = sum_{lm in l} a_lm G_lm   (a = dE/dA[n, lm])
//                        dE/dx' += z sum_l dR_l S_l,   g += z sum_l R_l T_l
//   per bond:            f = (dE/dx' dx'/dr) rhat + (g - rhat (rhat . g)) / r      (as the team kernels)
// Precision as the team kernels: geometry in fp64, network / radial / force sums in fp32, accumulation in KKA / fp64.
#pragma once

#include "grace/kk/fused_common.h"
#include "grace/kk/support.h"
#include "grace/kk/kk_mfma.h"    // GRACE_KK_GEMM_MFMA (CDNA build), mfma_switch, the matrix-core / wavefront helpers
#include "grace/coupling.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace grace {
namespace kk {
namespace GraceFusedKK {

constexpr int FBW_TEAM = 64;     // threads (bonds) per team = one wavefront on CDNA, two on RDNA (team-level code only)
constexpr int FBW_MINB = 2;      // launch bounds (HIP-Clang: minimum waves per SIMD)

// GRACE_FUSED_MFMA and GRACE_FUSED_WAVE_BWD, read once per process (env_switch, support.h): 1, 0, or -1 when unset
inline int fused_mfma_env()
{
  static const int v = env_switch("GRACE_FUSED_MFMA");
  return v;
}
inline int fused_wave_bwd_env()
{
  static const int v = env_switch("GRACE_FUSED_WAVE_BWD");
  return v;
}
// A matrix-core kernel of the HIP 1L / 2L engines (fits: the model is inside its limits): CDNA builds,
// GRACE_FUSED_MFMA=1 or the build's default (gfx90a) unless GRACE_FUSED_MFMA=0
inline KernelChoice fused_mfma(bool fits) { return mfma_switch("GRACE_FUSED_MFMA", fused_mfma_env(), fits); }
// The wave-per-atom layer-1 bond backward (GPU): the engine's default (def) unless GRACE_FUSED_WAVE_BWD=0 / =1
inline KernelChoice fused_wave_bwd(bool fits, bool def = true)
{
  const int e = fused_wave_bwd_env();
  const bool want = e < 0 ? def : e == 1;
  return {want && fits, e == 0  ? std::string("GRACE_FUSED_WAVE_BWD=0")
                        : !want ? std::string("default off for this engine and GPU; GRACE_FUSED_WAVE_BWD=1 enables it")
                        : !fits ? std::string("model outside the limits of the wave-per-atom kernel")
                        : e == 1 ? std::string("GRACE_FUSED_WAVE_BWD=1")
                                 : std::string("default; GRACE_FUSED_WAVE_BWD=0 disables it")};
}
// Launch bounds of the team-of-N bond backward kernels (g1lf:bond_bwd, g2lf:L1_bwd): on CUDA registers for 3 teams per
// SM (80 per thread for N = 256; without the bound 84 with the compact bond list, 2 teams per SM: A100 1L-large
// C-diamond-125k 2.39 -> 2.26 us/atom); HIP as hip_lb_t
template<class D, unsigned N>
using fbwd_lb_t = std::conditional_t<is_cuda_v<D>, Kokkos::LaunchBounds<N, 3>, hip_lb_t<D, N, 2>>;
// Launch a team-of-N bond backward kernel with fbwd_lb_t, on CUDA only where Kokkos admits teams of N with it
// (launch_bounds_fit: the shared memory of 3 teams has to fit an SM; Turing, 64 KB per SM: not with the 28 KB of
// 1L-large), else without the bound
template<class D, int N, class Body>
void fbwd_launch(const char *name, int league, size_t shmem, const Body &body)
{
  const auto pol = Kokkos::TeamPolicy<D, fbwd_lb_t<D, N>>(league, N).set_scratch_size(0, Kokkos::PerTeam(shmem));
  if constexpr (is_cuda_v<D>) {
    if (!launch_bounds_fit(pol, body, N)) {
      static std::once_flag said;
      std::call_once(said, [&] {
        std::fprintf(stderr, "grace kk: %s without launch bounds (the shared memory of 3 teams does not fit an SM, or "
                             "GRACE_LB_FALLBACK=1)\n", name);
      });
      Kokkos::parallel_for(name, Kokkos::TeamPolicy<D>(league, N).set_scratch_size(0, Kokkos::PerTeam(shmem)), body);
      return;
    }
  }
  Kokkos::parallel_for(name, pol, body);
}
// "<kernel> kernel (<why>)" of a matrix-core choice, for the engines' line on stderr
inline std::string fused_mfma_says(const KernelChoice &c)
{
  return std::string(c.on ? "matrix-core (MFMA)" : "vector-ALU") + " kernel (" + c.why + ")";
}

// The bonds of atom i at position ii of the ilist (see the top of the file)
template<class Traits, bool CMP, class NeighV, class NumV, class CnbV, class CnumV>
struct BondList {
  NeighV neigh;
  NumV numneigh;
  CnbV cnb;
  CnumV cnum;
  KOKKOS_INLINE_FUNCTION int count(const int i, const int ii) const { return CMP ? cnum(ii) : numneigh(i); }
  KOKKOS_INLINE_FUNCTION int operator()(const int i, const int ii, const int k) const
  {
    return CMP ? cnb(ii, k) : (neigh(i, k) & Traits::neighmask);
  }
};

template<class DeviceType, class Traits, bool CPL, class XV, class TV, class MapV, class NeighV, class NumV, class IlV,
         class BcV, class MaskV, class CnbV, class CnumV>
void compact_bonds_c(const int inum, const XV &x, const TV &type, const MapV &map, const NeighV &neigh,
                     const NumV &numneigh, const IlV &ilist, const BcV &bc, const int ceq, const int cb1, const int cb2,
                     const double clam, const MaskV &cmask, const CnbV &cnb, const CnumV &cnum)
{
  Kokkos::parallel_for("gfused:compact_bonds", Kokkos::RangePolicy<DeviceType>(0, inum), KOKKOS_LAMBDA(const int ii) {
    const int i = ilist[ii];
    const int mui = map(type(i));
    const auto xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);
    const int jnum = mui < 0 ? 0 : numneigh(i);    // unmapped (type2mu -1): no bonds to or from it (grace/type_map.h)
    int k = 0;
    for (int jj = 0; jj < jnum; jj++) {
      const int j = neigh(i, jj) & Traits::neighmask;
      const int muj = map(type(j));
      if (muj < 0) continue;
      const double dx = x(j, 0) - xi, dy = x(j, 1) - yi, dz = x(j, 2) - zi;
      const double r2 = dx * dx + dy * dy + dz * dz;
      const double rc = bc(mui, muj);
      if (r2 < rc * rc) {
        bool keep = true;
        if (CPL && coupling_cross(cmask(i), cmask(j), cb1, cb2)) {
          const double r = Kokkos::sqrt(r2);
          double rm = r, drm, dlm;
          coupling_shift(ceq, r, rc, clam, rm, drm, dlm);
          keep = rm < rc;
        }
        if (keep) cnb(ii, k++) = j;
      }
    }
    cnum(ii) = k;
  });
}

// cnb / cnum of the run: (re)allocated to inum x maxneigh / inum when smaller (MemKK::realloc_kokkos: uninitialised;
// only the first cnum(ii) entries of a row are read); ceq != 0: two-group coupling
template<class DeviceType, class Traits, class XV, class TV, class MapV, class NeighV, class NumV, class IlV, class BcV,
         class MaskV, class CnbV, class CnumV>
void compact_bonds(const int inum, const int maxneigh, const XV &x, const TV &type, const MapV &map,
                   const NeighV &neigh, const NumV &numneigh, const IlV &ilist, const BcV &bc, const int ceq,
                   const int cb1, const int cb2, const double clam, const MaskV &cmask, CnbV &cnb, CnumV &cnum)
{
  if ((int) cnb.extent(0) < inum || (int) cnb.extent(1) < maxneigh)
    MemKK::realloc_kokkos(cnb, "gfused:cnb", inum, maxneigh > 0 ? maxneigh : 1);
  if ((int) cnum.extent(0) < inum) MemKK::realloc_kokkos(cnum, "gfused:cnum", inum);
  if (ceq)
    compact_bonds_c<DeviceType, Traits, true>(inum, x, type, map, neigh, numneigh, ilist, bc, ceq, cb1, cb2, clam,
                                              cmask, cnb, cnum);
  else
    compact_bonds_c<DeviceType, Traits, false>(inum, x, type, map, neigh, numneigh, ilist, bc, ceq, cb1, cb2, clam,
                                               cmask, cnb, cnum);
}

// bond_bwd_wave: lmax 4 (Y, G in registers), the atom's dE/dA rows in shared memory
constexpr int FBW_LMAX = 4;
inline size_t bond_bwd_wave_shmem(int NR)
{
  return (size_t) NR * fused_lmp((FBW_LMAX + 1) * (FBW_LMAX + 1)) * sizeof(float) + 16;
}
template<class DeviceType> bool bond_bwd_wave_fits(int lmax, int NR)
{
  return lmax == FBW_LMAX && bond_bwd_wave_shmem(NR) <= team_scratch_max<DeviceType>(FBW_TEAM);
}

// nbl: the run's bonds (BondList over compact_bonds), the atom's row at its ilist position (league rank + chunk offset)
template<class DeviceType, class Traits, bool CPL, class AaV, class FV, class VirV, class XV, class TV, class MapV,
         class NblV, class IlV, class BcV, class ZV, class MaskV, class DedlV, class GV, class SplV, class VaV, class CvaV>
void bond_bwd_wave(const char *label, const int cs, const int co, const int NR, const float ascale, const int N,
                   const AaV &Aa, const FV &f, const VirV &vir, const XV &x, const TV &type, const MapV &map,
                   const NblV &nbl, const IlV &ilist, const BcV &bc, const ZV &ztr, const int ceq, const int cb1,
                   const int cb2, const double clam, const MaskV &cmask, const DedlV &dedl, const GV &idx_s,
                   const GV &alm, const GV &blm, const GV &cl, const GV &dl, const SplV &sc, const bool do_v,
                   const bool do_va, const bool do_cva, const VaV &va, const CvaV &cva)
{
  using F = float;
  using KKA = typename Traits::kk_acc_float;
  constexpr int LM = FBW_LMAX, NL = LM + 1, NLM = NL * NL;
  constexpr int LMP = fused_lmp(NLM);    // 28: dE/dA row of a channel as 7 float4
  using team_policy = Kokkos::TeamPolicy<DeviceType, hip_lb_t<DeviceType, FBW_TEAM, FBW_MINB>>;
  using member = typename team_policy::member_type;
  team_policy pol(cs, FBW_TEAM);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(bond_bwd_wave_shmem(NR)));

  Kokkos::parallel_for(label, pol, KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank();
    const int i = ilist[ii + co];
    const int mui = map(type(i));
    const auto xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);    // positions' precision (double)
    F *Aa_s = (F *) t.team_shmem().get_shmem_aligned(NR * LMP * sizeof(F), 16);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NR * LMP), [&](const int q) {
      const int c = q / LMP, lm = q - c * LMP;
      Aa_s[q] = (lm < NLM) ? Aa(ii, c, lm) * ascale : 0.0f;
    });
    t.team_barrier();

    double fi[3] = {0.0, 0.0, 0.0}, v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double dla = 0.0;    // dE/dlambda (coupling)
    const int jnum = nbl.count(i, ii + co);
    for (int jb = 0; jb < jnum; jb += FBW_TEAM) {
      const int b = jb + t.team_rank();
      int j = 0, mu = -1;
      F gx0 = 0.0f, gdr = 0.0f, grinv = 0.0f, rx = 0.0f, ry = 0.0f, rz = 0.0f, gr = 0.0f, gdl = 0.0f;
      if (b < jnum) {    // a bond of the compact list (no cutoff test)
        j = nbl(i, ii + co, b);
        const double dx = x(j, 0) - xi, dy = x(j, 1) - yi, dz = x(j, 2) - zi;    // bond geometry in fp64
        mu = map(type(j));
        const double rc = bc(mui, mu);
        const double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz), rinv = 1.0 / r;
        double rm = r, drm = 1.0, dlm = 0.0;    // coupled bond (grace/coupling.h): r', dr'/dr, dr'/dlambda
        if (CPL && coupling_cross(cmask(i), cmask(j), cb1, cb2)) coupling_shift(ceq, r, rc, clam, rm, drm, dlm);
        // radial part at r' (times dr'/dr), angular part and virial at the true r (as the team kernels)
        gx0 = F(rm / rc); gdr = F(drm / rc); grinv = F(rinv);
        rx = F(dx * rinv); ry = F(dy * rinv); rz = F(dz * rinv);
        gr = F(r);
        if (CPL) gdl = F(dlm / rc);
      }
      F Y[NLM], G[3 * NLM];
      real_sph_grad<1, 3, LM>(LM, rx, ry, rz, idx_s, alm, blm, cl, dl, Y, G);    // Y, G in registers
      if (mu < 0) {    // no bond: zero basis, so the channel loop adds exact zeros (radial at x = 0 is finite)
#pragma unroll
        for (int k = 0; k < NLM; k++) { Y[k] = 0.0f; G[3 * k] = G[3 * k + 1] = G[3 * k + 2] = 0.0f; }
      }
      const int muz = mu < 0 ? 0 : mu;
      F RAD = 0.0f, GX = 0.0f, GY = 0.0f, GZ = 0.0f;
#pragma unroll 1
      for (int n = 0; n < NR; n++) {
        F R[NL], dR[NL];
#pragma unroll
        for (int l = 0; l < NL; l++) spline_vd(sc, N, gx0, n * NL + l, R[l], dR[l]);
        const F z = ztr(muz, n);
        const GF4 *a4 = reinterpret_cast<const GF4 *>(Aa_s + n * LMP);
        F aa[LMP];
#pragma unroll
        for (int q = 0; q < LMP / 4; q++) {
          const GF4 w = a4[q];
          aa[4 * q] = w.x; aa[4 * q + 1] = w.y; aa[4 * q + 2] = w.z; aa[4 * q + 3] = w.w;
        }
        F rad = 0.0f, gx = 0.0f, gy = 0.0f, gz = 0.0f;
        if (is_hip_v<DeviceType>) {    // HIP-Clang unrolls the nested loops (the flat ones were 1-2 % slower)
#pragma unroll
          for (int l = 0; l < NL; l++) {
            F s = 0.0f, tx = 0.0f, ty = 0.0f, tz = 0.0f;
#pragma unroll
            for (int lm = l * l; lm < (l + 1) * (l + 1); lm++) {
              s += aa[lm] * Y[lm];
              tx += aa[lm] * G[3 * lm]; ty += aa[lm] * G[3 * lm + 1]; tz += aa[lm] * G[3 * lm + 2];
            }
            rad += s * dR[l]; gx += tx * R[l]; gy += ty * R[l]; gz += tz * R[l];
          }
        } else {    // one flat lm loop: nvcc keeps Y, G and aa in registers (nested: 512 B of local memory per thread)
          F S[NL], TX[NL], TY[NL], TZ[NL];
#pragma unroll
          for (int l = 0; l < NL; l++) S[l] = TX[l] = TY[l] = TZ[l] = 0.0f;
#pragma unroll
          for (int lm = 0; lm < NLM; lm++) {
            const int l = l_of_y(lm);
            S[l] += aa[lm] * Y[lm];
            TX[l] += aa[lm] * G[3 * lm]; TY[l] += aa[lm] * G[3 * lm + 1]; TZ[l] += aa[lm] * G[3 * lm + 2];
          }
#pragma unroll
          for (int l = 0; l < NL; l++) { rad += S[l] * dR[l]; gx += TX[l] * R[l]; gy += TY[l] * R[l]; gz += TZ[l] * R[l]; }
        }
        RAD += rad * z; GX += gx * z; GY += gy * z; GZ += gz * z;
      }
      if (mu >= 0) {
        F rad = RAD;
        if (CPL) dla += rad * gdl;    // dE/dlambda: dE/dx' dx'/dlambda
        rad *= gdr;                   // d/dr = d/dx' dx'/dr
        const F rg = rx * GX + ry * GY + rz * GZ;
        const F fx = rad * rx + (GX - rg * rx) * grinv;
        const F fy = rad * ry + (GY - rg * ry) * grinv;
        const F fz = rad * rz + (GZ - rg * rz) * grinv;
        fi[0] += (KKA) fx; fi[1] += (KKA) fy; fi[2] += (KKA) fz;
        Kokkos::atomic_add(&f(j, 0), (KKA) -fx);
        Kokkos::atomic_add(&f(j, 1), (KKA) -fy);
        Kokkos::atomic_add(&f(j, 2), (KKA) -fz);
        const double dx = -rx * gr, dy = -ry * gr, dz = -rz * gr;
        if (do_v) {
          v[0] += dx * fx; v[1] += dy * fy; v[2] += dz * fz;
          v[3] += dx * fy; v[4] += dx * fz; v[5] += dy * fz;
        }
        if (do_va || do_cva) tally_vatom(va, cva, do_va, do_cva, i, j, dx, dy, dz, fx, fy, fz);
      }
    }
    // the atom's own force, virial and dE/dlambda: sums over the team, one atomic each
    for (int k = 0; k < 3; k++) {
      double s = fi[k];
      t.team_reduce(Kokkos::Sum<double>(s));
      if (t.team_rank() == 0) Kokkos::atomic_add(&f(i, k), (KKA) s);
    }
    if (do_v)
      for (int k = 0; k < 6; k++) {
        double s = v[k];
        t.team_reduce(Kokkos::Sum<double>(s));
        if (t.team_rank() == 0) Kokkos::atomic_add(&vir(ii + co, k), s);
      }
    if (CPL) {
      double s = dla;
      t.team_reduce(Kokkos::Sum<double>(s));
      if (t.team_rank() == 0) Kokkos::atomic_add(&dedl(), s);
    }
  });
}

// a_fwd_mfma (below): lmax 4, batches of AFM_NB bonds, team of AFM_TEAM = 4 wavefronts, each with up to AFM_MAXP
// (16-channel tile, l) output blocks: the radial width NR fits when ceil(NR / 16) * 5 <= 4 AFM_MAXP (NR <= 48)
constexpr int AFM_LMAX = 4, AFM_NB = 32, AFM_TEAM = 256, AFM_MAXP = 4;
inline size_t a_fwd_mfma_shmem(int NR)
{
  constexpr int NL = AFM_LMAX + 1;
  return (size_t) AFM_NB * (NL * NL + NR * NL + 1) * sizeof(float) + AFM_NB * sizeof(int) + 16;
}
template<class DeviceType> bool a_fwd_mfma_fits(int lmax, int NR)
{
  return lmax == AFM_LMAX && (NR + 15) / 16 * (AFM_LMAX + 1) <= (AFM_TEAM / 64) * AFM_MAXP &&
         a_fwd_mfma_shmem(NR) <= team_scratch_max<DeviceType>(AFM_TEAM);
}

#if GRACE_KK_GEMM_MFMA
// The basis A[i, n, lm] = ascale sum_j z[mu_j, n] R_nl(x_ij) Y_lm(r_ij) (GRACE-1L g1lf:A, GRACE-2L g2lf:L1_A) on the
// CDNA matrix cores, lmax 4: team of 256 per atom, batches of 32 bonds staged as in the team kernels (Y, and
// P = R z per (bond, channel, l)); per l block the product A_l (channels x (2l+1)) = P_l^T Y_l with
// v_mfma_f32_16x16x4f32: k = lane group <-> bond, A operand rows = 16 channels, B operand columns = the 2l+1 lm of the
// block. The (channel tile, l) output blocks are split over the 4 wavefronts. Out of range bonds: P = 0, Y = 0.
// Only for a_fwd_mfma_fits(lmax, NR).
template<class DeviceType, class Traits, bool CPL, class AV, class XV, class TV, class MapV, class NblV, class IlV,
         class BcV, class ZV, class MaskV, class GV, class SplV>
void a_fwd_mfma(const char *label, const int cs, const int co, const int NR, const float ascale, const int N,
                const AV &Af, const XV &x, const TV &type, const MapV &map, const NblV &nbl, const IlV &ilist,
                const BcV &bc, const ZV &ztr, const int ceq, const int cb1, const int cb2, const double clam,
                const MaskV &cmask, const GV &idx_s, const GV &alm, const GV &blm, const GV &cl, const GV &dl,
                const SplV &sc)
{
  using F = float;
  constexpr int NB = AFM_NB, NL = AFM_LMAX + 1, NLM = NL * NL, TEAM = AFM_TEAM, MAXP = AFM_MAXP;
  const int nout = NR * NL, ntn = (NR + 15) / 16, npair = ntn * NL;
  using team_policy = Kokkos::TeamPolicy<DeviceType, Kokkos::LaunchBounds<TEAM, 2>>;
  using member = typename team_policy::member_type;
  team_policy pol(cs, TEAM);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(a_fwd_mfma_shmem(NR)));
  Kokkos::parallel_for(label, pol, KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank();
    const int i = ilist[ii + co];
    const int mui = map(type(i));
    const auto xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);    // positions' precision (double)
    F *Ys = (F *) t.team_shmem().get_shmem(NB * NLM * sizeof(F));
    F *Rs = (F *) t.team_shmem().get_shmem(NB * nout * sizeof(F));
    F *xs = (F *) t.team_shmem().get_shmem(NB * sizeof(F));
    int *mus = (int *) t.team_shmem().get_shmem(NB * sizeof(int));
    const int tid = t.team_rank(), w = tid >> 6, lane = tid & 63, g = lane >> 4, li = lane & 15;
    mfma_f4v acc[MAXP];
#pragma unroll
    for (int p = 0; p < MAXP; p++) acc[p] = mfma_f4v{0.0f, 0.0f, 0.0f, 0.0f};
    const int jnum = nbl.count(i, ii + co);
    for (int jb = 0; jb < jnum; jb += NB) {
      const int nb = (jnum - jb < NB) ? jnum - jb : NB;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NB), [&](const int b) {
        int mu = -1;
        if (b < nb) {    // a bond of the compact list (no cutoff test)
          const int j = nbl(i, ii + co, jb + b);
          const double dx = x(j, 0) - xi, dy = x(j, 1) - yi, dz = x(j, 2) - zi;    // bond geometry in fp64
          mu = map(type(j));
          const double rc = bc(mui, mu);
          const double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz), rinv = 1.0 / r;
          double rm = r, drm, dlm;    // the model's bond length: r' of a coupled bond (grace/coupling.h)
          if (CPL && coupling_cross(cmask(i), cmask(j), cb1, cb2)) coupling_shift(ceq, r, rc, clam, rm, drm, dlm);
          xs[b] = F(rm / rc);
          real_sph(AFM_LMAX, F(dx * rinv), F(dy * rinv), F(dz * rinv), idx_s, alm, blm, cl, dl, Ys + b * NLM);
        } else {
          for (int y = 0; y < NLM; y++) Ys[b * NLM + y] = 0.0f;
        }
        mus[b] = mu;
      });
      t.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NB * nout), [&](const int q) {
        const int b = q / nout, o = q - b * nout;
        Rs[q] = (mus[b] >= 0) ? spline_v(sc, N, xs[b], o) * ztr(mus[b], o / NL) : 0.0f;
      });
      t.team_barrier();
      const int nst = (nb + 3) / 4;
      for (int s = 0; s < nst; s++) {
        const int b = 4 * s + g;
        const F *Pb = Rs + b * nout, *Yb = Ys + b * NLM;
#pragma unroll
        for (int p = 0; p < MAXP; p++) {
          const int pr = w + 4 * p;
          if (pr >= npair) continue;    // uniform over the wavefront
          const int tn = pr / NL, l = pr - tn * NL, n = 16 * tn + li;
          const F a = (n < NR) ? Pb[n * NL + l] : 0.0f;
          const F yv = (li < 2 * l + 1) ? Yb[l * l + li] : 0.0f;
          mfma_16x16x4(a, yv, acc[p]);
        }
      }
      t.team_barrier();
    }
    // D[n][lm] of block (tn, l): register q of lane (li, g) = (n = 16 tn + 4 g + q, lm = l^2 + li)
#pragma unroll
    for (int p = 0; p < MAXP; p++) {
      const int pr = w + 4 * p;
      if (pr >= npair) continue;
      const int tn = pr / NL, l = pr - tn * NL;
      if (li < 2 * l + 1) {
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const int n = 16 * tn + 4 * g + q;
          if (n < NR) Af(ii, n, l * l + li) = acc[p][q] * ascale;
        }
      }
    }
  });
}
#endif

}    // namespace GraceFusedKK
}    // namespace kk
}    // namespace grace
