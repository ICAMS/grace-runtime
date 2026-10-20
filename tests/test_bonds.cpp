// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// LammpsBonds (positions + ghosts + masked neighbour list, as LAMMPS) and CsrBonds (caller r_ij, j = owner)
// gather the same bonds: same types, distances, unit vectors and spherical harmonics, bit for bit
#include "grace/bonds.h"
#include "test_util.h"
#include <vector>
struct W {    // the per-thread workspace fields the gatherer fills
  int nbat = 0;
  int atom[4];
  int boff[5];
  std::vector<int> bj, bt;
  std::vector<double> rh, rr, rc;
  std::vector<double> cp;    // two-group coupling
  std::vector<float> Y, G;
};
int main()
{
  // 2 central atoms (0: type 1, 1: type 2) and 3 ghosts: 2 = image of 1, 3 and 4 = images of 0
  double xs[5][3] = {{0, 0, 0}, {1.5, 0, 0}, {-1.5, 0, 0}, {0, 1.6, 0}, {3.0, 0.1, 0}};
  double *x[5];
  for (int a = 0; a < 5; a++) x[a] = xs[a];
  int type[5] = {1, 2, 2, 1, 1};
  const int MASK = 0x1FFFFFFF;
  int nl0[] = {1, 2, 3 | (1 << 30)}, nl1[] = {0, 4};    // one entry with special-bond bits set
  int *first[2] = {nl0, nl1};
  int numneigh[2] = {3, 2};
  double cs[3][3] = {{0, 0, 0}, {0, 25, 25}, {0, 25, 25}};
  double *cutsq[3] = {cs[0], cs[1], cs[2]};
  double rct[9] = {0, 0, 0, 0, 5, 5, 0, 5, 5};
  std::vector<double> a, b, c, d;
  grace::fused::sh_coefs(2, a, b, c, d);
  grace::fused::SHCoefs sh{2, 9, a.data(), b.data(), c.data(), d.data()};
  grace::BondFilter flt{cutsq, rct, 3};
  W w1;
  w1.nbat = 2; w1.atom[0] = 0; w1.atom[1] = 1;
  grace::BondGather<grace::LammpsBonds>::run(w1, grace::LammpsBonds{x, type, numneigh, first, MASK}, flt, sh, true);
  int64_t off[3] = {0, 3, 5};
  int jc[5] = {1, 1, 0, 0, 0};
  double rij[5][3] = {{1.5, 0, 0}, {-1.5, 0, 0}, {0, 1.6, 0}, {-1.5, 0, 0}, {1.5, 0.1, 0}};
  W w2;
  w2.nbat = 2; w2.atom[0] = 0; w2.atom[1] = 1;
  grace::BondGather<grace::CsrBonds>::run(w2, grace::CsrBonds{type, off, jc, &rij[0][0]}, flt, sh, true);
  CHECK(w1.boff[0] == 0 && w1.boff[1] == 3 && w1.boff[2] == 5);
  CHECK(w2.boff[0] == 0 && w2.boff[1] == 3 && w2.boff[2] == 5);
  for (int k = 0; k < 5; k++) {
    CHECK(w1.bt[k] == w2.bt[k]);
    CHECK_NEAR(w1.rr[k], w2.rr[k], 0);
    CHECK_NEAR(w1.rc[k], w2.rc[k], 0);
    for (int q = 0; q < 3; q++) CHECK_NEAR(w1.rh[3 * k + q], w2.rh[3 * k + q], 0);
    for (int q = 0; q < 9; q++) CHECK_NEAR(w1.Y[9 * k + q], w2.Y[9 * k + q], 0);
    for (int q = 0; q < 27; q++) CHECK_NEAR(w1.G[27 * k + q], w2.G[27 * k + q], 0);
  }
  CHECK(w1.bj[2] == 3);    // masked entry resolved to ghost 3
  CHECK(w2.bj[2] == 0);    // CSR: owner index (self-image)
  CHECK_NEAR(w1.Y[0], w1.Y[9], 0);    // l = 0 harmonic is the same constant for every bond
  // a bond beyond the pair cutoff is dropped by both
  cs[1][2] = cs[2][1] = 1.0;
  W w3;
  w3.nbat = 2; w3.atom[0] = 0; w3.atom[1] = 1;
  grace::BondGather<grace::CsrBonds>::run(w3, grace::CsrBonds{type, off, jc, &rij[0][0]}, flt, sh, false);
  CHECK(w3.boff[1] == 1 && w3.boff[2] == 1);    // only the 1-1 self-image bond of atom 0 stays
  return 0;
}
