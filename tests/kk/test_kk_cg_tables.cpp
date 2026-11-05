// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The compiled CG tables (cg_tables.h) against the generic CG-product kernels (fused_dense.h): every table of every
// engine product, forward and reverse as the engines call them (1L AA accumulates), on random operands, with other
// values beyond the table in the outputs. Needs no model.
#include "grace/kk/fused_dense.h"
#include "grace/kk/cg_tables.h"
#include "../test_util.h"

#include <algorithm>
#include <random>
#include <vector>

using Dev = Kokkos::DefaultExecutionSpace;
using V3 = Kokkos::View<float ***, Dev>;
namespace G = grace::kk::GraceFusedKK;
namespace cgt = grace::kk::cgt;

struct Prod {    // the fields of GRACE1LModel / GRACE2LModel::CGProduct that build_prod_meta and cgt_match read
  std::vector<int> left_ind, right_ind, m_sum_ind;
  std::vector<double> cg_coeff;
};

static V3 random_view(const char *name, int n, int nc, int w, unsigned seed)
{
  V3 v(name, n, nc, w);
  auto h = Kokkos::create_mirror_view(v);
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  for (int i = 0; i < n; i++)
    for (int k = 0; k < nc; k++)
      for (int x = 0; x < w; x++) h(i, k, x) = u(g);
  Kokkos::deep_copy(v, h);
  return v;
}

// max |a - b| / max |b|
static double rel_diff(const V3 &a, const V3 &b)
{
  auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  double d = 0.0, s = 1e-30;
  for (size_t i = 0; i < ha.size(); i++) {
    d = std::max(d, (double) std::fabs(ha.data()[i] - hb.data()[i]));
    s = std::max(s, (double) std::fabs(hb.data()[i]));
  }
  return d / s;
}

template<bool SELF, bool ACC, class T, class List>
void check_table(List list, const int id, const char *prod)
{
  Prod p;
  int nl = 0, nr = 0, no = 0;
  for (int t = 0; t < T::nt; t++) {
    p.left_ind.push_back(T::tab[t][0]); p.right_ind.push_back(T::tab[t][1]); p.m_sum_ind.push_back(T::tab[t][2]);
    p.cg_coeff.push_back(T::cg[t]);
    nl = std::max(nl, T::tab[t][0] + 1); nr = std::max(nr, T::tab[t][1] + 1); no = std::max(no, T::tab[t][2] + 1);
  }
  if (SELF) nl = nr = std::max(nl, nr);
  CHECK(G::cgt_match(list, p) == id);
  G::ProdMetaT<Dev> m;
  G::build_prod_meta(m, p, SELF);
  const int n = 5, nc = 3;
  const V3 L = random_view("L", n, nc, nl + 2, 1), R = SELF ? L : random_view("R", n, nc, nr + 2, 2);
  const V3 D = random_view("D", n, nc, no + 1, 3);
  V3 out = random_view("out", n, nc, no + 1, 4), out_g = random_view("out", n, nc, no + 1, 4);
  G::cg_prod_fwd(list, id, m, "test:cgt_fwd", n, L, R, out);
  G::cg_prod_fwd(list, -1, m, "test:prod_fwd", n, L, R, out_g);
  V3 dL = random_view("dL", n, nc, nl + 2, 5), dL_g = random_view("dL", n, nc, nl + 2, 5);
  V3 dR = SELF ? dL : random_view("dR", n, nc, nr + 2, 6), dR_g = SELF ? dL_g : random_view("dR", n, nc, nr + 2, 6);
  G::cg_prod_rev<ACC>(list, id, m, "test:cgt_rev", n, L, R, D, dL, dR);
  G::cg_prod_rev<ACC>(list, -1, m, "test:prod_rev", n, L, R, D, dL_g, dR_g);
  const double ef = rel_diff(out, out_g), el = rel_diff(dL, dL_g), er = SELF ? 0.0 : rel_diff(dR, dR_g);
  std::printf("%-6s table %d (%4d terms): forward %.1e, reverse dL %.1e dR %.1e\n", prod, id, T::nt, ef, el, er);
  CHECK(ef < 1e-5 && el < 1e-5 && er < 1e-5);
}

template<bool SELF, bool ACC, template<class...> class List, class... T>
void check_list(List<T...> list, const char *prod)
{
  int id = 0;
  (check_table<SELF, ACC, T>(list, id++, prod), ...);
}

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  check_list<true, true>(cgt::G1L_AA{}, "1L AA");
  check_list<false, false>(cgt::G1L_AAA{}, "1L AAA");
  check_list<true, false>(cgt::G1L_AAAA{}, "1L AAAA");
  check_list<true, false>(cgt::G2L_AA{}, "2L AA");
  check_list<false, false>(cgt::G2L_AAA{}, "2L AAA");
  check_list<true, false>(cgt::G2L_AAAA{}, "2L AAAA");
  check_list<true, false>(cgt::G2L_BB{}, "2L BB");
  check_list<false, false>(cgt::G2L_BBB{}, "2L BBB");
  check_list<true, false>(cgt::G2L_BBBB{}, "2L BBBB");
  std::printf("kk cg tables ok\n");
  return 0;
}
