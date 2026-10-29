// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The PACE .yaml reader (libgrace/src/pace_yaml.h), the Clebsch-Gordan coefficients and m-lists of the converter
// (pace_convert.h), and a fixture model loaded from its .yaml. The converted values are checked against pyace by the
// Python tests (python/tests/test_pace_ctilde.py and the engine tests).
#include "grace/error.h"
#include "grace/pace_model.h"
#include "pace_convert.h"
#include "pace_yaml.h"
#include "test_util.h"

#include <cmath>
#include <string>
#include <utility>

using grace::yaml::Node;

static bool throws(const std::string &text, const std::string &what)    // parse fails with `what` in the message
{
  try {
    grace::yaml::parse(text, "t.yaml");
  } catch (const grace::Error &e) {
    if (std::string(e.what()).find(what) != std::string::npos) return true;
    std::fprintf(stderr, "message: %s\n", e.what());
  }
  return false;
}

static void test_parser()
{
  // the yaml-cpp style of pacemaker files: sequences indented under their key, flow collections on one line
  const Node a = grace::yaml::parse("global:\n  DeltaSplineBins: 0.001\nspecies:\n  - speciesblock: C\n"
                                    "    parameters: [1, 1, 1, 0.5]   # comment\n"
                                    "    nbody:\n      - {type: C C, nr: [1], nl: [0], c: [-0.7, 2]}\n"
                                    "  - speciesblock: \"H H\"\n",
                                    "a.yaml");
  CHECK(a.kind == Node::Map && a.map.size() == 2);
  CHECK(a.find("global")->find("DeltaSplineBins")->str == "0.001");
  const Node &sp = *a.find("species");
  CHECK(sp.kind == Node::Seq && sp.seq.size() == 2);
  CHECK(sp.seq[0].find("parameters")->seq.size() == 4 && sp.seq[0].find("parameters")->seq[3].str == "0.5");
  const Node &f = sp.seq[0].find("nbody")->seq[0];
  CHECK(f.find("type")->str == "C C" && f.find("c")->seq[1].str == "2" && f.find("nr")->seq[0].str == "1");
  CHECK(sp.seq[1].find("speciesblock")->str == "H H" && sp.seq[1].find("speciesblock")->quoted);
  CHECK(f.line == 7);

  // the PyYAML safe_dump style: block sequences at their key's indentation, nested "- - x", quotes
  const Node b = grace::yaml::parse("species:\n- nbody:\n  - c:\n    - 1.5\n    - -2\n    type: 'A ''B'''\n"
                                    "  speciesblock: A\n- - 1\n  - [2, 3]\nempty:\nnull_v: ~\n",
                                    "b.yaml");
  const Node &s = *b.find("species");
  CHECK(s.seq.size() == 2);
  const Node &g = s.seq[0].find("nbody")->seq[0];
  CHECK(g.find("c")->seq.size() == 2 && g.find("c")->seq[1].str == "-2" && g.find("type")->str == "A 'B'");
  CHECK(s.seq[0].find("speciesblock")->str == "A");
  CHECK(s.seq[1].kind == Node::Seq && s.seq[1].seq[0].str == "1" && s.seq[1].seq[1].seq[1].str == "3");
  CHECK(b.find("empty")->is_null() && b.find("null_v")->is_null());

  // a flow collection over several lines, nested, with a trailing comma
  const Node c =
      grace::yaml::parse("r: [[[1, 2],\n      [3, 4]],\n   [[5, 6], [7, 8],]]\nnext: \"x#y\"  # c\n", "c.yaml");
  CHECK(c.find("r")->seq.size() == 2 && c.find("r")->seq[1].seq[1].seq[0].str == "7");
  CHECK(c.find("next")->str == "x#y");

  CHECK(throws("a: 1\n\tb: 2\n", "t.yaml:2: tab"));
  CHECK(throws("a: |\n  text\n", "t.yaml:1: block scalars"));
  CHECK(throws("a: &x 1\n", "anchors"));
  CHECK(throws("a: 1\na: 2\n", "t.yaml:2: duplicate key a"));
  CHECK(throws("a: 1\n   b: 2\n", "t.yaml:2: unexpected indentation"));
  CHECK(throws("a: [1, 2\n", "unclosed flow collection"));
  CHECK(throws("a: 1\n---\nb: 2\n", "several documents"));
  CHECK(throws("a: {b 1}\n", "expected ':'"));
}

static void test_clebsch_gordan()
{
  using grace::pace::clebsch_gordan;
  CHECK_NEAR(clebsch_gordan(1, 0, 1, 0, 0, 0), -1 / std::sqrt(3.0), 1e-15);
  CHECK_NEAR(clebsch_gordan(1, 1, 1, -1, 2, 0), 1 / std::sqrt(6.0), 1e-15);
  CHECK_NEAR(clebsch_gordan(1, 1, 1, 1, 2, 2), 1.0, 1e-15);
  CHECK(clebsch_gordan(2, 0, 1, 0, 2, 0) == 0.0);    // parity: exactly zero
  CHECK(clebsch_gordan(1, 1, 1, 0, 2, 0) == 0.0);    // m1 + m2 != M
  for (int j1 = 0; j1 <= 4; j1++)
    for (int j2 = 0; j2 <= 4; j2++)
      for (int J = std::abs(j1 - j2); J <= j1 + j2; J++)
        for (int M = -J; M <= J; M++) {
          double s = 0.0;
          for (int m1 = -j1; m1 <= j1; m1++)
            for (int m2 = -j2; m2 <= j2; m2++) s += std::pow(clebsch_gordan(j1, m1, j2, m2, J, M), 2);
          CHECK_NEAR(s, 1.0, 1e-13);
        }
  CHECK_NEAR(clebsch_gordan(9, 3, 9, -3, 18, 0), clebsch_gordan(9, -3, 9, 3, 18, 0), 1e-16);    // large l: symmetry
}

static void test_ms_lists()
{
  using grace::pace::ms_cg_list;
  const auto r1 = ms_cg_list({0}, {});
  CHECK(r1.size() == 1 && r1.ms[0] == 0 && r1.cg[0] == 1.0);
  const auto r2 = ms_cg_list({1, 1}, {});    // (0, 0) and (1, -1) doubled; (-1, 1) dropped (half basis)
  CHECK(r2.size() == 2 && r2.ms[0] == 0 && r2.ms[2] == 1 && r2.ms[3] == -1);
  CHECK_NEAR(r2.cg[0], -1 / std::sqrt(3.0), 1e-15);
  CHECK_NEAR(r2.cg[1], 2 / std::sqrt(3.0), 1e-15);
  for (auto [ls, lint] : {std::pair<std::vector<int>, std::vector<int>>{{1, 1, 2}, {2}}, {{1, 1, 1, 1}, {1, 1}},
                          {{1, 1, 1, 1, 2}, {0, 2, 2}}, {{1, 1, 1, 1, 1, 1}, {2, 2, 0, 0}}}) {
    const auto l = ms_cg_list(ls, lint);
    CHECK(l.size() > 0 && l.rank == (int) ls.size());
    for (size_t q = 0; q < l.size(); q++) {    // sum m = 0, first non-zero m positive, |m| <= l
      int s = 0, first = 0;
      for (int k = 0; k < l.rank; k++) {
        const int m = l.ms[q * l.rank + k];
        CHECK(std::abs(m) <= ls[k]);
        s += m;
        if (!first) first = m;
      }
      CHECK(s == 0 && first >= 0 && l.cg[q] != 0.0);
    }
  }
  bool threw = false;
  try {
    ms_cg_list({1, 2}, {});
  } catch (const grace::Error &e) {
    threw = std::string(e.what()).find("rank-2 function has equal ls") != std::string::npos;
  }
  CHECK(threw);
}

static void test_fixture_model()
{
  grace::PaceModel m;
  m.load("tests/fixtures/pace/pdh_zbl.yaml");
  CHECK(m.n_elements == 2 && m.element_names[0] == "H" && m.element_names[1] == "Pd");
  CHECK(m.element_z[0] == 1 && m.element_z[1] == 46 && m.inner_cutoff_type == grace::kInnerZbl);
  CHECK(m.elements[0].nrows > 0 && m.bond(0, 1).defined && m.bond(1, 0).defined && !m.has_uq);
  bool threw = false;
  try {
    m.load("tests/fixtures/pace/ref_pace.npz", "x.asi");
  } catch (const grace::Error &e) {
    threw = std::string(e.what()).find("goes with a PACE .yaml") != std::string::npos;
  }
  CHECK(threw);
}

int main()
{
  test_parser();
  test_clebsch_gordan();
  test_ms_lists();
  test_fixture_model();
  std::printf("test_pace_yaml: OK\n");
  return 0;
}
