// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// pacemaker B-basis yaml -> PACE npz arrays (pace_convert.h). The checks and defaults of ML-PACE's reader
// (lammps-user-pace ML-PACE/ace/ace_yaml_input.cpp, ace_b_basis.cpp); the C-tilde basis of to_ACECTildeBasisSet
// (ace_couplings.cpp: generate_coupling_tree, generate_equivariant_ms_cg_list, validate_ls_LS; ace_b_basis.cpp:
// group_basis_functions_by_index, summation_over_LS), the factors of each product in canonical order.
#include "pace_convert.h"
#include "pace_yaml.h"

#include "grace/error.h"
#include "grace/pace_model.h"
#include "npz3.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <tuple>

namespace grace::pace {

namespace {

// ---------------------------------------------------------------------------------------------------------------
// Exact integers for the Clebsch-Gordan coefficients: unsigned magnitudes, 32-bit limbs, little-endian, no leading
// zero limbs (empty: 0)
using Big = std::vector<uint32_t>;

void norm(Big &a)
{
  while (!a.empty() && a.back() == 0) a.pop_back();
}

void mul_small(Big &a, uint32_t m)
{
  uint64_t c = 0;
  for (auto &x : a) {
    const uint64_t t = (uint64_t) x * m + c;
    x = (uint32_t) t;
    c = t >> 32;
  }
  if (c) a.push_back((uint32_t) c);
  norm(a);
}

void add_to(Big &a, const Big &b)
{
  if (a.size() < b.size()) a.resize(b.size(), 0);
  uint64_t c = 0;
  for (size_t i = 0; i < a.size(); i++) {
    const uint64_t t = (uint64_t) a[i] + (i < b.size() ? b[i] : 0) + c;
    a[i] = (uint32_t) t;
    c = t >> 32;
  }
  if (c) a.push_back((uint32_t) c);
}

void sub_from(Big &a, const Big &b)    // a >= b
{
  int64_t c = 0;
  for (size_t i = 0; i < a.size(); i++) {
    int64_t t = (int64_t) a[i] - (i < b.size() ? b[i] : 0) - c;
    c = t < 0;
    if (t < 0) t += (int64_t) 1 << 32;
    a[i] = (uint32_t) t;
  }
  norm(a);
}

int cmp(const Big &a, const Big &b)
{
  if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
  for (size_t i = a.size(); i-- > 0;)
    if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
  return 0;
}

Big shl(const Big &a, int s)
{
  if (a.empty()) return a;
  const int limbs = s / 32, bits = s % 32;
  Big r(limbs, 0);
  uint32_t carry = 0;
  for (uint32_t x : a) {
    r.push_back(bits ? (x << bits) | carry : x);
    carry = bits ? x >> (32 - bits) : 0;
  }
  if (carry) r.push_back(carry);
  norm(r);
  return r;
}

int bitlen(const Big &a) { return a.empty() ? 0 : 32 * ((int) a.size() - 1) + (32 - __builtin_clz(a.back())); }

// num / den rounded correctly to double (round half to even), as Python's int / int and float(int)
double ratio(const Big &num, const Big &den)
{
  if (num.empty()) return 0.0;
  const int sh = 54 - (bitlen(num) - bitlen(den));    // floor(num 2^sh / den) has 54 or 55 bits
  Big A = sh > 0 ? shl(num, sh) : num;
  const Big B = sh < 0 ? shl(den, -sh) : den;
  uint64_t q = 0;
  for (int b = 56; b >= 0; b--) {
    const Big Bb = shl(B, b);
    if (cmp(A, Bb) >= 0) {
      sub_from(A, Bb);
      q |= (uint64_t) 1 << b;
    }
  }
  const int drop = (64 - __builtin_clzll(q)) - 53;
  uint64_t keep = q >> drop;
  const uint64_t rem = q & (((uint64_t) 1 << drop) - 1), half = (uint64_t) 1 << (drop - 1);
  if (rem > half || (rem == half && (!A.empty() || (keep & 1)))) keep++;
  return std::ldexp((double) keep, drop - sh);
}

// integers as exponents of the primes below kMaxFact (factorials, their ratios and lcm)
constexpr int kMaxFact = 256;

const std::vector<int> &primes()
{
  static const std::vector<int> p = [] {
    std::vector<int> v;
    for (int n = 2; n < kMaxFact; n++) {
      bool prime = true;
      for (int d : v) {
        if (d * d > n) break;
        if (n % d == 0) prime = false;
      }
      if (prime) v.push_back(n);
    }
    return v;
  }();
  return p;
}

using Exp = std::vector<int>;

void add_factorial(Exp &e, int n, int sign)
{
  if (n < 0 || n >= kMaxFact) throw Error("Clebsch-Gordan coefficient: factorial of " + std::to_string(n));
  const auto &p = primes();
  for (size_t k = 0; k < p.size() && p[k] <= n; k++)
    for (long q = p[k]; q <= n; q *= p[k]) e[k] += sign * (int) (n / q);
}

void add_int(Exp &e, int n)
{
  const auto &p = primes();
  for (size_t k = 0; k < p.size() && n > 1; k++)
    while (n % p[k] == 0) {
      e[k]++;
      n /= p[k];
    }
  if (n > 1) throw Error("Clebsch-Gordan coefficient: factor out of range");
}

Big product(const Exp &e, int sign)    // the product of p^(sign e) over the exponents with sign e > 0
{
  Big r{1};
  const auto &p = primes();
  for (size_t k = 0; k < p.size(); k++)
    for (int t = 0; t < sign * e[k]; t++) mul_small(r, (uint32_t) p[k]);
  return r;
}

// ---------------------------------------------------------------------------------------------------------------
// the B-basis

const char *const kRadbases[] = {"ChebExpCos", "ChebPow", "ChebLinear", "SBessel"};
const char *const kEmbeddings[] = {"FinnisSinclair", "FinnisSinclairShiftedScaled"};
const char *const kInnerCutoffs[] = {"density", "distance", "zbl"};

template<size_t N> int index_of(const char *const (&names)[N], const std::string &s)
{
  for (size_t i = 0; i < N; i++)
    if (s == names[i]) return (int) i;
  return -1;
}

template<size_t N> std::string joined(const char *const (&names)[N])
{
  std::string s;
  for (size_t i = 0; i < N; i++) s += (i ? ", " : "") + std::string(names[i]);
  return s;
}

// atomic numbers (ase.data.atomic_numbers, "X" 0)
const char *const kSymbols[] = {
    "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re",
    "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu",
    "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
    "Fl", "Mc", "Lv", "Ts", "Og"};

int atomic_number(const std::string &e)    // -1: unknown
{
  for (int z = 0; z < (int) (sizeof(kSymbols) / sizeof(kSymbols[0])); z++)
    if (e == kSymbols[z]) return z;
  return -1;
}

using yaml::Node;

std::vector<std::string> words(const std::string &s)
{
  std::vector<std::string> w;
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && std::isspace((unsigned char) s[i])) i++;
    const size_t a = i;
    while (i < s.size() && !std::isspace((unsigned char) s[i])) i++;
    if (i > a) w.push_back(s.substr(a, i - a));
  }
  return w;
}

std::string list_str(const std::vector<int> &v)
{
  std::string s = "[";
  for (size_t i = 0; i < v.size(); i++) s += (i ? ", " : "") + std::to_string(v[i]);
  return s + "]";
}

std::string names_str(std::vector<std::string> v)    // Python's repr of a sorted list of str
{
  std::sort(v.begin(), v.end());
  std::string s = "[";
  for (size_t i = 0; i < v.size(); i++) s += (i ? ", '" : "'") + v[i] + "'";
  return s + "]";
}

// values of the yaml: what is not a scalar of the right kind is an error naming ctx (the block) and the key
struct Ctx {
  std::string where;    // "<file>: <block>"
  [[noreturn]] void fail(const std::string &what) const { throw Error(where + ": " + what); }
};

const Node *opt(const Node &m, const std::string &key) { return m.kind == Node::Map ? m.find(key) : nullptr; }

// key as const char *: with a const std::string & the caller's temporary string made g++ >= 14 warn
// -Wdangling-reference, although the result refers into m, not into key
const Node &req(const Node &m, const char *key, const Ctx &c)
{
  const Node *v = opt(m, key);
  if (!v) c.fail(std::string("missing key ") + key);
  return *v;
}

std::string text(const Node &n, const std::string &key, const Ctx &c)
{
  if (n.kind != Node::Scalar) c.fail(key + " must be a scalar");
  return n.str;
}

double number(const Node &n, const std::string &key, const Ctx &c)
{
  if (n.kind != Node::Scalar) c.fail(key + " must be a number");
  std::string s = n.str;
  const bool neg = !s.empty() && s[0] == '-';
  const std::string a = !s.empty() && (s[0] == '-' || s[0] == '+') ? s.substr(1) : s;
  if (a == ".inf" || a == ".Inf" || a == ".INF") return neg ? -HUGE_VAL : HUGE_VAL;
  if (s == ".nan" || s == ".NaN" || s == ".NAN") return std::nan("");
  char *end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (s.empty() || *end != '\0') c.fail(key + " must be a number, not '" + s + "'");
  return v;
}

int integer(const Node &n, const std::string &key, const Ctx &c)
{
  const double v = number(n, key, c);
  if (v != std::floor(v) || std::fabs(v) > 1e9) c.fail(key + " must be an integer, not '" + n.str + "'");
  return (int) v;
}

std::vector<double> numbers(const Node &n, const std::string &key, const Ctx &c)    // a sequence of numbers
{
  if (n.is_null()) return {};
  if (n.kind != Node::Seq) c.fail(key + " must be a list of numbers");
  std::vector<double> v;
  for (const Node &x : n.seq) v.push_back(number(x, key, c));
  return v;
}

std::vector<int> integers(const Node &n, const std::string &key, const Ctx &c)
{
  if (n.kind != Node::Seq) c.fail(key + " must be a list of integers");
  std::vector<int> v;
  for (const Node &x : n.seq) v.push_back(integer(x, key, c));
  return v;
}

bool truthy(const Node &n, const std::string &key, const Ctx &c)    // a YAML 1.1 boolean (PyYAML), or a number
{
  if (n.is_null()) return false;
  if (n.kind != Node::Scalar) c.fail(key + " must be a boolean");
  static const std::set<std::string> yes = {"true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"};
  static const std::set<std::string> no = {"false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF"};
  if (!n.quoted && yes.count(n.str)) return true;
  if (!n.quoted && no.count(n.str)) return false;
  if (n.quoted) return !n.str.empty();
  return number(n, key, c) != 0.0;
}

struct Embedding {
  int ndensity = 0, npoti = 0;
  std::vector<double> parameters;
  double rho_core_cut = 0.0, drho_core_cut = 0.0;
};

struct Rad {
  std::string radbase;
  std::vector<double> radparameters, crad;    // crad [nradmax][lmax + 1][nradbase]
  int nradmax = 0, lmax = 0, nradbase = 0;
  double rcut = 0.0, dcut = 0.0;
};

struct Inner {
  std::string type = "density";
  double r_in = 0.0, delta_in = 0.0;
};

struct Bond {
  Rad rad;
  double r_in = 0.0, delta_in = 0.0, prehc = 0.0, lambdahc = 0.0;
};

struct BFunction {
  std::vector<int> mus, ns, ls, lint;    // mus: element indices, in the order of the file
  std::vector<double> coeffs;            // one per density
};

struct Spec {
  std::vector<std::string> elements;    // sorted, as ML-PACE
  int inner_cutoff_type = kInnerDensity;
  std::vector<Embedding> embeddings;
  std::map<std::pair<int, int>, Bond> bonds;    // a pair without one does not interact (as in pyace)
  std::vector<std::vector<BFunction>> functions;
};

using Key = std::pair<int, int>;

// the radial part of a block (none if it has none)
std::optional<Rad> block_rad(const Node &b, const Ctx &c)
{
  static const char *const radial_keys[] = {"nradmaxi", "nradbaseij", "radbase", "radparameters", "radcoefficients",
                                            "rcutij", "dcutij"};
  if (!opt(b, "lmaxi")) {
    for (const char *k : radial_keys)
      if (opt(b, k)) c.fail("the radial basis is partially defined (lmaxi missing)");
    return std::nullopt;
  }
  Rad r;
  r.radbase = text(req(b, "radbase", c), "radbase", c);
  if (index_of(kRadbases, r.radbase) < 0)
    c.fail("radbase '" + r.radbase + "' is not supported (supported: " + joined(kRadbases) + ")");
  const Node *cf = opt(b, "NameOfCutoffFunctionij");
  const std::string cut_fn = cf ? text(*cf, "NameOfCutoffFunctionij", c) : "cos";
  if (cut_fn != "cos") c.fail("NameOfCutoffFunctionij '" + cut_fn + "' is not supported (only 'cos')");
  r.nradmax = integer(req(b, "nradmaxi", c), "nradmaxi", c);
  r.lmax = integer(req(b, "lmaxi", c), "lmaxi", c);
  r.nradbase = integer(req(b, "nradbaseij", c), "nradbaseij", c);
  const Node &rc = req(b, "radcoefficients", c);
  const std::string want = "expected [nradmaxi, lmaxi + 1, nradbaseij] = [" + std::to_string(r.nradmax) + ", " +
                           std::to_string(r.lmax + 1) + ", " + std::to_string(r.nradbase) + "]";
  bool ok = rc.kind == Node::Seq && (int) rc.seq.size() == r.nradmax;
  for (const Node &a : rc.seq) {
    ok = ok && a.kind == Node::Seq && (int) a.seq.size() == r.lmax + 1;
    for (const Node &v : a.seq) {
      ok = ok && v.kind == Node::Seq && (int) v.seq.size() == r.nradbase;
      if (!ok) break;
      for (const Node &x : v.seq) r.crad.push_back(number(x, "radcoefficients", c));
    }
    if (!ok) break;
  }
  if (!ok) c.fail("radcoefficients does not have the shape " + want.substr(9));
  const Node *rp = opt(b, "radparameters");
  if (rp) r.radparameters = numbers(*rp, "radparameters", c);
  if ((r.radbase == "ChebExpCos" || r.radbase == "ChebPow") && r.radparameters.empty())
    c.fail("radbase " + r.radbase + " needs radparameters [lambda]");
  r.rcut = number(req(b, "rcutij", c), "rcutij", c);
  r.dcut = number(req(b, "dcutij", c), "dcutij", c);
  return r;
}

Inner block_inner(const Node &b, const Ctx &c)    // default of ace_yaml_input.cpp: density
{
  Inner in;
  if (const Node *t = opt(b, "inner_cutoff_type")) in.type = text(*t, "inner_cutoff_type", c);
  if (const Node *v = opt(b, "r_in")) in.r_in = number(*v, "r_in", c);
  if (const Node *v = opt(b, "delta_in")) in.delta_in = number(*v, "delta_in", c);
  return in;
}

// as ML-PACE (ace_yaml_input.cpp): a pair block without a radial basis takes its dual's (A-B <=> B-A), one without
// core repulsion takes its dual's with the inner cutoff; defined in both, they must agree
void complete_from_duals(const std::vector<Key> &order, std::map<Key, std::optional<Rad>> &rad,
                         std::map<Key, std::array<double, 2>> &core, std::map<Key, Inner> &inner,
                         const std::vector<std::string> &el, const std::string &path)
{
  for (const Key &k : order) {
    const Key d{k.second, k.first};
    if (k.first == k.second || !rad.count(d)) continue;
    const Ctx c{path + ": pair blocks " + el[k.first] + " " + el[k.second] + " and " + el[k.second] + " " +
                el[k.first]};
    std::optional<Rad> &a = rad[k];
    const std::optional<Rad> &b = rad[d];
    if (!a && !b) c.fail("neither defines the radial basis");
    if (!a) {
      a = b;
    } else if (b) {
      const auto bad = [&](const char *key) {
        c.fail(std::string("the radial basis is not consistent (") + key + ")");
      };
      if (a->radbase != b->radbase) bad("radbase");
      if (a->radparameters != b->radparameters) bad("radparameters");
      if (a->nradmax != b->nradmax || a->lmax != b->lmax || a->nradbase != b->nradbase || a->crad != b->crad)
        bad("radcoefficients");
      if (a->rcut != b->rcut) bad("rcutij");
      if (a->dcut != b->dcut) bad("dcutij");
      const Inner &x = inner[k], &y = inner[d];
      if (x.type != y.type) bad("inner_cutoff_type");
      if (x.r_in != y.r_in) bad("r_in");
      if (x.delta_in != y.delta_in) bad("delta_in");
    }
    if (!core.count(k) && core.count(d)) {
      core[k] = core[d];
      inner[k] = inner[d];
    } else if (core.count(k) && core.count(d)) {
      if (core[k] != core[d]) c.fail("core-repulsion is not consistent");
      if (inner[k].r_in != inner[d].r_in || inner[k].delta_in != inner[d].delta_in)
        c.fail("the inner cutoff (r_in, delta_in) is not consistent");
    }
  }
}

Spec load_spec(const std::string &path)
{
  const Node d = yaml::load_file(path);
  const Node *species = opt(d, "species");
  if (!species || species->kind != Node::Seq) throw Error(path + ": not a PACE B-basis yaml (no 'species')");
  const Ctx file{path};
  std::vector<std::vector<std::string>> names;
  std::set<std::string> all;
  for (const Node &b : species->seq) {
    names.push_back(words(text(req(b, "speciesblock", file), "speciesblock", file)));
    if (names.back().empty()) file.fail("empty speciesblock");
    all.insert(names.back().begin(), names.back().end());
  }
  Spec s;
  s.elements.assign(all.begin(), all.end());
  std::map<std::string, int> idx;
  for (size_t k = 0; k < s.elements.size(); k++) idx[s.elements[k]] = (int) k;
  const int nel = (int) s.elements.size();
  std::vector<std::optional<Embedding>> emb(nel);
  std::vector<Key> order;
  std::map<Key, std::optional<Rad>> rad;
  std::map<Key, std::array<double, 2>> core;
  std::map<Key, Inner> inner;
  for (size_t q = 0; q < species->seq.size(); q++) {
    const Node &b = species->seq[q];
    const auto &n = names[q];
    std::string bname;
    for (const auto &w : n) bname += (bname.empty() ? "" : " ") + w;
    const Ctx c{path + ": " + bname};
    if (n.size() == 1) {
      const int mu = idx[n[0]];
      if (emb[mu]) file.fail("element " + n[0] + " is defined twice");
      Embedding e;
      const std::string npoti = text(req(b, "npoti", c), "npoti", c);
      e.npoti = index_of(kEmbeddings, npoti);
      if (e.npoti < 0) c.fail("npoti '" + npoti + "' is not supported (supported: " + joined(kEmbeddings) + ")");
      e.ndensity = integer(req(b, "ndensityi", c), "ndensityi", c);
      e.parameters = numbers(req(b, "parameters", c), "parameters", c);
      if ((int) e.parameters.size() != 2 * e.ndensity)
        c.fail("parameters has " + std::to_string(e.parameters.size()) + " values, expected 2 * ndensityi = " +
               std::to_string(2 * e.ndensity));
      const Node *rc = opt(b, "rho_core_cut"), *drc = opt(b, "drho_core_cut");
      e.rho_core_cut = rc ? number(*rc, "rho_core_cut", c) : 100000.0;
      e.drho_core_cut = drc ? number(*drc, "drho_core_cut", c) : 250.0;
      emb[mu] = e;
    }
    if (std::set<std::string>(n.begin(), n.end()).size() <= 2) {
      const Key key{idx[n.front()], idx[n.back()]};
      if (!rad.count(key)) order.push_back(key);
      rad[key] = block_rad(b, c);
      if (const Node *cr = opt(b, "core-repulsion")) {
        const auto v = numbers(*cr, "core-repulsion", c);
        if (v.size() < 2) c.fail("core-repulsion must have two values");
        core[key] = {v[0], v[1]};
      }
      inner[key] = block_inner(b, c);
      if (index_of(kInnerCutoffs, inner[key].type) < 0)
        c.fail("inner_cutoff_type '" + inner[key].type + "' is not supported (supported: " + joined(kInnerCutoffs) +
               ")");
    }
  }
  for (int mu = 0; mu < nel; mu++) {
    if (!emb[mu]) file.fail("no one-element speciesblock (embedding) for " + s.elements[mu]);
    s.embeddings.push_back(*emb[mu]);
  }
  complete_from_duals(order, rad, core, inner, s.elements, path);

  std::set<std::string> kinds;
  for (const auto &kv : inner) kinds.insert(kv.second.type);
  if (kinds.size() > 1) {
    std::string f;
    for (const auto &k : kinds) f += (f.empty() ? "'" : ", '") + k + "'";
    file.fail("inner_cutoff_type must be the same in all one- and two-species blocks, found [" + f + "]");
  }
  s.inner_cutoff_type = kinds.empty() ? kInnerDensity : index_of(kInnerCutoffs, *kinds.begin());
  if (s.inner_cutoff_type == kInnerZbl)
    for (const auto &e : s.elements)
      if (atomic_number(e) < 0) file.fail("inner_cutoff_type zbl needs atomic numbers, unknown element '" + e + "'");

  for (int i = 0; i < nel; i++)
    for (int j = 0; j < nel; j++) {
      const Key key = rad.count({i, j}) ? Key{i, j} : Key{j, i};    // no block for the pair: its dual's, if any
      const auto it = rad.find(key);
      if (it == rad.end() || !it->second) continue;    // no radial basis: zero radial functions, no core repulsion
      Bond bd;
      bd.rad = *it->second;
      if (core.count(key)) {
        bd.prehc = core[key][0];
        bd.lambdahc = core[key][1];
      }
      bd.r_in = inner[key].r_in;
      bd.delta_in = inner[key].delta_in;
      s.bonds[{i, j}] = bd;
    }

  s.functions.resize(nel);
  for (size_t q = 0; q < species->seq.size(); q++) {
    const Node &b = species->seq[q];
    const auto &n = names[q];
    std::string bname;
    for (const auto &w : n) bname += (bname.empty() ? "" : " ") + w;
    const Ctx c{path + ": " + bname};
    const Node *nbody = opt(b, "nbody");
    if (!nbody || nbody->is_null()) {
      if (n.size() <= 2) c.fail("no nbody section (old file format?)");
      continue;
    }
    if (nbody->kind != Node::Seq) c.fail("nbody must be a list");
    const std::set<std::string> allowed(n.begin(), n.end());
    for (const Node &f : nbody->seq) {
      const std::string type = text(req(f, "type", c), "type", c);
      const auto t = words(type);
      bool belongs = !t.empty() && t[0] == n[0];
      for (size_t k = 1; belongs && k < t.size(); k++) belongs = allowed.count(t[k]) > 0;
      if (!belongs) c.fail("function of type '" + type + "' does not belong to this block");
      if (const Node *sk = opt(f, "skip"))
        if (truthy(*sk, "skip", c)) continue;
      const int rank = (int) t.size() - 1;
      if (rank < 1) c.fail("function of rank 0 ('" + type + "') is not supported");
      const int mu0 = idx[t[0]];
      const Node &cn = req(f, "c", c);
      BFunction bf;
      bf.coeffs = cn.kind == Node::Seq ? numbers(cn, "c", c) : std::vector<double>{number(cn, "c", c)};
      if ((int) bf.coeffs.size() != s.embeddings[mu0].ndensity)
        c.fail("function '" + type + "' has " + std::to_string(bf.coeffs.size()) + " coefficients, the embedding of " +
               t[0] + " expects ndensityi = " + std::to_string(s.embeddings[mu0].ndensity));
      bf.ns = integers(req(f, "nr", c), "nr", c);
      bf.ls = integers(req(f, "nl", c), "nl", c);
      if (rank > 2)
        if (const Node *li = opt(f, "lint")) bf.lint = integers(*li, "lint", c);
      if ((int) bf.ns.size() != rank || (int) bf.ls.size() != rank || (int) bf.lint.size() != std::max(0, rank - 2))
        c.fail("function '" + type + "' nr=" + list_str(bf.ns) + " nl=" + list_str(bf.ls) + " lint=" +
               list_str(bf.lint) + ": lengths do not match rank " + std::to_string(rank));
      for (int k = 0; k < rank; k++) {
        const int m = idx[t[k + 1]];
        bf.mus.push_back(m);
        const auto it = s.bonds.find({mu0, m});
        if (it == s.bonds.end())
          c.fail("function '" + type + "' uses the pair " + t[0] + "-" + s.elements[m] +
                 ", which has no radial basis (no pair block defines one)");
        const Rad &r = it->second.rad;
        const int nr = bf.ns[k], l = bf.ls[k];
        if ((rank == 1 && nr > r.nradbase) || (rank > 1 && nr > r.nradmax) || l > r.lmax || nr < 1 || l < 0)
          c.fail("function '" + type + "' nr=" + list_str(bf.ns) + " nl=" + list_str(bf.ls) +
                 " exceeds the radial basis of the bond (nradbase " + std::to_string(r.nradbase) + ", nradmax " +
                 std::to_string(r.nradmax) + ", lmax " + std::to_string(r.lmax) + ")");
      }
      s.functions[mu0].push_back(bf);
    }
  }
  return s;
}

// ---------------------------------------------------------------------------------------------------------------
// the C-tilde basis

// (i1, i2, i_coupled) nodes of ML-PACE's coupling tree of rank r: indices < r are the ls, r .. 2r-3 the LS, 2r-2
// the final L = 0
std::vector<std::array<int, 3>> coupling_tree(int r)
{
  std::vector<std::array<int, 3>> nodes;
  if (r < 2) return nodes;
  std::deque<int> leaves, inter, queue;
  for (int k = 0; k < r; k++) leaves.push_back(k);
  for (int k = r; k < 2 * r - 2; k++) inter.push_back(k);
  int i_int = 0;
  const auto pop = [](std::deque<int> &d) {
    const int v = d.front();
    d.pop_front();
    return v;
  };
  for (int k = 0; k < r / 2; k++) {
    const int i1 = pop(leaves), i2 = pop(leaves);
    i_int = inter.empty() ? -1 : pop(inter);
    nodes.push_back({i1, i2, i_int});
    queue.push_back(i_int);
  }
  while (queue.size() > 1 && i_int != -1) {
    const int i1 = pop(queue), i2 = pop(queue);
    i_int = inter.empty() ? -1 : pop(inter);
    nodes.push_back({i1, i2, i_int});
    queue.push_back(i_int);
  }
  if (i_int != -1) {
    const int i1 = pop(queue), i2 = pop(leaves);
    nodes.push_back({i1, i2, -1});
  }
  if ((int) nodes.size() != r - 1) throw Error("coupling tree of rank " + std::to_string(r));
  for (auto &n : nodes)
    if (n[2] == -1) n[2] = 2 * r - 2;
  return nodes;
}

// ML-PACE validate_ls_LS: the couplings a B function of rank len(ls) may have
void validate_ls_lint(const std::vector<int> &ls, const std::vector<int> &lint)
{
  const int r = (int) ls.size();
  const std::string w = "ls " + list_str(ls) + ", lint " + list_str(lint) + ": ";
  if ((int) lint.size() != std::max(0, r - 2))
    throw Error(w + "lint must have " + std::to_string(std::max(0, r - 2)) + " values");
  if (r == 1 && ls[0] != 0) throw Error(w + "a rank-1 function has l = 0");
  if (r == 2 && ls[0] != ls[1]) throw Error(w + "a rank-2 function has equal ls");
  if ((r == 3 || r == 5) && lint.back() != ls.back()) throw Error(w + "the last lint must equal the last l");
  if ((r == 4 || r == 6) && lint[r - 3] != lint[r - 4]) throw Error(w + "the last two lint must be equal");
  if (r > kPaceMaxRank) throw Error(w + "rank " + std::to_string(r) + " > 6 is not supported");
  int sum = 0;
  for (int l : ls) {
    if (l < 0 || l > kPaceMaxLmax) throw Error(w + "l outside 0 .. " + std::to_string(kPaceMaxLmax));
    sum += l;
  }
  for (int l : lint)
    if (l < 0 || l > kPaceMaxRank * kPaceMaxLmax) throw Error(w + "lint outside the couplings of the ls");
  if (sum % 2) throw Error(w + "the sum of ls must be even");
}

// the factors of every product in canonical order: positions sorted by (mu, n, l) and, among equal (mu, n, l), the
// ms ascending; equal m-combinations merged (in their order), |sum| < 1e-15 dropped. Rows in lexicographic order.
struct Canonical {
  std::vector<int> mus, ns, ls;
  std::vector<std::vector<int>> ms;
  std::vector<double> cg;
};

Canonical canonical(const BFunction &bf, const MsList &l)
{
  const int r = (int) bf.mus.size();
  std::vector<int> perm(r);
  for (int k = 0; k < r; k++) perm[k] = k;
  const auto key = [&](int p) { return std::array<int, 3>{bf.mus[p], bf.ns[p], bf.ls[p]}; };
  std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) { return key(a) < key(b); });
  Canonical c;
  for (int p : perm) {
    c.mus.push_back(bf.mus[p]);
    c.ns.push_back(bf.ns[p]);
    c.ls.push_back(bf.ls[p]);
  }
  std::map<std::vector<int>, double> sum;
  for (size_t q = 0; q < l.size(); q++) {
    std::vector<int> row(r);
    for (int k = 0; k < r; k++) row[k] = l.ms[q * r + perm[k]];
    for (int a = 0; a < r;) {    // equal keys: their ms ascending
      int b = a;
      while (b + 1 < r && key(perm[b + 1]) == key(perm[a])) b++;
      std::sort(row.begin() + a, row.begin() + b + 1);
      a = b + 1;
    }
    auto it = sum.emplace(row, 0.0).first;
    it->second += l.cg[q];
  }
  for (const auto &kv : sum)
    if (std::fabs(kv.second) >= 1e-15) {
      c.ms.push_back(kv.first);
      c.cg.push_back(kv.second);
    }
  return c;
}

struct CTilde {
  int rankmax = 1;
  std::vector<int> rank, mus, ns, ls, ms;    // [rows], [rows][rankmax] (padded with 0)
  std::vector<double> ctildes;               // [rows][ndensity]
  std::vector<int> func_off, func;           // the B functions merged into each row (CSR)
  std::vector<double> func_cg;
};

class Converter {
public:
  double cg(int j1, int m1, int j2, int m2, int J, int M)
  {
    const std::array<int, 6> k{j1, m1, j2, m2, J, M};
    auto it = cg_.find(k);
    if (it == cg_.end()) it = cg_.emplace(k, clebsch_gordan(j1, m1, j2, m2, J, M)).first;
    return it->second;
  }

  const MsList &ms_list(const std::vector<int> &ls, const std::vector<int> &lint)
  {
    const auto k = std::make_pair(ls, lint);
    auto it = ms_.find(k);
    if (it == ms_.end()) it = ms_.emplace(k, build_ms_list(ls, lint)).first;
    return it->second;
  }

  MsList build_ms_list(const std::vector<int> &ls, const std::vector<int> &lint);

  CTilde to_ctilde(const std::vector<BFunction> &funcs, int nd);
  std::vector<int> b_order(const std::vector<BFunction> &funcs);

private:
  std::map<std::array<int, 6>, double> cg_;
  std::map<std::pair<std::vector<int>, std::vector<int>>, MsList> ms_;
};

MsList Converter::build_ms_list(const std::vector<int> &ls, const std::vector<int> &lint)
{
  validate_ls_lint(ls, lint);
  const int r = (int) ls.size();
  std::vector<int> lsLS(ls);
  lsLS.insert(lsLS.end(), lint.begin(), lint.end());
  lsLS.push_back(0);
  const auto nodes = coupling_tree(r);
  // the order the nodes are evaluated in (as the leaves become known one by one): the order of the products
  std::vector<int> sched;
  {
    std::vector<bool> known(2 * r - 1, false), done(nodes.size(), false);
    for (int k = 0; k < r; k++) {
      known[k] = true;
      for (bool progress = true; progress;) {
        progress = false;
        for (size_t q = 0; q < nodes.size(); q++) {
          if (done[q] || !known[nodes[q][0]] || !known[nodes[q][1]]) continue;
          sched.push_back((int) q);
          done[q] = progress = true;
          known[nodes[q][2]] = true;
        }
      }
    }
  }
  MsList out;
  out.rank = r;
  std::vector<int> m(2 * r - 1, 0);
  for (int k = 0; k + 1 < r; k++) m[k] = -ls[k];
  while (true) {    // m_0 .. m_{r-2} in lexicographic order (m_0 slowest), m_{r-1} makes the sum zero
    int s = 0;
    for (int k = 0; k + 1 < r; k++) s += m[k];
    m[r - 1] = -s;
    if (std::abs(m[r - 1]) <= ls[r - 1]) {
      double c = 1.0;
      bool keep = true;
      for (int q : sched) {
        const auto &n = nodes[q];
        const int M = m[n[0]] + m[n[1]];
        if (std::abs(M) > lsLS[n[2]]) {
          keep = false;
          break;
        }
        m[n[2]] = M;
        c = c * cg(lsLS[n[0]], m[n[0]], lsLS[n[1]], m[n[1]], lsLS[n[2]], M);
        if (c == 0.0) {
          keep = false;
          break;
        }
      }
      int first = 0;    // the first non-zero m (half basis)
      for (int k = 0; k < r && !first; k++) first = m[k];
      if (keep && first >= 0) {
        out.ms.insert(out.ms.end(), m.begin(), m.begin() + r);
        out.cg.push_back(first > 0 ? 2.0 * c : c);
      }
    }
    int k = r - 2;
    while (k >= 0 && m[k] == ls[k]) {
      m[k] = -ls[k];
      k--;
    }
    if (k < 0) break;
    m[k]++;
  }
  return out;
}

CTilde Converter::to_ctilde(const std::vector<BFunction> &funcs, int nd)
{
  struct Row {
    std::vector<double> c;
    std::vector<int> f;
    std::vector<double> g;
  };
  // groups by the canonical (mus, ns, ls): [rank, mus, ns, ls] -> ms row -> row; contributions in function order
  std::map<std::vector<int>, std::map<std::vector<int>, Row>> groups;
  for (size_t fi = 0; fi < funcs.size(); fi++) {
    const BFunction &bf = funcs[fi];
    const Canonical c = canonical(bf, ms_list(bf.ls, bf.lint));
    std::vector<int> key{(int) c.mus.size()};
    for (const auto *v : {&c.mus, &c.ns, &c.ls}) key.insert(key.end(), v->begin(), v->end());
    auto &g = groups[key];
    for (size_t q = 0; q < c.ms.size(); q++) {
      Row &row = g[c.ms[q]];
      if (row.c.empty()) row.c.assign(nd, 0.0);
      for (int p = 0; p < nd; p++) {
        const volatile double x = c.cg[q] * bf.coeffs[p];    // rounded, then added: no FMA (as numpy)
        row.c[p] += x;
      }
      row.f.push_back((int) fi);
      row.g.push_back(c.cg[q]);
    }
  }
  CTilde t;
  for (const auto &kv : groups) t.rankmax = std::max(t.rankmax, kv.first[0]);
  const int R = t.rankmax;
  // rows sorted by (rank, mus, ns, ls, ms), each padded with 0 to rankmax
  std::vector<std::pair<std::vector<int>, const Row *>> rows;
  for (const auto &kv : groups) {
    const int r = kv.first[0];
    for (const auto &rv : kv.second) {
      std::vector<int> k{r};
      for (int part = 0; part < 3; part++) {
        k.insert(k.end(), kv.first.begin() + 1 + part * r, kv.first.begin() + 1 + (part + 1) * r);
        k.insert(k.end(), R - r, 0);
      }
      k.insert(k.end(), rv.first.begin(), rv.first.end());
      k.insert(k.end(), R - r, 0);
      rows.emplace_back(std::move(k), &rv.second);
    }
  }
  std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
  t.func_off.push_back(0);
  for (const auto &[k, row] : rows) {
    t.rank.push_back(k[0]);
    for (int part = 0; part < 4; part++) {
      std::vector<int> &dst = part == 0 ? t.mus : part == 1 ? t.ns : part == 2 ? t.ls : t.ms;
      dst.insert(dst.end(), k.begin() + 1 + part * R, k.begin() + 1 + (part + 1) * R);
    }
    t.ctildes.insert(t.ctildes.end(), row->c.begin(), row->c.end());
    t.func.insert(t.func.end(), row->f.begin(), row->f.end());
    t.func_cg.insert(t.func_cg.end(), row->g.begin(), row->g.end());
    t.func_off.push_back((int) t.func.size());
  }
  return t;
}

// indices of the functions in ML-PACE's order of the B functions (ACEBBasisSet: compare of ace_abstract_basis.cpp
// on rank, mus, ns, ls, the number of m-combinations and the m-combinations as generated; rank 1 first)
std::vector<int> Converter::b_order(const std::vector<BFunction> &funcs)
{
  struct K {
    int rank;
    std::vector<int> mus, ns, ls;
    size_t nms;
    std::vector<int> ms;
    bool operator<(const K &o) const
    {
      return std::tie(rank, mus, ns, ls, nms, ms) < std::tie(o.rank, o.mus, o.ns, o.ls, o.nms, o.ms);
    }
  };
  std::vector<K> keys;
  for (const auto &bf : funcs) {
    const MsList &l = ms_list(bf.ls, bf.lint);
    keys.push_back({(int) bf.ls.size(), bf.mus, bf.ns, bf.ls, l.size(), l.ms});
  }
  std::vector<int> order(funcs.size());
  for (size_t k = 0; k < order.size(); k++) order[k] = (int) k;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return keys[a] < keys[b]; });
  return order;
}

// ---------------------------------------------------------------------------------------------------------------
// the npz arrays

template<class T>
cnpy::NpyArray array(const std::vector<T> &v, const std::vector<size_t> &shape, const char *descr)
{
  cnpy::NpyArray a(shape, sizeof(T), false);
  a.descr = descr;
  if (a.num_vals != v.size()) throw Error("PACE npz array: size does not match its shape");
  if (!v.empty()) std::memcpy(a.data<char>(), v.data(), v.size() * sizeof(T));
  return a;
}

cnpy::NpyArray i4(const std::vector<int> &v, const std::vector<size_t> &shape) { return array(v, shape, "<i4"); }
cnpy::NpyArray i4(int x) { return array(std::vector<int>{x}, {1}, "<i4"); }
cnpy::NpyArray f8(const std::vector<double> &v, const std::vector<size_t> &shape) { return array(v, shape, "<f8"); }
cnpy::NpyArray f8(double x) { return array(std::vector<double>{x}, {1}, "<f8"); }

cnpy::NpyArray names_array(const std::vector<std::string> &names)    // numpy dtype |S<longest>
{
  size_t w = 1;
  for (const auto &s : names) w = std::max(w, s.size());
  cnpy::NpyArray a({names.size()}, w, false);
  a.descr = "|S" + std::to_string(w);
  for (size_t i = 0; i < names.size(); i++) {
    std::memset(a.data<char>() + i * w, 0, w);
    std::memcpy(a.data<char>() + i * w, names[i].data(), names[i].size());
  }
  return a;
}

std::string shape_str(const std::vector<size_t> &s)    // Python's repr of a shape tuple
{
  std::string r = "(";
  for (size_t i = 0; i < s.size(); i++) r += (i ? ", " : "") + std::to_string(s[i]);
  return r + (s.size() == 1 ? ",)" : ")");
}

// the active set inverses of a pacemaker .asi (npz, one square matrix per element name), as ML-PACE's
// load_active_set; the grade 0 linear (size nfunc) or 1 non-linear (size nfunc * ndensity), the same for all
int load_asi(const std::string &path, const Spec &s, std::vector<std::vector<double>> &mats)
{
  const cnpy::npz_t z = npz::npz_open(path);
  std::vector<std::string> names;
  for (const auto &kv : z) names.push_back(kv.first);
  if (names_str(names) != names_str(s.elements))
    throw Error(path + ": active set for elements " + names_str(names) + ", the model has " + names_str(s.elements));
  std::set<int> grades;
  for (size_t mu = 0; mu < s.elements.size(); mu++) {
    const std::string &e = s.elements[mu];
    const auto &sh = z.at(e).shape;
    if (sh.size() != 2 || sh[0] != sh[1])
      throw Error(path + ": the active set of " + e + " is not square: shape " + shape_str(sh));
    const size_t nf = s.functions[mu].size(), nd = s.embeddings[mu].ndensity;
    if (sh[0] == nf)
      grades.insert(0);
    else if (sh[0] == nf * nd)
      grades.insert(1);
    else
      throw Error(path + ": the active set of " + e + " has size " + std::to_string(sh[0]) + ", expected " +
                  std::to_string(nf) + " (linear) or " + std::to_string(nf * nd) + " (non-linear)");
    mats.push_back(npz::npz_get_double(z, e));
  }
  if (grades.size() != 1) throw Error(path + ": the grade (linear or non-linear) differs between the elements");
  return *grades.begin();
}

}    // namespace

double clebsch_gordan(int j1, int m1, int j2, int m2, int J, int M)
{
  if (m1 + m2 != M || !(std::abs(j1 - j2) <= J && J <= j1 + j2) || std::abs(m1) > j1 || std::abs(m2) > j2 ||
      std::abs(M) > J)
    return 0.0;
  // Racah: sqrt(pre / den) * sum_z (-1)^z / d_z, the sum as N / D with D the lcm of the d_z (as the Python version)
  const size_t np = primes().size();
  const int zmin = std::max({0, j2 - J - m1, j1 - J + m2}), zmax = std::min({j1 + j2 - J, j1 - m1, j2 + m2});
  std::vector<Exp> dz;
  Exp eD(np, 0);
  for (int z = zmin; z <= zmax; z++) {
    Exp e(np, 0);
    for (int n : {z, j1 + j2 - J - z, j1 - m1 - z, j2 + m2 - z, J - j2 + m1 + z, J - j1 - m2 + z})
      add_factorial(e, n, 1);
    for (size_t k = 0; k < np; k++) eD[k] = std::max(eD[k], e[k]);
    dz.push_back(e);
  }
  Big pos, neg;
  for (int z = zmin; z <= zmax; z++) {
    Exp e(np);
    for (size_t k = 0; k < np; k++) e[k] = eD[k] - dz[z - zmin][k];
    add_to((z % 2) ? neg : pos, product(e, 1));
  }
  norm(pos);
  norm(neg);
  const int c = cmp(pos, neg);
  if (c == 0) return 0.0;
  Big N = c > 0 ? pos : neg;
  sub_from(N, c > 0 ? neg : pos);
  Exp e(np, 0);
  add_int(e, 2 * J + 1);
  for (int n : {j1 + j2 - J, j1 - j2 + J, -j1 + j2 + J, J + M, J - M, j1 + m1, j1 - m1, j2 + m2, j2 - m2})
    add_factorial(e, n, 1);
  add_factorial(e, j1 + j2 + J + 1, -1);
  const Big one{1};
  const double s = std::sqrt(ratio(product(e, 1), product(e, -1)));
  return s * (c > 0 ? ratio(N, one) : -ratio(N, one)) / ratio(product(eD, 1), one);
}

MsList ms_cg_list(const std::vector<int> &ls, const std::vector<int> &lint)
{
  return Converter().build_ms_list(ls, lint);
}

cnpy::npz_t npz_from_yaml(const std::string &yaml_path, const std::string &asi_path)
{
  const Spec s = load_spec(yaml_path);
  const int nel = (int) s.elements.size();
  Converter cv;
  cnpy::npz_t a;
  std::vector<int> z(nel);
  for (int mu = 0; mu < nel; mu++) z[mu] = std::max(0, atomic_number(s.elements[mu]));
  a["pace_format_version"] = i4(kPaceFormatVersion);
  a["element_names"] = names_array(s.elements);
  a["element_z"] = i4(z, {(size_t) nel});
  a["inner_cutoff_type"] = i4(s.inner_cutoff_type);
  std::vector<CTilde> ct;
  for (int mu = 0; mu < nel; mu++) {
    const Embedding &e = s.embeddings[mu];
    const std::string p = "e" + std::to_string(mu) + "_";
    const CTilde t = cv.to_ctilde(s.functions[mu], e.ndensity);
    const size_t rows = t.rank.size(), R = t.rankmax;
    a[p + "ndensity"] = i4(e.ndensity);
    a[p + "npoti"] = i4(e.npoti);
    a[p + "fs_parameters"] = f8(e.parameters, {e.parameters.size()});
    a[p + "rho_core_cut"] = f8(e.rho_core_cut);
    a[p + "drho_core_cut"] = f8(e.drho_core_cut);
    a[p + "rank"] = i4(t.rank, {rows});
    a[p + "mus"] = i4(t.mus, {rows, R});
    a[p + "ns"] = i4(t.ns, {rows, R});
    a[p + "ls"] = i4(t.ls, {rows, R});
    a[p + "ms"] = i4(t.ms, {rows, R});
    a[p + "ctildes"] = f8(t.ctildes, {rows, (size_t) e.ndensity});
    ct.push_back(t);
  }
  if (!asi_path.empty()) {
    std::vector<std::vector<double>> mats;
    a["uq_grade"] = i4(load_asi(asi_path, s, mats));
    for (int mu = 0; mu < nel; mu++) {
      const auto &funcs = s.functions[mu];
      std::vector<int> pos(funcs.size());    // function of the yaml -> its index in ML-PACE's order
      const auto order = cv.b_order(funcs);
      for (size_t k = 0; k < order.size(); k++) pos[order[k]] = (int) k;
      std::vector<int> f(ct[mu].func.size());
      for (size_t k = 0; k < f.size(); k++) f[k] = pos[ct[mu].func[k]];
      const std::string p = "e" + std::to_string(mu) + "_";
      const size_t n = (size_t) std::sqrt((double) mats[mu].size());
      a[p + "uq_nfunc"] = i4((int) funcs.size());
      a[p + "uq_off"] = i4(ct[mu].func_off, {ct[mu].func_off.size()});
      a[p + "uq_func"] = i4(f, {f.size()});
      a[p + "uq_cg"] = f8(ct[mu].func_cg, {ct[mu].func_cg.size()});
      a[p + "asi"] = f8(mats[mu], {n, n});
    }
  }
  std::vector<int> defined((size_t) nel * nel, 0);
  for (const auto &[ij, b] : s.bonds) {
    const auto [i, j] = ij;
    defined[(size_t) i * nel + j] = 1;
    const std::string p = "b" + std::to_string(i) + "_" + std::to_string(j) + "_";
    const Rad &r = b.rad;
    a[p + "radbase"] = i4(index_of(kRadbases, r.radbase));
    a[p + "radparameters"] = f8(r.radparameters, {r.radparameters.size()});
    a[p + "crad"] = f8(r.crad, {(size_t) r.nradmax, (size_t) r.lmax + 1, (size_t) r.nradbase});
    a[p + "nradmax"] = i4(r.nradmax);
    a[p + "lmax"] = i4(r.lmax);
    a[p + "nradbase"] = i4(r.nradbase);
    a[p + "rcut"] = f8(r.rcut);
    a[p + "dcut"] = f8(r.dcut);
    a[p + "r_in"] = f8(b.r_in);
    a[p + "delta_in"] = f8(b.delta_in);
    a[p + "prehc"] = f8(b.prehc);
    a[p + "lambdahc"] = f8(b.lambdahc);
  }
  a["bond_defined"] = i4(defined, {(size_t) nel, (size_t) nel});
  return a;
}

}    // namespace grace::pace
