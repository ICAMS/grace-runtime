// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "%s:%d CHECK failed: %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)
#define CHECK_NEAR(a, b, tol) do { double _a = (a), _b = (b); if (!(std::fabs(_a - _b) <= (tol))) { \
  std::fprintf(stderr, "%s:%d %s=%.17g %s=%.17g tol %g\n", __FILE__, __LINE__, #a, _a, #b, _b, (double) (tol)); std::exit(1); } } while (0)
inline const char *env_or(const char *k, const char *d) { const char *v = std::getenv(k); return v ? v : d; }
// test model: $<env> if set, else <GRACE_CACHE or ~/.cache/grace>/<registry name>/kokkos.npz (grace-runtime models download
// <name> fetches it); a missing file skips the test (exit code 77, SKIP_RETURN_CODE in CMakeLists)
inline const char *test_model(const char *env, const char *name)
{
  static std::string keep[8];
  static int n = 0;
  if (const char *v = std::getenv(env)) return v;
  const char *c = std::getenv("GRACE_CACHE");
  std::string p = c ? std::string(c) : std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/.cache/grace";
  p += std::string("/") + name + "/kokkos.npz";
  struct stat st;
  if (stat(p.c_str(), &st) != 0) {
    std::fprintf(stderr, "SKIP: test model %s not found (%s); grace-runtime models download %s\n", name, p.c_str(), name);
    std::exit(77);
  }
  keep[n % 8] = p;
  return keep[n++ % 8].c_str();
}
