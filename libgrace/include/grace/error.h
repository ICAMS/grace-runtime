// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
#pragma once
#include <stdexcept>
#include <string>
namespace grace {
struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};
using LogFn = void (*)(void *user, int level, const char *msg);    // level 0 info, 1 warning
struct Logger {
  LogFn fn = nullptr;
  void *user = nullptr;
  void info(const std::string &m) const { if (fn) fn(user, 0, m.c_str()); }
  void warn(const std::string &m) const { if (fn) fn(user, 1, m.c_str()); }
};
}    // namespace grace
