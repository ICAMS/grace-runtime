// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
#include "grace/error.h"
#include "test_util.h"
#include <string>
static std::string last;
static void sink(void *, int level, const char *msg) { last = std::to_string(level) + ":" + msg; }
int main()
{
  grace::Logger log{sink, nullptr};
  log.info("hello");
  CHECK(last == "0:hello");
  log.warn("w");
  CHECK(last == "1:w");
  grace::Logger quiet{};          // no sink: must not crash
  quiet.info("x");
  bool caught = false;
  try { throw grace::Error("boom"); } catch (const std::runtime_error &e) { caught = std::string(e.what()) == "boom"; }
  CHECK(caught);
  return 0;
}
