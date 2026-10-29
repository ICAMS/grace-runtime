// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// fused::cgroup_memory_mb on fake cgroup trees (v2, v1, none); host_memory_mb stays within /proc/meminfo
#include "grace/fused_cpu.h"
#include "test_util.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;

static void put(const fs::path &p, const std::string &text)
{
  fs::create_directories(p.parent_path());
  std::ofstream(p) << text << "\n";
}

int main()
{
  using grace::fused::cgroup_memory_mb;
  const fs::path tmp = fs::temp_directory_path() / ("grace_test_cgroup_" + std::to_string(getpid()));
  const double GiB = 1024.0;    // in MB
  double total = 0.0, avail = 0.0;
  const auto cg = [&](const char *t) {    // the fake tree tmp/<t>: proc file and cgroup root
    return cgroup_memory_mb((tmp / t / "proc").c_str(), (tmp / t / "root").string(), total, avail);
  };
  fs::remove_all(tmp);    // a tree left by a failed run with the same pid

  // v2 (Slurm): the job limit (16 GiB, 4 GiB used) on an ancestor, the step without a limit ("max")
  put(tmp / "v2/proc", "0::/system.slice/job_1/step_0");
  put(tmp / "v2/root/system.slice/job_1/memory.max", "17179869184");
  put(tmp / "v2/root/system.slice/job_1/memory.current", "4294967296");
  put(tmp / "v2/root/system.slice/job_1/step_0/memory.max", "max");
  CHECK(cg("v2"));
  CHECK_NEAR(total, 16 * GiB, 1e-6);
  CHECK_NEAR(avail, 12 * GiB, 1e-6);
  // 1 GiB of the usage is inactive page cache (reclaimable): available again
  put(tmp / "v2/root/system.slice/job_1/memory.stat", "anon 3221225472\ninactive_file 1073741824\nactive_file 0");
  CHECK(cg("v2"));
  CHECK_NEAR(avail, 13 * GiB, 1e-6);
  // usage above the limit (a full job): 1 MB available, not 0 (host_memory_mb's "unreadable")
  put(tmp / "v2/root/system.slice/job_1/memory.current", "18253611008");
  CHECK(cg("v2"));
  CHECK_NEAR(avail, 1.0, 1e-6);

  // v1: the memory controller's line among others; the parent "unlimited" (near 2^63), the job 8 GiB with 1 GiB used
  // (a hybrid host: the v2 line "0::" is ignored when v1 has the memory controller)
  put(tmp / "v1/proc",
      "11:pids:/slurm/uid_1/job_2\n5:cpuacct,cpu:/slurm/uid_1/job_2\n4:memory:/slurm/uid_1/job_2\n0::/x");
  put(tmp / "v1/root/memory/slurm/memory.limit_in_bytes", "9223372036854771712");
  put(tmp / "v1/root/memory/slurm/uid_1/job_2/memory.limit_in_bytes", "8589934592");
  put(tmp / "v1/root/memory/slurm/uid_1/job_2/memory.usage_in_bytes", "1073741824");
  CHECK(cg("v1"));
  CHECK_NEAR(total, 8 * GiB, 1e-6);
  CHECK_NEAR(avail, 7 * GiB, 1e-6);
  // v1's inactive page cache key
  put(tmp / "v1/root/memory/slurm/uid_1/job_2/memory.stat", "cache 536870912\ntotal_inactive_file 536870912");
  CHECK(cg("v1"));
  CHECK_NEAR(avail, 7.5 * GiB, 1e-6);

  // a container (cgroup namespace: path "/"), limit at the root of the tree
  put(tmp / "ns/proc", "0::/");
  put(tmp / "ns/root/memory.max", "2147483648");
  put(tmp / "ns/root/memory.current", "0");
  CHECK(cg("ns"));
  CHECK_NEAR(total, 2 * GiB, 1e-6);
  // cgroup v2 memory.high below memory.max: the smaller one is the limit
  put(tmp / "ns/root/memory.high", "1073741824");
  CHECK(cg("ns"));
  CHECK_NEAR(total, 1 * GiB, 1e-6);

  // no limit anywhere, and no /proc file: false
  put(tmp / "none/proc", "0::/user.slice");
  put(tmp / "none/root/user.slice/memory.max", "max");
  CHECK(!cg("none"));
  CHECK(!cgroup_memory_mb((tmp / "missing").c_str(), (tmp / "none/root").string(), total, avail));
  // a path outside the cgroup namespace
  put(tmp / "out/proc", "0::/../../x");
  CHECK(!cgroup_memory_mb((tmp / "out/proc").c_str(), (tmp / "none/root").string(), total, avail));
  // a cgroup name that only starts with ".." is inside
  put(tmp / "dots/proc", "0::/..job");
  put(tmp / "dots/root/..job/memory.max", "1073741824");
  CHECK(cg("dots"));
  CHECK_NEAR(total, 1 * GiB, 1e-6);
  fs::remove_all(tmp);

#if defined(__linux__)
  // this host: within MemTotal, and the available part within the total
  grace::fused::host_memory_mb(total, avail);
  std::printf("host memory %.0f MB, available %.0f MB\n", total, avail);
  CHECK(total > 0.0 && avail > 0.0 && avail <= total);
#endif
  return 0;
}
