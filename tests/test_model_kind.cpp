// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// grace::model_kind: the kind of a model file from its grace_model_kind, its npz member names (GRACE-1L / 2L / 3L, PACE)
// or its .yaml suffix
#include "grace/error.h"
#include "grace/model_kind.h"
#include "test_util.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

// a stored (uncompressed) zip whose members are the given names, each empty. comment: an archive comment after the
// end record; zip64: the directory found through a ZIP64 end record and locator (as numpy writes past 4 GB); extra:
// bytes the end record claims for the directory beyond its entries (a directory that does not parse)
static void write_zip(const std::string &path, const std::vector<std::string> &names, const std::string &comment = "",
                      bool zip64 = false, uint32_t extra = 0)
{
  std::vector<char> zip, cd;
  auto u16 = [](std::vector<char> &b, uint16_t v) { b.push_back((char) (v & 0xff)); b.push_back((char) (v >> 8)); };
  auto u32 = [&](std::vector<char> &b, uint32_t v) { u16(b, (uint16_t) (v & 0xffff)); u16(b, (uint16_t) (v >> 16)); };
  for (const auto &n : names) {
    const uint32_t off = (uint32_t) zip.size();
    zip.insert(zip.end(), {'P', 'K', 3, 4});
    u16(zip, 20); u16(zip, 0); u16(zip, 0); u16(zip, 0); u16(zip, 0); u32(zip, 0); u32(zip, 0); u32(zip, 0);
    u16(zip, (uint16_t) n.size()); u16(zip, 0);
    zip.insert(zip.end(), n.begin(), n.end());
    cd.insert(cd.end(), {'P', 'K', 1, 2});
    u16(cd, 20); u16(cd, 20); u16(cd, 0); u16(cd, 0); u16(cd, 0); u16(cd, 0); u32(cd, 0); u32(cd, 0); u32(cd, 0);
    u16(cd, (uint16_t) n.size()); u16(cd, 0); u16(cd, 0); u16(cd, 0); u16(cd, 0); u32(cd, 0); u32(cd, off);
    cd.insert(cd.end(), n.begin(), n.end());
  }
  const uint32_t cd_off = (uint32_t) zip.size();
  zip.insert(zip.end(), cd.begin(), cd.end());
  if (zip64) {
    const uint32_t rec = (uint32_t) zip.size();
    auto u64 = [&](uint64_t v) { u32(zip, (uint32_t) v); u32(zip, (uint32_t) (v >> 32)); };
    zip.insert(zip.end(), {'P', 'K', 6, 6});
    u64(44); u16(zip, 45); u16(zip, 45); u32(zip, 0); u32(zip, 0);
    u64(names.size()); u64(names.size()); u64(cd.size()); u64(cd_off);
    zip.insert(zip.end(), {'P', 'K', 6, 7});
    u32(zip, 0); u64(rec); u32(zip, 1);
  }
  zip.insert(zip.end(), {'P', 'K', 5, 6});
  u16(zip, 0); u16(zip, 0); u16(zip, (uint16_t) names.size()); u16(zip, (uint16_t) names.size());
  u32(zip, zip64 ? 0xFFFFFFFFu : (uint32_t) cd.size() + extra); u32(zip, zip64 ? 0xFFFFFFFFu : cd_off);
  u16(zip, (uint16_t) comment.size());
  zip.insert(zip.end(), comment.begin(), comment.end());
  std::ofstream(path, std::ios::binary).write(zip.data(), (std::streamsize) zip.size());
}

static bool throws(const std::string &path)
{
  try {
    grace::model_kind(path);
  } catch (const grace::Error &) {
    return true;
  }
  return false;
}

int main()
{
  using grace::ModelKind;
  // model-independent cases first (the registry models below skip the test without the cache); files in the temp dir
  const std::string tmp = (std::filesystem::temp_directory_path() / ("grace_test_kind_" + std::to_string(getpid())))
                              .string();
  const std::string pace_npz = tmp + "_pace.npz", other_npz = tmp + "_other.npz", text = tmp + "_text.txt";
  const std::string comment_npz = tmp + "_comment.npz", zip64_npz = tmp + "_zip64.npz", cut_npz = tmp + "_cut.npz";
  const std::string yml = tmp + "_model.yml";
  write_zip(pace_npz, {"elements.npy", "pace_format_version.npy"});
  write_zip(other_npz, {"weights.npy"});
  write_zip(comment_npz, {"prod_names.npy"}, "an archive comment");
  write_zip(zip64_npz, {"x.npy", "B0_lin_transform_W.npy"}, "", true);
  write_zip(cut_npz, {"mlp_rad_W0.npy"}, "", false, 10);
  std::ofstream(text) << "not a model\n";
  std::ofstream(yml) << "global:\n";
  CHECK(grace::model_kind(pace_npz) == ModelKind::Pace);
  CHECK(grace::model_kind(comment_npz) == ModelKind::Grace3L);    // the end record searched back past the comment
  CHECK(grace::model_kind(zip64_npz) == ModelKind::Grace2L);
  CHECK(grace::model_kind(yml) == ModelKind::Pace);
  CHECK(throws(other_npz));    // a zip without a marker member
  CHECK(throws(cut_npz));      // a central directory that does not parse
  CHECK(throws(text));         // neither a zip nor a .yaml
  CHECK(throws(tmp + "_missing.npz"));
  CHECK(throws(tmp + "_missing.yaml"));
  for (const auto &p : {pace_npz, other_npz, comment_npz, zip64_npz, cut_npz, text, yml}) std::filesystem::remove(p);
  CHECK(grace::model_kind("tests/fixtures/pace/carbon_zbl.yaml") == ModelKind::Pace);
  // grace_model_kind decides over the marker members (tiny files written by np.savez, as the exporter writes them)
  CHECK(grace::model_kind("tests/fixtures/model_kind/kind_3l_with_1l_marker.npz") == ModelKind::Grace3L);
  CHECK(throws("tests/fixtures/model_kind/kind_unknown.npz"));
  CHECK(grace::model_kind("tests/fixtures/model_kind/kind_unreadable_1l_marker.npz") == ModelKind::Grace1L);
  CHECK(std::string(grace::model_kind_name(ModelKind::Grace3L)) == "GRACE-3L");
  CHECK(std::string(grace::model_kind_tag(ModelKind::Grace2L)) == "2l");

  CHECK(grace::model_kind(test_model("GRACE_TEST_M1", "GRACE-1L-OMAT-large-base")) == ModelKind::Grace1L);
  CHECK(grace::model_kind(test_model("GRACE_TEST_M2", "GRACE-2L-SMAX-OMAT-medium")) == ModelKind::Grace2L);
  CHECK(grace::model_kind(test_model("GRACE_TEST_M3", "GRACE-3L-OMAT-large")) == ModelKind::Grace3L);
  return 0;
}
