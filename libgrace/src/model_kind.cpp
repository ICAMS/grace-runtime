// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
#include "grace/model_kind.h"

#include "grace/error.h"
#include "grace/pace_model.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace grace {

namespace {
struct Kind {
  ModelKind kind;
  const char *member;    // a member only the export of this kind has; the first one present decides
  const char *name, *tag;
};
// in the order of the enum
constexpr Kind kKinds[] = {{ModelKind::Grace1L, "mlp_rad_W0.npy", "GRACE-1L", "1l"},
                           {ModelKind::Grace2L, "B0_lin_transform_W.npy", "GRACE-2L", "2l"},
                           {ModelKind::Grace3L, "prod_names.npy", "GRACE-3L", "3l"},
                           {ModelKind::Pace, "pace_format_version.npy", "PACE", "pace"}};
static_assert(kKinds[3].kind == ModelKind::Pace, "kKinds follows the enum");

uint64_t le(const uint8_t *p, int n)
{
  uint64_t v = 0;
  for (int k = n - 1; k >= 0; k--) v = v << 8 | p[k];
  return v;
}

bool read_at(std::ifstream &f, uint64_t off, std::vector<uint8_t> &b)
{
  f.clear();
  f.seekg((std::streamoff) off);
  return b.empty() || (bool) f.read(reinterpret_cast<char *>(b.data()), (std::streamsize) b.size());
}

struct Member {
  std::string name;
  uint64_t offset;    // of its local header; 0xFFFFFFFF: in a ZIP64 extra (past 4 GB), not read here
};

// the members of the zip's central directory (only the end of the file and the directory are read; ZIP64, which
// numpy writes past 4 GB, through its locator); false when the file is not a zip whose directory parses
bool zip_members(std::ifstream &f, std::vector<Member> &members)
{
  f.seekg(0, std::ios::end);
  const uint64_t size = (uint64_t) f.tellg();
  if (size < 22) return false;
  std::vector<uint8_t> b((size_t) std::min<uint64_t>(size, 65536 + 22));
  const uint64_t b0 = size - b.size();
  if (!read_at(f, b0, b)) return false;
  int64_t e = (int64_t) b.size() - 22;    // the end-of-central-directory record, searched backwards past a comment
  while (e >= 0 && le(&b[e], 4) != 0x06054b50u) e--;
  if (e < 0) return false;
  uint64_t cd_size = le(&b[e + 12], 4), cd_off = le(&b[e + 16], 4);
  if (cd_size == 0xFFFFFFFFu || cd_off == 0xFFFFFFFFu) {    // ZIP64: the locator just before the record
    std::vector<uint8_t> loc(20), rec(56);
    if (b0 + e < 20 || !read_at(f, b0 + e - 20, loc) || le(&loc[0], 4) != 0x07064b50u) return false;
    if (!read_at(f, le(&loc[8], 8), rec) || le(&rec[0], 4) != 0x06064b50u) return false;
    cd_size = le(&rec[40], 8);
    cd_off = le(&rec[48], 8);
  }
  if (cd_off > size || cd_size > size - cd_off) return false;
  std::vector<uint8_t> cd((size_t) cd_size);
  if (!read_at(f, cd_off, cd)) return false;
  size_t q = 0;
  while (q + 46 <= cd.size() && le(&cd[q], 4) == 0x02014b50u) {
    const size_t nl = le(&cd[q + 28], 2), xl = le(&cd[q + 30], 2), cl = le(&cd[q + 32], 2);
    if (q + 46 + nl + xl + cl > cd.size()) return false;
    members.push_back({std::string(reinterpret_cast<const char *>(&cd[q + 46]), nl), le(&cd[q + 42], 4)});
    q += 46 + nl + xl + cl;
  }
  return q == cd.size();
}

// the string of a stored 1-element byte-string member (np.array([b"GRACE-2L"]): dtype |S<n>, shape (1,)); "" when the
// member is not that. Found through the central directory: cnpy's npz_load(path, name) skips members by the 32-bit
// sizes of their local headers, which np.savez (ZIP64 members) sets to 0xFFFFFFFF
std::string member_string(std::ifstream &f, uint64_t offset)
{
  std::vector<uint8_t> h(30), n(10);
  if (offset == 0xFFFFFFFFu || !read_at(f, offset, h) || le(&h[0], 4) != 0x04034b50u || le(&h[8], 2) != 0) return "";
  const uint64_t data = offset + 30 + le(&h[26], 2) + le(&h[28], 2);
  if (!read_at(f, data, n) || n[0] != 0x93 || std::string(n.begin() + 1, n.begin() + 6) != "NUMPY") return "";
  const bool v1 = n[6] == 1;
  std::vector<uint8_t> l(v1 ? 2 : 4);
  if (!read_at(f, data + 8, l)) return "";
  const uint64_t hlen = le(l.data(), (int) l.size());
  if (hlen > 1024) return "";
  std::vector<uint8_t> hb((size_t) hlen);
  if (!read_at(f, data + 8 + l.size(), hb)) return "";
  const std::string hdr(hb.begin(), hb.end());
  const size_t d = hdr.find("'descr': '|S"), s = hdr.find("'shape': (1,)");
  if (d == std::string::npos || s == std::string::npos) return "";
  const int width = std::atoi(hdr.c_str() + d + 12);
  if (width < 1 || width > 64) return "";
  std::vector<uint8_t> v((size_t) width);
  if (!read_at(f, data + 8 + l.size() + hlen, v)) return "";
  std::string out(v.begin(), v.end());
  out.erase(std::find(out.begin(), out.end(), '\0'), out.end());
  return out;
}
}    // namespace

ModelKind model_kind(const std::string &path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error("cannot open " + path);
  if (is_pace_yaml(path)) return ModelKind::Pace;
  std::vector<Member> members;
  if (zip_members(f, members)) {
    const auto find_member = [&](const char *name) {
      return std::find_if(members.begin(), members.end(), [&](const Member &m) { return m.name == name; });
    };
    // the kind the exporter wrote (tensorpotential, grace_format_version >= 1; its first member); older exports, and a
    // member this reader cannot read (compressed, another dtype): the marker members, the loader then says what is wrong
    if (auto it = find_member("grace_model_kind.npy"); it != members.end()) {
      if (const std::string kind = member_string(f, it->offset); !kind.empty()) {
        for (const Kind &k : kKinds)
          if (kind == k.name && k.kind != ModelKind::Pace) return k.kind;
        throw Error("unknown grace_model_kind '" + kind + "' (GRACE-1L, GRACE-2L or GRACE-3L) in " + path);
      }
    }
    for (const Kind &k : kKinds)
      if (find_member(k.member) != members.end()) return k.kind;
  }
  throw Error("not a GRACE-1L/2L/3L kokkos.npz or a PACE npz / yaml: " + path);
}

const char *model_kind_name(ModelKind k) { return kKinds[static_cast<int>(k)].name; }
const char *model_kind_tag(ModelKind k) { return kKinds[static_cast<int>(k)].tag; }

}    // namespace grace
