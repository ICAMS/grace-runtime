// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// A reader of the YAML subset of pacemaker B-basis files (the yaml-cpp emitter of ML-PACE / pyace, PyYAML safe_dump):
// block mappings and sequences by indentation (a sequence may sit at its key's indentation), flow sequences and
// mappings (also over several lines), plain and quoted scalars, comments. Anchors, aliases, tags, block scalars
// (| >) and several documents are a grace::Error naming the line.
#ifndef GRACE_PACE_YAML_H
#define GRACE_PACE_YAML_H

#include <string>
#include <vector>

namespace grace::yaml {

struct Node {
  enum Kind { Null, Scalar, Seq, Map };
  struct Entry;    // key: value of a Map (std::pair<std::string, Node> would need Node complete here)
  Kind kind = Null;
  std::string str;    // Scalar: the text (quotes removed, escapes resolved)
  bool quoted = false;
  std::vector<Node> seq;
  std::vector<Entry> map;    // in file order
  int line = 0;    // 1-based, for messages

  const Node *find(const std::string &key) const;    // Map: the value of key, or nullptr
  bool is_null() const;    // Null, or a plain ~ / null / Null / NULL
};

struct Node::Entry {
  std::string key;
  Node value;
};

Node parse(const std::string &text, const std::string &name);    // name: for messages ("<name>:<line>: ...")
Node load_file(const std::string &path);

}    // namespace grace::yaml

#endif
