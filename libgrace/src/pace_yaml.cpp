// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The YAML subset of pacemaker B-basis files (pace_yaml.h): lines with their indentation (comments stripped), then
// recursive descent over the blocks; flow collections are parsed character by character after their lines are joined.
#include "pace_yaml.h"
#include "grace/error.h"

#include <fstream>
#include <sstream>
#include <utility>

namespace grace::yaml {

const Node *Node::find(const std::string &key) const
{
  for (const auto &kv : map)
    if (kv.key == key) return &kv.value;
  return nullptr;
}

bool Node::is_null() const
{
  return kind == Null || (kind == Scalar && !quoted && (str == "~" || str == "null" || str == "Null" || str == "NULL"));
}

namespace {

struct Line {
  int indent;
  std::string text;    // without the indentation, the comment and trailing blanks
  int no;
};

std::string trim(const std::string &s)
{
  const size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t");
  return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

bool is_seq_item(const std::string &t) { return t == "-" || (t.size() >= 2 && t[0] == '-' && t[1] == ' '); }

class Parser {
public:
  Parser(const std::string &text, const std::string &name) : name_(name) { split(text); }

  Node document()
  {
    if (lines_.empty()) return Node();
    Node n = block();
    if (pos_ < lines_.size()) fail(lines_[pos_].no, "unexpected content (indentation?)");
    return n;
  }

private:
  std::string name_;
  std::vector<Line> lines_;
  size_t pos_ = 0;

  [[noreturn]] void fail(int no, const std::string &what) const
  {
    throw Error(name_ + ":" + std::to_string(no) + ": " + what);
  }

  void split(const std::string &text)
  {
    std::istringstream in(text);
    std::string raw;
    for (int no = 1; std::getline(in, raw); no++) {
      if (!raw.empty() && raw.back() == '\r') raw.pop_back();
      size_t ind = 0;
      while (ind < raw.size() && (raw[ind] == ' ' || raw[ind] == '\t')) {
        if (raw[ind] == '\t') fail(no, "tab in the indentation");
        ind++;
      }
      std::string t = trim(strip_comment(raw.substr(ind)));
      if (t.empty() || (ind == 0 && t[0] == '%')) continue;    // blank, comment, directive
      if (ind == 0 && (t == "---" || t == "...")) {
        if (!lines_.empty() && t == "---") fail(no, "several documents are not supported");
        continue;
      }
      lines_.push_back({(int) ind, t, no});
    }
  }

  static std::string strip_comment(const std::string &s)
  {
    char q = 0;
    for (size_t i = 0; i < s.size(); i++) {
      const char c = s[i];
      if (q) {
        if (q == '"' && c == '\\') i++;
        else if (c == q) q = 0;
      } else if (c == '"' || c == '\'') {
        q = c;
      } else if (c == '#' && (i == 0 || s[i - 1] == ' ' || s[i - 1] == '\t')) {
        return s.substr(0, i);
      }
    }
    return s;
  }

  std::string quoted(const std::string &s, size_t &i, int no) const    // s[i] is the opening quote
  {
    const char q = s[i++];
    std::string out;
    while (i < s.size()) {
      const char c = s[i++];
      if (q == '\'' && c == '\'') {
        if (i < s.size() && s[i] == '\'') {
          out += '\'';
          i++;
          continue;
        }
        return out;
      }
      if (q == '"' && c == '"') return out;
      if (q == '"' && c == '\\') {
        if (i >= s.size()) break;
        const char e = s[i++];
        switch (e) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case '0': out += '\0'; break;
          case '\\': case '"': case '/': out += e; break;
          default: fail(no, std::string("unsupported escape \\") + e);
        }
        continue;
      }
      out += c;
    }
    fail(no, "unclosed quoted scalar");
  }

  // "key: rest" (rest may be empty); false for anything else (a scalar or a flow collection)
  bool split_key(const std::string &t, std::string &key, std::string &rest, int no) const
  {
    if (t.empty() || t[0] == '[' || t[0] == '{') return false;
    size_t p;
    if (t[0] == '"' || t[0] == '\'') {
      size_t i = 0;
      std::string k = quoted(t, i, no);
      while (i < t.size() && t[i] == ' ') i++;
      if (i >= t.size() || t[i] != ':' || (i + 1 < t.size() && t[i + 1] != ' ')) return false;
      key = k;
      p = i;
    } else {
      p = t.find(':');
      while (p != std::string::npos && p + 1 < t.size() && t[p + 1] != ' ') p = t.find(':', p + 1);
      if (p == std::string::npos) return false;
      key = trim(t.substr(0, p));
    }
    rest = trim(t.substr(p + 1));
    return true;
  }

  Node block()
  {
    const Line &L = lines_[pos_];
    std::string k, r;
    if (is_seq_item(L.text)) return block_seq(L.indent);
    if (split_key(L.text, k, r, L.no)) return block_map(L.indent);
    pos_++;
    return inline_value(L.text, L.no);
  }

  Node block_map(int ind)
  {
    Node m;
    m.kind = Node::Map;
    m.line = lines_[pos_].no;
    while (pos_ < lines_.size()) {
      const Line L = lines_[pos_];
      if (L.indent < ind || (L.indent == ind && is_seq_item(L.text))) break;
      if (L.indent > ind) fail(L.no, "unexpected indentation");
      std::string key, rest;
      if (!split_key(L.text, key, rest, L.no)) fail(L.no, "expected 'key: value'");
      if (m.find(key)) fail(L.no, "duplicate key " + key);
      pos_++;
      Node v;
      v.line = L.no;
      if (!rest.empty())
        v = inline_value(rest, L.no);
      else if (pos_ < lines_.size() &&
               (lines_[pos_].indent > ind || (lines_[pos_].indent == ind && is_seq_item(lines_[pos_].text))))
        v = block();    // a nested block; a sequence may sit at the key's indentation
      m.map.push_back({std::move(key), std::move(v)});
    }
    return m;
  }

  Node block_seq(int ind)
  {
    Node s;
    s.kind = Node::Seq;
    s.line = lines_[pos_].no;
    while (pos_ < lines_.size()) {
      Line &L = lines_[pos_];
      if (L.indent != ind || !is_seq_item(L.text)) {
        if (L.indent > ind) fail(L.no, "unexpected indentation");
        break;
      }
      const std::string rest = trim(L.text.substr(1));
      std::string k, r;
      Node item;
      item.line = L.no;
      if (rest.empty()) {
        pos_++;
        if (pos_ < lines_.size() && lines_[pos_].indent > ind) item = block();
      } else if (is_seq_item(rest) || split_key(rest, k, r, L.no)) {    // "- key: v", "- - v": a block at its column
        L.indent = ind + (int) (L.text.size() - rest.size());
        L.text = rest;
        item = block();
      } else {
        const int no = L.no;
        pos_++;
        item = inline_value(rest, no);
      }
      s.seq.push_back(std::move(item));
    }
    return s;
  }

  static int flow_depth(const std::string &s)
  {
    int d = 0;
    char q = 0;
    for (size_t i = 0; i < s.size(); i++) {
      const char c = s[i];
      if (q) {
        if (q == '"' && c == '\\') i++;
        else if (c == q) q = 0;
      } else if (c == '"' || c == '\'') {
        q = c;
      } else if (c == '[' || c == '{') {
        d++;
      } else if (c == ']' || c == '}') {
        d--;
      }
    }
    return d;
  }

  // a value after "key:" / "- " or on its own line; the line is consumed already, a flow collection may take more
  Node inline_value(std::string t, int no)
  {
    const char c = t[0];
    if (c == '|' || c == '>') fail(no, "block scalars (| >) are not supported");
    if (c == '&' || c == '*') fail(no, "anchors and aliases are not supported");
    if (c == '!') fail(no, "tags are not supported");
    if (c == '[' || c == '{') {
      while (flow_depth(t) > 0) {
        if (pos_ >= lines_.size()) fail(no, "unclosed flow collection");
        t += " " + lines_[pos_++].text;
      }
      size_t i = 0;
      Node n = flow(t, i, no);
      skip_ws(t, i);
      if (i < t.size()) fail(no, "unexpected text after a flow collection: " + t.substr(i));
      return n;
    }
    Node n;
    n.kind = Node::Scalar;
    n.line = no;
    if (c == '"' || c == '\'') {
      size_t i = 0;
      n.str = quoted(t, i, no);
      n.quoted = true;
      if (i < t.size()) fail(no, "unexpected text after a quoted scalar");
    } else {
      n.str = t;
    }
    return n;
  }

  static void skip_ws(const std::string &s, size_t &i)
  {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) i++;
  }

  Node flow(const std::string &s, size_t &i, int no) const
  {
    skip_ws(s, i);
    Node n;
    n.line = no;
    if (i >= s.size()) fail(no, "missing value in a flow collection");
    const char c = s[i];
    if (c == '[' || c == '{') {
      const char close = c == '[' ? ']' : '}';
      n.kind = c == '[' ? Node::Seq : Node::Map;
      i++;
      while (true) {
        skip_ws(s, i);
        if (i < s.size() && s[i] == close) {
          i++;
          return n;
        }
        if (n.kind == Node::Seq) {
          n.seq.push_back(flow(s, i, no));
        } else {
          std::string key;
          if (i < s.size() && (s[i] == '"' || s[i] == '\'')) {
            key = quoted(s, i, no);
          } else {
            const size_t a = i;
            while (i < s.size() && s[i] != ':' && s[i] != ',' && s[i] != '}') i++;
            key = trim(s.substr(a, i - a));
          }
          skip_ws(s, i);
          if (i >= s.size() || s[i] != ':') fail(no, "expected ':' after the key " + key + " in a flow mapping");
          i++;
          skip_ws(s, i);
          Node v;
          v.line = no;
          if (i < s.size() && s[i] != ',' && s[i] != '}') v = flow(s, i, no);
          if (n.find(key)) fail(no, "duplicate key " + key);
          n.map.push_back({std::move(key), std::move(v)});
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
          i++;
          continue;
        }
        if (i < s.size() && s[i] == close) continue;
        fail(no, std::string("expected ',' or '") + close + "' in a flow collection");
      }
    }
    n.kind = Node::Scalar;
    if (c == '"' || c == '\'') {
      n.str = quoted(s, i, no);
      n.quoted = true;
      return n;
    }
    if (c == '&' || c == '*' || c == '!') fail(no, "anchors, aliases and tags are not supported");
    const size_t a = i;
    while (i < s.size() && s[i] != ',' && s[i] != ']' && s[i] != '}') i++;
    n.str = trim(s.substr(a, i - a));
    if (n.str.empty()) fail(no, "missing value in a flow collection");
    return n;
  }
};

}    // namespace

Node parse(const std::string &text, const std::string &name) { return Parser(text, name).document(); }

Node load_file(const std::string &path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error("cannot read " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return parse(ss.str(), path);
}

}    // namespace grace::yaml
