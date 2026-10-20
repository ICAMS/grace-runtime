// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Energy MLPs the fixed-size buffers of energy_atom cannot hold (more than 7 layers, widths above 256) are
// rejected at setup with grace::Error instead of overrunning a stack buffer inside the parallel region.
#include "grace/fused_1l.h"
#include "grace/fused_2l.h"
#include "test_util.h"
#include <memory>
#include <vector>

template<class Drv, class Model> bool rejects(std::shared_ptr<Model> m)
{
  try {
    Drv d(m, {}, {});
    d.setup_types(1, std::vector<int>{-1, 0});
  } catch (const grace::Error &) {
    return true;
  }
  return false;
}
template<class Model> void deepen(Model &m)    // 8 extra 1 -> 1 layers after the output layer: 9+ layers
{
  auto L = m.energy_mlp_layers.back();
  L.n_in = 1; L.n_out = 1; L.W.assign(1, 1.0);
  for (int k = 0; k < 8; k++) m.energy_mlp_layers.push_back(L);
}
template<class Model> void widen(Model &m)     // first hidden layer 300 wide, second layer reads 300 inputs
{
  auto &a = m.energy_mlp_layers[0], &b = m.energy_mlp_layers[1];
  a.n_out = 300; a.W.assign((size_t) a.n_in * 300, 0.0);
  b.n_in = 300; b.W.assign((size_t) 300 * b.n_out, 0.0);
}
int main()
{
  auto base2 = std::make_shared<grace::GRACE2LModel>();
  base2->load(test_model("GRACE_TEST_M2", "GRACE-2L-SMAX-OMAT-medium"));
  auto base1 = std::make_shared<grace::GRACE1LModel>();
  base1->load(test_model("GRACE_TEST_M1", "GRACE-1L-OMAT-large-base"));
  CHECK(!rejects<grace::Grace2LFused>(base2));      // the foundation models stay accepted
  CHECK(!rejects<grace::Grace1LFused>(base1));
  {
    auto m = std::make_shared<grace::GRACE2LModel>(*base2); deepen(*m);
    CHECK(rejects<grace::Grace2LFused>(m));
  }
  {
    auto m = std::make_shared<grace::GRACE2LModel>(*base2); widen(*m);
    CHECK(rejects<grace::Grace2LFused>(m));
  }
  {
    auto m = std::make_shared<grace::GRACE1LModel>(*base1); deepen(*m);
    CHECK(rejects<grace::Grace1LFused>(m));
  }
  {
    auto m = std::make_shared<grace::GRACE1LModel>(*base1); widen(*m);
    CHECK(rejects<grace::Grace1LFused>(m));
  }
  return 0;
}
