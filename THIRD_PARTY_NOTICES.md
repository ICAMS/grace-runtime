# Third-party notices

grace-runtime is under the Academic Software Licence (`LICENSE.md`). The following third-party components are
bundled with it, in source form in this repository or compiled into the wheels; each keeps its own licence.

| Component | Where | Licence | Text |
|---|---|---|---|
| cnpy (Carl Rogers) | `libgrace/third_party/grace_cnpy/`, compiled into libgrace and all wheels | MIT | `libgrace/third_party/grace_cnpy/LICENSE` |
| Kokkos (National Technology & Engineering Solutions of Sandia, LLC) | compiled into the `grace-runtime-cuda12` wheels (`_kk_<arch>` modules) | Apache-2.0 WITH LLVM-exception | `python-cuda/LICENSE-Kokkos` |
| nanobind (Wenzel Jakob) | compiled into the Python extension modules of all wheels | BSD-3-Clause | below |
| tsl::robin_map (Thibaut Goetghebuer-Planchon), part of nanobind | compiled into the Python extension modules of all wheels | MIT | below |
| GCC runtime: libstdc++ (linked statically), libgomp (vendored by auditwheel) | all wheels | GPL-3.0 WITH GCC-exception-3.1 (GCC Runtime Library Exception) | https://www.gnu.org/licenses/gcc-exception-3.1.html |

Not bundled, loaded at run time when present: the CUDA runtime and driver, cuBLAS (`nvidia-cublas-cu12`, UQ only).

## nanobind

Copyright (c) 2022 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## tsl::robin_map

MIT License

Copyright (c) 2017 Thibaut Goetghebuer-Planchon <tessil@gmx.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
