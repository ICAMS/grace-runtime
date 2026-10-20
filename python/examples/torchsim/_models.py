# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Models of the TorchSim examples: a GRACE model (registry name or kokkos.npz)."""
import numpy as np    # noqa: F401  (before torch, see the README)
import torch


def load(spec: str, device="cuda", dtype=torch.float64):
    """(model, dtype of its states, label)"""
    from grace_runtime.torchsim import GRACERuntimeModel
    return GRACERuntimeModel(spec, device=device, dtype=dtype), dtype, spec.split("/")[-2] if "/" in spec else spec
