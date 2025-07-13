# SPDX-License-Identifier: MPL-2.0 AND LicenseRef‑Commons‑Clause‑License‑Condition‑1.0
# -*- coding: utf‑8 -*-
#
# Tests for nnll.dissect.Dissector
#
# Run with:
#   pytest -vv tests/test_dissect.py
# or:
#   python -m unittest tests.test_dissect

import re
import unittest

import torch
import torch._dynamo  # noqa: F401  (required side‑effect import for torch.export)
from torch import nn
import networkx as nx

from nnll.dissect import Dissector


# --------------------------------------------------------------------------- #
# Helper models                                                               #
# --------------------------------------------------------------------------- #

class ResidualBlock(nn.Module):
    """
    A minimal ResNet‑style residual block identical to the one shown
    in the example code under test.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # identity mapping when channel sizes differ
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class TinyMLP(nn.Module):
    """A 2‑layer MLP with ReLU non‑linearity."""

    def __init__(self, in_dim=8, hid=16, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ParallelAdd(nn.Module):
    """Two parallel conv branches that are summed."""

    def __init__(self, in_ch=3, ch=8):
        super().__init__()
        self.a = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.b = nn.Conv2d(in_ch, ch, 1)

    def forward(self, x):
        return self.a(x) + self.b(x)


# --------------------------------------------------------------------------- #
# Test‑cases                                                                  #
# --------------------------------------------------------------------------- #


class TestDissector(unittest.TestCase):
    """Smoke‑tests for graph & tree generation."""

    @staticmethod
    def _dissect(model: nn.Module):
        """Utility: create Dissector, return (graph, tree) tuple."""
        dissect = Dissector(model)
        graph, tree = dissect()
        # basic sanity
        assert isinstance(graph, nx.DiGraph)
        return graph, tree

    # ---------- Residual block ------------------------------------------------

    def test_residual_block_tree_structure(self):
        """Tree for the residual block should contain the relu–add cascade."""
        with torch.device("meta"):
            model = ResidualBlock(3, 64)

        _, tree = self._dissect(model)
        t = str(tree)

        # Root node must be the graph output
        self.assertEqual(str(tree).splitlines()[0].strip(), "output")

        # Expect a relu → add → conv2d chain at the top of the indentation
        pattern = (
            r"output\n"            # root
            r"\s+relu\n"           # first child
            r"\s+add\n"            # skip connection merge
            r"[\s\S]*?conv2d"      # at least one conv2d somewhere beneath add
        )
        self.assertRegex(t, re.compile(pattern, re.MULTILINE))

        # The residual path contains at least three convolutions in total
        self.assertGreaterEqual(t.lower().count("conv2d"), 3)

    # ---------- Tiny MLP ------------------------------------------------------

    def test_tiny_mlp_has_linear_nodes(self):
        """The MLP’s tree should mention linear (or addmm) ops."""
        with torch.device("meta"):
            model = TinyMLP()

        _, tree = self._dissect(model)
        t = str(tree).lower()

        # Linear layers should surface either as 'linear' or 'addmm'
        self.assertTrue(
            any(tag in t for tag in ("linear", "addmm")),
            msg=f"Expected 'linear' or 'addmm' in tree, got:\n{t}",
        )

    # ---------- Parallel add --------------------------------------------------

    def test_parallel_add_contains_add_node(self):
        """A model with a parallel addition must show an 'add' op."""
        with torch.device("meta"):
            model = ParallelAdd()

        _, tree = self._dissect(model)
        t = str(tree)

        self.assertIn("add", t)
        # There should be exactly one top‑level add below the root
        first_two_lines = t.splitlines()[:3]  # output + its first child
        self.assertIn("add", "\n".join(first_two_lines).lower())


# --------------------------------------------------------------------------- #
# Stand‑alone runner                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import pytest

    pytest.main(["-vv", __file__])
