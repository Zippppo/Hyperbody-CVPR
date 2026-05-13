#!/usr/bin/env python3
"""Compatibility wrapper for the S2I-Dataset graph precompute tool."""

from __future__ import annotations

import runpy
from pathlib import Path


TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "S2I-Dataset"
    / "tools"
    / "precompute_graph_distance.py"
)


if __name__ == "__main__":
    runpy.run_path(str(TOOL_PATH), run_name="__main__")
