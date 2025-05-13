"""Verilog backend generator for Assassyn."""

from .elaborate import elaborate
from .utils import namify, camelize, dtype_to_verilog_type
from enum import Enum


class Simulator(Enum):
    """Verilog simulator options."""
    VCS = "vcs"
    Verilator = "verilator"
    None_ = None