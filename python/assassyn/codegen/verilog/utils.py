"""
Utilities for the Verilog backend.
Ported from Rust's src/backend/verilog/utils.rs.
"""

from collections import deque
from typing import List, Iterator, Optional, Tuple

from assassyn.ir import DataType


class DisplayInstance:
    """
    Display instance for various IR elements in Verilog code generation.
    """

    def __init__(self, prefix: str, id_: str):
        self.prefix = prefix
        self.id = id_

    @classmethod
    def from_module(cls, module):
        """Create a DisplayInstance from a module."""
        return cls("", namify(module.get_name()))

    @classmethod
    def from_array(cls, array):
        """Create a DisplayInstance from an array."""
        return cls("array", namify(array.get_name()))

    @classmethod
    def from_fifo(cls, fifo, global_):
        """Create a DisplayInstance from a FIFO."""
        raw = namify(fifo.get_name())
        if global_:
            fifo_name = f"{namify(fifo.get_module().get_name())}_{raw}"
        else:
            fifo_name = raw
        return cls("fifo", fifo_name)

    def field(self, attr: str) -> str:
        """Get a field of this instance with the given attribute name."""
        return f"{self}_{attr}"

    def __str__(self) -> str:
        """String representation of this instance."""
        if not self.prefix:
            return self.id
        return f"{self.prefix}_{self.id}"


class Edge:
    """
    Edge between a display instance and a driver module.
    """

    def __init__(self, instance: DisplayInstance, driver):
        self.instance = instance
        self.driver = namify(driver.get_name())

    def field(self, field: str) -> str:
        """Get a field of this edge with the given field name."""
        return f"{self.instance}_driver_{self.driver}_{field}"


def broadcast(value: str, bits: int) -> str:
    """Broadcast a value to a given bit width."""
    return f"{{ {bits} {{ {value} }} }}"


def select_1h(iter_: Iterator[Tuple[str, str]], bits: int) -> str:
    """One-hot select between values based on predicates."""
    return reduce(
        (f"({pred} & {broadcast(value, bits)})" for pred, value in iter_),
        " | "
    )


def reduce(iter_: Iterator[str], concat: str) -> str:
    """Reduce an iterator of strings to a single string with a concatenator."""
    result = concat.join(list(iter_))
    if not result:
        return "'x"
    return result


def bool_ty() -> DataType:
    """Get a boolean type (1-bit integer)."""
    return DataType.int_ty(1)


def declare_impl(decl_prefix: str, ty: DataType, id_: str, term: str) -> str:
    """Implementation helper for declaring variables."""
    bits = ty.get_bits() - 1
    return f"  {decl_prefix} [{bits}:0] {id_}{term}\n"


def declare_logic(ty: DataType, id_: str) -> str:
    """Declare a logic variable."""
    return declare_impl("logic", ty, id_, ";")


def declare_in(ty: DataType, id_: str) -> str:
    """Declare an input port."""
    return declare_impl("input logic", ty, id_, ",")


def declare_out(ty: DataType, id_: str) -> str:
    """Declare an output port."""
    return declare_impl("output logic", ty, id_, ",")


def declare_array(prefix: str, array, id_: str, term: str) -> str:
    """Declare an array variable."""
    size = array.get_size()
    ty = array.scalar_ty()
    prefix_str = f"{prefix} " if prefix else ""
    return f"  {prefix_str}logic [{(ty.get_bits() * size) - 1}:0] {id_}{term}\n"


def connect_top(display, edge, fields: List[str]) -> str:
    """Connect fields between a display instance and an edge."""
    result = ""
    for field in fields:
        result += f"    .{display.field(field)}({edge.field(field)}),\n"
    return result


def type_to_fmt(ty: DataType) -> str:
    """Convert a DataType to a format specifier."""
    if ty.is_int() or ty.is_uint() or ty.is_bits():
        return "d"
    raise ValueError(f"Invalid type for type: {ty}")


def parse_format_string(args: List, sys) -> str:
    """Parse a format string for printing."""
    raw = args[0].as_ref("StrImm", sys).get_value()
    fmt = deque(raw)
    result = ""
    arg_idx = 1

    while fmt:
        c = fmt.popleft()
        if c == '{':
            if not fmt:
                raise ValueError(f"Invalid format string: {raw}")

            c = fmt.popleft()
            if c == '{':
                result += '{'
            else:
                dtype = args[arg_idx].get_dtype(sys)
                substr = c

                # Handle "{}"
                if substr == "}":
                    result += f"%{type_to_fmt(dtype)}"
                    arg_idx += 1
                    continue

                closed = False
                while fmt:
                    c = fmt.popleft()
                    if c == '}':
                        closed = True
                        break
                    substr += c

                if not closed:
                    raise ValueError(f"Invalid format string, unclosed braces: {raw}")

                if not substr:
                    new_fmt = type_to_fmt(dtype)
                elif substr.startswith(':'):
                    width_idx = 1
                    pad = "0" if substr[1:2] == "0" else ""
                    if pad:
                        width_idx += 1

                    width = ""
                    if width_idx < len(substr) and substr[width_idx].isdigit():
                        width = substr[width_idx]
                        width_idx += 1

                    vfmt = substr[width_idx] if width_idx < len(substr) and substr[width_idx].isalpha() else type_to_fmt(dtype)
                    new_fmt = f"%{pad}{width}{vfmt}"
                else:
                    raise ValueError(f"Invalid format string: {raw}")

                result += new_fmt
                arg_idx += 1
        elif c == '}':
            if not fmt or fmt.popleft() != '}':
                raise ValueError(f"Invalid format string, single closing brace: {raw}")
            result += '}'
        else:
            result += c

    return result


def namify(name: str) -> str:
    """
    Convert a name to a valid Verilog identifier.
    This should be implemented based on the common.namify function in Rust.
    """
    # Simple implementation - actual implementation should match Rust code
    return name.replace('.', '_')
