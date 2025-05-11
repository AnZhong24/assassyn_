"""Node reference dumper for simulator code generation."""

from ...visitor import Visitor
from .utils import namify, int_imm_dumper_impl, fifo_name
from ...expr import (
    Expr,
    PureIntrinsic,
)
from ...array import Array
from ...module import Module
from ...value import IntImm, StrImm
from ...module import FIFO


def dump_rval_ref(module_ctx, sys, node):
    """Dispatch to appropriate handler based on node kind."""
    node_kind = node.get_kind()

    if isinstance(node, Array):
        return namify(node.get_name())

    if isinstance(node, FIFO):
        return fifo_name(node)

    if isinstance(node, IntImm):
        return int_imm_dumper_impl(node.get_dtype(), node.get_value())

    if isinstance(node, StrImm):
        value = node.get_value()
        # Using Python's repr to get quote-escaped string
        return repr(value)

    if isinstance(node, Module):
        return namify(node.get_name())

    if isinstance(node, Expr):
        # Figure out the ID format based on context
        parent_block = node.get_parent()
        if self.module_ctx != parent_block.get_module():
            # Expression from another module
            raw = namify(expr.get_name())
            field_id = f"{raw}_value"
            panic_log = f"Value {raw} invalid!"
            return f"""if let Some(x) = &sim.{field_id} {{
                        x
                      }} else {{
                        panic!("{panic_log}");
                      }}.clone()"""
        elif expr.dtype().get_bits() <= 64:
            # Simple value
            return namify(expr.get_name())
        else:
            # Large value needs cloning
            return f"{namify(expr.get_name())}.clone()"

        # Handle FIFO peek special case
        if isinstance(node, PureIntrinsic) and node.subcode == "FIFOPeek":
            id = namify(node.get_name())
            id += ".clone().unwrap()"
            return id

        return namify(node.get_name())

    else:
        # Default case
        return namify(node.to_string())


def externally_used_combinational(expr):
    """Check if an expression is used outside its module.

    This matches the Rust function in src/backend/simulator/elaborate.rs
    """
    # Push is NOT a combinational operation
    if expr.get_opcode() == "FIFOPush":
        return False

    this_module = expr.get_block().get_module()

    # Check if any user is in a different module
    for user in expr.users():
        if hasattr(user, 'get_parent'):
            parent = user.get_parent()
            if parent and isinstance(parent, Expr):
                if parent.get_block().get_module() != this_module:
                        return True

    return False
