"""Node reference dumper for simulator code generation."""

from .utils import namify, int_imm_dumper_impl, fifo_name
from ...expr import Expr
from ...array import Array
from ...module import Module, Port
from ...const import Const
from ...expr import FIFOPush


def dump_rval_ref( # pylint: disable=too-many-branches, too-many-return-statements
        module_ctx, _, node):
    """Dispatch to appropriate handler based on node kind."""

    if isinstance(node, Array):
        return namify(node.get_name())

    if isinstance(node, Port):
        return fifo_name(node)

    if isinstance(node, Const):
        return int_imm_dumper_impl(node.get_dtype(), node.get_value())

    if isinstance(node, Module):
        return namify(node.get_name())

    if isinstance(node, Expr):
        # Figure out the ID format based on context
        parent_block = node.get_parent()
        if module_ctx != parent_block.get_module():
            # Expression from another module
            raw = namify(node.get_name())
            field_id = f"{raw}_value"
            panic_log = f"Value {raw} invalid!"
            return f"""if let Some(x) = &sim.{field_id} {{
                        x
                      }} else {{
                        panic!("{panic_log}");
                      }}.clone()"""
        if node.dtype().get_bits() <= 64:
            # Simple value
            return namify(node.get_name())
        # Large value needs cloning
        return f"{namify(node.get_name())}.clone()"

    # Default case
    return namify(node.to_string())


def externally_used_combinational(expr: Expr) -> bool:
    """Check if an expression is used outside its module.

    This matches the Rust function in src/backend/simulator/elaborate.rs
    """

    # Push is NOT a combinational operation
    if isinstance(expr, FIFOPush):
        return False

    this_module = expr.parent.module

    # Check if any user is in a different module
    for user in expr.users():
        if hasattr(user, 'get_parent'):
            parent = user.get_parent()
            if parent and isinstance(parent, Expr):
                if parent.get_block().get_module() != this_module:
                    return True

    return False
