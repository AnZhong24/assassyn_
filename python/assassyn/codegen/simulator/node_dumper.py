"""Node reference dumper for simulator code generation."""

from .utils import namify, int_imm_dumper_impl, fifo_name
from ...utils import unwrap_operand
from ...ir.expr import Expr
from ...ir.array import Array
from ...ir.const import Const
from ...ir.expr import FIFOPush
from ...ir.module import Module, Port

def dump_rval_ref( # pylint: disable=too-many-branches, too-many-return-statements
        module_ctx, _, node):
    """Dispatch to appropriate handler based on node kind."""

    unwrapped = unwrap_operand(node)

    if isinstance(unwrapped, Array):
        return namify(unwrapped.name)

    if isinstance(unwrapped, Port):
        return fifo_name(unwrapped)

    if isinstance(unwrapped, Const):
        return int_imm_dumper_impl(unwrapped.dtype, unwrapped.value)

    if isinstance(unwrapped, Module):
        return namify(unwrapped.as_operand())

    if isinstance(unwrapped, Expr):
        # Figure out the ID format based on context
        parent_block = unwrapped.parent
        if module_ctx != parent_block.module:
            # Expression from another module
            raw = namify(unwrapped.as_operand())
            field_id = f"{raw}_value"
            panic_log = f"Value {raw} invalid!"
            return f"""if let Some(x) = &sim.{field_id} {{
                        x
                      }} else {{
                        panic!("{panic_log}");
                      }}.clone()"""

        ref = namify(unwrapped.as_operand())

        dtype = unwrapped.dtype
        if dtype.bits <= 64:
            # Simple value
            return namify(ref)

        # Large value needs cloning
        return f"{ref}.clone()"

    if isinstance(unwrapped, str):
        return f'"{unwrapped}"'

    # Default case
    return namify(unwrapped.as_operand())


def externally_used_combinational(expr: Expr) -> list:
    """Check if an expression is used outside its module.

    This matches the Rust function in src/backend/simulator/elaborate.rs
    """

    # Push is NOT a combinational operation
    if isinstance(expr, FIFOPush):
        return False

    this_module = expr.parent.module

    # Check if any user is in a different module
    for user in expr.users:
        parent_module = user.user.parent.module
        if parent_module != this_module:
            return True

    return False
