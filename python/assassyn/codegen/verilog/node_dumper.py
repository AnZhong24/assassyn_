"""Node reference dumper for Verilog code generation."""

from .utils import namify, int_imm_dumper_impl, fifo_name
from ...utils import unwrap_operand
from ...ir.expr import Expr
from ...ir.array import Array
from ...ir.const import Const
from ...ir.expr import FIFOPush
from ...ir.module import Module, Port


def dump_rval_ref(node):
    """Generate Verilog code for a node reference.
    
    This matches the Rust function in src/backend/verilog/elaborate.rs
    """
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
        return namify(unwrapped.as_operand())
    
    if isinstance(unwrapped, str):
        return f'"{unwrapped}"'
    
    # Default case
    return namify(unwrapped.as_operand())


def externally_used_combinational(expr: Expr) -> bool:
    """Check if an expression is used outside its module.
    
    This matches the Rust function in src/backend/verilog/elaborate.rs
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