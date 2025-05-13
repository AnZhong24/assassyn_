"""Expression gathering utilities for Verilog generation."""

from collections import defaultdict
from ...ir.expr import Expr


class Gather:
    """Gather conditional values for Verilog generation.
    
    This matches the Rust class in src/backend/verilog/gather.rs
    """
    
    def __init__(self):
        """Initialize with empty conditions and values."""
        self.conditions = []
        self.values = []
    
    def add(self, condition, value):
        """Add a new condition-value pair."""
        self.conditions.append(condition)
        self.values.append(value)
    
    def is_empty(self):
        """Check if the gather is empty."""
        return not self.conditions
    
    def conditions_str(self):
        """Get the conditions as a string."""
        return ", ".join(self.conditions)
    
    def values_str(self):
        """Get the values as a string."""
        return ", ".join(self.values)


def gather_conditional_values(exprs):
    """Gather conditional values from a list of expressions.
    
    This matches the Rust function in src/backend/verilog/gather.rs
    """
    result = defaultdict(Gather)
    
    for expr in exprs:
        if not isinstance(expr, Expr):
            continue
        
        if expr.is_binary() or expr.is_unary():
            # Handle binary and unary operations
            result[expr.opcode].add(expr.condition, expr)
        elif expr.is_array_op():
            # Handle array operations
            if hasattr(expr, 'array'):
                key = (expr.opcode, expr.array)
                result[key].add(expr.condition, expr)
        elif expr.is_fifo_op():
            # Handle FIFO operations
            if hasattr(expr, 'fifo'):
                key = (expr.opcode, expr.fifo)
                result[key].add(expr.condition, expr)
    
    return result