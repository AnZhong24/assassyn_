"""Gather classes for the Verilog backend."""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ...builder import SysBuilder
from ...ir.module import Module
from ...ir.visitor import Visitor
from ...ir.expr import Operand, Expr

from .utils import select_1h

class Gather:
    """Gather multiple conditional values into a single value."""
    
    def __init__(self, cond: str, value: str, bits: int):
        """Initialize a Gather instance."""
        self.bits = bits
        self.condition = [cond]
        self.value = [value]
    
    def and_(self, cond: str, join: str) -> str:
        """Combine condition with the gathered conditions."""
        if self.is_conditional():
            gather_cond = " || ".join([f"({x})" for x in self.condition])
            return f"({cond}) && ({gather_cond})"
        return cond
    
    def select_1h(self) -> str:
        """Select a value based on one-hot encoding."""
        if self.is_conditional():
            return select_1h(
                zip(self.condition, self.value),
                self.bits
            )
        return self.value[0]
    
    def is_conditional(self) -> bool:
        """Check if this gather is conditional."""
        assert self.condition, "Condition list cannot be empty"
        return bool(self.condition[0])
    
    def push(self, cond: str, value: str, bits: int):
        """Push a new conditional value into the gather."""
        assert self.is_conditional(), "Cannot push to a non-conditional gather"
        assert self.bits == bits, f"Bit width mismatch: {self.bits} != {bits}"
        self.condition.append(cond)
        self.value.append(value)

class ExternalUsage:
    """Track expressions used by external modules."""
    
    def __init__(self):
        """Initialize an ExternalUsage instance."""
        self.module_use_external_expr = defaultdict(set)
        self.expr_externally_used = defaultdict(set)
    
    def is_externally_used(self, expr: Expr) -> bool:
        """Check if an expression is externally used."""
        module = expr.parent.module
        return module in self.expr_externally_used and expr in self.expr_externally_used[module]
    
    def out_bounds(self, module: Module):
        """Get expressions used externally from this module."""
        if module in self.expr_externally_used:
            return self.expr_externally_used[module]
        return None
    
    def in_bounds(self, module: Module):
        """Get external expressions used by this module."""
        if module in self.module_use_external_expr:
            return self.module_use_external_expr[module]
        return None

    def visit_expr(self, expr: Expr) -> None:
        """Visit an expression to check for external usage."""
        module = expr.parent.module
        externals = set()
        
        for user in expr.users:
            user_expr = user.get_expr()
            ext_module = user_expr.parent.module
            if ext_module != module:
                externals.add(ext_module)
        
        if not expr.is_valued() or expr.opcode == Opcode.BIND:
            return
        
        if externals:
            self.expr_externally_used[module].add(expr)
            
            for elem in externals:
                self.module_use_external_expr[elem].add(expr)
    
    def enter(self, sys: SysBuilder):
        """Entry point for visiting the system."""
        for module in sys.modules:
            for expr in module.collect_expressions():
                self.visit_expr(expr)

def gather_exprs_externally_used(sys: SysBuilder) -> ExternalUsage:
    """Gather all expressions used by external modules."""
    result = ExternalUsage()
    result.enter(sys)
    return result