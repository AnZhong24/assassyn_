"""
Gather module for Verilog backend, ported from src/backend/verilog/gather.rs.
This handles gathering data for Verilog code generation.
"""

from collections import defaultdict
from typing import Dict, Set, List, Optional, Tuple

from assassyn.ir import Opcode
from assassyn.ir.visitor import Visitor

from .utils import select_1h


class Gather:
    """
    Gather is a data structure that gathers multiple conditional values into a single value.
    Typically used by FIFO and Array writes.
    """
    
    def __init__(self, cond: str, value: str, bits: int):
        """
        Create a new Gather with the given condition and value.
        
        Args:
            cond: The condition
            value: The value
            bits: The bit width
        """
        self.bits = bits
        self.condition: List[str] = [cond]
        self.value: List[str] = [value]
    
    def and_(self, cond: str, join: str) -> str:
        """
        Combine the gathered conditions with the given condition.
        
        Args:
            cond: The condition to combine with
            join: The join operator
            
        Returns:
            The combined condition
        """
        if self.is_conditional():
            gather_cond = join.join([f"({x})" for x in self.condition])
            return f"({cond}) && ({gather_cond})"
        return cond
    
    def select_1h(self) -> str:
        """
        Select one-hot between the gathered values based on conditions.
        
        Returns:
            The one-hot selection expression
        """
        if self.is_conditional():
            return select_1h(
                zip(self.condition, self.value),
                self.bits
            )
        return self.value[0]
    
    def is_conditional(self) -> bool:
        """
        Check if this gather is conditional.
        
        Returns:
            True if conditional, False otherwise
        """
        assert self.condition, "Condition list cannot be empty"
        return bool(self.condition[0])
    
    def push(self, cond: str, value: str, bits: int) -> None:
        """
        Push a new conditional value into the gather.
        
        Args:
            cond: The condition
            value: The value
            bits: The bit width
        """
        assert self.is_conditional(), "Cannot push to unconditional gather"
        assert self.bits == bits, f"Bit width mismatch: {self.bits} != {bits}"
        self.condition.append(cond)
        self.value.append(value)


class ExternalUsage:
    """
    Tracks expressions used by external modules.
    """
    
    def __init__(self):
        """Initialize an empty external usage tracker."""
        self.module_use_external_expr: Dict[object, Set[object]] = defaultdict(set)
        self.expr_externally_used: Dict[object, Set[object]] = defaultdict(set)
    
    def is_externally_used(self, expr) -> bool:
        """
        Check if the given expression is used externally.
        
        Args:
            expr: The expression to check
            
        Returns:
            True if the expression is used externally, False otherwise
        """
        module = expr.get_block().get_module()
        return module in self.expr_externally_used and expr.upcast() in self.expr_externally_used[module]
    
    def out_bounds(self, module) -> Optional[Set]:
        """
        Get the expressions from this module that are used externally.
        
        Args:
            module: The module to check
            
        Returns:
            The set of expressions, or None if none
        """
        return self.expr_externally_used.get(module.upcast())
    
    def in_bounds(self, module) -> Optional[Set]:
        """
        Get the external expressions used by this module.
        
        Args:
            module: The module to check
            
        Returns:
            The set of expressions, or None if none
        """
        return self.module_use_external_expr.get(module.upcast())


class ExternalUsageVisitor(Visitor):
    """
    Visitor to collect external usage information.
    """
    
    def __init__(self, sys):
        """Initialize the visitor."""
        self.sys = sys
        self.external_usage = ExternalUsage()
    
    def visit_expr(self, expr) -> None:
        """
        Visit an expression and collect external usage information.
        
        Args:
            expr: The expression to visit
        """
        m = expr.get_block().get_module()
        externals = set()
        
        for user in expr.users():
            ext = user.as_ref("Operand", self.sys).get_expr().get_block().get_module()
            if ext != m:
                externals.add(ext)
        
        if not expr.get_opcode().is_valued() or expr.get_opcode() == Opcode.Bind:
            return
        
        if externals:
            self.external_usage.expr_externally_used[m].add(expr.upcast())
            
            for elem in externals:
                self.external_usage.module_use_external_expr[elem].add(expr.upcast())


def gather_exprs_externally_used(sys) -> ExternalUsage:
    """
    Gather all expressions used by external modules.
    
    Args:
        sys: The system builder
        
    Returns:
        An ExternalUsage object containing the collected information
    """
    visitor = ExternalUsageVisitor(sys)
    visitor.enter(sys)
    return visitor.external_usage