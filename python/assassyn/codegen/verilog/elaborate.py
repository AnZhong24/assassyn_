"""
Verilog elaboration module for Assassyn.
This is ported from src/backend/verilog/elaborate.rs.
"""

import os
import re
from collections import defaultdict, deque
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional, Any, Iterator

from ...ir import DType
from ...ir.dtype import Int, UInt, Bits
from ...ir.visitor import Visitor
from ...ir.const import Const
from ...ir.expr import FIFOPush

from .gather import Gather, ExternalUsage, gather_exprs_externally_used
from .utils import (
    DisplayInstance, Edge, bool_ty, declare_array, declare_in,
    declare_logic, declare_out, reduce, select_1h, parse_format_string
)


# A simple simulator enum similar to the Rust version
class Simulator(Enum):
    VCS = auto()
    VERILATOR = auto()
    NONE = auto()


def fifo_name(fifo) -> str:
    """Get a formatted name for a FIFO."""
    return namify(fifo.get_name())


def namify(name: str) -> str:
    """Make a name valid for Verilog, should be imported from common."""
    # Simple implementation - should match backend/common.rs namify
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def upstreams(module, topo: Dict) -> List:
    """Get the upstream modules of a module in topological order."""
    # This is simplified, actual implementation depends on the topology
    return [upstream for upstream in topo if topo[upstream] < topo[module.upcast()]]


class VerilogDumper:
    """
    Main class for dumping Verilog code from Assassyn IR.
    """
    
    def __init__(self, sys, config, external_usage, topo, array_memory_params_map, module_expr_map):
        """Initialize the Verilog dumper."""
        self.sys = sys
        self.config = config
        self.pred_stack = deque()
        self.fifo_pushes = {}
        self.array_stores = {}
        self.triggers = {}
        self.current_module = ""
        self.external_usage = external_usage
        self.before_wait_until = False
        self.topo = topo
        self.array_memory_params_map = array_memory_params_map
        self.module_expr_map = module_expr_map
    
    @staticmethod
    def collect_array_memory_params_map(sys) -> Dict:
        """Collect memory parameters for arrays."""
        memory_map = {}
        
        for module in sys.downstreams[:]:
            for attr in module.get_attrs():
                if hasattr(attr, 'MemoryParams'):
                    mem = attr.MemoryParams
                    if module.is_downstream():
                        for interf, _ in module.ext_interf_iter():
                            if interf.get_kind() == 'Array':
                                array_ref = interf.as_ref('Array', sys)
                                memory_map[array_ref.upcast()] = mem.clone()
        
        return memory_map
    
    def dump_memory_nodes(self, node, res: List[str]) -> None:
        """Dump memory-related nodes."""
        kind = node.get_kind()
        
        if kind == 'Expr':
            expr = node.as_ref('Expr', self.sys)
            if expr.get_opcode() == Opcode.Load:
                id_ = namify(expr.upcast().to_string(self.sys))
                ty = expr.dtype()
                res.append(declare_logic(ty, id_))
                res.append(f"  assign {id_} = dataout;\n")
            else:
                res.append(self.print_body(node))
        elif kind == 'Block':
            block = node.as_ref('Block', self.sys)
            skip = 0
            
            if condition := block.get_condition():
                self.pred_stack.append(
                    dump_ref(self.sys, condition, True) 
                    if condition.get_dtype(block.sys).get_bits() == 1 
                    else f"(|{dump_ref(self.sys, condition, False)})"
                )
                skip = 1
            elif cycle := block.get_cycle():
                self.pred_stack.append(f"(cycle_cnt == {cycle})")
                skip = 1
            
            for elem in list(block.body_iter())[skip:]:
                self.dump_memory_nodes(elem, res)
            
            if skip:
                self.pred_stack.pop()
        else:
            raise ValueError(f"Unexpected node kind: {kind}")
    
    def get_pred(self) -> Optional[str]:
        """Get the current predicate stack as a string."""
        if not self.pred_stack:
            return None
        return f"({' && '.join(self.pred_stack)})"
    
    def dump_array(self, array, mem_init_path=None) -> str:
        """Dump an array instance."""
        res = []
        display = DisplayInstance.from_array(array)
        
        # Field names
        w = display.field("w")          # write enable
        widx = display.field("widx")    # write index
        d = display.field("d")          # write data
        q = display.field("q")          # array buffer
        
        res.append(f"  /* {array} */\n")
        
        # Check if this is a memory-mapped array
        if array.upcast() in self.array_memory_params_map:
            res.append(declare_logic(array.scalar_ty(), q))
        else:
            res.append(declare_array("", array, q, ";"))
        
        # Collect drivers
        seen = set()
        drivers = []
        
        for user in array.users():
            operand = user.as_ref('Operand', array.sys)
            expr = operand.get_expr()
            
            if expr.get_opcode() == Opcode.Store:
                module = expr.get_block().get_module()
                module_key = module.get_key()
                
                if module_key not in seen:
                    seen.add(module_key)
                    module_ref = module.as_ref('Module', array.sys)
                    drivers.append(Edge(display, module_ref))
        
        scalar_bits = array.scalar_ty().get_bits()
        array_size = array.get_size()
        
        # Declare driver fields
        for edge in drivers:
            res.append(declare_logic(array.scalar_ty(), edge.field("d")))
            res.append(declare_logic(Int(1), edge.field("w")))
            res.append(declare_logic(array.get_idx_type(), edge.field("widx")))
        
        # If not a memory-mapped array, implement array logic
        if array.upcast() not in self.array_memory_params_map:
            res.append(declare_logic(array.scalar_ty(), d))
            res.append(declare_logic(array.get_idx_type(), widx))
            res.append(declare_logic(Int(1), w))
            
            # Write data selection
            write_data = select_1h(
                ((edge.field("w"), edge.field("d")) for edge in drivers),
                scalar_bits
            )
            res.append(f"  assign {d} = {write_data};\n")
            
            # Write index selection
            write_idx = select_1h(
                ((edge.field("w"), edge.field("widx")) for edge in drivers),
                array.get_idx_type().get_bits()
            )
            res.append(f"  assign {widx} = {write_idx};\n")
            
            # Write enable
            write_enable = reduce((edge.field("w") for edge in drivers), " | ")
            res.append(f"  assign {w} = {write_enable};\n")
            
            # Array write logic
            res.append("  always_ff @(posedge clk or negedge rst_n)\n")
            res.append("    if (!rst_n)\n")
            
            # Initialize array
            if mem_init_path:
                res.append(f'      $readmemh("{mem_init_path}", {q});\n')
            elif initializer := array.get_initializer():
                res.append("    begin\n")
                for idx, value in enumerate(initializer):
                    elem_init = value.as_ref('IntImm', self.sys).get_value()
                    slice_fmt = f"{(idx + 1) * scalar_bits - 1}:{idx * scalar_bits}"
                    res.append(f"      {q}[{slice_fmt}] <= {scalar_bits}'d{elem_init};\n")
                res.append("    end\n")
            else:
                init_bits = array.get_flattened_size()
                res.append(f"      {q} <= {init_bits}'d0;\n")
            
            # Array write
            res.append(f"    else if ({w}) begin\n\n")
            res.append(f"      case ({widx})\n")
            
            for i in range(array_size):
                slice_fmt = f"{(i + 1) * scalar_bits - 1}:{i * scalar_bits}"
                res.append(f"        {i} : {q}[{slice_fmt}] <= {d};\n")
            
            res.append("        default: ;\n")
            res.append("      endcase\n")
            res.append("    end\n")
        
        return ''.join(res)
    
    # More methods would be implemented here...
    # The Rust version has many more methods that would need to be ported
    
    def dump_exposed_array(self, array, exposed_kind, mem_init_path=None) -> str:
        """Dump an exposed array instance."""
        # Implementation would be similar to dump_array but with exposed ports
        # This is a simplified placeholder
        return f"  /* Exposed array {array} */\n"
    
    def dump_fifo(self, fifo) -> str:
        """Dump a FIFO instance."""
        # Implementation would be similar to the Rust version
        return f"  /* FIFO {fifo} */\n"
    
    def dump_trigger(self, module) -> str:
        """Dump the trigger event state machine's instantiation."""
        # Implementation would be similar to the Rust version
        return f"  /* Trigger SM for Module: {module.get_name()} */\n"
    
    def dump_module_instance(self, module) -> str:
        """Dump a module instance."""
        # Implementation would be similar to the Rust version
        return f"  /* Module instance {module.get_name()} */\n"
    
    def dump_runtime(self, outfile, sim_threshold: int) -> None:
        """Dump the runtime code."""
        # Implementation would be similar to the Rust version
        outfile.write("/* Runtime module would be implemented here */\n")
    
    def print_body(self, node) -> str:
        """Print the body of a node."""
        kind = node.get_kind()
        
        if kind == 'Expr':
            expr = node.as_ref('Expr', self.sys)
            return self.visit_expr(expr)
        elif kind == 'Block':
            block = node.as_ref('Block', self.sys)
            return self.visit_block(block)
        else:
            raise ValueError(f"Unexpected reference type: {node}")
    
    def visit_module(self, module) -> str:
        """Visit a module and generate Verilog code for it."""
        self.current_module = namify(module.get_name())
        res = []
        
        # Start module definition
        res.append(f"""
module {self.current_module} (
  input logic clk,
  input logic rst_n,
""")
        
        # Handle FIFO ports
        for port in module.fifo_iter():
            name = fifo_name(port)
            ty = port.scalar_ty()
            display = DisplayInstance.from_fifo(port, False)
            
            res.append(f"  // Port FIFO {name}\n")
            res.append(declare_in(bool_ty(), display.field("pop_valid")))
            res.append(declare_in(ty, display.field("pop_data")))
            res.append(declare_out(bool_ty(), display.field("pop_ready")))
        
        # Handle external interfaces
        has_memory_params = False
        has_memory_init_path = False
        memory_params = None
        init_file_path = None
        
        # Rest of the implementation would follow the Rust version
        # This would be quite lengthy to fully implement
        
        # Simplify for now
        res.append("""
  output logic expose_executed);

  logic executed;
  
  // Module body would be implemented here
  
  assign executed = 1'b1;
  assign expose_executed = executed;

endmodule // %s
""" % self.current_module)
        
        return ''.join(res)
    
    def visit_block(self, block) -> str:
        """Visit a block and generate Verilog code for it."""
        res = []
        skip = 0
        
        if cond := block.get_condition():
            self.pred_stack.append(
                dump_ref(self.sys, cond, True) 
                if cond.get_dtype(block.sys).get_bits() == 1 
                else f"(|{dump_ref(self.sys, cond, False)})"
            )
            skip = 1
        elif cycle := block.get_cycle():
            self.pred_stack.append(f"(cycle_cnt == {cycle})")
            skip = 1
        
        for elem in list(block.body_iter())[skip:]:
            if elem.get_kind() == 'Expr':
                expr = elem.as_ref('Expr', self.sys)
                res.append(self.visit_expr(expr))
            elif elem.get_kind() == 'Block':
                inner_block = elem.as_ref('Block', self.sys)
                res.append(self.visit_block(inner_block))
            else:
                raise ValueError(f"Unexpected reference type: {elem}")
        
        if skip:
            self.pred_stack.pop()
        
        return ''.join(res)
    
    def visit_expr(self, expr) -> str:
        """Visit an expression and generate Verilog code for it."""
        # This would be a lengthy method with many cases based on opcode
        # Similar to the Rust version, it would handle all the expression types
        
        opcode = expr.get_opcode()
        
        # Simplified version - a full implementation would handle all opcodes
        if opcode == Opcode.Binary:
            return f"  // Binary operation: {expr}\n"
        elif opcode == Opcode.Unary:
            return f"  // Unary operation: {expr}\n"
        elif opcode == Opcode.Load:
            return f"  // Load operation: {expr}\n"
        elif opcode == Opcode.Store:
            return f"  // Store operation: {expr}\n"
        else:
            return f"  // Expression with opcode {opcode}: {expr}\n"


def node_dump_ref(sys, node, node_kinds, immwidth, signed) -> Optional[str]:
    """Dump a node reference with options."""
    kind = node.get_kind()
    
    if kind == 'Array':
        array = node.as_ref('Array', sys)
        return namify(array.get_name())
    elif kind == 'FIFO':
        return namify(node.as_ref('FIFO', sys).get_name())
    elif kind == 'IntImm':
        int_imm = node.as_ref('IntImm', sys)
        dbits = int_imm.dtype().get_bits()
        value = int_imm.get_value()
        if immwidth:
            return f"{dbits}'d{value}"
        else:
            return str(value)
    elif kind == 'StrImm':
        str_imm = node.as_ref('StrImm', sys)
        return f'"{str_imm.get_value()}"'
    elif kind == 'Expr':
        dtype = node.get_dtype(sys)
        raw = namify(node.to_string(sys))
        if dtype.is_int() and signed:
            return f"$signed({raw})"
        else:
            return raw
    else:
        raise ValueError(f"Unknown node of kind {kind}")


def dump_ref(sys, value, with_imm_width) -> str:
    """Dump a reference with optional immediate width."""
    return node_dump_ref(sys, value, [], with_imm_width, False)


def dump_arith_ref(sys, value) -> str:
    """Dump an arithmetic reference."""
    return node_dump_ref(sys, value, [], True, True)


def elaborate(sys, **config) -> None:
    """Elaborate a system into Verilog."""

    # Collect exposed nodes in modules
    module_expr_map = {}
    for m in sys.modules[:] + sys.downstreams[:]:
        exposed_map = {}
        # Collect exposed nodes for this module
        for node, kind in sys.exposed_nodes():
            # Logic to collect exposed nodes
            # This would depend on implementation details
            pass
        module_expr_map[m.upcast()] = exposed_map
    
    # Create output directory
    dirname = config.dirname(sys, "verilog")
    os.makedirs(dirname, exist_ok=True)
    
    # Clear directory if override_dump is set
    if config.override_dump:
        for item in os.listdir(dirname):
            path = os.path.join(dirname, item)
            if os.path.isfile(path):
                os.unlink(path)
    
    # Generate Verilog file path
    fname = os.path.join(dirname, f"{sys.get_name()}.sv")
    print(f"Writing verilog rtl to {fname}")
    
    # Generate testbench if needed
    if config.verilog == Simulator.VERILATOR:
        # Generate C++ testbench
        pass
    
    # Get the topological order for acyclic combinational logic
    topo = {}  # This would be populated by the topological sort
    
    # Collect external usage
    external_usage = gather_exprs_externally_used(sys)
    
    # Collect array memory params
    array_memory_params_map = VerilogDumper.collect_array_memory_params_map(sys)
    
    # Create dumper and generate Verilog
    vd = VerilogDumper(
        sys, config, external_usage, topo, 
        array_memory_params_map, module_expr_map
    )
    
    with open(fname, 'w') as fd:
        # Generate module code
        for module in vd.sys.modules[:] + vd.sys.downstreams[:]:
            fd.write(vd.visit_module(module))
        
        # Generate runtime
        vd.dump_runtime(fd, config.sim_threshold)
