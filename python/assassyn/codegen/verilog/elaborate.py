"""Elaborate function for Assassyn Verilog generator."""

from __future__ import annotations

import os
import shutil
import subprocess
import typing
from pathlib import Path
from collections import defaultdict
from .utils import namify, dtype_to_verilog_type, int_imm_dumper_impl, fifo_name, DisplayInstance
from .node_dumper import dump_rval_ref, externally_used_combinational
from .gather import gather_conditional_values

from ...ir.visitor import Visitor
from ...ir.block import Block, CondBlock, CycledBlock
from ...ir.expr import (
    Expr,
    BinaryOp,
    UnaryOp,
    ArrayRead,
    ArrayWrite,
    Cast,
    Intrinsic,
    PureIntrinsic,
    Bind,
    AsyncCall,
    FIFOPop,
    FIFOPush,
    Log,
    Select,
    Select1Hot,
    Slice,
    Concat,
)
from ...ir.module import Module, Downstream, Port, SRAM
from ...ir.array import Array

if typing.TYPE_CHECKING:
    from ...builder import SysBuilder


class VerilogDumper(Visitor):
    """Visitor for generating Verilog code.
    
    This matches the Rust class in src/backend/verilog/elaborate.rs
    """
    
    def __init__(self, sys):
        """Initialize the Verilog dumper."""
        self.sys = sys
        self.code = []
        self.wire_decls = {}
        self.reg_decls = {}
        self.module_decls = []
        self.instances = []
        self.module_instances = {}
        self.array_decls = []
        self.fifo_decls = []
        self.comb_logic = []
        self.seq_logic = []
        self.module_ctx = None
        self.indent = 0
        self.fifo_depth = 2  # Default FIFO depth
        self.next_counter = 0
        
    def _indent_str(self):
        """Return indentation string for current level."""
        return "  " * self.indent
    
    def _next_temp(self):
        """Generate a unique temporary variable name."""
        temp = f"temp_{self.next_counter}"
        self.next_counter += 1
        return temp
    
    def add_wire(self, name, dtype):
        """Add a wire declaration."""
        vlog_type = dtype_to_verilog_type(dtype)
        self.wire_decls[name] = f"{vlog_type} {name}"
    
    def add_reg(self, name, dtype):
        """Add a register declaration."""
        vlog_type = dtype_to_verilog_type(dtype)
        self.reg_decls[name] = f"{vlog_type} {name}"
    
    def get_binary_op(self, op):
        """Convert IR binary operation to Verilog operator."""
        op_map = {
            BinaryOp.ADD: "+",
            BinaryOp.SUB: "-",
            BinaryOp.MUL: "*",
            BinaryOp.DIV: "/",
            BinaryOp.MOD: "%",
            BinaryOp.ILT: "<",
            BinaryOp.IGT: ">",
            BinaryOp.ILE: "<=",
            BinaryOp.IGE: ">=",
            BinaryOp.EQ: "==",
            BinaryOp.NEQ: "!=",
            BinaryOp.BITWISE_OR: "|",
            BinaryOp.BITWISE_AND: "&",
            BinaryOp.BITWISE_XOR: "^",
            BinaryOp.SHL: "<<",
            BinaryOp.SHR: ">>",
        }
        return op_map.get(op, f"/* Unsupported operator: {op} */")
    
    def get_unary_op(self, op):
        """Convert IR unary operation to Verilog operator."""
        op_map = {
            UnaryOp.FLIP: "~",
            UnaryOp.NEG: "-",
        }
        return op_map.get(op, f"/* Unsupported operator: {op} */")
    
    def visit_system(self, sys):
        """Visit the system and generate Verilog code for the top module and all submodules."""
        # Start with the module name
        self.code.append(f"// Top module for {sys.name}")
        
        # Generate module declarations for each module in the system
        all_modules = sys.modules[:] + sys.downstreams[:]
        for module in all_modules:
            self.visit_module_decl(module)
        
        # Create top-level module
        ports = []
        port_list = []
        
        # Add standard ports
        port_list.append("  input  logic        clk")
        port_list.append("  input  logic        rst_n")
        
        # Add user-exposed ports based on exposed nodes
        for node, kind in sys.exposed_nodes.items():
            port_name = namify(node.as_operand())
            dtype = dtype_to_verilog_type(node.dtype)
            
            if kind == "Input":
                port_list.append(f"  input  logic {dtype} {port_name}")
            elif kind == "Output":
                port_list.append(f"  output logic {dtype} {port_name}")
            else:  # Inout
                port_list.append(f"  inout  logic {dtype} {port_name}")
                
            ports.append(port_name)
        
        # Create module declaration
        self.code.append(f"module {sys.name} (")
        self.code.append(",\n".join(port_list))
        self.code.append(");")
        
        # Process arrays
        for array in sys.arrays:
            for part in array.partition:
                self.visit_array(part)
        
        # Process all modules
        for module in all_modules:
            name = namify(module.name)
            instance_name = f"{name}_inst"
            
            # Setup module instance
            params = {}
            ports = {
                "clk": "clk",
                "rst_n": "rst_n"
            }
            
            # Add ports for modules
            if isinstance(module, Module):
                for port in module.ports:
                    port_name = namify(port.name)
                    fifo_name = f"{name}_{port_name}"
                    
                    # Create FIFO instance
                    fifo_width = port.dtype.bits
                    fifo_depth = self.fifo_depth
                    
                    fifo_params = {
                        "WIDTH": str(fifo_width),
                        "DEPTH_LOG2": str(fifo_depth.bit_length() - 1)
                    }
                    
                    fifo_ports = {
                        "clk": "clk",
                        "rst_n": "rst_n",
                        "push_valid": f"{fifo_name}_push_valid",
                        "push_data": f"{fifo_name}_push_data",
                        "push_ready": f"{fifo_name}_push_ready",
                        "pop_valid": f"{fifo_name}_pop_valid",
                        "pop_data": f"{fifo_name}_pop_data",
                        "pop_ready": f"{fifo_name}_pop_ready"
                    }
                    
                    # Add wire declarations for FIFO connections
                    self.add_wire(f"{fifo_name}_push_valid", port.dtype)
                    self.add_wire(f"{fifo_name}_push_data", port.dtype)
                    self.add_wire(f"{fifo_name}_push_ready", port.dtype)
                    self.add_wire(f"{fifo_name}_pop_valid", port.dtype)
                    self.add_wire(f"{fifo_name}_pop_data", port.dtype)
                    self.add_wire(f"{fifo_name}_pop_ready", port.dtype)
                    
                    # Create FIFO instance
                    fifo_inst = DisplayInstance.module(
                        "fifo", f"{fifo_name}_fifo", fifo_params, fifo_ports)
                    self.fifo_decls.append(fifo_inst)
                    
                    # Connect FIFO to module ports
                    # All ports are input ports in Assassyn
                    ports[f"{port_name}_valid"] = f"{fifo_name}_pop_valid"
                    ports[f"{port_name}_data"] = f"{fifo_name}_pop_data"
                    ports[f"{port_name}_ready"] = f"{fifo_name}_pop_ready"
            
            # Add parameters for memory modules
            elif isinstance(module, SRAM):
                params["WIDTH"] = str(module.width)
                params["DEPTH"] = str(module.depth)
                
                # Connect memory ports
                if module.re is not None:
                    ports["re"] = dump_rval_ref(module.re)
                if module.we is not None:
                    ports["we"] = dump_rval_ref(module.we)
                if module.addr is not None:
                    ports["addr"] = dump_rval_ref(module.addr)
                if module.wdata is not None:
                    ports["wdata"] = dump_rval_ref(module.wdata)
                if module.payload is not None:
                    ports["rdata"] = dump_rval_ref(module.payload)
            
            # Create module instance
            module_inst = DisplayInstance.module(
                name, instance_name, params, ports)
            self.instances.append(module_inst)
        
        # Add wire and register declarations
        for wire in self.wire_decls.values():
            self.code.append(f"  wire {wire};")
        
        for reg in self.reg_decls.values():
            self.code.append(f"  reg {reg};")
        
        # Add array declarations
        for array_decl in self.array_decls:
            self.code.append(f"  {array_decl}")
        
        # Add FIFO declarations
        for fifo_decl in self.fifo_decls:
            self.code.append(f"  {fifo_decl}")
        
        # Add module instances
        for instance in self.instances:
            self.code.append(f"  {instance}")
        
        # Add combinational logic
        if self.comb_logic:
            self.code.append("  // Combinational logic")
            for logic in self.comb_logic:
                self.code.append(f"  {logic}")
        
        # Add sequential logic
        if self.seq_logic:
            self.code.append("  // Sequential logic")
            for logic in self.seq_logic:
                self.code.append(f"  {logic}")
        
        # Close the module
        self.code.append("endmodule")
        
        # Add all module definitions
        for module_decl in self.module_decls:
            self.code.append(module_decl)
        
        return "\n".join(self.code)
    
    def visit_module_decl(self, module):
        """Generate module declaration for a module."""
        name = namify(module.name)
        self.module_decls.append(f"// Module: {name}")
        
        # Create port list
        ports = ["  input  logic        clk", "  input  logic        rst_n"]
        
        # Add ports based on module type
        if isinstance(module, Module):
            for port in module.ports:
                port_name = namify(port.name)
                dtype = dtype_to_verilog_type(port.dtype)
                
                # All ports are input ports in Assassyn
                ports.append(f"  input  logic              {port_name}_valid")
                ports.append(f"  input  logic {dtype}      {port_name}_data")
                ports.append(f"  output logic              {port_name}_ready")
        
        # Add ports for memory modules
        elif isinstance(module, SRAM):
            # Memory module parameters
            width = module.width
            depth = module.depth
            addr_width = (depth - 1).bit_length()
            
            self.module_decls.append(f"// Memory module: width={width}, depth={depth}")
            
            # Standard memory ports
            ports.append(f"  input  logic                re")
            ports.append(f"  input  logic                we")
            ports.append(f"  input  logic [{addr_width-1}:0]  addr")
            ports.append(f"  input  logic [{width-1}:0]       wdata")
            ports.append(f"  output logic [{width-1}:0]       rdata")
        
        # Create module declaration
        module_decl = [f"module {name} ("]
        module_decl.append(",\n".join(ports))
        module_decl.append(");")
        
        # Add module implementation based on type
        if isinstance(module, SRAM):
            # Basic memory implementation
            width = module.width
            depth = module.depth
            addr_width = (depth - 1).bit_length()
            
            module_decl.append(f"  // Memory array")
            module_decl.append(f"  reg [{width-1}:0] mem [{depth-1}:0];")
            
            # Initialize memory if init file provided
            if module.init_file:
                module_decl.append(f"  // Initialize memory from file")
                module_decl.append(f"  initial begin")
                module_decl.append(f"    $readmemh(\"{module.init_file}\", mem);")
                module_decl.append(f"  end")
            
            # Read and write logic
            module_decl.append(f"  // Read logic")
            module_decl.append(f"  always @(posedge clk) begin")
            module_decl.append(f"    if (re) begin")
            module_decl.append(f"      rdata <= mem[addr];")
            module_decl.append(f"    end")
            module_decl.append(f"  end")
            
            module_decl.append(f"  // Write logic")
            module_decl.append(f"  always @(posedge clk) begin")
            module_decl.append(f"    if (we) begin")
            module_decl.append(f"      mem[addr] <= wdata;")
            module_decl.append(f"    end")
            module_decl.append(f"  end")
        
        # For regular modules, we'll generate their implementation when we visit them
        elif isinstance(module, Module):
            # Create temp storage for module implementation
            prev_comb_logic = self.comb_logic
            prev_seq_logic = self.seq_logic
            prev_wire_decls = self.wire_decls.copy()
            prev_reg_decls = self.reg_decls.copy()
            
            # Reset storage for this module
            self.comb_logic = []
            self.seq_logic = []
            self.wire_decls = {}
            self.reg_decls = {}
            
            # Visit module to generate implementation
            self.module_ctx = module
            if hasattr(module, 'body'):
                self.visit_block(module.body)
            
            # Add wire and register declarations
            for wire in self.wire_decls.values():
                module_decl.append(f"  wire {wire};")
            
            for reg in self.reg_decls.values():
                module_decl.append(f"  reg {reg};")
            
            # Add combinational logic
            if self.comb_logic:
                module_decl.append("  // Combinational logic")
                for logic in self.comb_logic:
                    module_decl.append(f"  {logic}")
            
            # Add sequential logic
            if self.seq_logic:
                module_decl.append("  // Sequential logic")
                for logic in self.seq_logic:
                    module_decl.append(f"  {logic}")
            
            # Restore context
            self.comb_logic = prev_comb_logic
            self.seq_logic = prev_seq_logic
            self.wire_decls = prev_wire_decls
            self.reg_decls = prev_reg_decls
        
        # Close the module
        module_decl.append("endmodule\n")
        self.module_decls.extend(module_decl)
    
    def visit_array(self, array):
        """Visit an array and generate its declaration."""
        name = namify(array.name)
        size = array.size
        dtype = dtype_to_verilog_type(array.scalar_ty)
        
        decl = f"reg {dtype} {name} [{size-1}:0];"
        self.array_decls.append(decl)
        
        # Handle array initialization if provided
        if array.initializer:
            init_block = []
            init_block.append("initial begin")
            
            for i, value in enumerate(array.initializer):
                init_block.append(f"  {name}[{i}] = {int_imm_dumper_impl(array.scalar_ty, value)};")
            
            init_block.append("end")
            self.array_decls.append("\n  ".join(init_block))
    
    def visit_module(self, module):
        """Visit a module and generate its implementation."""
        self.module_ctx = module
        
        # Create function implementation
        if hasattr(module, 'body'):
            self.visit_block(module.body)
    
    def visit_block(self, block):
        """Visit a block and generate its Verilog representation."""
        if isinstance(block, CondBlock):
            cond = dump_rval_ref(block.cond)
            self.comb_logic.append(f"// Conditional block: {cond}")
            self.comb_logic.append(f"if ({cond}) begin")
            self.indent += 1
            
            for elem in block.iter():
                self.dispatch(elem)
            
            self.indent -= 1
            self.comb_logic.append("end")
        
        elif isinstance(block, CycledBlock):
            self.seq_logic.append(f"// Cycled block: {block.cycle}")
            self.seq_logic.append("always @(posedge clk) begin")
            self.indent += 1
            
            for elem in block.iter():
                self.dispatch(elem)
            
            self.indent -= 1
            self.seq_logic.append("end")
        
        else:
            # Regular block
            for elem in block.iter():
                self.dispatch(elem)
    
    def visit_expr(self, node):
        """Visit an expression and generate its Verilog representation."""
        # Generate different code based on expression type
        if node.is_binary():
            self.visit_binary_expr(node)
        elif node.is_unary():
            self.visit_unary_expr(node)
        elif isinstance(node, ArrayRead):
            self.visit_array_read(node)
        elif isinstance(node, ArrayWrite):
            self.visit_array_write(node)
        elif isinstance(node, FIFOPop):
            self.visit_fifo_pop(node)
        elif isinstance(node, FIFOPush):
            self.visit_fifo_push(node)
        elif isinstance(node, Bind):
            self.visit_bind(node)
        elif isinstance(node, AsyncCall):
            self.visit_async_call(node)
        elif isinstance(node, Select):
            self.visit_select(node)
        elif isinstance(node, Select1Hot):
            self.visit_select_1hot(node)
        elif isinstance(node, Slice):
            self.visit_slice(node)
        elif isinstance(node, Concat):
            self.visit_concat(node)
        elif isinstance(node, Cast):
            self.visit_cast(node)
        elif isinstance(node, Log):
            self.visit_log(node)
        elif isinstance(node, Intrinsic):
            self.visit_intrinsic(node)
        elif isinstance(node, PureIntrinsic):
            self.visit_pure_intrinsic(node)
    
    def visit_binary_expr(self, node):
        """Visit a binary expression."""
        lhs = dump_rval_ref(node.lhs)
        rhs = dump_rval_ref(node.rhs)
        op = self.get_binary_op(node.opcode)
        result = namify(node.as_operand())
        
        # Add wire declaration
        self.add_wire(result, node.dtype)
        
        # Generate assignment
        self.comb_logic.append(f"assign {result} = {lhs} {op} {rhs};")
    
    def visit_unary_expr(self, node):
        """Visit a unary expression."""
        x = dump_rval_ref(node.x)
        op = self.get_unary_op(node.opcode)
        result = namify(node.as_operand())
        
        # Add wire declaration
        self.add_wire(result, node.dtype)
        
        # Generate assignment
        self.comb_logic.append(f"assign {result} = {op}{x};")
    
    def visit_array_read(self, node):
        """Visit an array read expression."""
        array = dump_rval_ref(node.array)
        idx = dump_rval_ref(node.idx)
        result = namify(node.as_operand())
        
        # Add wire declaration
        self.add_wire(result, node.dtype)
        
        # Generate assignment
        self.comb_logic.append(f"assign {result} = {array}[{idx}];")
    
    def visit_array_write(self, node):
        """Visit an array write expression."""
        array = dump_rval_ref(node.array)
        idx = dump_rval_ref(node.idx)
        val = dump_rval_ref(node.val)
        
        # Generate assignment in sequential block
        self.seq_logic.append(f"{array}[{idx}] <= {val};")
    
    def visit_fifo_pop(self, node):
        """Visit a FIFO pop expression."""
        fifo = dump_rval_ref(node.fifo)
        result = namify(node.as_operand())
        
        # Add wire declaration
        self.add_wire(result, node.dtype)
        
        # Generate FIFO pop logic
        # In Verilog, we'll need to create the handshaking for FIFO pop
        fifo_name = fifo
        self.comb_logic.append(f"// Pop from FIFO {fifo_name}")
        self.comb_logic.append(f"assign {fifo_name}_pop_ready = 1'b1;  // Always ready to pop")
        self.comb_logic.append(f"assign {result} = {fifo_name}_pop_data;")
    
    def visit_fifo_push(self, node):
        """Visit a FIFO push expression.
        
        In Assassyn's design, a FIFO push is used to call another module,
        which means we're connecting to the input port of another module.
        """
        fifo = namify(node.fifo.name)
        val = dump_rval_ref(node.val)
        module_name = namify(node.bind.callee.name)
        
        # Generate FIFO push logic (to the target module's input)
        fifo_name = f"{module_name}_{fifo}"
        self.comb_logic.append(f"// Push to module {module_name} through port {fifo}")
        self.comb_logic.append(f"assign {fifo_name}_push_valid = 1'b1;  // Always valid for push")
        self.comb_logic.append(f"assign {fifo_name}_push_data = {val};  // Send the data")
    
    def visit_bind(self, node):
        """Visit a bind expression."""
        # Bind expressions are handled at the module level
        pass
    
    def visit_async_call(self, node):
        """Visit an async call expression."""
        # Async calls are handled through FIFO connections
        pass
    
    def visit_select(self, node):
        """Visit a select expression."""
        cond = dump_rval_ref(node.cond)
        true_val = dump_rval_ref(node.true_value)
        false_val = dump_rval_ref(node.false_value)
        result = namify(node.as_operand())
        
        # Add wire declaration
        self.add_wire(result, node.dtype)
        
        # Generate assignment with ternary operator
        self.comb_logic.append(f"assign {result} = {cond} ? {true_val} : {false_val};")
    
    def visit_select_1hot(self, node):
        """Visit a one-hot select expression."""
        cond = dump_rval_ref(node.cond)
        result = namify(node.as_operand())
        
        # Add wire declaration
        self.add_wire(result, node.dtype)
        
        # Generate case statement for one-hot select
        self.comb_logic.append(f"// One-hot select")
        self.comb_logic.append(f"always @(*) begin")
        self.comb_logic.append(f"  case (1'b1)")
        
        for i, value in enumerate(node.values):
            val = dump_rval_ref(value)
            self.comb_logic.append(f"    {cond}[{i}]: {result} = {val};")
        
        self.comb_logic.append(f"    default: {result} = 'x;")
        self.comb_logic.append(f"  endcase")
        self.comb_logic.append(f"end")
    
    def visit_slice(self, node):
        """Visit a slice expression."""
        x = dump_rval_ref(node.x)
        l = dump_rval_ref(node.l)
        r = dump_rval_ref(node.r)
        result = namify(node.as_operand())
        
        # Add wire declaration
        self.add_wire(result, node.dtype)
        
        # Generate assignment
        self.comb_logic.append(f"assign {result} = {x}[{l}:{r}];")
    
    def visit_concat(self, node):
        """Visit a concatenation expression."""
        msb = dump_rval_ref(node.msb)
        lsb = dump_rval_ref(node.lsb)
        result = namify(node.as_operand())
        
        # Add wire declaration
        self.add_wire(result, node.dtype)
        
        # Generate assignment
        self.comb_logic.append(f"assign {result} = {{{msb}, {lsb}}};")
    
    def visit_cast(self, node):
        """Visit a cast expression."""
        x = dump_rval_ref(node.x)
        result = namify(node.as_operand())
        dtype = dtype_to_verilog_type(node.dtype)
        
        # Add wire declaration
        self.add_wire(result, node.dtype)
        
        # Generate assignment based on cast type
        if node.opcode == Cast.BITCAST:
            self.comb_logic.append(f"assign {result} = {x};")
        elif node.opcode == Cast.ZEXT:
            self.comb_logic.append(f"assign {result} = {dtype}'({x});")
        elif node.opcode == Cast.SEXT:
            self.comb_logic.append(f"assign {result} = $signed({x});")
    
    def visit_log(self, node):
        """Visit a log expression."""
        fmt = node.args[0]
        args = [dump_rval_ref(arg) for arg in node.args[1:]]
        
        # Generate $display statement
        arg_str = ", ".join(args)
        self.seq_logic.append(f"$display(\"{fmt}\", {arg_str});")
    
    def visit_intrinsic(self, node):
        """Visit an intrinsic expression."""
        if node.opcode == Intrinsic.WAIT_UNTIL:
            cond = dump_rval_ref(node.args[0])
            self.seq_logic.append(f"// Wait until {cond}")
            self.seq_logic.append(f"if (!({cond})) begin")
            self.seq_logic.append(f"  // Stall until condition is met")
            self.seq_logic.append(f"end")
        
        elif node.opcode == Intrinsic.ASSERT:
            cond = dump_rval_ref(node.args[0])
            self.seq_logic.append(f"// Assert {cond}")
            self.seq_logic.append(f"if (!({cond})) begin")
            self.seq_logic.append(f"  $error(\"Assertion failed: {cond}\");")
            self.seq_logic.append(f"end")
        
        elif node.opcode == Intrinsic.FINISH:
            self.seq_logic.append(f"// Finish simulation")
            self.seq_logic.append(f"$finish;")
    
    def visit_pure_intrinsic(self, node):
        """Visit a pure intrinsic expression."""
        result = namify(node.as_operand())
        
        # Add wire declaration
        self.add_wire(result, node.dtype)
        
        if node.opcode == PureIntrinsic.FIFO_PEEK:
            fifo = dump_rval_ref(node.args[0])
            self.comb_logic.append(f"assign {result} = {fifo}_pop_data;")
        
        elif node.opcode == PureIntrinsic.FIFO_VALID:
            fifo = dump_rval_ref(node.args[0])
            self.comb_logic.append(f"assign {result} = {fifo}_pop_valid;")


def elaborate_impl(sys, config):
    """Internal implementation of the elaborate function."""
    # Create output directory
    verilog_dir = config.get('dirname', f"{sys.name}_verilog")
    verilog_path = Path(config.get('path', os.getcwd())) / verilog_dir
    
    # Clean directory if it exists and override is enabled
    if verilog_path.exists() and config.get('override_dump', True):
        shutil.rmtree(verilog_path)
    
    # Create directories
    verilog_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing Verilog code to: {verilog_path}")
    
    # Generate Verilog code
    dumper = VerilogDumper(sys)
    dumper.fifo_depth = config.get('fifo_depth', 2)
    verilog_code = dumper.visit_system(sys)
    
    # Write the main Verilog file
    main_file = verilog_path / f"{sys.name}.sv"
    with open(main_file, 'w', encoding="utf-8") as fd:
        fd.write(verilog_code)
    
    # Copy runtime files (FIFO implementation, testbench, etc.)
    runtime_src = Path(__file__).parent / "runtime.sv"
    if runtime_src.exists():
        shutil.copy(runtime_src, verilog_path / "runtime.sv")
    else:
        # If runtime.sv doesn't exist in the Python module, copy it from the Rust backend
        rust_runtime = Path("/Users/were/repos/assassyn-dev/src/backend/verilog/runtime.sv")
        if rust_runtime.exists():
            shutil.copy(rust_runtime, verilog_path / "runtime.sv")
    
    # Create a simple testbench
    tb_file = verilog_path / f"{sys.name}_tb.sv"
    with open(tb_file, 'w', encoding="utf-8") as fd:
        fd.write(f"""
`timescale 1ns/1ps

module {sys.name}_tb;
  // Clock and reset
  logic clk;
  logic rst_n;
  
  // Instantiate the DUT
  {sys.name} dut (
    .clk(clk),
    .rst_n(rst_n)
    // Add other ports here
  );
  
  // Clock generation
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end
  
  // Reset generation
  initial begin
    rst_n = 0;
    #20 rst_n = 1;
  end
  
  // Test stimulus
  initial begin
    // Wait for reset to complete
    wait(rst_n);
    
    // Add test stimulus here
    
    // Run for some cycles
    repeat(100) @(posedge clk);
    
    // End simulation
    $display("Simulation completed");
    $finish;
  end
  
  // Add waveform dumping
  initial begin
    $dumpfile("{sys.name}.vcd");
    $dumpvars(0, {sys.name}_tb);
  end
endmodule
""")
    
    return verilog_path


def elaborate(sys, **config):
    """Generate Verilog code for the given Assassyn system.
    
    This function is the main entry point for Verilog generation. It takes
    an Assassyn system builder and configuration options, and generates Verilog
    code that implements the system.
    
    Args:
        sys: The Assassyn system builder
        **config: Configuration options including:
            - dirname: Output directory name (default: {sys.name}_verilog)
            - path: Base path for output (default: current directory)
            - override_dump: Whether to overwrite existing files (default: True)
            - fifo_depth: Default FIFO depth (default: 2)
            - simulator: Verilog simulator to target (vcs, verilator, None)
    
    Returns:
        Path to the generated Verilog directory
    """
    # Generate the Verilog code
    verilog_path = elaborate_impl(sys, config)
    
    # If a verilog simulator is specified, create simulator-specific files
    verilog_sim = config.get('verilog')
    if verilog_sim:
        if verilog_sim == 'verilator':
            # Create Verilator testbench
            create_verilator_testbench(sys, verilog_path)
        elif verilog_sim == 'vcs':
            # Create VCS simulation script
            create_vcs_script(sys, verilog_path)
    
    return verilog_path


def create_verilator_testbench(sys, verilog_path):
    """Create a Verilator C++ testbench."""
    cpp_file = verilog_path / "main.cpp"
    with open(cpp_file, 'w', encoding="utf-8") as fd:
        fd.write(f"""
#include <iostream>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "V{sys.name}.h"

int main(int argc, char** argv) {{
    // Initialize Verilator
    Verilated::commandArgs(argc, argv);
    
    // Create an instance of the model
    V{sys.name}* top = new V{sys.name};
    
    // Enable waveform tracing
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    top->trace(tfp, 99);
    tfp->open("{sys.name}.vcd");
    
    // Initialize simulation inputs
    top->clk = 0;
    top->rst_n = 0;
    
    // Run simulation
    for (int i = 0; i < 1000; i++) {{
        // Toggle clock
        top->clk = !top->clk;
        
        // Release reset after 20 timesteps
        if (i == 40) {{
            top->rst_n = 1;
        }}
        
        // Evaluate model
        top->eval();
        
        // Dump waveforms
        tfp->dump(i);
    }}
    
    // Clean up
    tfp->close();
    delete tfp;
    delete top;
    
    return 0;
}}
""")
    
    # Create Makefile
    makefile = verilog_path / "Makefile"
    with open(makefile, 'w', encoding="utf-8") as fd:
        fd.write(f"""
TARGET = {sys.name}
VERILATOR = verilator
VERILATOR_FLAGS = --trace -Wall --top-module {sys.name}

all: $(TARGET)

$(TARGET): main.cpp $(TARGET).sv runtime.sv
	$(VERILATOR) $(VERILATOR_FLAGS) --cc $(TARGET).sv runtime.sv --exe main.cpp
	make -C obj_dir -f V$(TARGET).mk

clean:
	rm -rf obj_dir *.vcd

run: $(TARGET)
	obj_dir/V$(TARGET)

.PHONY: all clean run
""")


def create_vcs_script(sys, verilog_path):
    """Create a VCS simulation script."""
    tcl_file = verilog_path / "run_vcs.tcl"
    with open(tcl_file, 'w', encoding="utf-8") as fd:
        fd.write(f"""
# VCS simulation script for {sys.name}

# Compile the design
vcs -full64 -sverilog -debug_acc+all -LDFLAGS -Wl,--no-as-needed \\
    {sys.name}.sv {sys.name}_tb.sv runtime.sv -o {sys.name}_sim

# Run the simulation
./{sys.name}_sim -gui &
""")
    
    # Create a shell script to run the VCS simulation
    sh_file = verilog_path / "run_vcs.sh"
    with open(sh_file, 'w', encoding="utf-8") as fd:
        fd.write(f"""#!/bin/bash
# Run VCS simulation for {sys.name}
vcs -full64 -sverilog -debug_acc+all -LDFLAGS -Wl,--no-as-needed \\
    {sys.name}.sv {sys.name}_tb.sv runtime.sv -o {sys.name}_sim

# Run the simulation
./{sys.name}_sim -gui
""")
    
    # Make the shell script executable
    sh_file.chmod(0o755)
