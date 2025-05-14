"""Verilog dumper for Assassyn."""

from collections import defaultdict, deque
import os
from typing import Dict, List, Optional, Set, Tuple, Any

from ...builder import SysBuilder
from ...ir.block import Block
from ...ir.expr import Expr, Intrinsic, Operand
from ...ir.module import Module
from ...ir.visitor import Visitor
from ...utils import identifierize

from .utils import bool_ty, declare_array, declare_in, declare_out
from .visit_expr import visit_expr_impl, dump_ref, dump_arith_ref
from .gather import Gather, ExternalUsage

def fifo_name(fifo):
    """Get the name of a FIFO."""
    return identifierize(fifo.name)

class VerilogDumper(Visitor):
    """Dumps Verilog code for Assassyn modules."""
    
    def __init__(self, sys, config, external_usage, topo, array_memory_params_map, module_expr_map):
        """Initialize a VerilogDumper."""
        self.sys = sys
        self.config = config
        self.pred_stack = deque()
        self.fifo_pushes = {}
        self.array_stores = {}
        self.triggers = {}
        self.external_usage = external_usage
        self.current_module = ""
        self.before_wait_until = False
        self.topo = topo
        self.array_memory_params_map = array_memory_params_map
        self.module_expr_map = module_expr_map
    
    @classmethod
    def new(cls, sys, config, external_usage, topo, array_memory_params_map, module_expr_map):
        """Create a new VerilogDumper."""
        return cls(sys, config, external_usage, topo, array_memory_params_map, module_expr_map)
    
    @classmethod
    def collect_array_memory_params_map(cls, sys):
        """Collect array memory parameters."""
        result = {}
        for module in sys.modules:
            for attr in module.attrs:
                if isinstance(attr, MemoryParams):
                    result[module.payload] = attr
        return result
    
    def get_pred(self) -> Optional[str]:
        """Get the current predicate."""
        if not self.pred_stack:
            return None
        return self.pred_stack[-1]
    
    def print_body(self, node) -> str:
        """Print the body of a node."""
        if node.kind == NodeKind.EXPR:
            expr = node.as_ref(Expr, self.sys)
            return self.visit_expr(expr) or ""
        elif node.kind == NodeKind.BLOCK:
            block = node.as_ref(Block, self.sys)
            return self.visit_block(block) or ""
        else:
            raise ValueError(f"Unexpected reference type: {node.kind}")
    
    def dump_memory_nodes(self, node, res: str) -> None:
        """Dump memory nodes."""
        # This is a placeholder for the actual implementation
        # TODO: Implement memory node dumping
        pass
    
    def dump_runtime(self, fd, sim_threshold) -> None:
        """Dump the runtime.sv file."""
        # Just copy the runtime.sv file for now
        runtime_path = os.path.join(os.path.dirname(__file__), "runtime.sv")
        with open(runtime_path, "r") as src:
            fd.write(src.read())
    
    def visit_module(self, module) -> Optional[str]:
        """Visit a module and generate Verilog code."""
        self.current_module = identifierize(module.name)
        
        result = []
        
        # Module header
        result.append(f"""
module {self.current_module} (
  input logic clk,
  input logic rst_n,
""")
        
        # FIFO ports
        for port in module.fifo_iter():
            name = fifo_name(port)
            ty = port.scalar_ty
            display = utils.DisplayInstance.from_fifo(port, False)
            
            result.append(f"  // Port FIFO {name}")
            result.append(declare_in(bool_ty(), display.field("pop_valid")))
            result.append(declare_in(ty, display.field("pop_data")))
            result.append(declare_out(bool_ty(), display.field("pop_ready")))
        
        # Memory parameters
        has_memory_params = False
        has_memory_init_path = False
        empty_pins = MemoryPins(
            unknown(),  # array
            unknown(),  # re
            unknown(),  # we
            unknown(),  # addr
            unknown(),  # wdata
        )
        
        memory_params = MemoryParams(
            0,          # width
            0,          # depth
            range(0, 1),  # lat
            None,       # init_file
            empty_pins, # pins
        )
        
        init_file_path = self.config.get("resource_base", ".")
        
        # External interfaces
        for interf, ops in module.ext_interf_iter():
            if interf.kind == NodeKind.FIFO:
                fifo = interf.as_ref(FIFO, self.sys)
                parent_name = fifo.module.name
                display = utils.DisplayInstance.from_fifo(fifo, True)
                
                result.append(f"  // External FIFO {parent_name}.{fifo.name}")
                result.append(declare_out(bool_ty(), display.field("push_valid")))
                result.append(declare_out(fifo.scalar_ty, display.field("push_data")))
                result.append(declare_in(bool_ty(), display.field("push_ready")))
            
            elif interf.kind == NodeKind.ARRAY:
                array = interf.as_ref(Array, self.sys)
                display = utils.DisplayInstance.from_array(array)
                result.append(f"  /* {array} */")
                
                # Check for memory parameters
                for attr in module.attrs:
                    if isinstance(attr, MemoryParams):
                        has_memory_params = True
                        memory_params = attr
                        
                        if attr.init_file:
                            init_file_path = os.path.join(init_file_path, attr.init_file)
                            result.append(f"  /* {init_file_path} */")
                            has_memory_init_path = True
                
                if has_memory_params:
                    pass
                else:
                    if self.sys.user_contains_opcode(ops, Opcode.LOAD):
                        result.append(declare_array("input", array, display.field("q"), ","))
                    
                    if self.sys.user_contains_opcode(ops, Opcode.STORE):
                        result.append(declare_out(bool_ty(), display.field("w")))
                        result.append(declare_out(array.get_idx_type(), display.field("widx")))
                        result.append(declare_out(array.scalar_ty, display.field("d")))
            
            elif interf.kind == NodeKind.MODULE:
                module_ref = interf.as_ref(Module, self.sys)
                display = utils.DisplayInstance.from_module(module_ref)
                result.append(f"  // Module {module_ref.name}")
                
                # FIXME: Don't hardcode counter delta width
                result.append(declare_out(Int(8), display.field("counter_delta")))
                result.append(declare_in(bool_ty(), display.field("counter_delta_ready")))
            
            elif interf.kind == NodeKind.EXPR:
                # Handled below in module_expr_map
                pass
            
            else:
                raise ValueError(f"Unknown interf kind {interf.kind}")
            
            result.append("")
        
        # Downstream upstream signals
        if module.is_downstream:
            result.append("  // Declare upstream executed signals")
            for upstream in upstreams(module, self.topo):
                name = identifierize(upstream.as_ref(Module, module.sys).name)
                result.append(declare_in(bool_ty(), f"{name}_executed"))
        
        # External usage out bounds
        out_bounds = self.external_usage.out_bounds(module)
        if out_bounds:
            for elem in out_bounds:
                id_ = identifierize(str(elem))
                dtype = elem.get_dtype(module.sys)
                result.append(declare_out(dtype, f"expose_{id_}"))
                result.append(declare_out(bool_ty(), f"expose_{id_}_valid"))
        
        # External usage in bounds
        in_bounds = self.external_usage.in_bounds(module)
        if in_bounds:
            for elem in in_bounds:
                id_ = identifierize(str(elem))
                dtype = elem.get_dtype(module.sys)
                result.append(declare_in(dtype, id_))
                result.append(declare_in(bool_ty(), f"{id_}_valid"))
        
        # Exposed expressions
        if module in self.module_expr_map:
            exposed_map = self.module_expr_map[module]
            for exposed_node, kind in exposed_map.items():
                if exposed_node.kind == NodeKind.EXPR:
                    expr = exposed_node.as_ref(Expr, self.sys)
                    id_ = identifierize(str(expr))
                    dtype = exposed_node.get_dtype(self.sys)
                    bits = dtype.bits - 1
                    
                    if kind == "Output" or kind == "Inout":
                        result.append(f"  output logic [{bits}:0] {id_}_exposed_o,")
                    
                    if kind == "Input" or kind == "Inout":
                        result.append(f"  input logic [{bits}:0] {id_}_exposed_i,")
                        result.append(f"  input logic {id_}_exposed_i_valid,")
        
        # Event queue for non-downstream modules
        if not module.is_downstream:
            result.append("  // self.event_q")
            result.append("  input logic counter_pop_valid,")
            result.append("  input logic counter_delta_ready,")
            result.append("  output logic counter_pop_ready,")
        
        # End of port declarations
        result.append("  output logic expose_executed);\n")
        
        # Wait until handling
        wait_until = ""
        skip = 0
        
        if module.body.has_wait_until():
            wu_intrin = module.body.get_wait_until()
            self.before_wait_until = True
            
            body_iter = module.body.body_iter()
            for i, elem in enumerate(body_iter):
                if elem == wu_intrin:
                    skip = i + 1
                    break
                result.append(self.print_body(elem))
            
            bi = wu_intrin.as_ref(Intrinsic, self.sys)
            value = bi.operands[0].value
            wait_until = f" && ({identifierize(str(value))})"
            
        self.before_wait_until = False
        
        # Executed logic
        result.append("  logic executed;")
        
        # Testbench cycle counter
        if self.current_module == "testbench":
            result.append("""
  int cycle_cnt;
  always_ff @(posedge clk or negedge rst_n) if (!rst_n) cycle_cnt <= 0;
  else if (executed) cycle_cnt <= cycle_cnt + 1;
""")
        
        # Clear data structures for gathering
        self.fifo_pushes.clear()
        self.array_stores.clear()
        self.triggers.clear()
        
        # Handle memory parameters
        if has_memory_params:
            result.append(f"  logic [{memory_params.width - 1}:0] dataout;")
            self.dump_memory_nodes(module.body, result)
        else:
            for elem in list(module.body.body_iter())[skip:]:
                result.append(self.print_body(elem))
        
        # Generate triggers
        for m, g in self.triggers.items():
            trigger_str = "1"
            if g.is_conditional():
                bits = g.bits
                trigger_str = " + ".join([f"{{ {bits-1}'b0, |{x} }}" for x in g.condition])
                
            result.append(f"  assign {m}_counter_delta = executed ? {trigger_str} : 0;\n")
        
        # Generate FIFO pushes
        result.append("  // Gather FIFO pushes")
        for fifo, g in self.fifo_pushes.items():
            result.append(f"""  assign fifo_{fifo}_push_valid = {g.and_("executed", " || ")};
  assign fifo_{fifo}_push_data = {g.select_1h()};
""")
        
        # Generate array writes
        result.append("  // Gather Array writes")
        if has_memory_params:
            result.append("  // this is Mem Array")
            
            for a, (idx, data) in self.array_stores.items():
                result.append(f"  logic array_{a}_w;")
                result.append(f"  logic [{memory_params.width - 1}:0] array_{a}_d;")
                addr_bits = (63 - (memory_params.depth - 1).bit_length())
                result.append(f"  logic [{addr_bits}:0] array_{a}_widx;")
                
                result.append(f"""  assign array_{a}_w = {idx.and_("executed", " || ")};
  assign array_{a}_d = {data.select_1h()};
  assign array_{a}_widx = {idx.value[0]};
""")
                
                result.append(f"""
  memory_blackbox_{a} #(
        .DATA_WIDTH({memory_params.width}),
        .ADDR_WIDTH({addr_bits})
    ) memory_blackbox_{a}(
    .clk     (clk),
    .address (array_{a}_widx),
    .wd      (array_{a}_d),
    .banksel (1'd1),
    .read    (1'd1),
    .write   (array_{a}_w),
    .dataout (dataout),
    .rst_n   (rst_n)
    );
          """)
        else:
            for a, (idx, data) in self.array_stores.items():
                result.append(f"""  assign array_{a}_w = {idx.and_("executed", " || ")};
    assign array_{a}_d = {data.select_1h()};
    assign array_{a}_widx = {idx.select_1h()};

  """)
        
        # Executed logic
        if not module.is_downstream:
            result.append(f"  assign executed = counter_pop_valid{wait_until};")
            result.append("  assign counter_pop_ready = executed;")
        else:
            upstream_execs = [f"{identifierize(x.as_ref(Module, module.sys).name)}_executed" 
                            for x in upstreams(module, self.topo)]
            result.append(f"  assign executed = {' || '.join(upstream_execs)};")
        
        result.append("  assign expose_executed = executed;")
        result.append(f"endmodule // {self.current_module}\n")
        
        # Memory blackbox modules
        if has_memory_params:
            for a, (_, _) in self.array_stores.items():
                data_width = memory_params.width
                addr_bits = (63 - (memory_params.depth).bit_length())
                
                result.append(f"""
`ifdef SYNTHESIS
(* blackbox *)
`endif
module memory_blackbox_{a} #(
    parameter DATA_WIDTH = {data_width},
    parameter ADDR_WIDTH = {addr_bits}
)(
    input clk,
    input [ADDR_WIDTH-1:0] address,
    input [DATA_WIDTH-1:0] wd,
    input banksel,
    input read,
    input write,
    output reg [DATA_WIDTH-1:0] dataout,
    input rst_n
);

    localparam DEPTH = 1 << ADDR_WIDTH;
    reg [DATA_WIDTH-1:0] mem [DEPTH-1:0];

  """)
                
                if has_memory_init_path:
                    result.append(f"""  initial begin
          $readmemh("{init_file_path}", mem);
      end
        always @ (posedge clk) begin
            if (write & banksel) begin
                mem[address] <= wd;
            end
        end

        assign dataout = (read & banksel) ? mem[address] : {{DATA_WIDTH{{1'b0}}}};

    endmodule
              """)
                else:
                    result.append("""

        always @ (posedge clk) begin
            if (!rst_n) begin
                mem[address] <= {DATA_WIDTH{1'b0}};
            end
            else if (write & banksel) begin
                mem[address] <= wd;
            end
        end

        assign dataout = (read & banksel) ? mem[address] : {DATA_WIDTH{1'b0}};

    endmodule
              """)
        
        return "\n".join(result)
    
    def visit_block(self, block) -> Optional[str]:
        """Visit a block and generate Verilog code."""
        result = []
        skip = 0
        
        # Handle block condition
        if block.condition:
            cond = block.condition
            dtype = cond.get_dtype(block.sys)
            
            if dtype.bits == 1:
                pred = dump_ref(self.sys, cond, True)
            else:
                pred = f"(|{dump_ref(self.sys, cond, False)})"
                
            self.pred_stack.append(pred)
            skip = 1
            
        # Handle cycled block
        elif block.cycle is not None:
            cycle = block.cycle
            self.pred_stack.append(f"(cycle_cnt == {cycle})")
            skip = 1
        
        # Process block body
        for elem in list(block.body_iter())[skip:]:
            if elem.kind == NodeKind.EXPR:
                expr = elem.as_ref(Expr, self.sys)
                result.append(self.visit_expr(expr) or "")
            elif elem.kind == NodeKind.BLOCK:
                sub_block = elem.as_ref(Block, self.sys)
                result.append(self.visit_block(sub_block) or "")
            else:
                raise ValueError(f"Unexpected reference type: {elem.kind}")
        
        if skip > 0:
            self.pred_stack.pop()
            
        return "".join(result)
    
    def visit_expr(self, expr: Expr) -> Optional[str]:
        """Visit an expression and generate Verilog code."""
        return visit_expr_impl(self, expr)


def generate_cpp_testbench(dir_path, sys, config):
    """Generate C++ testbench for Verilator simulation."""
    if config.get("verilog") == "Verilator":
        # Copy main.cpp and Makefile
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        main_fname = os.path.join(dir_path, "main.cpp")
        with open(os.path.join(current_dir, "main.cpp"), "r") as src:
            with open(main_fname, "w") as dst:
                dst.write(src.read())
        
        make_fname = os.path.join(dir_path, "Makefile")
        with open(os.path.join(current_dir, "Makefile"), "r") as src:
            content = src.read().format(sys.name)
            with open(make_fname, "w") as dst:
                dst.write(content)
    
    return True

class ExposeGather:
    """Gathers exposed expressions."""
    
    def __init__(self, sys):
        """Initialize an ExposeGather."""
        self.exposed_map = {}
        self.sys = sys
    
    def visit_expr(self, expr: Expr) -> None:
        """Visit an expression to check if it's exposed."""
        for node, kind in self.sys.exposed_nodes:
            if node == expr:
                self.exposed_map[expr] = kind