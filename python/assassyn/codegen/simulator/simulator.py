"""Simulator generation for Assassyn."""

import os
from collections import defaultdict
from .utils import namify, dtype_to_rust_type, int_imm_dumper_impl, fifo_name
from .node_dumper import externally_used_combinational
from ...module import Downstream, Module, SRAM
from ...builder import SysBuilder
from ...block import Block
from ...expr import Expr


def dump_simulator( #pylint: disable=too-many-locals, too-many-branches, too-many-statements
                   sys: SysBuilder, config, fd):
    """Generate the simulator module.

    This matches the Rust function in src/backend/simulator/elaborate.rs
    """
    # Write imports
    fd.write("use std::collections::VecDeque;\n")
    fd.write("use super::runtime::*;\n")
    fd.write("use num_bigint::{BigInt, BigUint};\n")
    fd.write("use rand::seq::SliceRandom;\n\n")

    # Initialize data structures
    simulator_init = []
    downstream_reset = []
    registers = []

    # Begin simulator struct definition
    fd.write("pub struct Simulator { pub stamp: usize, ")

    # Add array fields to simulator struct
    for array in sys.arrays:
        name = namify(array.name)
        dtype = dtype_to_rust_type(array.scalar_ty)
        fd.write(f"pub {name} : Array<{dtype}>, ")

        # Handle array initialization
        if array.initializer:
            init_values = []
            for x in array.initializer:
                init_values.append(int_imm_dumper_impl(array.scalar_ty, x))
            init_str = ", ".join(init_values)
            simulator_init.append(f"{name} : Array::new_with_init(vec![{init_str}]),")
        else:
            simulator_init.append(f"{name} : Array::new({array.size}),")

        registers.append(name)

    # Track expressions with external visibility
    expr_validities = set()

    # Add module fields to simulator struct
    for module in sys.modules:
        assert isinstance(module, Module)

        module_name = namify(module.name)

        # Add triggered flag for all modules
        fd.write(f"pub {module_name}_triggered : bool, ")
        simulator_init.append(f"{module_name}_triggered : false,")
        downstream_reset.append(f"self.{module_name}_triggered = false;")

        if isinstance(module, Downstream):
            # Add event queue for non-downstream modules
            fd.write(f"pub {module_name}_event : VecDeque<usize>, ")
            simulator_init.append(f"{module_name}_event : VecDeque::new(),")

            # Add FIFO fields for each FIFO
            for fifo in module.fifo_iter():
                name = fifo_name(fifo)
                ty = dtype_to_rust_type(fifo.scalar_ty())
                fd.write(f"pub {name} : FIFO<{ty}>, ")
                simulator_init.append(f"{name} : FIFO::new(),")
                registers.append(name)
        else:
            # Gather expressions with external visibility for downstream modules
            for expr in module.externals:
                if isinstance(expr, Expr) and externally_used_combinational(expr):
                    expr_validities.add(expr)

    # Add value validity tracking for expressions with external visibility
    for expr in expr_validities:
        name = namify(expr.get_name())
        dtype = dtype_to_rust_type(expr.dtype())
        fd.write(f"pub {name}_value : Option<{dtype}>, ")
        simulator_init.append(f"{name}_value : None,")
        downstream_reset.append(f"self.{name}_value = None;")

    # Close simulator struct
    fd.write("}\n\n")

    # Begin simulator implementation
    fd.write("impl Simulator {\n")

    # Constructor
    fd.write("  pub fn new() -> Self {\n")
    fd.write("    Simulator {\n")
    fd.write("      stamp: 0,\n")
    for init in simulator_init:
        fd.write(f"      {init}\n")
    fd.write("    }\n")
    fd.write("  }\n\n")

    # Event validity check
    fd.write("  fn event_valid(&self, event: &VecDeque<usize>) -> bool {\n")
    fd.write("    event.front().map_or(false, |x| *x <= self.stamp)\n")
    fd.write("  }\n\n")

    # Reset downstream method
    fd.write("  pub fn reset_downstream(&mut self) {\n")
    for reset in downstream_reset:
        fd.write(f"    {reset}\n")
    fd.write("  }\n\n")

    # Tick registers method
    fd.write("  pub fn tick_registers(&mut self) {\n")
    for reg in registers:
        fd.write(f"    self.{reg}.tick(self.stamp);\n")
    fd.write("  }\n\n")

    # Critical path analysis
    # TODO(@derui): Implement critical path analysis equivalent to Rust

    # Get topological order for downstream modules
    downstreams = analyze_topological_order(sys)
    topo_map = {module: i for i, module in enumerate(downstreams)}

    # Module simulation functions
    simulators = []
    for module in sys.modules:
        module_name = namify(module.name)
        fd.write(f"  fn simulate_{module_name}(&mut self) {{\n")

        if not isinstance(module, Downstream):
            # Event based triggering for non-downstream modules
            fd.write(f"    if self.event_valid(&self.{module_name}_event) {{\n")
        else:
            # Dependency based triggering for downstream modules
            upstream_conds = []
            for upstream in get_upstreams(module, topo_map, sys):
                upstream_name = namify(upstream.name)
                upstream_conds.append(f"self.{upstream_name}_triggered")

            conds = " || ".join(upstream_conds) if upstream_conds else "false"
            fd.write(f"    if {conds} {{\n")

        # Call module function and handle result
        fd.write(f"      let succ = super::modules::{module_name}(self);\n")

        if not isinstance(module, Downstream):
            # Pop event on success
            fd.write(f"      if succ {{ self.{module_name}_event.pop_front(); }}\n")
            fd.write("      else {\n")

            # Reset externally used values on failure
            for expr in expr_validities:
                if expr.get_block().get_module() == module:
                    name = namify(expr.get_name())
                    fd.write(f"        self.{name}_value = None;\n")

            fd.write("      }\n")
            simulators.append(module_name)

        # Update trigger state and close condition
        fd.write(f"      self.{module_name}_triggered = succ;\n")
        fd.write("    } // close event condition\n")
        fd.write("  } // close function\n\n")

    # Close simulator impl
    fd.write("}\n\n")

    # Generate simulate function
    fd.write("pub fn simulate() {\n")
    fd.write("  let mut sim = Simulator::new();\n")

    # Handle randomization if enabled
    if config.get('random', False):
        fd.write("  let mut rng = rand::thread_rng();\n")
        fd.write("  let mut simulators : Vec<fn(&mut Simulator)> = vec![")
    else:
        fd.write("  let simulators : Vec<fn(&mut Simulator)> = vec![")

    # Add simulators for all non-downstream modules
    for sim in simulators:
        fd.write(f"Simulator::simulate_{sim}, ")
    fd.write("];\n")

    # Add simulators for downstream modules
    fd.write("  let downstreams : Vec<fn(&mut Simulator)> = vec![")
    for downstream in downstreams:
        module_name = downstream.get_name()
        fd.write(f"Simulator::simulate_{module_name}, ")
    fd.write("];\n")

    # Initialize memory from files if needed
    for sram in [m for m in sys.modules if isinstance(m, SRAM)]:
        init_file_path = os.path.join(config.get('resource_base', '.'), sram.init_file)
        array = sram.payload 
        array_name = namify(array.name)
        fd.write(f'  load_hex_file(&mut sim.{array_name}.payload, "{init_file_path}");\n')

    # Set simulation threshold
    sim_threshold = config.get('sim_threshold', 100)

    # Add initial events for driver if present
    fd.write(f"""
        for i in 1..={sim_threshold} {{ sim.driver_event.push_back(i * 100); }} """)

    # Add initial events for testbench if present
    if sys.has_module("testbench") is not None:
        testbench = sys.get_module("testbench")
        cycles = []

        # Collect cycles from testbench blocks
        for block in testbench.get_body().body_iter():
            if isinstance(block, Block):
                if block.get_cycle() is not None:
                    cycles.append(block.get_cycle())

        if cycles:
            fd.write(f"""
              let tb_cycles = vec![{', '.join(map(str, cycles))}];
              for cycle in tb_cycles {{
                sim.testbench_event.push_back(cycle * 100);
              }}
            """)

    # Generate main simulation loop
    randomization = ""
    if config.get('random', False):
        randomization = "    simulators.shuffle(&mut rng);\n"

    fd.write(f"""
      for i in 1..={sim_threshold} {{
        sim.stamp = i * 100;
        sim.reset_downstream();
{randomization}
        for simulate in simulators.iter() {{
          simulate(&mut sim);
        }}
        for simulate in downstreams.iter() {{
          simulate(&mut sim);
        }}
        sim.stamp += 50;
        sim.tick_registers();
      }}
    """)

    # Close simulate function
    fd.write("}\n")

    return True


def analyze_topological_order(sys):
    """Analyze the topological order of modules.

    This is a simplified implementation of the topo_sort function in Rust.
    """
    # Get all downstream modules
    modules = [m for m in sys.modules if isinstance(m, Downstream)]

    # Build dependency graph
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for module in modules:
        deps = module.get_dependencies()
        for dep in deps:
            graph[dep].append(module)
            in_degree[module] += 1

    # Topological sort
    queue = [m for m in modules if in_degree[m] == 0]
    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result


def get_upstreams(module, topo_map, sys):
    """Get upstream modules of a given module.

    This matches the upstreams function in Rust.
    """
    result = []

    # Each module depends on all modules with a lower topo_map index
    module_idx = topo_map.get(module, float('inf'))
    for other_module in sys.module_iter():
        if other_module == module:
            continue

        other_idx = topo_map.get(other_module, float('inf'))
        if other_idx < module_idx:
            result.append(other_module)

    return result


def dump_main(fd):
    """Generate the main.rs file.

    This matches the Rust function in src/backend/simulator/elaborate.rs
    """
    fd.write("""
mod runtime;
mod modules;
mod simulator;

fn main() {
  simulator::simulate();
}
    """)
    return True
