"""Elaborate Assassyn IR to Verilog."""

import os
import shutil
from typing import Dict, List, Optional, Set, Tuple, Any

from ...builder import SysBuilder
from ...analysis import topo_downstream_modules

from .gather import gather_exprs_externally_used
from .verilog_dumper import VerilogDumper, generate_cpp_testbench

from ...utils import create_and_clean_dir

def elaborate(sys: SysBuilder, **kwargs) -> str:
    """Elaborate the system into Verilog.
    
    Args:
        sys: The system to elaborate
        **kwargs: Configuration options including:
            - verilog: The simulator to use ("Verilator", "VCS", or None)
            - resource_base: Path to resources
            - override_dump: Whether to override existing files
            - sim_threshold: Simulation threshold
            - idle_threshold: Idle threshold
            - random: Whether to randomize execution
            - fifo_depth: Default FIFO depth
    
    Returns:
        Path to the generated Verilog files
    """
    # Verify simulator is specified
    verilog = kwargs.get("verilog")
    if not verilog:
        raise ValueError("No simulator specified for verilog generation")
    
    # Process configuration
    config = {
        "verilog": verilog,
        "resource_base": kwargs.get("resource_base", "."),
        "override_dump": kwargs.get("override_dump", False),
        "sim_threshold": kwargs.get("sim_threshold", 100),
    }
    
    # Create output directory
    verilog_dir = os.path.join(kwargs.get("output_dir", "."), "verilog")
    create_and_clean_dir(verilog_dir)
    
    # Generate full path for the output file
    verilog_file = os.path.join(verilog_dir, f"{sys.name}.sv")
    print(f"Writing verilog rtl to {verilog_file}")
    
    # Generate C++ testbench if needed
    generate_cpp_testbench(verilog_dir, sys, config)
   
    # Calculate topological order of modules
    topo = topo_downstream_modules(sys)
    topo_dict = {node: i for i, node in enumerate(topo)}
    
    # Gather externally used expressions
    external_usage = gather_exprs_externally_used(sys)
    
    # Collect array memory parameters
    array_memory_params_map = VerilogDumper.collect_array_memory_params_map(sys)
    
    # Collect exposed expressions
    module_expr_map = {}
    for module in sys.modules:
        exposed_map = {}
        for expr in module.collect_expressions():
            for node, kind in sys.exposed_nodes:
                if node == expr:
                    exposed_map[expr] = kind
        if exposed_map:
            module_expr_map[module] = exposed_map
    
    # Create Verilog dumper
    vd = VerilogDumper.new(
        sys,
        config,
        external_usage,
        topo_dict,
        array_memory_params_map,
        module_expr_map
    )
    
    # Generate Verilog code
    with open(verilog_file, "w") as fd:
        # Generate modules
        for module in sys.modules:
            verilog_code = vd.visit_module(module)
            fd.write(verilog_code)
        
        # Generate runtime
        vd.dump_runtime(fd, config["sim_threshold"])
    
    # Copy support files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for file_name in ["runtime.sv"]:
        src_path = os.path.join(current_dir, file_name)
        if os.path.exists(src_path):
            dst_path = os.path.join(verilog_dir, file_name)
            shutil.copy(src_path, dst_path)
    
    return verilog_dir