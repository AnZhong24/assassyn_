"""Utility functions for Verilog generation."""

from ...ir.dtype import DType, Void, ArrayType, Record, Bits, Int, UInt


def namify(name: str) -> str:
    """Convert a name to a valid Verilog identifier.
    
    This matches the Rust function in src/backend/verilog/utils.rs
    """
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in name)


def camelize(name: str) -> str:
    """Convert a name to camelCase.
    
    This matches the Rust function in src/backend/verilog/utils.rs
    """
    result = ""
    capitalize = True
    for c in name:
        if c == '_':
            capitalize = True
        elif capitalize:
            result += c.upper()
            capitalize = False
        else:
            result += c
    return result


def dtype_to_verilog_type(dtype: DType) -> str:
    """Convert an Assassyn data type to a Verilog type.
    
    This matches the Rust function in src/backend/verilog/utils.rs
    """
    
    if isinstance(dtype, Record):
        dtype = Bits(dtype.bits)
    
    if isinstance(dtype, (Int, UInt, Bits)):
        bits = dtype.bits
        return f"[{bits-1}:0]"
    
    if isinstance(dtype, Void):
        return ""
    
    if isinstance(dtype, ArrayType):
        elem_ty = dtype_to_verilog_type(dtype.scalar_ty)
        size = dtype.size
        return f"{elem_ty} [{size-1}:0]"
    
    raise ValueError(f"Unsupported data type: {dtype}")


def int_imm_dumper_impl(ty: DType, value: int) -> str:
    """Generate Verilog code for integer immediate values.
    
    This matches the Rust function in src/backend/verilog/utils.rs
    """
    if ty.bits == 1:
        return "1'b1" if value != 0 else "1'b0"
    
    bits = ty.bits
    return f"{bits}'d{value}"


def fifo_name(fifo) -> str:
    """Generate a name for a FIFO.
    
    This matches the Rust function in src/backend/verilog/utils.rs
    """
    module = fifo.module
    return f"{namify(module.name)}_{namify(fifo.name)}"


class DisplayInstance:
    """Format strings for Verilog instance generation."""
    
    @staticmethod
    def module(module_name, instance_name, params, ports):
        """Format a module instance with improved readability."""
        # Format parameters with newlines if any exist
        params_str = ""
        if params:
            param_items = [f".{k}({v})" for k, v in params.items()]
            items = ',\n    '.join(param_items)
            params_str = f"""#(
    {items}
  )"""
        
        # Format ports with newlines for better readability
        port_items = [f".{k}({v})" for k, v in ports.items()]
        ports_str = ',\n    '.join(port_items)
        
        # The complete module instantiation with proper indentation
        return f"{module_name} {params_str}\n  {instance_name} (\n    {ports_str}\n  );"
