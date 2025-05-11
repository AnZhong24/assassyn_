'''The module to generate the assassyn IR builder for the given system'''

# Import all the necessary items from the implementation
from .impl import (
    # Main function
    codegen,

    # Helper functions
    generate_dtype,
    const_int_wrapper,
    generate_init_value,
    generate_port,
    opcode_to_ib,

    # Classes
    CodeGen,
    EmitBinds,

    # Constants
    CG_OPCODE,
    CG_MIDFIX,
    CG_SIMULATOR
)

# Export all the necessary items
__all__ = [
    'codegen',
    'generate_dtype',
    'const_int_wrapper',
    'generate_init_value',
    'generate_port',
    'opcode_to_ib',
    'CodeGen',
    'EmitBinds',
    'CG_OPCODE',
    'CG_MIDFIX',
    'CG_SIMULATOR'
]
