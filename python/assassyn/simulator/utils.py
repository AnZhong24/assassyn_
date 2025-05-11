"""Utility functions for simulator generation."""

from assassyn.dtype import DType


def namify(name: str) -> str:
    """Convert a name to a valid identifier.

    This matches the Rust function in src/backend/simulator/utils.rs
    """
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in name)


def camelize(name: str) -> str:
    """Convert a name to camelCase.

    This matches the Rust function in src/backend/simulator/utils.rs
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


def dtype_to_rust_type(dtype: DType) -> str:
    """Convert an Assassyn data type to a Rust type.

    This matches the Rust function in src/backend/simulator/utils.rs
    """
    if dtype.is_int() or dtype.is_raw():
        prefix = "u" if not dtype.is_signed() or dtype.is_raw() else "i"
        bits = dtype.get_bits()

        if 8 <= bits <= 64:
            # Round up to next power of 2
            bits = 1 << (bits - 1).bit_length()
            return f"{prefix}{bits}"
        elif bits == 1:
            return "bool"
        elif bits < 8:
            return f"{prefix}8"
        elif bits > 64:
            if not dtype.is_signed() or dtype.is_raw():
                return "BigUint"
            else:
                return "BigInt"
        else:
            raise ValueError(f"Unsupported data type: {dtype}")

    if dtype.is_module():
        return "Box<EventKind>"
    elif dtype.is_array():
        elem_ty = dtype_to_rust_type(dtype.element_type())
        size = dtype.get_size()
        return f"[{elem_ty}; {size}]"
    else:
        raise ValueError(f"Unsupported data type: {dtype}")


def int_imm_dumper_impl(ty: DType, value: int) -> str:
    """Generate Rust code for integer immediate values.

    This matches the Rust function in src/backend/simulator/elaborate.rs
    """
    if ty.get_bits() == 1:
        return "true" if value != 0 else "false"

    if ty.get_bits() <= 64:
        return f"{value}{dtype_to_rust_type(ty)}"
    else:
        scalar_ty = "i64" if ty.is_signed() else "u64"
        return f"ValueCastTo::<{dtype_to_rust_type(ty)}>::cast(&({value} as {scalar_ty}))"


def fifo_name(fifo):
    """Generate a name for a FIFO.

    This matches the Rust macro in src/backend/simulator/elaborate.rs
    """
    module = fifo.get_parent().as_module()
    return f"{namify(module.get_name())}_{namify(fifo.get_name())}"
