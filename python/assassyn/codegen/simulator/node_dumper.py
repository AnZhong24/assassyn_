"""Node reference dumper for simulator code generation."""

from ..visitor import Visitor
from .utils import namify, int_imm_dumper_impl, fifo_name


def dump_rval_ref(module_ctx, sys, node):
    """Dump a reference to a node as an rvalue.

    This matches the Rust function in src/backend/simulator/elaborate.rs
    """
    return NodeRefDumper(module_ctx).dispatch(sys, node, [])


class NodeRefDumper(Visitor):
    """Visitor for dumping node references.

    This matches the Rust class in src/backend/simulator/elaborate.rs
    """

    def __init__(self, module_ctx):
        """Initialize the node reference dumper."""
        self.module_ctx = module_ctx

    def dispatch(self, sys, node, _):
        """Dispatch to appropriate handler based on node kind."""
        node_kind = node.get_kind()

        if node_kind == "Array":
            array = node.as_array()
            return namify(array.get_name())

        elif node_kind == "FIFO":
            fifo = node.as_fifo()
            return fifo_name(fifo)

        elif node_kind == "IntImm":
            int_imm = node.as_int_imm()
            return int_imm_dumper_impl(int_imm.dtype(), int_imm.get_value())

        elif node_kind == "StrImm":
            str_imm = node.as_str_imm()
            value = str_imm.get_value()
            # Using Python's repr to get quote-escaped string
            return repr(value)

        elif node_kind == "Module":
            module = node.as_module()
            return namify(module.get_name())

        elif node_kind == "Expr":
            expr = node.as_expr()

            # Figure out the ID format based on context
            parent_block = expr.get_parent().as_block()
            if self.module_ctx != parent_block.get_module():
                # Expression from another module
                raw = namify(expr.get_name())
                field_id = f"{raw}_value"
                panic_log = f"Value {raw} invalid!"
                return f"""if let Some(x) = &sim.{field_id} {{
                            x
                          }} else {{
                            panic!("{panic_log}");
                          }}.clone()"""
            elif expr.dtype().get_bits() <= 64:
                # Simple value
                return namify(expr.get_name())
            else:
                # Large value needs cloning
                return f"{namify(expr.get_name())}.clone()"

            # Handle FIFO peek special case
            if expr.as_sub_expr_type() == PureIntrinsic and expr.get_subcode() == "FIFOPeek":
                id = namify(expr.get_name())
                id += ".clone().unwrap()"
                return id

            return namify(expr.get_name())

        else:
            # Default case
            return namify(node.to_string())


def externally_used_combinational(expr):
    """Check if an expression is used outside its module.

    This matches the Rust function in src/backend/simulator/elaborate.rs
    """
    # Push is NOT a combinational operation
    if expr.get_opcode() == "FIFOPush":
        return False

    this_module = expr.get_block().get_module()

    # Check if any user is in a different module
    for user in expr.users():
        operand = user.as_operand()
        if operand:
            parent = operand.get_parent()
            if parent:
                user_expr = parent.as_expr()
                if user_expr:
                    if user_expr.get_block().get_module() != this_module:
                        return True

    return False
