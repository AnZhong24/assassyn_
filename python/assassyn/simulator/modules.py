"""Module elaboration for simulator code generation."""

from ..visitor import Visitor
from ..expr import (
        Expr,
        BinaryOp,
        UnaryOp,
        ArrayRead,
        ArrayWrite,
        Cast,
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
from .utils import namify, dtype_to_rust_type, fifo_name
from .node_dumper import dump_rval_ref, externally_used_combinational


class ElaborateModule(Visitor):
    """Visitor for elaborating modules.

    This matches the Rust class in src/backend/simulator/elaborate.rs
    """

    def __init__(self, sys):
        """Initialize the module elaborator."""
        self.sys = sys
        self.indent = 0
        self.module_name = ""
        self.module_ctx = None

    def visit_module(self, module):
        """Visit a module and generate its implementation."""
        self.module_name = module.get_name()
        self.module_ctx = module

        # Create function header
        result = [f"\n// Elaborating module {namify(module.get_name())}"]
        result.append(f"pub fn {namify(module.get_name())}(sim: &mut Simulator) -> bool {{")

        # Increase indentation for function body
        self.indent += 2

        # Visit the module body
        body = self.visit_block(module.get_body())
        result.append(body)

        # Decrease indentation and add function closing
        self.indent -= 2
        result.append(" true }")

        return "\n".join(result)

    def visit_expr(self, expr):
        """Visit an expression and generate its implementation."""
        # Determine if the expression produces a value and if it needs exposure
        id_and_exposure = None
        if expr.get_opcode().is_valued():
            need_exposure = externally_used_combinational(expr)
            id_expr = namify(expr.to_string())
            id_and_exposure = (id_expr, need_exposure)

        # Handle different expression types
        open_scope = False
        code = []

        if isinstance(expr, BinaryOp):
            binary = expr.as_sub_binary()
            binop = BinaryOp.OPERATORS[expr.opcode]
            rust_ty = dtype_to_rust_type(ty)
            lhs = dump_rval_ref(self.module_ctx, self.sys, binary.a())
            rhs = dump_rval_ref(self.module_ctx, self.sys, binary.b())
            # Special handling for shift operations
            if binop in [Binary.SHL, Binary.SHR]:
                rhs = f"ValueCastTo::<u64>::cast(&{rhs})"
            else:
                rhs = f"ValueCastTo::<{rust_ty}>::cast(&{rhs})"

            code.append(f"{lhs} {binop} {rhs}")

        elif isinstance(expr, UnaryOp):
            unary = expr.opcode
            operand = dump_rval_ref(self.module_ctx, self.sys, unary.x())
            code.append(f"{unary.get_opcode()}{operand}")

        elif isinstance(expr, ArrayRead):
            array = expr.arr
            idx = expr.idx
            array_name = namify(array.get_name())
            idx_val = dump_rval_ref(self.module_ctx, self.sys, idx)
            code.append(f"sim.{array_name}.payload[{idx_val} as usize].clone()")

        elif isinstance(expr, ArrayWrite):
            array = store.arr
            idx = store.idx
            value = store.value

            array_name = namify(array.get_name())
            idx_val = dump_rval_ref(self.module_ctx, self.sys, idx)
            value_val = dump_rval_ref(self.module_ctx, self.sys, value)
            module_writer = self.module_name

            code.append(f"""{{
              let stamp = sim.stamp - sim.stamp % 100 + 50;
              sim.{array_name}.write.push(
                ArrayWrite::new(stamp, {idx_val} as usize, {value_val}.clone(), "{module_writer}"));
            }}""")

        elif isinstance(expr, AsyncCall):

            bind = expr.bind

            event_q = f"{namify(bind.callee.get_name())}_event"

            code.append(f"""{{
              let stamp = sim.stamp - sim.stamp % 100 + 100;
              sim.{event_q}.push_back(stamp)
            }}""")

        elif isinstance(expr, FIFOPop):
            fifo = expr.fifo
            fifo_id = fifo_name(fifo)
            module_name = self.module_name

            code.append(f"""{{
              let stamp = sim.stamp - sim.stamp % 100 + 50;
              sim.{fifo_id}.pop.push(FIFOPop::new(stamp, "{module_name}"));
              sim.{fifo_id}.payload.front().unwrap().clone()
            }}""")

        elif isinstance(expr, PureIntrinsic):

            intrinsic = expr.opcode

            if intrinsic == PureIntrinsic.FIFO_PEEK:
                port_self = dump_rval_ref(self.module_ctx, self.sys, call.get_operand_value(0))
                code.append(f"sim.{port_self}.front().cloned()")

            elif intrinsic == PureIntrinsic.FIFO_VALID:
                port_self = dump_rval_ref(self.module_ctx, self.sys, call.get_operand_value(0))
                code.append(f"!sim.{port_self}.is_empty()")

            elif intrinsic == PureIntrinsic.VALUE_VALID:
                value = call.get_operand_value(0)
                value_expr = value.as_expr()
                code.append(f"sim.{namify(value_expr.get_name())}_value.is_some()")

            elif intrinsic == PureIntrinsic.MODULE_TRIGGERED:
                port_self = dump_rval_ref(self.module_ctx, self.sys, call.get_operand_value(0))
                code.append(f"sim.{port_self}_triggered")

        elif isinstance(expr, FIFOPush):
            push = expr.as_sub_fifo_push()
            fifo = push.fifo()
            fifo_id = fifo_name(fifo)
            value = dump_rval_ref(self.module_ctx, self.sys, push.value())
            module_writer = self.module_name

            code.append(f"""{{
              let stamp = sim.stamp;
              sim.{fifo_id}.push.push(
                FIFOPush::new(stamp + 50, {value}.clone(), "{module_writer}"));
            }}""")

        elif expr.get_opcode() == "Log":
            mn = self.module_name
            result = [f'print!("@line:{{:<5}} {{:<10}}: [{mn}]\\t", line!(), cyclize(sim.stamp));']
            result.append("println!(")

            for elem in expr.operand_iter():
                dump = dump_rval_ref(self.module_ctx, self.sys, elem.get_value())

                # Special handling for boolean display
                if elem.get_value().get_dtype() and elem.get_value().get_dtype().get_bits() == 1:
                    dump = f"if {dump} {{ 1 }} else {{ 0 }}"

                result.append(f"{dump}, ")

            result.append(")")
            code.append("".join(result))

        elif expr.get_opcode() == "Slice":
            slice_expr = expr.as_sub_slice()
            a = dump_rval_ref(self.module_ctx, self.sys, slice_expr.x())
            l = slice_expr.l()
            r = slice_expr.r()
            dtype = slice_expr.dtype()
            mask_bits = "1" * (r - l + 1)

            if l < 64 and r < 64:
                result_a = f'''let a = ValueCastTo::<u64>::cast(&{a});
                               let mask = u64::from_str_radix("{mask_bits}", 2).unwrap();'''
            else:
                result_a = f'''let a = ValueCastTo::<BigUint>::cast(&{a});
                               let mask = BigUint::parse_bytes("{mask_bits}".as_bytes(), 2).unwrap();'''

            code.append(f"""{{
                {result_a}
                let res = (a >> {l}) & mask;
                ValueCastTo::<{dtype_to_rust_type(dtype)}>::cast(&res)
            }}""")

        elif expr.get_opcode() == "Concat":
            concat = expr.as_sub_concat()
            dtype = expr.dtype()
            a = dump_rval_ref(self.module_ctx, self.sys, concat.msb())
            b = dump_rval_ref(self.module_ctx, self.sys, concat.lsb())
            b_bits = concat.lsb().get_dtype().get_bits()

            code.append(f"""{{
                let a = ValueCastTo::<BigUint>::cast(&{a});
                let b = ValueCastTo::<BigUint>::cast(&{b});
                let c = (a << {b_bits}) | b;
                ValueCastTo::<{dtype_to_rust_type(dtype)}>::cast(&c)
            }}""")

        elif expr.get_opcode() == "Select":
            select = expr.as_sub_select()
            cond = dump_rval_ref(self.module_ctx, self.sys, select.cond())
            true_value = dump_rval_ref(self.module_ctx, self.sys, select.true_value())
            false_value = dump_rval_ref(self.module_ctx, self.sys, select.false_value())

            code.append(f"if {cond} {{ {true_value} }} else {{ {false_value} }}")

        elif expr.get_opcode() == "Select1Hot":
            select1hot = expr.as_sub_select1hot()
            cond = dump_rval_ref(self.module_ctx, self.sys, select1hot.cond())

            result = [f"{{ let cond = {cond}; assert!(cond.count_ones() == 1, \"Select1Hot: condition is not 1-hot\");"]

            for i, value in enumerate(select1hot.value_iter()):
                if i != 0:
                    result.append(" else ")

                result.append(f"if cond >> {i} & 1 != 0 {{ {dump_rval_ref(self.module_ctx, self.sys, value)} }}")

            result.append(" else { unreachable!() } }")
            code.append("".join(result))

        elif expr.get_opcode() == "Cast":
            cast = expr.as_sub_cast()
            dest_dtype = cast.dest_type()
            a = dump_rval_ref(self.module_ctx, self.sys, cast.x())

            if cast.get_subcode() in [Cast.ZEXT, Cast.BITCAST, Cast.SEXT]:
                code.append(f"ValueCastTo::<{dtype_to_rust_type(dest_dtype)}>::cast(&{a})")

        elif expr.get_opcode() == "Bind":
            code.append("()")

        # elif expr.get_opcode() == "BlockIntrinsic":
        #     bi = expr.as_sub_block_intrinsic()
        #     intrinsic = bi.get_subcode()

        #     value = ""
        #     if bi.value():
        #         value = dump_rval_ref(self.module_ctx, self.sys, bi.value())

        #     if intrinsic == BlockIntrinsic.VALUE:
        #         code.append(value)

        #     elif intrinsic == BlockIntrinsic.CYCLED:
        #         open_scope = True
        #         code.append(f"if sim.stamp / 100 == ({value} as usize) {{")

        #     elif intrinsic == BlockIntrinsic.WAIT_UNTIL:
        #         code.append(f"if !{value} {{ return false; }}")

        #     elif intrinsic == BlockIntrinsic.CONDITION:
        #         open_scope = True
        #         code.append(f"if {value} {{")

        #     elif intrinsic == BlockIntrinsic.FINISH:
        #         code.append("std::process::exit(0);")

        #     elif intrinsic == BlockIntrinsic.ASSERT:
        #         code.append(f"assert!({value});")

        #     elif intrinsic == BlockIntrinsic.BARRIER:
        #         code.append(f"/* Barrier: {value} */")

        # Format the result with proper indentation and variable assignment
        indent_str = " " * self.indent
        result = ""

        if id_and_exposure:
            id_expr, need_exposure = id_and_exposure
            code_block = "\n".join(code)

            valid_update = ""
            if need_exposure:
                valid_update = f"sim.{id_expr}_value = Some({id_expr}.clone());"

            result = f"{indent_str}let {id_expr} = {{ {code_block} }}; {valid_update}\n"
        else:
            for line in code:
                result += f"{indent_str}{line};\n"

        # Adjust indentation if we opened a scope
        if open_scope:
            self.indent += 2

        return result

    def visit_int_imm(self, int_imm):
        """Visit an integer immediate value."""
        return f"ValueCastTo::<{dtype_to_rust_type(int_imm.dtype())}>::cast(&{int_imm.get_value()})"

    def visit_block(self, block):
        """Visit a block and generate its implementation."""
        result = []

        # Save current indentation
        restore_indent = self.indent

        # Visit each element in the block
        for elem in block.body_iter():
            if elem.get_kind() == "Expr":
                expr = elem.as_expr()
                result.append(self.visit_expr(expr))
            elif elem.get_kind() == "Block":
                sub_block = elem.as_block()
                result.append(self.visit_block(sub_block))
            else:
                raise ValueError(f"Unexpected reference type: {elem.get_kind()}")

        # Restore indentation and close scope if needed
        if restore_indent != self.indent:
            self.indent -= 2
            result.append(f"{' ' * self.indent}}}\n")

        # Handle block value if present
        if block.get_value():
            return f"{{ {''.join(result)} }}"

        return "".join(result)
