"""Expression visitor for the Verilog backend."""

from typing import Dict, Optional, Tuple

from ...utils import identifierize
from ...builder import SysBuilder
from ...ir.expr import (
    Expr, BinaryOp, UnaryOp, FIFOPop, Log, ArrayRead, ArrayWrite,
    FIFOPush, PureIntrinsic, AsyncCall, Slice, Concat, Cast, Select,
    Select1Hot, Intrinsic, Bind
)
from ...ir import const

from .utils import declare_logic
from .gather import Gather

def visit_expr_impl(vd, expr: Expr) -> Optional[str]:
    """Implement the visit_expr method for the VerilogDumper."""
    # Handle expressions that are externally used
    decl, expose = None, ""
    if expr.is_valued() and expr.opcode != Opcode.BIND:
        id_ = identifierize(str(expr))
        expose_str = ""
        if vd.external_usage.is_externally_used(expr):
            pred = vd.get_pred() or "1"
            expose_str = f"  assign expose_{id_} = {id_};\n  assign expose_{id_}_valid = executed && {pred};\n"
        decl = (id_, expr.dtype)
        expose = expose_str
    
    # Handle exposed expressions
    exposed_map = dict(vd.sys.exposed_nodes)
    expose_out_str = ""
    if expr in exposed_map:
        exposed_kind = exposed_map[expr]
        id_ = identifierize(str(expr))
        if exposed_kind == "Output" or exposed_kind == "Inout":
            expose_out_str = f"  assign {id_}_exposed_o = {id_};\n"
    
    is_pop = None
    body = ""
    
    # Handle different expression types
    if isinstance(expr, Binary):
        binop = expr.opcode
        dtype = expr.dtype
        a = dump_arith_ref(vd.sys, expr.lhs)
        
        op_str = str(binop)
        if binop == Opcode.SHR:
            op_str = ">>>" if dtype.is_signed else ">>"
            
        b = dump_arith_ref(vd.sys, expr.rhs)
        body = f"{a} {op_str} {b}"
    
    elif isinstance(expr, Unary):
        uop = expr.opcode
        op_str = "~" if uop == Opcode.FLIP else "-"
        x = dump_arith_ref(vd.sys, expr.x)
        body = f"{op_str}{x}"
    
    elif isinstance(expr, Compare):
        a = dump_arith_ref(vd.sys, expr.lhs)
        op_str = str(expr.opcode)
        b = dump_arith_ref(vd.sys, expr.rhs)
        body = f"{a} {op_str} {b}"
    
    elif isinstance(expr, FIFOPop):
        fifo = expr.fifo
        display = utils.DisplayInstance.from_fifo(fifo, False)
        pred = vd.get_pred() or ""
        pred_str = f" && {pred}" if pred else ""
        is_pop = f"  assign {display.field('pop_ready')} = executed{pred_str};"
        body = display.field("pop_data")
    
    elif isinstance(expr, Log):
        res = []
        if vd.before_wait_until:
            condition = "1'b1"
        else:
            condition = "executed"
        
        pred = vd.get_pred() or ""
        if pred:
            condition += f" && {pred}"
        
        res.append(f"  always_ff @(posedge clk) if ({condition})")
        
        args = [arg for arg in expr.operands]
        format_str = utils.parse_format_string(args, expr.sys)
        
        res.append(f"$display(\"%t\\t[{vd.current_module}]\\t\\t")
        res.append(format_str)
        res.append("\",")
        res.append("`ifndef SYNTHESIS")
        res.append("  $time - 200")
        res.append("`else")
        res.append("  $time")
        res.append("`endif")
        res.append(", ")
        
        for arg in expr.operands[1:]:
            res.append(f"{dump_ref(vd.sys, arg, False)}, ")
        
        # Remove last comma and space
        res[-1] = res[-1][:-2]
        res.append(");")
        res.append("")
        
        body = "\n".join(res)
    
    elif isinstance(expr, ArrayRead):
        array_ref = expr.array
        array_idx = expr.idx
        size = array_ref.size
        bits = array_ref.scalar_ty.bits
        name = f"array_{identifierize(array_ref.name)}_q"

        if array_idx.kind == NodeKind.INT_IMM:
            imm = array_idx.value
            body = f"{name}[{bits * (imm + 1) - 1}:{imm * bits}]"
        elif array_idx.kind == NodeKind.EXPR:
            res = "'x"
            idx = dump_ref(vd.sys, array_idx, True)
            for i in range(size):
                slice_ = f"{name}[{((i + 1) * bits) - 1}:{i * bits}]"
                res = f"{i} == {idx} ? {slice_} : ({res})"
            body = res
        else:
            raise ValueError(f"Unexpected reference type: {array_idx.kind}")
    
    elif isinstance(expr, ArrayWrite):
        array = expr.array
        array_idx = expr.idx
        array_name = identifierize(array.name)
        pred = vd.get_pred() or ""
        idx = dump_ref(vd.sys, array_idx, True)
        idx_bits = array_idx.get_dtype(vd.sys).bits
        value = dump_ref(vd.sys, expr.val, True)
        value_bits = expr.val.get_dtype(vd.sys).bits
        
        if array_name in vd.array_stores:
            g_idx, g_value = vd.array_stores[array_name]
            g_idx.push(pred, idx, idx_bits)
            g_value.push(pred, value, value_bits)
        else:
            vd.array_stores[array_name] = (
                Gather(pred, idx, idx_bits),
                Gather(pred, value, value_bits)
            )
        
        body = ""
    
    elif isinstance(expr, FIFOPush):
        fifo = expr.fifo
        fifo_name = f"{identifierize(fifo.module.name)}_{identifierize(fifo.name)}"
        pred = vd.get_pred() or ""
        value = dump_ref(vd.sys, expr.val, False)
        
        if fifo_name in vd.fifo_pushes:
            vd.fifo_pushes[fifo_name].push(pred, value, fifo.scalar_ty.bits)
        else:
            vd.fifo_pushes[fifo_name] = Gather(pred, value, fifo.scalar_ty.bits)
        
        body = ""
    
    elif isinstance(expr, PureIntrinsic):
        intrinsic = expr.opcode
        if intrinsic in (Opcode.FIFO_VALID, Opcode.FIFO_PEEK):
            fifo = expr.operands[0].value
            fifo_name = identifierize(fifo.name)
            
            if intrinsic == Opcode.FIFO_VALID:
                body = f"fifo_{fifo_name}_pop_valid"
            elif intrinsic == Opcode.FIFO_PEEK:
                body = f"fifo_{fifo_name}_pop_data"
        
        elif intrinsic == Opcode.VALUE_VALID:
            value = expr.operands[0].value
            value_expr = value
            
            if value_expr.parent.module != expr.parent.module:
                body = f"{identifierize(str(value_expr))}_valid"
            else:
                pred = vd.get_pred() or ""
                pred_str = f" && {pred}" if pred else ""
                body = f"(executed{pred_str})"
        else:
            # TODO: Handle other intrinsics
            body = ""
    
    elif isinstance(expr, AsyncCall):
        bind = expr.bind
        callee = identifierize(bind.callee.name)
        pred = vd.get_pred() or ""
        
        if callee in vd.triggers:
            vd.triggers[callee].push(pred, "", 8)
        else:
            vd.triggers[callee] = Gather(pred, "", 8)
        
        body = ""
    
    elif isinstance(expr, Slice):
        a = dump_ref(vd.sys, expr.x, False)
        l = dump_ref(vd.sys, expr.l, False)
        r = dump_ref(vd.sys, expr.r, False)
        body = f"{a}[{r}:{l}]"
    
    elif isinstance(expr, Concat):
        a = dump_ref(vd.sys, expr.msb, True)
        b = dump_ref(vd.sys, expr.lsb, True)
        body = f"{{{a}, {b}}}"
    
    elif isinstance(expr, Cast):
        dbits = expr.dtype.bits
        a = dump_ref(vd.sys, expr.x, False)
        src_dtype = expr.src_type
        pad = dbits - src_dtype.bits
        
        if expr.cast_kind == Opcode.BITCAST:
            body = a
        elif expr.cast_kind == Opcode.ZEXT:
            body = f"{{{pad}'b0, {a}}}"
        elif expr.cast_kind == Opcode.SEXT:
            dest_dtype = expr.dtype
            if (src_dtype.is_int() and src_dtype.is_signed and 
                dest_dtype.is_int() and dest_dtype.is_signed and 
                dest_dtype.bits > src_dtype.bits):
                # perform sext
                body = f"{{{pad}'{{{a}[{src_dtype.bits - 1}]}}, {a}}}"
            else:
                body = f"{{{pad}'b0, {a}}}"
    
    elif isinstance(expr, Select):
        cond = dump_ref(vd.sys, expr.cond, True)
        true_value = dump_ref(vd.sys, expr.true_value, True)
        false_value = dump_ref(vd.sys, expr.false_value, True)
        body = f"{cond} ? {true_value} : {false_value}"
    
    elif isinstance(expr, Bind):
        # handled in AsyncCall
        body = ""
    
    elif isinstance(expr, Select1Hot):
        dbits = expr.dtype.bits
        cond = dump_ref(vd.sys, expr.cond, False)
        
        terms = []
        for i, elem in enumerate(expr.values):
            value = dump_ref(vd.sys, elem, False)
            terms.append(f"({{{dbits}{{{{cond}}[{i}] == 1'b1}}}} & {value})")
        
        body = " | ".join(terms)
    
    elif isinstance(expr, Intrinsic):
        intrinsic = expr.opcode
        
        if intrinsic == Opcode.FINISH:
            pred = vd.get_pred() or "1"
            body = f"\n`ifndef SYNTHESIS\n  always_ff @(posedge clk) if (executed && {pred}) $finish();\n`endif\n"
        
        elif intrinsic == Opcode.ASSERT:
            pred = vd.get_pred() or "1"
            cond = dump_ref(vd.sys, expr.operands[0].value, False)
            body = f"  always_ff @(posedge clk) if (executed && {pred}) assert({cond});\n"
        
        else:
            raise ValueError(f"Unknown block intrinsic: {intrinsic}")
    
    else:
        raise ValueError(f"Unhandled expression type: {type(expr).__name__}")
    
    # Process the body based on exposed values
    if decl:
        id_, _ = decl
        if expr in exposed_map:
            kind = exposed_map[expr]
            if kind == "Inout" or kind == "Input":
                body = f"{id_}_exposed_i_valid ? {id_}_exposed_i :({body}) "
    
    # Build the final result
    result = ""
    if decl:
        id_, ty = decl
        result = f"{declare_logic(ty, id_)}  assign {id_} = {body};\n{expose}\n{expose_out_str}\n"
    else:
        result = body
    
    # Add FIFO pop logic if needed
    if is_pop:
        result += f"{is_pop}\n"
    
    return result

def dump_ref(sys: SysBuilder, node, with_imm_width: bool) -> str:
    """Dump a reference to a node."""
    return node_dump_ref(sys, node, [], with_imm_width, False)

def dump_arith_ref(sys: SysBuilder, node) -> str:
    """Dump an arithmetic reference."""
    return node_dump_ref(sys, node, [], True, True)

def node_dump_ref(
    sys: SysBuilder,
    node,
    _node_kinds,
    immwidth: bool,
    signed: bool
) -> Optional[str]:
    """Dump a reference to a node with options."""
    if node.kind == NodeKind.ARRAY:
        array = node
        return identifierize(array.name)
    
    elif node.kind == NodeKind.FIFO:
        fifo = node
        return identifierize(fifo.name)
    
    elif node.kind == NodeKind.INT_IMM:
        int_imm = node
        dbits = int_imm.dtype.bits
        value = int_imm.value
        
        if immwidth:
            return f"{dbits}'d{value}"
        return str(value)
    
    elif node.kind == NodeKind.STR_IMM:
        str_imm = node
        value = str_imm.value
        return f'"{value}"'
    
    elif node.kind == NodeKind.EXPR:
        dtype = node.get_dtype(sys)
        raw = identifierize(str(node))
        
        if isinstance(dtype, Int) and signed:
            return f"$signed({raw})"
        return raw
    
    else:
        raise ValueError(f"Unknown node of kind {node.kind}")