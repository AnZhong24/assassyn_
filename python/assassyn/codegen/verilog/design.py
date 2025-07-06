"""Verilog design generation and code dumping."""

from typing import List, Dict, Tuple
from string import Formatter

from ...analysis import expr_externally_used
from ...ir.module import Module, Downstream, Port
from ...builder import SysBuilder
from ...ir.visitor import Visitor
from ...ir.block import Block, CondBlock,CycledBlock
from ...ir.const import Const
from ...ir.array import Array
from ...ir.dtype import Int, UInt, Bits, DType
from ...utils import namify, unwrap_operand
from ...ir.expr import (
    Expr,
    BinaryOp,
    UnaryOp,
    FIFOPop,
    Log,
    ArrayRead,
    ArrayWrite,
    FIFOPush,
    PureIntrinsic,
    AsyncCall,
    Slice,
    Concat,
    Cast,
    Select,
    Bind,
    Select1Hot,
    Intrinsic
)

def dump_rval(node, with_namespace: bool) -> str:  # pylint: disable=too-many-return-statements
    """Dump a reference to a node with options."""

    node = unwrap_operand(node)

    if isinstance(node, Module):
        return namify(node.name)
    if isinstance(node, Array):
        array = node
        return namify(array.name)
    if isinstance(node, Port):
        return namify(node.name)
    if isinstance(node, FIFOPop):
        if not with_namespace:
            return f'self.{namify(node.fifo.name)}'
        return namify(node.fifo.module.name) + "_" + namify(node.fifo.name)
    if isinstance(node, Const):
        int_imm = node
        value = int_imm.value
        ty = dump_type(int_imm.dtype)
        return f"{ty}({value})"
    if isinstance(node, str):
        value = node
        return f'"{value}"'
    if isinstance(node, Expr):
        raw = namify(node.as_operand())
        if with_namespace:
            owner_module_name = namify(node.parent.module.name)
            return f"{owner_module_name}_{raw}"
  

        return raw
    raise ValueError(f"Unknown node of kind {type(node).__name__}")

def dump_type(ty: DType) -> str:
    """Dump a type to a string."""
    if isinstance(ty, Int):
        return f"SInt({ty.bits})"
    if isinstance(ty, UInt):
        return f"UInt({ty.bits})"
    if isinstance(ty, Bits):
        return f"Bits({ty.bits})"
    raise ValueError(f"Unknown type: {type(ty)}")

def dump_type_cast(ty: DType) -> str:
    """Dump a type to a string."""
    if isinstance(ty, Int):
        return "as_sint()"
    if isinstance(ty, UInt):
        return "as_uint()"
    if isinstance(ty, Bits):
        return "as_bits()"
    raise ValueError(f"Unknown type: {type(ty)}")

class CIRCTDumper(Visitor):  # pylint: disable=too-many-instance-attributes
    """Dumps IR to CIRCT-compatible Verilog code."""

    wait_until: bool
    indent: int
    code: List[str]
    cond_stack: List[str]
    _exposes: Dict[Expr, List[Tuple[Expr, str]]]
    logs: List[str]
    connections: List[Tuple[Module, str, str]]
    current_module: Module
    sys: SysBuilder
    async_callees: Dict[Module, List[Module]]

    def __init__(self):
        super().__init__()
        self.wait_until = None
        self.indent = 0
        self.code = []
        self._exposes = {}
        self.cond_stack = []
        self.logs = []
        self.connections = []
        self.current_module = None
        self.sys = None
        self.async_callees = {}
        self.exposed_ports_to_add = []

    def get_pred(self) -> str:
        """Get the current predicate for conditional execution."""
        if not self.cond_stack:
            return "Bits(1)(1)" 
        return " & ".join([s for s, _ in self.cond_stack])

    def append_code(self, code: str):
        """Append code with proper indentation."""
        if code.strip() == '':
            self.code.append('')
        else:
            self.code.append(self.indent * ' ' + code)

    def expose(self, kind: str, expr: Expr):
        ''' Expose an expression out of the module.'''
        key = None
        if kind == 'expr':
            key = expr
            
        elif kind == 'array':
            assert isinstance(expr, (ArrayRead, ArrayWrite))
            key = expr.array
        elif kind == 'fifo':
            assert isinstance(expr, FIFOPush)
            key = expr.fifo
        elif kind == 'fifo_pop':
            assert isinstance(expr, FIFOPop)
            key = expr.fifo
        elif kind == 'trigger':
            assert isinstance(expr, AsyncCall)
            key = expr.bind.callee
       
        assert key is not None
        if key not in self._exposes:
            self._exposes[key] = []
        self._exposes[key].append((expr, self.get_pred()))

    def visit_block(self, node: Block):
        is_cond = isinstance(node, CondBlock)
        is_cycle = isinstance(node, CycledBlock)

        if is_cond: 
            cond_str = dump_rval(node.cond, False)
            self.cond_stack.append((f"({cond_str})", node))
            def has_side_effect(block: Block) -> bool:
                for item in block.body:
                    if isinstance(item, Log):
                        return True
                    if isinstance(item, Block) and has_side_effect(item):
                        return True
                return False
            
            if has_side_effect(node):
                self.expose('expr', node.cond)
                 
 
        elif is_cycle: 
            self.cond_stack.append((f"(self.cycle_count == {node.cycle})", node))
            
        for i in node.body:
            if isinstance(i, Expr):
                self.visit_expr(i)
            elif isinstance(i, Block):
                self.visit_block(i)
            else:
                raise ValueError(f'Unknown node type: {type(node)}')
        
        if is_cond or is_cycle: 
            self.cond_stack.pop()


    def visit_expr(self, expr: Expr):  # pylint: disable=arguments-renamed,too-many-locals,too-many-branches,too-many-statements
        self.append_code(f'# {expr}')
        body = None
        rval = dump_rval(expr, False)
        
        if isinstance(expr, BinaryOp):
            binop = expr.opcode 
             
            a = dump_rval(expr.lhs, False)
            b = dump_rval(expr.rhs, False)
            dtype = expr.dtype
            if binop in [BinaryOp.SHL, BinaryOp.SHR] or 'SHR' in str(binop):
                
                a = f"{a}.as_bits()"
                b = f"{b}.as_bits()"
 
                op_class_name = None
                if binop == BinaryOp.SHL:
                    op_class_name = "comb.ShlOp"
                elif binop == BinaryOp.SHR:
                    if expr.lhs.dtype.is_signed(): 
                        op_class_name = "comb.ShrSOp"
                    else: 
                        op_class_name = "comb.ShrUOp"
                
                if op_class_name is None:
                    raise TypeError(f"Unhandled shift operation: {binop}")
                 
                body = f"{rval} = {op_class_name}({a}, {b}).as_bits()[0:{dtype.bits}].{dump_type_cast(dtype)}"
         
            elif binop == BinaryOp.MOD: 
                if expr.dtype.is_signed():
                    op_class_name = "comb.ModSOp"
                else:
                    op_class_name = "comb.ModUOp"
                
                body = f"{rval} = {op_class_name}({a}.as_bits(), {b}.as_bits()).as_bits()[0:{dtype.bits}].{dump_type_cast(dtype)}"
            else: 
                op_str = BinaryOp.OPERATORS[expr.opcode]
                
                op_body = f"(({a} {op_str} {b}).as_bits()[0:{dtype.bits}]).{dump_type_cast(dtype)}"
                body = f'{rval} = {op_body}'
        elif isinstance(expr, UnaryOp):
            uop = expr.opcode
            op_str = "~" if uop == UnaryOp.FLIP else "-"
            x = dump_rval(expr.x, False)
            if uop == UnaryOp.FLIP:
                x = f"({x}.as_bits())" 
            body = f"{op_str}{x}"
            body = f'{rval} = {body}'
        
        elif isinstance(expr, Log):
            formatter_str = expr.operands[0].value

            arg_print_snippets = []
            condition_snippets = []
            module_name = namify(self.current_module.name)
 
            for i in expr.operands[1:]:
                operand = unwrap_operand(i)
                if not isinstance(operand, Const):
                    self.expose('expr', operand)
                    exposed_name = dump_rval(operand, True)
                    valid_signal = f'dut.{module_name}.valid_{exposed_name}.value'
                    condition_snippets.append(valid_signal)
                    
                    base_value = f"dut.{module_name}.expose_{exposed_name}.value"
                    if isinstance(operand.dtype, Int):
                        bits = operand.dtype.bits
                        expose_signal = f"({base_value} - (1 << {bits}) if ({base_value} >> ({bits} - 1)) & 1 else int({base_value}))"
                    else: 
                        expose_signal = f"int({base_value})"
                    arg_print_snippets.append(expose_signal)
 
            f_string_content_parts = []
            arg_iterator = iter(arg_print_snippets)
 
            for literal_text, field_name, format_spec, conversion in Formatter().parse(formatter_str):
  
                if literal_text:
                    f_string_content_parts.append(literal_text)
 
                if field_name is not None: 
                    arg_code = next(arg_iterator)
                    new_placeholder =  f"{{{arg_code}"
                    if conversion:  # for !s, !r, !a
                        new_placeholder += f"!({conversion})"
                    if format_spec:  # for :b, :08x,  
                        new_placeholder += f":{format_spec}"
                    new_placeholder += "}"
                    f_string_content_parts.append(new_placeholder)

            f_string_content = "".join(f_string_content_parts)
            

            block_condition = self.get_pred() 
            block_condition=block_condition.replace('cycle_count','dut.global_cycle_count')
            final_conditions = []
             
            for cond_str, cond_obj in self.cond_stack: 
                if isinstance(cond_obj, CycledBlock): 
                    tb_cond_path = cond_str.replace("self.cycle_count", f"dut.global_cycle_count.value")
                    final_conditions.append(tb_cond_path)
                 
                elif isinstance(cond_obj, CondBlock):  
                    exposed_name = dump_rval(cond_obj.cond, False)
                    
                    tb_expose_path = f"(dut.{module_name}.expose_{exposed_name}.value)"
                    tb_valid_path = f"(dut.{module_name}.valid_{exposed_name}.value)"
                    
                    combined_cond = f"({tb_valid_path} & {tb_expose_path})"
                    final_conditions.append(combined_cond)
 
            if condition_snippets:
                final_conditions.append(" and ".join(condition_snippets))
             
            if_condition = " and ".join(final_conditions)

            self.logs.append(f'# {expr}')

            line_info = f"@line:{expr.loc.rsplit(':', 1)[-1]}"   
 
            module_info = f"[{namify(self.current_module.name)}]"
 
            cycle_info = f"Cycle @{{float(dut.global_cycle_count.value):.2f}}:"
 
            final_print_string = (
                 f'f"{line_info} {cycle_info} {module_info:<20} {f_string_content}"'
             )

            self.logs.append(f'#@ line {expr.loc}: {expr}')
            if if_condition:
                self.logs.append(f'if ( {if_condition} ):') 
                self.logs.append(f'    print({final_print_string})')
            else:
                self.logs.append(f'print({final_print_string})')
        
        elif isinstance(expr, ArrayRead):
            array_ref = expr.array
            array_idx = unwrap_operand(expr.idx)
            array_idx = (dump_rval(array_idx, False)
                         if not isinstance(array_idx, Const) else array_idx.value)
            array_name = dump_rval(array_ref, False)
            body = f'{rval} = self.{array_name}_payload[{array_idx}]'
            self.expose('array', expr)
        elif isinstance(expr, ArrayWrite):
            
            self.expose('array', expr)
        elif isinstance(expr, FIFOPush):
            self.expose('fifo', expr)
        elif isinstance(expr,FIFOPop):
            rval = namify(expr.as_operand())
            fifo_name = dump_rval( expr.fifo, False) 
            body = f'{rval} = self.{fifo_name}' 
            
            self.expose('fifo_pop', expr)
 
        elif isinstance(expr, PureIntrinsic):
            intrinsic = expr.opcode
            if intrinsic in [PureIntrinsic.FIFO_VALID, PureIntrinsic.FIFO_PEEK]:
                fifo = expr.args[0]
                fifo_name = dump_rval(fifo, False)
                if intrinsic == PureIntrinsic.FIFO_PEEK:
                    body = f'{rval} = self.{fifo_name}'
                    self.expose('expr', expr)
                elif intrinsic == PureIntrinsic.FIFO_VALID:
                    body = f'{rval} = self.{fifo_name}_valid'
            elif intrinsic == PureIntrinsic.VALUE_VALID:
                 value_expr = expr.operands[0].value
                 if value_expr.parent.module != expr.parent.module:
                     body = f"{rval} = self.{namify(str(value_expr).as_operand())}_valid"
                 else:
                     body = f"{rval} = self.executed"
            else:
                raise ValueError(f"Unknown intrinsic: {expr}")
        elif isinstance(expr, AsyncCall):
            self.expose('trigger', expr)
        elif isinstance(expr, Slice):
            a = dump_rval(expr.x, False)
            l = expr.l.value.value
            r = expr.r.value.value
            body = f"{rval} = {a}.as_bits()[{l}:{r+1}]"
        elif isinstance(expr, Concat):
            a = dump_rval(expr.msb, False)
            b = dump_rval(expr.lsb, False)
            body = f"{rval} = BitsSignal.concat([{a}.as_bits(), {b}.as_bits()])"

 
        elif isinstance(expr, Cast):
            dbits = expr.dtype.bits
            a = dump_rval(expr.x, False)
            src_dtype = expr.x.dtype
            
            pad = dbits - src_dtype.bits
            cast_body = ""
            cast_kind =  expr.opcode 
            if cast_kind == Cast.BITCAST:
                # assert pad == 0
                cast_body = f"{a}.{dump_type_cast(expr.dtype)}"
            elif cast_kind == Cast.ZEXT:
                cast_body = f" BitsSignal.concat( [Bits({pad})(0) , {a}.as_bits()]).{dump_type_cast(expr.dtype)} "
        
            elif cast_kind == Cast.SEXT: 
                cast_body =  f"BitsSignal.concat( [BitsSignal.concat([ {a}.as_bits()[{src_dtype.bits-1}] ] * {pad}) , {a}.as_bits()]).{dump_type_cast(expr.dtype)}"
                 
            body = f"{rval} = {cast_body}"
        elif isinstance(expr, Select):
            cond = dump_rval(expr.cond, False)
            true_value = dump_rval(expr.true_value, False)
            false_value = dump_rval(expr.false_value, False)
            body = f'{rval} = Mux({cond}, {false_value}, {true_value})'
        elif isinstance(expr, Bind):
            body = None
        elif isinstance(expr, Select1Hot):
            rval = dump_rval(expr, False)
            cond = dump_rval(expr.cond, False)
            values = [dump_rval(v, False) for v in expr.values]
             
            value_type = dump_type(expr.values[0].dtype)
            zero_value = f"{value_type}(0)"
            
            gated_terms = []
            for i, value_name in enumerate(values):
                term = f"Mux({cond}.as_bits()[{i}], {zero_value}, {value_name})"
                gated_terms.append(f"({term})")
             
            if not gated_terms:
                body = f"{rval} = {zero_value}"
            else:
                final_expr = " | ".join(gated_terms)
                body = f"{rval} = {final_expr}"
            # value_type = dump_type(expr.values[0].dtype)
            # zero_value = f"{value_type}(0)"
  
            # for i, value_name in enumerate(values): 
            #     gated_term_name = f"{rval}_gated_{i}"
            #     if i == 0: 
            #         mux_code = (
            #             f"{gated_term_name} = Mux({cond}.as_bits()[{i}], {zero_value}, {value_name})"
            #         )
            #     else :
            #         mux_code = (
            #             f"{gated_term_name} = Mux({cond}.as_bits()[{i}], {last_gated_term_name}, {value_name})"
            #         )
                
            #     last_gated_term_name = gated_term_name
            #     self.append_code(mux_code)
                 
            # body = f"{rval} = {last_gated_term_name} "
        elif isinstance(expr, Intrinsic):
            intrinsic = expr.opcode
            if intrinsic == Intrinsic.FINISH:
                pred = self.get_pred() or "1"
                body = (f"\n`ifndef SYNTHESIS\n  always_ff @(posedge clk) "
                        f"if (self.executed && {pred}) $finish();\n`endif\n")
            elif intrinsic == Intrinsic.ASSERT:
                self.expose('expr', expr.args[0])
            elif intrinsic == Intrinsic.WAIT_UNTIL:
                cond = dump_rval(expr.args[0], False)
                is_async_callee = self.current_module in self.async_callees
                
                final_cond = cond
                if is_async_callee:
                    final_cond = f"({cond}.as_bits() & self.trigger_counter_pop_valid)"
                 
                self.wait_until = final_cond
            else:
                raise ValueError(f"Unknown block intrinsic: {expr}")
        else:
            raise ValueError(f"Unhandled expression type: {type(expr).__name__}")

        if expr.is_valued() and expr_externally_used(expr, True):
            self.expose('expr', expr)

        if body is not None:
            self.append_code(body)
    
    def cleanup_post_generation(self):
        self.append_code('')
        exec_conditions = []
        # is_driver = self.current_module not in self.async_callees
        
        # Condition 1: The module's own trigger counter must be valid.
        exec_conditions.append("self.trigger_counter_pop_valid")
            
        # Condition 2: Any 'wait_until' condition must be met.
        if self.wait_until:
            exec_conditions.append(f"({self.wait_until})")

        # Condition 3: If the module pops from any FIFOs, those FIFOs must be valid.
        # ports_being_popped = set()
        # for key, exposes in self._exposes.items():
            
        #     if isinstance(key, Port) and any(isinstance(e, FIFOPop) for e, p in exposes):
        #         ports_being_popped.add(key)
        
        # for port in ports_being_popped:
        #     exec_conditions.append(f"self.{namify(port.name)}_valid")

        # --- Generate the final executed_wire ---
        if not exec_conditions:
            self.append_code('executed_wire = Bits(1)(1)')
        else:
            self.append_code(f"executed_wire = {' & '.join(exec_conditions)}")

        for key, exposes in self._exposes.items():
            if isinstance(key, Array):
                array = dump_rval(key, False)
                has_write = any(isinstance(e, ArrayWrite) for e, p in exposes)
                if not has_write: continue
                 
                ce_terms = []
                widx_terms = [f"{dump_type(key.index_type())}(0)"]
                wdata_terms = [f"Bits({key.scalar_ty.bits})(0)"]
                
                for expr, pred in exposes:
                    if isinstance(expr, ArrayWrite):
                        self.append_code(f'# Expose: {expr}')
                        ce_terms.append(pred)
                        idx = dump_rval(expr.idx, False)
                        data = dump_rval(expr.val, False)
                     
                        widx_terms.insert(0, f"Mux({pred}, {widx_terms[0]}, {idx})")
                        wdata_terms.insert(0, f"Mux({pred}, {wdata_terms[0]}, {data}.as_bits())")

                final_ce = " | ".join(ce_terms) if ce_terms else "Bits(1)(0)"
                # Use executed_wire
                self.append_code(f'self.{array}_ce = executed_wire & ({final_ce})')
                self.append_code(f'self.{array}_wdata = {wdata_terms[0]}')
                if key.index_bits > 0:
                    self.append_code(f'self.{array}_widx = {widx_terms[0]}')

            elif isinstance(key, Port):
                # Check what kind of operations were exposed for this port
                has_push = any(isinstance(e, FIFOPush) for e, p in exposes)
                has_pop = any(isinstance(e, FIFOPop) for e, p in exposes)

                if has_push: 
                    fifo = dump_rval(key, False)
                    pushes = [(e, p) for e, p in exposes if isinstance(e, FIFOPush)]
                    
                    final_push_predicate = " | ".join([f"({p})" for _, p in pushes]) if pushes else "Bits(1)(0)"
 
                    if len(pushes) == 1:
                        final_push_data = dump_rval(pushes[0][0].val, False)
                    else:
                        mux_data = f"{dump_type(key.dtype)}(0)"
                        for expr, pred in pushes:
                            rval = dump_rval(expr.val, False)
                             
                            mux_data = f"Mux({pred}, {rval}, {mux_data})"
                        final_push_data = mux_data
 
                    self.append_code(f'# Push logic for port: {fifo}')
                    ready_signal = f"self.fifo_{namify(key.module.name)}_{fifo}_push_ready"
                     
                    self.append_code(f"self.{namify(key.module.name)}_{fifo}_push_valid = executed_wire & ({final_push_predicate}) & {ready_signal}")
                    self.append_code(f"self.{namify(key.module.name)}_{fifo}_push_data = {final_push_data}")
                    
                if has_pop: 
                    fifo = dump_rval(key, False)
                    pop_expr = [e for e, p in exposes if isinstance(e, FIFOPop)][0]
                     
                    self.append_code(f'# {pop_expr}')
                    self.append_code(f'self.{fifo}_pop_ready = executed_wire')
                    
                    
            elif isinstance(key, Module): # This is for AsyncCall triggers
                rval = dump_rval(key, False)
                ce_terms = []
                for expr, pred in exposes:
                    self.append_code(f'# {expr}')
                    ce_terms.append(pred)
                
                final_ce = " | ".join(ce_terms) if ce_terms else "Bits(1)(0)"
                self.append_code(f'self.{rval}_trigger = executed_wire & ({final_ce}) & self.{rval}_trigger_counter_delta_ready')

            # elif isinstance(key, Expr)  :
            else:
                expr, pred = exposes[0]
                rval = dump_rval(expr, False)
                exposed_name = dump_rval(expr, True)
                if not isinstance(key,ArrayWrite ):
                    dtype_str = dump_type(expr.dtype)
                else :
                    dtype_str = dump_type(expr.x.dtype)

                # Add port declaration strings to our list
                self.exposed_ports_to_add.append(f'expose_{exposed_name} = Output({dtype_str})')
                self.exposed_ports_to_add.append(f'valid_{exposed_name} = Output(Bits(1))')

                # Generate the logic assignment
                self.append_code(f'# Expose: {expr}')
                self.append_code(f'self.expose_{exposed_name} = {rval}')
                self.append_code(f'self.valid_{exposed_name} = executed_wire')

        self.append_code('self.executed = executed_wire')
    
    def visit_module(self, node: Module):
        # STAGE 1: ANALYSIS & BODY GENERATION
        # Generate the 'construct' method body into a temporary buffer to discover
        # all necessary ports before writing the final class definition.

        original_code_buffer = self.code
        original_indent = self.indent

        self.code = []
        self.indent = original_indent + 8
        
        self.wait_until = None
        self._exposes = {}
        self.cond_stack = []
        self.current_module = node
        self.exposed_ports_to_add = []

        self.visit_block(node.body)
        self.cleanup_post_generation()
        
        construct_method_body = self.code
        
        self.code = original_code_buffer
        self.indent = original_indent
        self.current_module = node
 
        is_async_callee = node in self.async_callees
        is_driver = node not in self.async_callees
        
        self.append_code(f'class {namify(node.name)}(Module):')
        self.indent += 4
        
        self.append_code('clk = Clock()')
        self.append_code('rst = Reset()')
        self.append_code('executed = Output(Bits(1))')
        self.append_code('cycle_count = Input(UInt(64))')
        if is_driver or node in self.async_callees:
            self.append_code('trigger_counter_pop_valid = Input(Bits(1))')

        for i in node.ports:
            name = namify(i.name)
            self.append_code(f'{name} = Input({dump_type(i.dtype)})')
            self.append_code(f'{name}_valid = Input(Bits(1))')
            has_pop = any(isinstance(e, FIFOPop) and e.fifo == i for e in self._walk_expressions(node.body))
            if has_pop:
                self.append_code(f'{name}_pop_ready = Output(Bits(1))')
 
        pushes = [e for e in self._walk_expressions(node.body) if isinstance(e, FIFOPush)]
        calls = [e for e in self._walk_expressions(node.body) if isinstance(e, AsyncCall)]
        
        unique_push_handshake_targets = {(p.fifo.module, p.fifo.name) for p in pushes}
        unique_call_handshake_targets = {c.bind.callee for c in calls}
        unique_output_push_ports = {p.fifo for p in pushes}

        for module, fifo_name in unique_push_handshake_targets:
            port_name = f'fifo_{namify(module.name)}_{namify(fifo_name)}_push_ready'
            self.append_code(f'{port_name} = Input(Bits(1))')
        for callee in unique_call_handshake_targets:
            port_name = f'{namify(callee.name)}_trigger_counter_delta_ready'
            self.append_code(f'{port_name} = Input(Bits(1))')

        for fifo_port in unique_output_push_ports: 
            self.append_code(f'{namify(fifo_port.module.name)}_{namify(fifo_port.name)}_push_valid = Output(Bits(1))')
            dtype = [p.val.dtype for p in pushes if p.fifo == fifo_port][0]
            self.append_code(f'{namify(fifo_port.module.name)}_{namify(fifo_port.name)}_push_data = Output({dump_type(dtype)})')
        for callee in unique_call_handshake_targets:
            self.append_code(f'{namify(callee.name)}_trigger = Output(Bits(1))')
    
        for arr in self.sys.arrays:
            if node in self.array_users.get(arr, []):
                self.append_code(f'{namify(arr.name)}_payload = Input(Array({dump_type(arr.scalar_ty)}, {arr.size}))')
                is_writer = any(isinstance(e, ArrayWrite) and e.array == arr for e in self._walk_expressions(node.body))
                if is_writer:
                    self.append_code(f'{namify(arr.name)}_ce = Output(Bits(1))')
                    self.append_code(f'{namify(arr.name)}_wdata = Output(Bits({arr.scalar_ty.bits}))')
                    if arr.index_bits > 0:
                        self.append_code(f'{namify(arr.name)}_widx = Output({dump_type(arr.index_type())})')
 
        for port_code in self.exposed_ports_to_add:
            self.append_code(port_code)
 
        self.append_code('')
        self.append_code('@generator')
        self.append_code('def construct(self):')
        
        self.code.extend(construct_method_body)
        
        self.indent -= 4
        self.append_code('') 

    def _walk_expressions(self, block: Block):
        """Recursively walks a block and yields all expressions."""
        for item in block.body:
            if isinstance(item, Expr):
                yield item
            elif isinstance(item, Block):
                yield from self._walk_expressions(item)

    def visit_system(self, sys: SysBuilder):
        self.sys = sys
         
        for module in sys.modules:
            for expr in self._walk_expressions(module.body):
                if isinstance(expr, AsyncCall):
                    callee = expr.bind.callee
                    if callee not in self.async_callees:
                        self.async_callees[callee] = []
                     
                    if module not in self.async_callees[callee]:
                        self.async_callees[callee].append(module)

        self.array_users = {}
        for arr in self.sys.arrays:
            self.array_users[arr] = []
            for mod in self.sys.modules:
                for expr in self._walk_expressions(mod.body):
                    if isinstance(expr, (ArrayRead, ArrayWrite)) and expr.array == arr:
                        if mod not in self.array_users[arr]:
                            self.array_users[arr].append(mod)

        for elem in sys.arrays:
            self.visit_array(elem)
        for elem in sys.modules:
            self.current_module = elem
            self.visit_module(elem)
        self.current_module = None
        for elem in sys.downstreams:
            self.visit_module(elem)
 
        self._generate_top_harness()

 
    def _generate_top_harness(self):
        """
        Generates a generic Top-level harness that connects all modules based on
        the analyzed dependencies (async calls, array usage).
        """
        self.append_code('class Top(Module):')
        self.indent += 4
        self.append_code('clk = Clock()')
        self.append_code('rst = Reset()')
        self.append_code('global_cycle_count = Output(UInt(64))')
        self.append_code('')
        self.append_code('@generator')
        self.append_code('def construct(self):')
        self.indent += 4

        self.append_code('\n# --- Global Cycle Counter ---')
        self.append_code('# A free-running counter for testbench control')
         
        self.append_code('cycle_count = Reg(UInt(64), clk=self.clk, rst=self.rst, rst_value=0)')
        self.append_code('cycle_count.assign( (cycle_count + UInt(64)(1)).as_bits()[0:64].as_uint() )')
        self.append_code('self.global_cycle_count = cycle_count')
        # --- 1. Wire Declarations (Generic) ---
        self.append_code('# --- Wires for FIFOs, Triggers, and Arrays ---')
        # Wires for FIFOs (one per callee's port)
        for callee in self.async_callees:
            for port in callee.ports:
                fifo_base_name = f'fifo_{namify(callee.name)}_{namify(port.name)}'
                self.append_code(f'# Wires for FIFO connected to {callee.name}.{port.name}')
                self.append_code(f'{fifo_base_name}_push_valid = Wire(Bits(1))')
                self.append_code(f'{fifo_base_name}_push_data = Wire(Bits({port.dtype.bits}))')
                self.append_code(f'{fifo_base_name}_push_ready = Wire(Bits(1))')
                self.append_code(f'{fifo_base_name}_pop_valid = Wire(Bits(1))')
                self.append_code(f'{fifo_base_name}_pop_data = Wire(Bits({port.dtype.bits}))')
                self.append_code(f'{fifo_base_name}_pop_ready = Wire(Bits(1))')

        # Wires for TriggerCounters (one per module)
        for module in self.sys.modules:
            tc_base_name = f'{namify(module.name)}_trigger_counter'
            self.append_code(f'# Wires for {module.name}\'s TriggerCounter')
            self.append_code(f'{tc_base_name}_delta = Wire(Bits(8))')
            self.append_code(f'{tc_base_name}_delta_ready = Wire(Bits(1))')
            self.append_code(f'{tc_base_name}_pop_valid = Wire(Bits(1))')
            self.append_code(f'{tc_base_name}_pop_ready = Wire(Bits(1))')

        # Wires for Arrays (one per global array)
        for array in self.sys.arrays:
            arr_name = namify(array.name) 
            self.append_code(f'# Wires for Array {array.name}')
            self.append_code(f'{arr_name}_ce = Wire(Bits(1))')
            self.append_code(f'{arr_name}_wdata = Wire(Bits({array.scalar_ty.bits}))')
            if array.index_bits > 0:
                self.append_code(f'{arr_name}_widx = Wire(Bits({array.index_bits}))')


        # --- 2. Hardware Instantiations (Generic) ---
        self.append_code('\n# --- Hardware Instantiations ---')
        # Instantiate Regs for each Array
        array_init = {}
        for array in self.sys.arrays:
            arr_name = namify(array.name)
            arr_type = dump_type(array.scalar_ty)
            
            if hasattr(array, 'initializer') and array.initializer is not None: 
                rst_vals = [f"{arr_type}({val})" for val in array.initializer]
                rst_value_str = f"[{', '.join(rst_vals)}]"
            else: 
                rst_value_str = f"[{', '.join([f'{arr_type}(0)'] * array.size)}]"
            array_init[arr_name] = rst_value_str
            self.append_code(
                f'reg_{arr_name} = Reg(Array({arr_type}, {array.size}), '
                f'clk=self.clk, rst=self.rst, rst_value={rst_value_str}, '
                f'ce={arr_name}_ce)'
            )
        # Instantiate FIFOs
        for callee in self.async_callees:
            for port in callee.ports:
                fifo_base_name = f'fifo_{namify(callee.name)}_{namify(port.name)}'
                self.append_code(f'{fifo_base_name}_inst = FIFO(WIDTH={port.dtype.bits}, DEPTH_LOG2=2)(clk=self.clk, rst_n=~self.rst, push_valid={fifo_base_name}_push_valid, push_data={fifo_base_name}_push_data, pop_ready={fifo_base_name}_pop_ready)')
                self.append_code(f'{fifo_base_name}_push_ready.assign({fifo_base_name}_inst.push_ready)')
                self.append_code(f'{fifo_base_name}_pop_valid.assign({fifo_base_name}_inst.pop_valid)')
                self.append_code(f'{fifo_base_name}_pop_data.assign({fifo_base_name}_inst.pop_data)')

        # Instantiate TriggerCounters
        for module in self.sys.modules:
            tc_base_name = f'{namify(module.name)}_trigger_counter'
            self.append_code(f'{tc_base_name}_inst = TriggerCounter(WIDTH=8)(clk=self.clk, rst_n=~self.rst, delta={tc_base_name}_delta, pop_ready={tc_base_name}_pop_ready)')
            self.append_code(f'{tc_base_name}_delta_ready.assign({tc_base_name}_inst.delta_ready)')
            self.append_code(f'{tc_base_name}_pop_valid.assign({tc_base_name}_inst.pop_valid)')


        # --- 3. Module Instantiations and Connections (Generic) ---
        self.append_code('\n# --- Module Instantiations and Connections ---')
        for module in self.sys.modules:
            mod_name = namify(module.name)
            self.append_code(f'# Instantiation for {module.name}')
            
            # Build the port map for the module instance
            port_map = [f'clk=self.clk', f'rst=self.rst']
            
            # Connect required trigger/cycle ports
            if module in self.async_callees or module not in self.async_callees:
                port_map.append(f"trigger_counter_pop_valid={mod_name}_trigger_counter_pop_valid")
            port_map.append(f"cycle_count=cycle_count")

            # Connect input payloads from arrays
            for arr, users in self.array_users.items():
                if module in users:
                    port_map.append(f"{namify(arr.name)}_payload=reg_{namify(arr.name)}")
            
            pushes = [e for e in self._walk_expressions(module.body) if isinstance(e, FIFOPush)]
            calls = [e for e in self._walk_expressions(module.body) if isinstance(e, AsyncCall)]
             
            unique_push_targets = {(p.fifo.module, p.fifo) for p in pushes}
            unique_call_targets = {c.bind.callee for c in calls}
 
            for (callee_mod, callee_port) in unique_push_targets:
                port_map.append(f"fifo_{namify(callee_mod.name)}_{namify(callee_port.name)}_push_ready=fifo_{namify(callee_mod.name)}_{namify(callee_port.name)}_push_ready")
            
            for callee_mod in unique_call_targets:
                port_map.append(f"{namify(callee_mod.name)}_trigger_counter_delta_ready={namify(callee_mod.name)}_trigger_counter_delta_ready")
            
            # Connect input data channels (if it's a callee)
            if module in self.async_callees:
                for port in module.ports:
                    fifo_base_name = f'fifo_{mod_name}_{namify(port.name)}'
                    port_map.append(f"{namify(port.name)}={fifo_base_name}_pop_data.{dump_type_cast(port.dtype)}")
                    port_map.append(f"{namify(port.name)}_valid={fifo_base_name}_pop_valid")
            
            # Instantiate the module with the complete port map
            self.append_code(f"inst_{mod_name} = {mod_name}({', '.join(port_map)})")

            # --- Connect the module's OUTPUTS back to wires ---
            
            # Connect main 'executed' output 
            self.append_code(f"{mod_name}_trigger_counter_pop_ready.assign(inst_{mod_name}.executed)")
            
            # Connect data channel pop_ready outputs
            for port in module.ports:
                if any(isinstance(e, FIFOPop) and e.fifo == port for e in self._walk_expressions(module.body)):
                    self.append_code(f"fifo_{mod_name}_{namify(port.name)}_pop_ready.assign(inst_{mod_name}.{namify(port.name)}_pop_ready)")

            # Connect data channel push outputs
            for (callee_mod, callee_port) in unique_push_targets:
                callee_mod_name = namify(callee_mod.name)
                callee_port_name = namify(callee_port.name) 
                self.append_code(f"fifo_{callee_mod_name}_{callee_port_name}_push_valid.assign(inst_{mod_name}.{callee_mod_name}_{callee_port_name}_push_valid)")
                self.append_code(f"fifo_{callee_mod_name}_{callee_port_name}_push_data.assign(inst_{mod_name}.{callee_mod_name}_{callee_port_name}_push_data.as_bits())")
        
        # --- 4. Array Write-Back Connections (Generic) ---
        self.append_code('\n# --- Array Write-Back Connections ---')
        for arr, users in self.array_users.items():
            arr_name = namify(arr.name)
             
            writers = [
                m for m in users
                if any(isinstance(e, ArrayWrite) and e.array == arr
                       for e in self._walk_expressions(m.body))
            ]
 
            if not writers:
                # Case 0: No writers (read-only array)
                self.append_code(f'# Tying off write ports for read-only array {arr_name}')
                self.append_code(f"{arr_name}_ce.assign(Bits(1)(0))")
                self.append_code(f"{arr_name}_wdata.assign(Bits({arr.scalar_ty.bits})(0))")
                if arr.index_bits > 0:
                    self.append_code(f"{arr_name}_widx.assign(Bits({arr.index_bits})(0))")
                
            elif len(writers) == 1:
                # Case 1: Single writer - direct connection
                writer_mod_name = namify(writers[0].name)
                self.append_code(f'# Connecting single writer for array {arr_name}')
                self.append_code(f"{arr_name}_ce.assign(inst_{writer_mod_name}.{arr_name}_ce)")
                self.append_code(f"{arr_name}_wdata.assign(inst_{writer_mod_name}.{arr_name}_wdata)")
                if arr.index_bits > 0:
                    self.append_code(f"{arr_name}_widx.assign(inst_{writer_mod_name}.{arr_name}_widx)")
            
            else:
                # Case 2: Multiple writers - generate arbitration logic
                self.append_code(f'# Arbitrating multiple writers for array {arr_name}')
 
                ce_terms = [f"inst_{namify(w.name)}.{arr_name}_ce" for w in writers]
                self.append_code(f"{arr_name}_ce.assign({' | '.join(ce_terms)})")
 
                wdata_mux = f"Bits({arr.scalar_ty.bits})(0)" 
                for writer in reversed(writers):
                    writer_mod_name = namify(writer.name)
                    cond = f"inst_{writer_mod_name}.{arr_name}_ce"
                    true_val = f"inst_{writer_mod_name}.{arr_name}_wdata"
                    wdata_mux = f"Mux({cond}, {wdata_mux}, {true_val})"
                self.append_code(f"{arr_name}_wdata.assign({wdata_mux})")
 
                if arr.index_bits > 0:
                    widx_mux = f"Bits({arr.index_bits})(0)" # Default value
                    for writer in reversed(writers):
                        writer_mod_name = namify(writer.name)
                        cond = f"inst_{writer_mod_name}.{arr_name}_ce"
                        true_val = f"inst_{writer_mod_name}.{arr_name}_widx"
                        widx_mux = f"Mux({cond},  {widx_mux},{true_val})"
                    self.append_code(f"{arr_name}_widx.assign({widx_mux})")

            # Final assignment to the register's input port
            if writers:
                self.append_code(f"reg_{arr_name}.assign([{arr_name}_wdata.{dump_type_cast(arr.scalar_ty)}])")
            else:
                self.append_code(f"reg_{arr_name}.assign({array_init[arr_name]})")
        # --- 5. Trigger Counter Delta Connections  ---
        self.append_code('\n# --- Trigger Counter Delta Connections ---')
        for module in self.sys.modules:
            mod_name = namify(module.name)
            
            if module in self.async_callees:  
                callers_of_this_module = self.async_callees[module]
                 
                trigger_terms = [
                    f"inst_{namify(c.name)}.{mod_name}_trigger.as_uint()"
                    for c in callers_of_this_module
                ] 
                if len(trigger_terms) > 1:
                    summed_triggers = f"({' + '.join(trigger_terms)})"
                else:
                    summed_triggers = trigger_terms[0]
                 
                self.append_code(f"{mod_name}_trigger_counter_delta.assign({summed_triggers}.as_bits(8))")
            else:   
                self.append_code(f"{mod_name}_trigger_counter_delta.assign(Bits(8)(1))")

        self.indent -= 8
        self.append_code('')
        self.append_code(f'system = System([Top], name="Top", output_directory="sv")')
        self.append_code('system.compile()')

# The HEADER constant. NOTE the updated TriggerCounter definition.
HEADER = '''from pycde import Input, Output, Module, System, Clock, Reset
from pycde import generator, modparams
from pycde.constructs import Reg, Array, Mux,Wire
from pycde.types import Bits, SInt, UInt
from pycde.signals import Struct, BitsSignal
from pycde.dialects import comb

@modparams
def FIFO(WIDTH: int, DEPTH_LOG2: int):
    class FIFOImpl(Module):
        module_name = f"fifo"
        # Define inputs
        clk = Clock()
        rst_n = Input(Bits(1))
        push_valid = Input(Bits(1))
        push_data = Input(Bits(WIDTH))
        pop_ready = Input(Bits(1))
        # Define outputs
        push_ready = Output(Bits(1))
        pop_valid = Output(Bits(1))
        pop_data = Output(Bits(WIDTH))
    return FIFOImpl


@modparams
def TriggerCounter(WIDTH: int):
    class TriggerCounterImpl(Module):
        module_name = f"trigger_counter"
        clk = Clock()
        rst_n = Input(Bits(1))
        delta = Input(Bits(WIDTH))
        delta_ready = Output(Bits(1))
        pop_ready = Input(Bits(1))
        pop_valid = Output(Bits(1))
    return TriggerCounterImpl

'''

def generate_design(fname: str, sys: SysBuilder):
    """Generate a complete Verilog design file for the system."""
    with open(fname, 'w', encoding='utf-8') as fd:
        fd.write(HEADER)
        dumper = CIRCTDumper()
        dumper.visit_system(sys)
        code = '\n'.join(dumper.code)
        fd.write(code)
    logs = dumper.logs
     
    return logs