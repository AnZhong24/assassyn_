
import ast
import typing
from dataclasses import dataclass
 

@dataclass
class NamingContext:
    """Context information for naming decisions"""
    ast_node: ast.AST
    target_names: typing.List[str]
    lineno: int

class UnifiedNamingStrategy:
    """Single unified recursive strategy for all naming patterns"""
    
    def __init__(self):
        self.collected_names = []
        self.temp_counter = 0
    
    def generate_names(self, context: NamingContext) -> typing.List[str]:
        """Generate names for any assignment pattern"""
        # Reset state
        self.collected_names = []
        self.temp_counter = 0
        
        node = context.ast_node

        if isinstance(node, ast.Assign):
            assign = context.ast_node
            target = assign.targets[0]
            value = assign.value
        
            if isinstance(target, ast.Name):
                target_name = target.id
            elif isinstance(target, ast.Attribute):
                # a.is_ood = ... → record_a_is_ood
                target_name = f"record_{target.value.id}_{target.attr}"
            elif isinstance(target, ast.Subscript):
                # a[0] = ... → array_a_0
                target_name = f"array_{target.value.id}_{target.slice.value}"
            elif isinstance(target, ast.Tuple):
                # Multiple targets - handled specially
                target_name = None
            else:
                target_name = "result"
         
            self._process_value(value, target_name, target)
        
        elif isinstance(node, ast.Expr): 
            self._process_value(node.value, None, None)
        
        return self.collected_names
    
    def _process_value(self, node: ast.AST, target_name: str, target: ast.AST) -> None:
        """Recursively process the value expression"""
        
        if isinstance(node, ast.BinOp):
            # Binary operation - process recursively
            self._process_binop(node, target_name)
            
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # Method calls
            method = node.func.attr
            
            if method == 'pop_all_ports':
                # Special handling for tuple unpacking
                self._process_pop_all_ports(target)
                
            elif method == 'select' or method == 'select1hot':
                # Conditional select
                self._process_method_object(node.func.value)
                self._process_select(node, target_name)
            
            elif method == 'async_called':
                print(f"async called: {node}")
                # Handle async_called method calls
                self._process_async_called(node)

            else:
                # Other method calls - just use target name
                if target_name:
                    self.collected_names.append(target_name)
                    
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            # Array slice: value[0:0] → just use target name
            if target_name:
                self.collected_names.append(target_name)
                
        else:
            # Simple assignments - just use target name
            if target_name:
                self.collected_names.append(target_name)
    

    def _process_method_object(self, node: ast.AST) -> None:
        """Process the object that a method is called on"""
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Slice):
                # Handle cnt[0][0:0] pattern
                if isinstance(node.value, ast.Subscript):
                    # First process the inner subscript: cnt[0] → cnt_0
                    if isinstance(node.value.value, ast.Name):
                        self.collected_names.append(f"{node.value.value.id}_{node.value.slice.value}")
                    # Then process the slice: cnt[0][0:0] → cnt_0_0
                    start = node.slice.lower.value if node.slice.lower else 0
                    self.collected_names.append(f"{node.value.value.id}_{node.value.slice.value}_{start}")
                elif isinstance(node.value, ast.Name):
                    # Simple slice: value[0:0] → value_0
                    start = node.slice.lower.value if node.slice.lower else 0
                    self.collected_names.append(f"{node.value.id}_{start}")
            else:
                # Simple subscript
                if isinstance(node.value, ast.Name):
                    self.collected_names.append(f"{node.value.id}_{node.slice.value}")
        # elif isinstance(node, ast.Name):
        #     # Simple name
        #     self.collected_names.append(node.id)

    def _process_binop(self, node: ast.BinOp, base_name: str, is_root: bool = True) -> None:
        """Process binary operations recursively"""
        # Process left operand
        self._process_operand(node.left, base_name)
        
        # Process right operand
        self._process_operand(node.right, base_name)
        
        # Add name for this BinOp result
        if is_root:
            self.collected_names.append(base_name)
        else:
            self.temp_counter += 1
            self.collected_names.append(f"{base_name}_temp{self.temp_counter}")
    
    
    def _process_async_called(self, node: ast.Call) -> None:
        """Process async_called method calls"""
        # Process each keyword argument
        for kw in node.keywords:
            if kw.value:
                self.collected_names.append(kw.arg)
                self._process_async_arg(kw.value)
    
    def _process_async_arg(self, node: ast.AST) -> None:
        """Process an argument in async_called"""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # Method call like f[0:31].bitcast(Int(32))
            if node.func.attr == 'bitcast':
                # Process the object being bitcast
                if isinstance(node.func.value, ast.Subscript) and isinstance(node.func.value.slice, ast.Slice):
                    # f[0:31] → f_bits_0to31
                    obj = node.func.value
                    start = obj.slice.lower.value if obj.slice.lower else 0
                    stop = obj.slice.upper.value if obj.slice.upper else 32
                    base_name = f"{obj.value.id}_bits_{start}to{stop}"
                    self.collected_names.append(base_name)
                    # Add cast version
                    self.collected_names.append(f"{base_name}_i32")
        elif isinstance(node, ast.Subscript):
            # Simple subscript like cnt[0] → array_cnt_0
            if isinstance(node.value, ast.Name):
                self.collected_names.append(f"array_{node.value.id}_{node.slice.value}")
        # elif isinstance(node, ast.Name):
        #     # Simple variable reference
        #     self.collected_names.append(node.id)



    def _process_operand(self, node: ast.AST, base_name: str) -> None:
        """Process an operand in a binary operation"""
        
        if isinstance(node, ast.BinOp):
            # Nested BinOp
            self._process_binop(node, base_name, is_root=False)
            
        elif isinstance(node, ast.Attribute):
            # Attribute access: a.payload
            if isinstance(node.value, ast.Name):
                attr_name = f"{node.value.id}_{node.attr}"
                self.collected_names.append(attr_name)
                
                # # Some attributes need casting
                # if node.attr in ['payload', 'data']:
                #     self.collected_names.append(f"{attr_name}_i32")
                    
            elif isinstance(node.value, ast.Subscript):
                # bundle[0].payload pattern
                self.collected_names.append("value")
                if node.attr in ['payload', 'data']:
                    self.collected_names.append("value_i32")
                    
        elif isinstance(node, ast.Subscript):
            # Handle both slice and simple subscript
            if isinstance(node.slice, ast.Slice):
                # Array slice in expression
                if isinstance(node.value, ast.Name):
                    start = node.slice.lower.value if node.slice.lower else 0
                    stop = node.slice.upper.value if node.slice.upper else 32
                    self.collected_names.append(f"{node.value.id}_bits_{start}to{stop}")
            else:
                # Simple subscript: cnt[0] → array_cnt_0
                if isinstance(node.value, ast.Name):
                    self.collected_names.append(f"array_{node.value.id}_{node.slice.value}")
        
        # Other nodes (Name, Constant, Call) don't generate intermediate names
    
    def _process_pop_all_ports(self, target: ast.Tuple) -> None:
        """Process pop_all_ports tuple unpacking"""
        target_names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
        n = 0
        
        # Individual validations
        for name in target_names:
            self.collected_names.append(f"{name}_valid")
            if n > 0:
                combined_name = "_".join(target_names[:n+1]) + "_valid"
                self.collected_names.append(combined_name)
            n += 1

        self.collected_names.extend(target_names)
    
    def _process_select(self, node: ast.Call, target_name: str) -> None:
        """Process select() calls"""
        # Process each argument
        for arg in node.args:
            if isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Slice):
                # Array slice: value[0:0] → value_0
                start = arg.slice.lower.value if arg.slice.lower else 0
                self.collected_names.append(f"{arg.value.id}_{start}")
            elif isinstance(arg, ast.Attribute):
                # Attribute: rand0.is_addr → rand0_is_addr
                self.collected_names.append(f"{arg.value.id}_{arg.attr}")
            elif isinstance(arg, ast.Subscript):
                # Simple subscript: values[0] → values_0
                self.collected_names.append(f"{arg.value.id}_{arg.slice.value}")
           
        self.collected_names.append(target_name)
     
class NamingManager:
    """Simplified naming manager with single unified strategy"""
    
    def __init__(self):
        self.strategy = UnifiedNamingStrategy()
        self.line_contexts = {}
    
    def analyze_assignment(self, ast_node: ast.Assign) -> NamingContext:
        """Create naming context"""
        if isinstance(ast_node, ast.Expr):
            target_names = None
        else:   
            target_names = self._extract_target_names(ast_node)
        
        return NamingContext(
            ast_node=ast_node,
            target_names=target_names,
            lineno=getattr(ast_node, 'lineno', 0)
        )
    
    def _extract_target_names(self, ast_node: ast.Assign) -> list:
        """Extract target variable names"""
        target_names = []
        
        if len(ast_node.targets) == 1:
            target = ast_node.targets[0]
            
            if isinstance(target, ast.Name):
                target_names = [target.id]
            elif isinstance(target, ast.Tuple):
                target_names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
            elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                target_names = [target.value.id]
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                target_names = [target.attr]
        
        return target_names
    
    def generate_source_names(self, lineno: int, target_ast_node: ast.Assign) -> typing.List[str]:
        """Generate source names using unified strategy"""
        
        context = self.analyze_assignment(target_ast_node)
        context.lineno = lineno
        
        # Track context
        if lineno not in self.line_contexts:
            self.line_contexts[lineno] = []
        self.line_contexts[lineno].append(context)
         
        names = self.strategy.generate_names(context)
        
        # Debug output
        # print(f"Line {lineno}: Generated {len(names)} names: {names}")
        
        return names
   