'''The module provides the implementation of a class that is both IR builder and the system.'''

#pylint: disable=cyclic-import

from __future__ import annotations
import os
import typing
import site
import inspect
from decorator import decorator
import ast

if typing.TYPE_CHECKING:
    from .ir.module import Module
    from .ir.array import Array
    from .ir.dtype import DType
    from .ir.value import Value


from dataclasses import dataclass
from enum import Enum

class AssignmentType(Enum):
    """Types of assignments that require different naming strategies"""
    SIMPLE_ASSIGN = "simple"           # x = expr
    TUPLE_POP_ALL = "tuple_pop_all"    # a, b = self.pop_all_ports(True)
    NESTED_BINOP = "nested_binop"      # d = (a + b) + c
    SUBSCRIPT_ASSIGN = "subscript"     # cnt[0] = expr
    CONDITIONAL_SELECT = "conditional_select"  # cnt_div2 = cnt[0][0:0].select(cnt[0], cnt_div2_temp)
    COMPLEX_CALL = "complex_call"      # x = obj.method(args)
    # ATTRIBUTE_BINOP = "attribute_binop"

@dataclass
class NamingContext:
    """Context information for naming decisions"""
    ast_node: ast.AST
    assignment_type: AssignmentType
    target_names: typing.List[str]
    lineno: int
    is_first_occurrence: bool = True
    nested_level: int = 0
    sibling_count: int = 0
    

@dataclass
class ExpressionNode:
    ast_node: ast.AST
    name: str
    children: typing.List['ExpressionNode']
    is_leaf: bool = False
    order: int = 0

class NamingStrategy:
    """Base class for naming strategies"""
    
    def generate_names(self, context: NamingContext) -> typing.List[str]:
        """Generate a list of names for the given context"""
        raise NotImplementedError

class SimpleAssignNamingStrategy(NamingStrategy):
    """Strategy for simple assignments like: x = expr"""
    
    def generate_names(self, context: NamingContext) -> typing.List[str]:
        if len(context.target_names) == 1:
            return [context.target_names[0]]
        return context.target_names


class TuplePopAllNamingStrategy(NamingStrategy):

    def generate_names(self, context: NamingContext) -> typing.List[str]:
        
        target_names = context.target_names
        n = len(target_names)
        generated_names = []
        
        for i in range(n): 
            generated_names.append(f"{target_names[i]}_valid")
            if i > 0:  
                combined_name = "_".join(target_names[:i+1]) + "_valid"
                generated_names.append(combined_name)
 
        generated_names.extend(target_names)
 
        return generated_names
    

class NestedBinOpNamingStrategy(NamingStrategy):
    """Strategy for nested binary operations"""
    
    def generate_names(self, context: NamingContext) -> typing.List[str]:
        base_name = context.target_names[0] if context.target_names else "temp"
        generated_names = []
        
        # Count the actual nesting levels by analyzing the AST
        nesting_count = self._count_intermediate_expressions(context.ast_node.value)
        
        # Generate intermediate names
        for i in range(1, nesting_count + 1):
            generated_names.append(f"{base_name}{i}")
         
        generated_names.append(base_name)
        
        return generated_names
    
    def _count_intermediate_expressions(self, node: ast.AST) -> int:
        """Count intermediate expressions needed for nested binary operations"""
        if not isinstance(node, ast.BinOp):
            return 0
        
        count = 0
        
        # Check if left operand is a nested BinOp
        if isinstance(node.left, ast.BinOp):
            count += 1 + self._count_intermediate_expressions(node.left)
        
        # Check if right operand is a nested BinOp
        if isinstance(node.right, ast.BinOp):
            count += self._count_intermediate_expressions(node.right)
        
        return count

class ConditionalSelectNamingStrategy(NamingStrategy):
    """Strategy for conditional select assignments like cnt_div2 = cnt[0][0:0].select(cnt[0], cnt_div2_temp)"""
    
    def generate_names(self, context: NamingContext) -> typing.List[str]:
        base_name = context.target_names[0] if context.target_names else "select"
         
        return [
            f"{base_name}_cond_array",    # condition array access
            f"{base_name}_cond_slice",    # condition bit slice
            f"{base_name}_true_val",      # true value array access
            base_name                     # final result
        ]

class SubscriptAssignNamingStrategy(NamingStrategy):
    """Strategy for subscript assignments like cnt[0] = expr"""
    
    def generate_names(self, context: NamingContext) -> typing.List[str]:
        # For subscript assignments, use a descriptive name
        if context.target_names:
            base_name = context.target_names[0]
            return [f"{base_name}_update"]
        return ["subscript_update"]

# class AttributeBinOpNamingStrategy(NamingStrategy):
class NamingManager:
    """naming manager that analyzes AST patterns and assigns appropriate names"""
    
    def __init__(self):
        self.strategies = {
            AssignmentType.SIMPLE_ASSIGN: SimpleAssignNamingStrategy(),
            AssignmentType.TUPLE_POP_ALL: TuplePopAllNamingStrategy(),
            AssignmentType.NESTED_BINOP: NestedBinOpNamingStrategy(),
            AssignmentType.SUBSCRIPT_ASSIGN: SubscriptAssignNamingStrategy(),
            AssignmentType.CONDITIONAL_SELECT: ConditionalSelectNamingStrategy(),
            # AssignmentType.ATTRIBUTE_BINOP: AttributeBinOpNamingStrategy(),
        }
        self.line_contexts = {}  # Track contexts per line number
        self.name_counters = {}  # Track name usage for disambiguation
        

    def analyze_assignment(self, ast_node: ast.Assign) -> NamingContext:
        """Analyze an assignment AST node to determine its type and context"""
        
        target_names = [] 
        assignment_type = AssignmentType.SIMPLE_ASSIGN  #TO DO: add more assignment types
         
        if len(ast_node.targets) == 1:
            target = ast_node.targets[0]
            
            if isinstance(target, ast.Name):
                target_names = [target.id]
                assignment_type = AssignmentType.SIMPLE_ASSIGN
            elif isinstance(target, ast.Tuple):
                target_names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
                
                # Check if this is a pop_all_ports call
                if (isinstance(ast_node.value, ast.Call) and 
                    isinstance(ast_node.value.func, ast.Attribute) and
                    ast_node.value.func.attr == 'pop_all_ports'):
                    assignment_type = AssignmentType.TUPLE_POP_ALL
                    
            elif isinstance(target, ast.Subscript):
                if isinstance(target.value, ast.Name):
                    target_names = [target.value.id]
                    assignment_type = AssignmentType.SUBSCRIPT_ASSIGN
        
        # Check for conditional select pattern: target = obj[...].select(...)
        if (isinstance(ast_node.value, ast.Call) and 
            isinstance(ast_node.value.func, ast.Attribute) and
            ast_node.value.func.attr == 'select'):
            assignment_type = AssignmentType.CONDITIONAL_SELECT
        
        # Check for nested binary operations
        elif isinstance(ast_node.value, ast.BinOp):
            nested_level = self._count_nested_binops(ast_node.value)
            if nested_level > 0:
                assignment_type = AssignmentType.NESTED_BINOP
        
        return NamingContext(
            ast_node=ast_node,
            assignment_type=assignment_type,
            target_names=target_names,
            lineno=getattr(ast_node, 'lineno', 0),
            # nested_level=self._count_nested_binops(ast_node.value) if isinstance(ast_node.value, ast.BinOp) else 0
        )
    
    def _count_nested_binops(self, node: ast.AST) -> int:
        """Count the level of nested binary operations"""
        if not isinstance(node, ast.BinOp):
            return 0
        
        left_depth = self._count_nested_binops(node.left) if isinstance(node.left, ast.BinOp) else 0
        right_depth = self._count_nested_binops(node.right) if isinstance(node.right, ast.BinOp) else 0
        
        return max(left_depth, right_depth) + (1 if isinstance(node.left, ast.BinOp) or isinstance(node.right, ast.BinOp) else 0)
    
    
    def generate_source_names(self, lineno: int, target_ast_node: ast.Assign) -> typing.List[str]:
        """Generate appropriate source names for a given line and AST node"""
        
        context = self.analyze_assignment(target_ast_node)
        context.lineno = lineno
        
        # Check if this is the first occurrence of this line
        if lineno in self.line_contexts:
            context.is_first_occurrence = False
            context.sibling_count = len(self.line_contexts[lineno])
        else:
            self.line_contexts[lineno] = []
        
        self.line_contexts[lineno].append(context)
        
        # Generate names using appropriate strategy
        strategy = self.strategies.get(context.assignment_type, self.strategies[AssignmentType.SIMPLE_ASSIGN])
        names = strategy.generate_names(context)
        
        # Disambiguate names if necessary
        disambiguated_names = []
        for name in names:
            final_name = self._disambiguate_name(name, lineno)
            disambiguated_names.append(final_name)
        
        return disambiguated_names
    
    def _disambiguate_name(self, name: str, lineno: int) -> str:
        """Add suffixes to names if they conflict"""
        key = f"{name}_{lineno}"
        
        if key not in self.name_counters:
            self.name_counters[key] = 0
            return name
        else:
            self.name_counters[key] += 1
            return f"{name}_{self.name_counters[key]}"


NAMING_MANAGER = NamingManager()
LINE_EXPRESSION_TRACKER = {}  # Maps lineno to list of expressions
LAST_PROCESSED_LINE = None


def process_naming(expr, line_of_code: str, lineno: int) -> typing.Dict[str, typing.Any]:
    """Process naming for an expression based on line context"""
    
    global LINE_EXPRESSION_TRACKER, LAST_PROCESSED_LINE
    
    try:
        parsed_ast = ast.parse(line_of_code)
        
        if parsed_ast.body and isinstance(parsed_ast.body[0], ast.Assign):
            assign_node = parsed_ast.body[0]
            print(ast.dump(assign_node, indent=2))
           
            if lineno not in LINE_EXPRESSION_TRACKER:
                LINE_EXPRESSION_TRACKER[lineno] = {
                    'expressions': [],
                    'assign_node': assign_node,
                    'names_generated': False,
                    'generated_names': []
                }
            
            line_data = LINE_EXPRESSION_TRACKER[lineno]
            line_data['expressions'].append(expr)
            
            # Generate names when we encounter a new line or when forced
            if not line_data['names_generated'] or LAST_PROCESSED_LINE != lineno:
                generated_names = NAMING_MANAGER.generate_source_names(
                    lineno, assign_node
                )
                line_data['generated_names'] = generated_names
                line_data['names_generated'] = True
                LAST_PROCESSED_LINE = lineno
            
            # Assign name to current expression
            expr_position = len(line_data['expressions']) - 1
            generated_names = line_data['generated_names']
            
            if expr_position < len(generated_names):
                source_name = generated_names[expr_position]
            else:
                # Fallback naming for extra expressions
                base_name = generated_names[0] if generated_names else "expr"
                source_name = f"{base_name}_extra_{expr_position}"
                print(f"ERROE  Fallback naming: {source_name}")
            # Determine assignment type for metadata
            context = NAMING_MANAGER.analyze_assignment(assign_node)
            
            return {
                'source_name': source_name, 
                'assignment_type': context.assignment_type.value,
                'position': expr_position,
                'total_count': len(generated_names)
            }
            
    except SyntaxError: 
            pass
    
    return None


@decorator
def ir_builder(func, *args, **kwargs):
    '''The decorator annotates the function whose return value will be inserted into the AST.'''
    res = func(*args, **kwargs)
    global LAST_RES, LAST_RES_LINENO
    # This indicates this res is handled somewhere else, so we do not need to rehandle it
    if res is None:
        return res

    #pylint: disable=cyclic-import,import-outside-toplevel
    from .ir.const import Const
    from .utils import package_path
    from .ir.expr import Expr

    if not isinstance(res, Const):
        if isinstance(res, Expr):
            res.parent = Singleton.builder.current_block
            for i in res.operands:
                Singleton.builder.current_module.add_external(i)
        Singleton.builder.insert_point.append(res)

    package_dir = os.path.abspath(package_path())

    Singleton.initialize_dirs_to_exclude()
    
    for i in inspect.stack()[2:]:
        fname, lineno = i.filename, i.lineno
        fname_abs = os.path.abspath(fname)

        if not fname_abs.startswith(package_dir) \
            and not any(fname_abs.startswith(exclude_dir) \
                         for exclude_dir in Singleton.all_dirs_to_exclude):
            res.loc = f'{fname}:{lineno}'
            if isinstance(res, Expr):
                if res.is_valued() and i.code_context:
                    line_of_code = i.code_context[0].strip()

                    naming_result = process_naming(
                    res, line_of_code, lineno
                ) 
                    if naming_result:
                        res.source_name = naming_result.get('source_name')
                    
                     
                    # res.naming_metadata = {
                    #     'assignment_type': naming_result.get('assignment_type'),
                    #     'position_in_sequence': naming_result.get('position'),
                    #     'total_expressions': naming_result.get('total_count')
                    # }

            break
        
    assert hasattr(res, 'loc')
    return res


#pylint: disable=too-many-instance-attributes
class SysBuilder:
    '''The class serves as both the system and the IR builder.'''

    name: str  # Name of the system
    modules: typing.List[Module]  # List of modules
    downstreams: list  # List of downstream modules
    arrays: typing.List[Array]  # List of arrays
    _ctx_stack: dict  # Stack for context tracking
    _exposes: dict  # Dictionary of exposed nodes

    @property
    def current_module(self):
        '''Get the current module being built.'''
        return None if not self._ctx_stack['module'] else self._ctx_stack['module'][-1]

    @property
    def current_block(self):
        '''Get the current block being built.'''
        return None if not self._ctx_stack['block'] else self._ctx_stack['block'][-1]

    @property
    def insert_point(self):
        '''Get the insert point.'''
        return self.current_block.body

    def enter_context_of(self, ty, entry):
        '''Enter the context of the given type.'''
        #pylint: disable=import-outside-toplevel
        from .ir.block import CondBlock
        if isinstance(entry, CondBlock):
            self.current_module.add_external(entry.cond)
        self._ctx_stack[ty].append(entry)

    def exit_context_of(self, ty):
        '''Exit the context of the given type.'''
        self._ctx_stack[ty].pop()

    def has_driver(self):
        '''Check if the system has a driver module.'''
        for i in self.modules:
            if i.__class__.__name__ == 'Driver':
                return True
        return False

    def has_module(self, name):
        '''Check if a module with the given name exists.'''
        for i in self.modules:
            if i.name == name:
                return i
        return None

    def __init__(self, name):
        self.name = name
        self.modules = []
        self.downstreams = []
        self.arrays = []
        self._ctx_stack = {'module': [], 'block': []}
        self._exposes = {}

    def expose_on_top(self, node, kind=None):
        '''Expose the given node in the top function with the given kind.'''
        self._exposes[node] = kind

    @property
    def exposed_nodes(self):
        '''Get the exposed nodes.'''
        return self._exposes

    def __enter__(self):
        '''Designate the scope of this system builder.'''
        assert Singleton.builder is None
        Singleton.builder = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        '''Leave the scope of this system builder.'''
        assert Singleton.builder is self
        Singleton.builder = None

    def __repr__(self):
        body = '\n\n'.join(map(repr, self.modules))
        body = body + '\n\n' + '\n\n'.join(map(repr, self.downstreams))
        array = '  ' + '\n  '.join(repr(elem) for elem in self.arrays)
        return f'system {self.name} {{\n{array}\n\n{body}\n}}'

class Singleton(type):
    '''The class maintains the global singleton instance of the system builder.'''
    builder: SysBuilder = None  # Global singleton instance of the system builder
    repr_ident: int = None  # Indentation level for string representation
    id_slice: slice = slice(-6, -1)  # Slice for identifiers
    with_py_loc: bool = False  # Whether to include Python location in string representation
    all_dirs_to_exclude: list = []  # Directories to exclude for stack inspection

    
    @classmethod
    def initialize_dirs_to_exclude(mcs):
        '''Initialize the directories to exclude if not already initialized.'''
        if not mcs.all_dirs_to_exclude:
            site_package_dirs = site.getsitepackages()
            user_site_package_dir = site.getusersitepackages()
            mcs.all_dirs_to_exclude = site_package_dirs + [user_site_package_dir]
