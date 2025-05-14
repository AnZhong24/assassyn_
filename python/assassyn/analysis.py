"""Analysis utilities for Assassyn."""

from collections import defaultdict
from .ir.expr import Expr
from .ir.module import Downstream


def topo_downstream_modules(sys):
    """Analyze the topological order of modules.

    This is a simplified implementation of the topo_sort function in Rust.
    """
    # Get all downstream modules
    downstreams = sys.downstreams[:]

    # Build dependency graph
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for module in downstreams:
        deps = set()
        for elem in module.externals.keys():
            if isinstance(elem, Expr):
                depend = elem.parent.module
                if isinstance(depend, Downstream):
                    deps.add(depend)
        for dep in deps:
            graph[dep].append(module)
            in_degree[module] += 1

    # Topological sort
    queue = [m for m in downstreams if in_degree[m] == 0]
    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result 

def expr_externally_used(expr: Expr) -> typing.Set[Module]:
    """Check if an expression is used outside its module.
    Returns the module uses this expression.
    """

    # Push is NOT a combinational operation
    if isinstance(expr, FIFOPush):
        return False

    this_module = expr.parent.module

    res = set()

    # Check if any user is in a different module
    for user in expr.users:
        user_parent_module = user.user.parent.module
        if user_parent_module != this_module:
            res.add(user_parent_module)

    return res
