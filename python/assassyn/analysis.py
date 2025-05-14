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