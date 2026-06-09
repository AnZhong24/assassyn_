"""Shared sizing helpers for the Verilog backend."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ...builder import SysBuilder


def fifo_depth_log2(depth: int) -> int:
    """Return the FIFO primitive DEPTH_LOG2 for a requested entry count."""
    entries = max(1, int(depth))
    if entries <= 1:
        return 0
    return (entries - 1).bit_length()


def fifo_trigger_width(depth: int) -> int:
    """Return trigger-counter width for a requested FIFO entry count."""
    return max(1, fifo_depth_log2(depth) + 1)


def _all_schedulable_modules(sys: SysBuilder) -> list[Any]:
    """Return modules that can own FIFOs."""
    return list(sys.modules) + list(sys.downstreams)


def _initial_fifo_depths(
    modules: list[Any],
    default_fifo_depth: int,
) -> Dict[Any, Dict[Any, int]]:
    """Create default per-port FIFO depth maps."""
    return {
        mod: {port: default_fifo_depth for port in getattr(mod, "ports", [])}
        for mod in modules
    }


def _apply_explicit_fifo_depths(
    modules: list[Any],
    module_metadata: Dict[Any, Any],
    default_fifo_depth: int,
    module_fifo_depths: Dict[Any, Dict[Any, int]],
) -> None:
    """Merge metadata-requested FIFO depths into default maps."""
    for module in modules:
        metadata = module_metadata.get(module)
        if metadata is None:
            continue
        for push in metadata.interactions.pushes:
            fifo_port = push.fifo
            owner = fifo_port.module
            if owner not in module_fifo_depths:
                continue
            depth = push.fifo_depth
            if not isinstance(depth, int) or depth <= 0:
                depth = default_fifo_depth
            current = module_fifo_depths[owner].get(fifo_port, default_fifo_depth)
            module_fifo_depths[owner][fifo_port] = max(current, depth)


def _requested_module_depth(
    module: Any,
    depth_map: Dict[Any, int],
    default_fifo_depth: int,
) -> int:
    """Return the uniform requested FIFO depth for one module."""
    if not depth_map:
        return default_fifo_depth

    depths = list(depth_map.values())
    requested_depth = depths[0]
    if any(depth != requested_depth for depth in depths):
        raise RuntimeError(
            f"Inconsistent FIFO depths for module {module.name}: {depths}"
        )
    return requested_depth


def compute_fifo_sizing(
    sys: SysBuilder,
    module_metadata: Dict[Any, Any],
    default_fifo_depth: int,
) -> Tuple[Dict[Any, Dict[Any, int]], Dict[Any, int]]:
    """Compute per-port requested FIFO depths and per-module trigger widths."""
    modules = _all_schedulable_modules(sys)
    module_fifo_depths = _initial_fifo_depths(modules, default_fifo_depth)
    _apply_explicit_fifo_depths(
        modules,
        module_metadata,
        default_fifo_depth,
        module_fifo_depths,
    )

    module_trigger_widths: Dict[Any, int] = {}
    for module in sys.modules:
        requested_depth = _requested_module_depth(
            module,
            module_fifo_depths.get(module, {}),
            default_fifo_depth,
        )

        # Keep the historical 8-bit trigger datapath for small FIFOs. The narrower
        # version increased mapped area on ASAP7 because it changed the adder/OR
        # structure enough to defeat later sharing.
        module_trigger_widths[module] = max(8, fifo_trigger_width(requested_depth))

    return module_fifo_depths, module_trigger_widths
