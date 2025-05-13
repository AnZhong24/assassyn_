'''The base class for the module definition.'''

from __future__ import annotations
import typing

from ..utils import identifierize
from ..builder import ir_builder
from ..expr import PureIntrinsic, Operand, Expr


# pylint: disable=too-few-public-methods, cyclic-import
class ModuleBase:
    '''The base class for the module definition.'''
    # Base class with no attributes of its own - attributes are added by derived classes

    _externals: typing.Dict[Expr, typing.List[Operand]] # External usage of this module

    def __init__(self):
        self._externals = {}

    def as_operand(self):
        '''Dump the module as a right-hand side reference.'''
        return f'_{identifierize(self)}'

    @ir_builder
    def triggered(self):
        '''The frontend API for creating a triggered node,
        which checks if this module is triggered this cycle.
        NOTE: This operation is only usable in downstream modules.'''
        return PureIntrinsic(PureIntrinsic.MODULE_TRIGGERED, self)

    @property
    def externals(self):
        '''Expose the external interfaces of this module.'''
        return self._externals

    def add_external(self, operand: Operand):
        '''Add an external operand to this module.'''
        # pylint: disable=import-outside-toplevel
        from .module import Module
        from ..array import Array
        is_external = False
        if isinstance(operand, Operand):
            value = operand.value
            if isinstance(value, (Array, Module)):
                is_external = True
            if isinstance(value, Expr):
                is_external = value.parent.module != self
            if is_external:
                if value not in self._externals:
                    self._externals[value] = []
                self._externals[value].append(operand)
