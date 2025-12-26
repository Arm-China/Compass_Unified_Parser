# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from sympy.printing import StrPrinter


class CompassPrinter(StrPrinter):
    def _print_Pow(self, expr):
        base = self._print(expr.base)
        exp = self._print(expr.exp)
        return f'({base}**{exp})'

    def _print_floor(self, expr):
        expr_str = str(expr.args[0]).replace("/", "//")
        return f'({expr_str})'

    def _print_Add(self, expr):
        terms = [self._print(term) for term in expr.args]
        return '(' + '+'.join(terms) + ')'

    def _print_Mul(self, expr):
        terms = [self._print(term) for term in expr.args]
        return '(' + '*'.join(terms) + ')'

    def _print_Symbol(self, expr):
        return str(expr.name)

    def _print_Integer(self, expr):
        return str(expr.p)


def compass_str_expr(expr):
    return CompassPrinter().doprint(expr)
