# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from sympy.printing import StrPrinter
from sympy.printing.precedence import precedence


class CompassPrinter(StrPrinter):
    def _print_Pow(self, expr):
        base = self._print(expr.base)
        exp = self._print(expr.exp)
        return f'({base}**{exp})'

    def _print_floor(self, expr):
        expr_str = str(expr.args[0]).replace("/", "//")
        return f'({expr_str})'

    def _print_ceiling(self, expr):
        expr_str = str(expr.args[0])
        return f'ceil({expr_str})'

    def _print_Add(self, expr, order=None):
        terms = self._as_ordered_terms(expr, order=order)

        prec = precedence(expr)
        l = []
        for term in terms:
            t = self._print(term)
            if t.startswith('-') and not term.is_Add:
                last = l.pop(-1)
                l.append(f'({last}{t})')
                continue
            else:
                sign = '+'
            if precedence(term) < prec or term.is_Add:
                l.extend([sign, '(%s)' % t])
            else:
                l.extend([sign, t])
        sign = l.pop(0)
        if sign == '+':
            sign = ''
        return '(' + sign + ''.join(l) + ')'

    def _print_Mul(self, expr):
        mul_str = super()._print_Mul(expr)
        if mul_str[0] == '-':
            if '*' in mul_str or '/' in mul_str:
                if len(mul_str) >= 2 and mul_str[1] == '(' and mul_str[-1] == ')':
                    return mul_str
                else:
                    return f'-({mul_str[1:]})'
            else:
                return mul_str
        else:
            return f'({mul_str})'

    def _print_Symbol(self, expr):
        return str(expr.name)

    def _print_Integer(self, expr):
        return str(expr.p)

    def _print_Max(self, expr):
        terms = [self._print(term) for term in expr.args]
        return 'max(' + ','.join(terms) + ')'

    def _print_Min(self, expr):
        terms = [self._print(term) for term in expr.args]
        return 'min(' + ','.join(terms) + ')'


def compass_str_expr(expr):
    return CompassPrinter().doprint(expr)
