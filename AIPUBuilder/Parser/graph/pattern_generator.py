# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2023 Arm Technology (China) Co. Ltd.


import re
import ast
import typing
import copy
from enum import Enum
from numpy.random import ranf
from itertools import permutations, combinations, product
from ..logger import INFO, DEBUG, WARN, ERROR, FATAL
from ..common.defs import FLOAT_EQUAL
from .pattern_match import matched_patterns


class TokenType(str, Enum):
    NAME = 'NAME'
    CONSTANT = 'CONST'


class Node:
    def __init__(self, token_type, inputs=[], name=None, expr=None):
        super().__init__()
        self.token_type = token_type
        self.inputs = inputs
        self.name = name
        self.expr = expr

    def serialize(self):
        ''' TODO: Convert its expr to original format.
        '''
        pass

    def show_all_info(self):
        '''Show node's info, including token type, inputs, name and expr. This function is for debug.
        '''
        DEBUG('Token Type | Inputs | Name | Expr')
        input_names = [inp.name for inp in self.inputs]
        DEBUG(' | '.join([self.token_type, str(input_names), self.name, self.expr]))

    @staticmethod
    def eval_expression(token_type: str, inputs) -> str:
        '''Return expr constructed by token_type and inputs. If token_type is +-*/, calculate its output
        if all the inputs are const; return non-const input if token_type is */ and another input is 1;
        if token_type is Custom, calculate its output use a special formula and return the output in the
        format of string.
        '''
        if not isinstance(inputs, list):
            inputs = [inputs]
        ret = token_type + '(' + ','.join(inputs) + ')'
        token_map = {'Mult': '*', 'Add': '+', 'Div': '/', 'Sub': '-'}
        if token_type not in token_map and token_type != 'Custom':
            return ret
        input_values = []
        for inp in inputs:
            assert isinstance(inp, str), 'input %s of %s is not str, but %s' % (str(inp), token_type, type(inp))
            input_values.append(str2float(inp))
        if token_type in token_map:
            if all(value is not None for value in input_values):
                result = eval(token_map[token_type].join(inputs))
                return str(result)
            if token_type == 'Mult':
                if input_values[0] is not None and FLOAT_EQUAL(input_values[0], 1):
                    return inputs[1]
                if input_values[1] is not None and FLOAT_EQUAL(input_values[1], 1):
                    return inputs[0]
            if token_type == 'Div' and input_values[1] is not None and FLOAT_EQUAL(input_values[1], 1):
                return inputs[0]
        elif token_type == 'Custom':
            if all(value is not None for value in input_values):
                result = eval('sum([((idx*25.+10.)*val) for idx, val in enumerate(input_values)])')
                return str(result)
        return ret

    def calculate(self, inputs_map={}, op_replace_to='Custom') -> str:
        '''Get inputs value from inputs_map, replace unknown ops to Custom op, and call eval_expression
        to get the output(a numeric string).
        '''
        token_type = self.token_type
        if token_type == TokenType.CONSTANT:
            return self.expr
        if token_type == TokenType.NAME:
            if self.name not in inputs_map:
                value = float(ranf([]))
                DEBUG('Use ramdom input for name %s: %s' % (self.name, value))
                inputs_map.update({self.name: value})
            self.token_type = TokenType.CONSTANT
            self.expr = str(inputs_map[self.name])
            return self.expr
        if len(self.inputs) > 0 and token_type not in ('Mult', 'Add', 'Div', 'Sub'):
            self.token_type = op_replace_to
        inputs_expr = []
        for obj in self.inputs:
            obj.expr = obj.calculate(inputs_map, op_replace_to)
            obj.inputs = []
            obj.token_type = TokenType.CONSTANT
            inputs_expr.append(obj.expr)
        return Node.eval_expression(self.token_type, inputs_expr)

    def match_with_target(self, target_node, strict_mode, vars_map={}):
        '''
        '''
        if target_node.token_type == TokenType.NAME:
            if target_node.expr in vars_map:
                return vars_map[target_node.expr] == self.expr
            if strict_mode and self.expr in [val for val in vars_map.values()]:
                return False
            vars_map[target_node.expr] = self.expr
            return True
        if self.token_type != target_node.token_type:
            return False
        if len(self.inputs) != len(target_node.inputs):
            return False
        if target_node.token_type == TokenType.CONSTANT:
            return target_node.expr == self.expr
        num = len(target_node.inputs)
        if target_node.token_type in ['Mult', 'Add']:
            for idx in permutations(list(range(num))):
                matched = True
                new_vars_map = copy.copy(vars_map)
                for i in range(num):
                    if not self.inputs[idx[i]].match_with_target(target_node.inputs[i], strict_mode, new_vars_map):
                        matched = False
                        break
                    matched = True
                if matched:
                    vars_map.update(new_vars_map)
                    return True
            return False

        new_vars_map = copy.copy(vars_map)
        for i in range(num):
            if not self.inputs[i].match_with_target(target_node.inputs[i], strict_mode, new_vars_map):
                return False
        vars_map.update(new_vars_map)
        return True

    def get_variant(self, vars_map):
        '''Replace the variables in current node's expr according to vars_map and call eval_expression
        to get a new expr.
        '''
        if self.token_type == TokenType.NAME:
            assert self.name in vars_map, 'Cannot find var %s from vars_map!' % self.name
            return vars_map[self.name]
        if len(self.inputs) == 0:
            return self.expr
        new_vars = []
        for inp in self.inputs:
            new_vars.append(inp.get_variant(vars_map))
        return Node.eval_expression(self.token_type, new_vars)

    def match_rules(self, rules, strict_mode):
        '''Convert self to other expressions(variants) according to rules(format: {match_expr : target_expr}).
        '''
        variants = []
        for match_expr, target_expr in rules:
            matched_vars_map = {}
            matched = self.match_with_target(parse_expression(match_expr), strict_mode, matched_vars_map)
            if not matched:
                matched_vars_map.clear()
                matched = self.match_with_target(parse_expression(target_expr), strict_mode, matched_vars_map)
                if not matched:
                    continue
                target_expr = match_expr
            target_node = parse_expression(target_expr)
            var = target_node.get_variant(matched_vars_map)
            # if var:
            #     DEBUG('Match: match_expr -> %s, target_expr -> %s, result -> %s' % (match_expr, target_expr, var))
            variants.append(var)
        return variants


class Graph(ast.NodeVisitor):
    def __init__(self, expression: str):
        super().__init__()
        self.expression = expression
        self.nodes = dict()  # key is node name, value is node object
        tree = ast.parse(self.expression)
        # print(ast.dump(tree))
        self.root = self.visit(tree)

    def show_all_info(self):
        '''For each node in the graph, show their information. This function is used for debug.
        '''
        for node_name, node in self.nodes.items():
            DEBUG('------------- %s -------------' % node_name)
            node.show_all_info()

    def add_node(self, node):
        '''Add node to graph and return the node object. If node name is not set, set its name based on its token_type and
        the number of nodes in graph.
        '''
        node_name = node.name
        if not node_name:
            node_name = node.token_type + '_' + \
                str(len([val for val in self.nodes.values() if val.token_type == node.token_type]))
            node.name = node_name
        if node_name not in self.nodes:
            self.nodes[node_name] = node
        return node

    def update_node_name(self, node, new_name):
        '''Update node name to a new one and update it in graph.nodes.
        '''
        if node.name in self.nodes:
            self.nodes.pop(node.name)
        node.name = new_name
        self.nodes[new_name] = node

    def remove_node(self, node):
        '''Remove node rom graph.nodes if it's existed in graph.
        '''
        node_name = node.name
        if node_name in self.nodes:
            self.nodes.pop(node_name)
        return node

    def visit_Constant(self, node):
        node = Node(TokenType.CONSTANT, expr=str(node.value))
        self.add_node(node)
        return node

    def visit_Name(self, node):
        if node.id in self.nodes:
            return self.nodes[node.id]
        node = Node(TokenType.NAME, name=node.id, expr=node.id)
        self.add_node(node)
        return node

    def visit_UnaryOp(self, node):
        token_type = type(node.op).__name__
        input_node = self.visit(node.operand)
        assert input_node.expr is not None, 'expr of input node for %s cannot be None!' % token_type
        node_expr = Node.eval_expression(token_type, input_node.expr)
        node = Node(token_type, inputs=[input_node], expr=node_expr)
        self.add_node(node)
        return node

    def visit_BinOp(self, node):
        token_type = type(node.op).__name__
        left_node = self.visit(node.left)
        assert left_node.expr is not None, 'expr of left input node for %s cannot be None!' % token_type
        right_node = self.visit(node.right)
        assert right_node.expr is not None, 'expr of right input node for %s cannot be None!' % token_type
        node_expr = Node.eval_expression(token_type, [left_node.expr, right_node.expr])
        node = Node(token_type, inputs=[left_node, right_node], expr=node_expr)
        self.add_node(node)
        return node

    def visit_BoolOp(self, node):
        token_type = type(node.op).__name__
        inputs = []
        inputs_expr = []
        for value in node.values:
            input_node = self.visit(value)
            assert input_node.expr is not None, 'expr of input node for %s cannot be None!' % token_type
            inputs.append(input_node)
            inputs_expr.append(input_node.expr)
        node_expr = Node.eval_expression(token_type, inputs_expr)
        node = Node(token_type, inputs=inputs, expr=node_expr)
        self.add_node(node)
        return node

    def visit_Call(self, node):
        func_node = self.visit(node.func)
        func_name = func_node.expr
        inputs = []
        inputs_expr = []
        for arg in node.args:
            input_node = self.visit(arg)
            assert input_node.expr is not None, 'expr of input node for %s cannot be None!' % func_name
            inputs.append(input_node)
            inputs_expr.append(input_node.expr)
        node_expr = Node.eval_expression(func_name, inputs_expr)
        self.remove_node(func_node)
        node = Node(func_name, inputs=inputs, expr=node_expr)
        self.add_node(node)
        return node

    def visit_Assign(self, node):
        assert len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Name), 'only 1 named target of ast.Assign is supported!'
        var_name = node.targets[0].id
        assert var_name is not None, 'target expr of ast.Assign cannot be None!'
        input_node = self.visit(node.value)
        assert input_node.expr is not None, 'expr of input node for %s cannot be None!' % token_type
        self.update_node_name(input_node, var_name)
        return input_node

    def visit_Module(self, node):
        token_type = type(node).__name__
        assert len(node.body) >= 1, 'empty body of ast.Module is not supported!'
        nodes = []
        for body_node in node.body:
            nodes.append(self.visit(body_node))
        return nodes[-1]

    def visit_Expr(self, node):
        return self.visit(node.value)

    def generic_visit(self, node):
        token_type = type(node).__name__
        WARN('[Parser]: ast type %s is not yet supported!' % token_type)
        node = Node(token_type)
        self.add_node(node)
        return node


def str2float(value_str: str):
    '''Convert numeric string to float, for instance, convert '-4.0' to -4.0, '4e-1' to 0.4.
    For non numeric string, return None(for 'rt4' None will be returned).
    '''
    try:
        value = float(value_str)
    except:
        value = None
    return value


def cal_expression(expression: str, inputs_map={}) -> str:
    '''Eval the output for expression and return a numeric string. If inputs_map is empty,
    random scalar will be used for the evalation. This function is for debug.
    '''
    graph = Graph(expression)
    root_node = graph.root
    root_expr = root_node.calculate(inputs_map)
    # DEBUG('root: ', root_node.name, ' , result: ', root_expr)
    return root_expr


def check_variants(node_variants):
    '''Calculate each variant and compare their outputs. Report error and return false if outputs
    are not numeric or different.
    '''
    ret = True
    if len(node_variants) < 1:
        return ret
    inputs_map = {}  # use random values
    res = cal_expression(node_variants[0], inputs_map)
    exp_res = str2float(res)
    assert exp_res is not None, 'Expect result of %s to be float, but got %s!' % (node_variants[0], res)
    for var in node_variants[1:]:
        res = str2float(cal_expression(var, inputs_map))
        assert res is not None, 'result of %s is not float!' % var
        if not FLOAT_EQUAL(exp_res, res):
            ret = False
            ERROR('[Parser]: Different results for expression %s and %s: %s, %s' %
                  (node_variants[0], var, exp_res, res))
    return ret


def parse_expression(expression: str):
    '''Construct graph for expression and return its root node.
    '''
    return Graph(expression).root


def apply_rules(node):
    '''Apply rules for node and return all the variants.
    '''
    # Vars in common rules could be different or same
    common_rules = [  # ('x*x', 'Pow(x,2)'),  # TODO: cannot check whether the second input of Pow is 2 in pattern match
        # ('x+x', 'x*2'),
        ('x/y', 'x*Reciprocal(y)'),
    ]
    # All the vars in strict rules need to be different
    strict_rules = [('x*(a+b)', 'x*a+x*b'),  # x*(a+b)=x*a+x*b
                    ('x*(x+a)', 'x*x+x*a'),
                    # ('x*(x+a)', 'Pow(x,2)+x*a'),
                    ('x*(y*z)', 'y*(x*z)'),  # x*(y*z)=y*(x*z)
                    ('x*(y*z)', 'z*(x*y)'),
                    ('x*(x*z)', 'z*(x*x)'),
                    # ('x*(x*z)', 'z*Pow(x,2)'),
                    ('x+(a+b)', 'x+a+b'),  # x+(a+b)=x+a+b
                    ('x+(a+b)', 'x+b+a'),
                    ('x+(x+b)', 'x+x+b'),
                    # ('x+(x+b)', 'x*2+b'),
                    ]
    variants = node.match_rules(common_rules, False)
    variants.extend(node.match_rules(strict_rules, True))
    return variants


def transform(node):
    '''Simplify node and transform its input nodes recursively to get all the variants.
    TODO: Consider opposite solution. Get expressions in model graph and only match once.
    https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html
    '''
    if not node.inputs:
        return [node.expr]
    if all(inp.token_type == TokenType.CONSTANT for inp in node.inputs):
        inputs = [inp.expr for inp in node.inputs]
        return [Node.eval_expression(node.token_type, inputs)]
    # transform its child nodes to get more variants
    child_variants = []
    for inp in node.inputs:
        possible_variants = transform(inp)
        child_variants.append(possible_variants)
    node_variants = []
    for variant in product(*child_variants):
        variant_expr = Node.eval_expression(node.token_type, list(variant))
        node_variants.append(variant_expr)
        variant_node = parse_expression(variant_expr)
        node_variants.extend(apply_rules(variant_node))
    # DEBUG('---- %s ----' % node.expr)
    # DEBUG('%s' % str(node_variants))
    # DEBUG('-----------------------')
    assert len(node_variants) > 0, 'At least one expression should be got!'
    # check_variants(node_variants)  # for debug
    return node_variants


def get_node_patterns(node, cur_roots_name):
    '''Generate pattern for node. If node's token type is NAME, only add it to node_info
    if it's not already created(not in cur_roots_name).
    '''
    node_info = None
    if node.token_type == TokenType.CONSTANT:
        node_info = (node.name, {'op': 'Constant', 'unique': False})
    elif node.token_type == TokenType.NAME:
        if node.name not in cur_roots_name:
            node_info = (node.name, {'unique': False})
            cur_roots_name.append(node.name)
        # else: node already exists in graph
    else:
        op_type = 'Mul' if node.token_type == 'Mult' else node.token_type
        node_info = (node.name, {'op': op_type, 'unique': False})
    edges = []
    if node.token_type in ('Mult', 'Add'):
        for inp in node.inputs:
            edges.append((inp.name, node.name))
    else:
        for idx, inp in enumerate(node.inputs):
            edges.append((inp.name, node.name, {'dst_in_port': idx}))
    return node_info, edges


def match_expression(graph, variants):
    '''Generate patterns for all the variants and use matched_patterns to match all the
    patterns. Return all the matched outputs in a list.
    '''
    all_matches = []
    for expressions in product(*variants):
        nodes, edges = [], []
        cur_roots_name = []
        for expression in expressions:
            g = Graph(expression)
            for node in g.nodes.values():
                _node, _edges = get_node_patterns(node, cur_roots_name)
                if _node is not None:
                    nodes.append(_node)
                if len(_edges) > 0:
                    edges.extend(_edges)
            cur_roots_name.append(g.root.name)
        # DEBUG('nodes: %s' % str(nodes))
        # DEBUG('edges: %s' % str(edges))
        matches = matched_patterns(graph, nodes, edges)
        if matches:
            DEBUG('Expression %s is matched!' % str(expressions))
            all_matches.extend(matches)
    return all_matches


def match_patterns_from_expression(graph, expression: str):
    '''The entry of matching patterns from expression.
    '''
    root_node = None
    variants = []
    expr_list = [expr.strip() for expr in expression.split(';')]
    for idx, single_expr in enumerate(expr_list):
        # step 1: parse expression and get info
        root_node = parse_expression(single_expr)
        # root_node.show_all_info()
        # step 2: transform
        variants.append([(root_node.name + ' = ' + var) for var in transform(root_node)])
    if root_node is None:
        return []

    # step 3: convert to patterns(nodes, edges) and match
    matches = match_expression(graph, variants)
    return matches
