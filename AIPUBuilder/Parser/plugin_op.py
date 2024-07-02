# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


from .logger import *
import copy
from collections import OrderedDict


class ParserOpNotImplement(Exception):
    pass


class ParserOp(object):
    # a string type of Op type
    op_type = None

    # the alias_op_type write to op_type in IR if you specified it, a string
    alias_op_type = None
    # the plugin use priority
    priority = 0
    '''
    the named subgraph is define by start_nodes and end_nodes,
    the defined subgraph is combined by all nodes which in any of path from any of start_nodes to any of end_nodes.
    '''
    start_nodes = None
    end_nodes = None

    pattern_nodes = None
    pattern_edges = None
    '''
    input_nodes and input_shapes are used when preprocess custom op is needed.
    input_nodes is a list of input node names, and input_shapes is a list of shapes of the preprocess inputs.
    The length of input_nodes should be same as input_shapes.
    '''
    input_nodes = None
    input_shapes = None

    @classmethod
    def _check_(cls):
        '''
        the function would check the field is correctly be set.
        '''
        subgraph_type = {
            'named_subgraph': [cls.start_nodes, cls.end_nodes],
            'pattern_subgraph': [cls.pattern_nodes, cls.pattern_edges]
        }
        is_subgraph = []
        for name, sg in subgraph_type.items():
            is_sg = all([i is not None for i in sg])
            is_subgraph.append(is_sg)
            if any([i is not None for i in sg]) and \
                    not is_sg:
                WARN_EXCEPTION('plugin %s defines a wrong %s, where some necessary fields are None.' % (
                    cls.__name__, name))
                return False
            if is_sg:
                cls._subgraph_type = name
        if is_subgraph.count(True) > 1:
            WARN_EXCEPTION('plugin %s defines over 1 subgraph type, we only accept one subgraph type, this plugin would be disabled.' % (
                cls.__name__))
            return False
        if cls.input_nodes is not None:
            if cls.input_shapes is None:
                WARN_EXCEPTION('input_shapes could not be None for plugin %s.' % (cls.__name__))
                return False
            if not isinstance(cls.input_nodes, (list, tuple)) or not isinstance(cls.input_shapes, (list, tuple)):
                WARN_EXCEPTION('input_nodes and input_shapes should be a list or tuple for plugin %s.' % (cls.__name__))
                return False
            if len(cls.input_nodes) != len(cls.input_shapes):
                WARN_EXCEPTION('the number of input_nodes and input_shapes should be same for plugin %s.' %
                               (cls.__name__))
                return False
        return True

    def __init__(self, framework, op_infos):
        '''
        the framework is a string type, possible value is :
            * TensorFlow
            * TFLite
            * ONNX
            * Caffe

        if the plugin is a simple op plugin:
            The op_infos is a dict type to store all attributes,parameters in models'pb file.

        if the plugin is a Subgraph op, the op_infos is a dict. 
            if the plugin is a named subgraph op:
                the key is nodes'name, and value is a dict of all attributes in original model.
            if the plugin is a type-pattern subgraph:
                the key is integer from 0 to n that is the index of pattern, and value is a dict of all attributes in original model.
        '''
        # for hyper
        self.params = OrderedDict()
        # if 'name' in op_infos:
        #   self.params.update({'name': op_infos['name']})
        # for weights: key is str, value is numpy.array
        self.constants = OrderedDict()
        # for removing input tensors(edges). Its element is a tuple: (node_index, tensor_index).
        self._inputs_to_remove = []

    def remove_inputs_at(self, tensor_index, node_index=0):
        '''
        remove input tensors at the specific tensor index and node index(for subgraph);
        the specific input tensor is removed after infer_shape is called;
        call this function with different args if there are multiple input tensors to remove.
        '''
        assert isinstance(tensor_index, int), \
            'Expect tensor_index to be a integer, but got %s' % type(tensor_index)
        assert isinstance(node_index, int), \
            'Expect node_index to be a integer, but got %s' % type(node_index)
        remove_inputs = (node_index, tensor_index)
        self._inputs_to_remove.append(remove_inputs)

    def infer_shape(self, input_tensors, *args):
        '''
        inference the output shape according to the input shape(s)
        return the output shape(s), a list of shape
        '''
        raise ParserOpNotImplement(
            'Plugin Node (' + self.params.get('name', '') + ') infer_shape not implemented!')


class ParserSubgraphNotImplement(Exception):
    pass


class ParserSubgraph(object):
    op_type = 'Subgraph'
    alias = None

    def __init__(self, framework, params):
        self.framework = framework.upper()
        self.params = copy.deepcopy(params)

    def replacing_nodes(self):
        raise ParserSubgraphNotImplement('')
