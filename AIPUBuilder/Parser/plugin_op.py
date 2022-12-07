# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from .logger import *
import copy
from collections import OrderedDict


class ParserOpNotImplement(Exception):
    pass


class ParserOp(object):
    # a string type of Op type
    op_type = None
    layout = None
    alias = None
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
                warn(None, 'plugin %s defines a wrong %s, where some necessary fields are None.' % (
                    cls.__name__, name))
                return False
            if is_sg:
                cls._subgraph_type = name
        if is_subgraph.count(True) > 1:
            warn(None, 'plugin %s defines over 1 subgraph type, we only accept one subgraph type, this plugin would be disabled.' % (
                cls.__name__))
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
        # for weights: str,numpy.array
        self.constants = OrderedDict()

    def infer_shape(self, input_tensors, *args):
        '''
        inference the output shape according to the input shape(s)
        return the output shape(s), a list of shape
        '''
        raise ParserOpNotImplement(
            'Plugin Node (' + self.params.get('name', '') + ') infer_shape not implemented!')

    def nchw_to_nhwc(self):
        '''
        convert current layout to NHWC
        in this function, you need to change some parameters due to the layout conversion, for example:
        the permutes of transpose, axis of concat, etc.
        '''
        raise ParserOpNotImplement('')

    def multi_batch_transform(self, origin_batch_size, new_batch_size):
        raise ParserOpNotImplement('')


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
