# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from collections import OrderedDict, Iterable
import tensorflow.compat.v1 as tf
import onnx
import os
import itertools
import multiprocessing as mp
import threading
from ...common.defs import Framework, get_opset_version, Tensor
from ...common.utils import is_dir, is_file, get_version, extend_lists
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL
from ...graph.graph import Graph
from ...graph.graph_algo import clear_redundant_nodes, all_simple_paths, has_path, get_valid_node_name
from ...graph.node_wrap import NodeWrap
from .buffer import *
from .buffer_tf2 import parse_keras
from .utils import trim_tensor_name

tf_attr_names_map = {'activation': 'activations',
                     'bias': 'biases',
                     'dilation_rate': 'dilations',
                     'groups': 'group',
                     'keep_dims': 'keepdims',
                     'kernel_size': 'kernel_shape',
                     'ksize': 'kernel_shape',
                     'padding': 'auto_pad',
                     'rates': 'dilations',
                     'reduction_indices': 'axes'
                     }


def attr_to_list(key, value):
    '''Convert value of attr to list and update key if it can be found in tf_attr_names_map.
    '''
    value = [value] if isinstance(value, int) else list(value[:])
    new_k = tf_attr_names_map.get(key, key)
    return {new_k: value}


def attr_to_int(key, value):
    '''Convert value of attr to int type and update key if it can be found in tf_attr_names_map.
    '''
    return {tf_attr_names_map.get(key, key): int(value)}


def attr_rename(key, value):
    '''Rename attr only. Keep the original value.
    '''
    return {tf_attr_names_map.get(key, key): value}


def padding_attr_rename(key, value):
    '''Rename padding attr. Update both key and value.
    '''
    value = value.upper()
    new_attr = {}
    if value == 'VALID':
        new_attr = {tf_attr_names_map.get(key, key): 'VALID'}
    elif value == 'SAME':
        new_attr = {tf_attr_names_map.get(key, key): 'SAME_UPPER'}
    elif value == 'EXPLICIT':
        new_attr = {tf_attr_names_map.get(key, key): 'NOTSET'}
    else:
        WARN(
            '[Parser]: Unsupported TF padding type (%s) in padding_attr_rename' % value)
    return new_attr


def convert_keras_attr_to_onnx(attr_dict):
    new_attr = copy.deepcopy(attr_dict)
    for k, v in attr_dict.items():
        updated_attr = {}
        if k == 'activation' and 'recurrent_activation' not in attr_dict:
            v = 'NONE' if v is None or v == 'linear' else v.upper()
            updated_attr = attr_rename(k, v)
        elif k == 'axis' and isinstance(v, (list, tuple)):
            updated_attr = {k: None, 'axes': list(v[:])}
        elif k in ('bias', 'groups'):
            updated_attr = attr_rename(k, v)
        elif k == 'data_format':
            if v == 'channels_first':
                if 'input_shape' in attr_dict \
                        and isinstance(attr_dict['input_shape'], (list, tuple)) \
                        and len(attr_dict['input_shape']) == 5:
                    data_format = 'NCDHW'
                else:
                    data_format = 'NCHW'
            elif v == 'channels_last':
                if 'input_shape' in attr_dict \
                        and isinstance(attr_dict['input_shape'], (list, tuple)) \
                        and len(attr_dict['input_shape']) == 5:
                    data_format = 'NDHWC'
                else:
                    data_format = 'NHWC'
            else:
                data_format = v
            updated_attr = {k: data_format}
        elif k in ('dilation_rate', 'kernel_size', 'strides'):
            updated_attr = attr_to_list(k, v)
        elif k == 'keep_dims':
            updated_attr = attr_to_int(k, v)
        elif k == 'padding' and isinstance(v, str):
            updated_attr = padding_attr_rename(k, v)
        else:
            continue
        if k not in updated_attr:
            new_attr.pop(k)
        new_attr.update(updated_attr)
    return new_attr


def convert_attr_to_onnx(attr_dict, is_keras_op=False):
    if is_keras_op:
        return convert_keras_attr_to_onnx(attr_dict)

    new_attr = copy.deepcopy(attr_dict)
    for k, v in attr_dict.items():
        updated_attr = {}
        if k in ('ksize', 'reduction_indices', 'strides'):
            updated_attr = attr_to_list(k, v)
        elif k in ('keep_dims', 'parallel_iterations'):
            updated_attr = attr_to_int(k, v)
        elif k == 'padding' and isinstance(v, str):
            updated_attr = padding_attr_rename(k, v)
        elif k == 'rates':
            updated_attr = attr_rename(k, v)
        elif k in ('shape', 'element_shape'):
            updated_attr = {k: v['dim'].tolist()}
        elif k == 'T' and 'dtype' not in attr_dict:
            updated_attr = {'dtype': v}
        else:
            continue
        if k not in updated_attr:
            new_attr.pop(k)
        new_attr.update(updated_attr)
    return new_attr


def parse_pb(graph, model_path, params, anchor_tensor):
    def _remove_unneeded_nodes(graph, nodes):
        if not graph._attr['output_names']:
            return nodes
        simple_graph = copy.deepcopy(graph)
        for n in nodes:
            if n['name'] in graph._attr['input_names']:
                continue
            for in_port, (src_name, src_out_port, is_control) in enumerate(n.get('input', [])):
                if is_control:
                    continue
                simple_graph.add_edge(
                    src_name, n['name'], **{'src_out_port': src_out_port, 'dst_in_port': in_port})
        clear_redundant_nodes(simple_graph)
        reduced_nodes = [n for n in nodes if n['name'] in simple_graph.nodes]
        return reduced_nodes

    if not is_file(model_path):
        FATAL('[Parser]: Invalid pb file %s in parse_pb!' %
              model_path)
    try:
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph_def, params['output_names'])
            default_graph = tf.get_default_graph()
            nodes = list(parse_proto(
                default_graph.get_operations(), get_op_content))
            for func in graph_def.library.function:
                func_name = func.signature.name
                if any((node['type'] == func_name) for node in nodes):
                    nodes = get_function_node_content(func, nodes)

            if anchor_tensor is not None:
                anchor_node = [n for n in nodes if n['name']
                               == trim_tensor_name(anchor_tensor)]
                if not anchor_node:
                    WARN(
                        '[Parser]: Invalid anchor (%s) in convert_tf_to_graph!' % anchor_tensor)
            nodes = _remove_unneeded_nodes(graph, nodes)
            nodes_dict = OrderedDict()
            input_shapes = params['input_shapes'].copy()
            for n in nodes:
                nodes_dict.update({n['name']: n})
                if n['type'] == 'Placeholder':
                    tensor_shape = n['output'][0][1]
                    if all([d is not None for d in tensor_shape]):
                        if n['name'] not in input_shapes:
                            input_shapes.update({n['name']: tensor_shape})
                        elif input_shapes[n['name']] != tensor_shape:
                            WARN('[Parser]: Original model expects input shape of input %s to be %s. '
                                 'Now reset it to %s basing on config file!' % (
                                     n['name'], str(tensor_shape), str(input_shapes[n['name']])))

            tensors, feed_dict = OrderedDict(), OrderedDict()
            for k, v in input_shapes.items():
                if k not in nodes_dict.keys():
                    WARN(
                        '[Parser]: Ignore input (%s) as it does not exist in graph!' % k)
                    params['input_shapes'].pop(k)
                    continue
                tensor_name = k + ':0'
                try:
                    t = default_graph.get_tensor_by_name(tensor_name)
                    np_type = t.dtype.as_numpy_dtype
                except Exception as e:
                    WARN('[Parser]: Meets error when getting input tensor (%s): %s!' % (
                        tensor_name, str(e)))
                    np_type = np.float32
                np_tensor = np.random.randint(0, 1, size=v, dtype=np_type) \
                    if re.search(r'int', str(np_type)) \
                    else np.random.ranf(v).astype(np_type)
                feed_dict.update({tensor_name: np_tensor})

            for n in nodes:
                for i, out in enumerate(n['output']):
                    if n.get('type', '') in ('FusedBatchNorm', 'FusedBatchNormV3') and i > 0:
                        continue
                    elif n.get('type', '') in ('Enter', 'Merge', 'TensorArrayReadV3',):
                        continue
                    elif n.get('from_function', False):
                        continue
                    tensors.update(
                        {out[0]: default_graph.get_tensor_by_name(out[0])})

            if anchor_tensor and anchor_tensor not in tensors:
                tensors.update(
                    {anchor_tensor: default_graph.get_tensor_by_name(anchor_tensor)})

            np_tensors = OrderedDict()
            try:
                np_tensors = sess.run(tensors, feed_dict=feed_dict)
            except:
                def _run_tensor(meta_tensors, sess, feed_dict, ret):
                    for mt in meta_tensors:
                        try:
                            np_res = sess.run(mt[1], feed_dict=feed_dict)
                        except:
                            np_res = None
                        ret.append((mt[0], np_res))

                tensors_list = [(k, v) for k, v in tensors.items()]
                tensors_num = len(tensors_list)
                threads_num = max(1, mp.cpu_count() - 1)
                tensor_num_per_thread = tensors_num // threads_num \
                    if tensors_num % threads_num == 0 \
                    else int(np.ceil(tensors_num / threads_num))
                np_tensors_list = []

                coord = tf.train.Coordinator()
                threads = []
                for thread_idx in range(threads_num):
                    cur_range = slice(thread_idx * tensor_num_per_thread,
                                      min((thread_idx + 1) * tensor_num_per_thread, tensors_num))
                    args = (tensors_list[cur_range],
                            sess, feed_dict, np_tensors_list)
                    t = threading.Thread(target=_run_tensor, args=args)
                    t.start()
                    threads.append(t)
                coord.join(threads)
                np_tensors = {nt[0]: nt[1] for nt in np_tensors_list}
    except Exception as e:
        FATAL('[Parser]: Meet error in parse_pb: %s' % str(e))
    return nodes, nodes_dict, tensors, np_tensors, input_shapes


def convert_tf_to_graph(model_path, params):
    '''Parse the tensorflow model into a graph structure.'''

    from ...plugin_loader import PARSER_OP_DICT
    is_keras_model = model_path.endswith('.h5') or model_path.endswith('.hdf5') or model_path.endswith(
        '.keras') or is_dir(model_path)

    graph = Graph(name=params.get('model_name', ''))
    graph._attr['framework'] = Framework.TENSORFLOW
    graph._attr['output_tensor_names'] = params.get('output_tensor_names', [])
    graph._attr['is_keras_model'] = is_keras_model

    if not is_keras_model:
        if params.get('output_names', []):
            params['output_names'] = list(
                map(trim_tensor_name, params['output_names']))

        if params.get('input_names', []):
            params['input_names'] = list(
                map(trim_tensor_name, params['input_names']))
            params['input_shapes'] = {trim_tensor_name(
                k): v for k, v in params['input_shapes'].items()}

    anchor_tensor = None
    if params.get('anchor_tensor_name') is not None:
        anchor_tensor = params['anchor_tensor_name'] if ':' in params['anchor_tensor_name'] else params[
            'anchor_tensor_name'] + ':0'

    consumer_ver = get_version(onnx)
    if consumer_ver >= 1.04:
        opset_version = get_opset_version(consumer_ver)
        graph._attr['opset_version'] = opset_version
        graph._attr['input_tensors'] = OrderedDict()
        graph._attr['input_names'] = copy.deepcopy(
            params.get('input_names', []))
        graph._attr['output_names'] = copy.deepcopy(
            params.get('output_names', []))

        meta_ret = True

        try:
            if not is_keras_model:
                nodes, nodes_dict, tensors, np_tensors, input_shapes \
                    = parse_pb(graph, model_path, params, anchor_tensor)
            else:
                nodes, nodes_dict, tensors, np_tensors, input_shapes \
                    = parse_keras(model_path, params)
            if anchor_tensor and anchor_tensor in np_tensors:
                graph._attr['anchors'] = np_tensors[anchor_tensor]
            nodes_outputs = {n['name']: n['output'] for n in nodes}
            nodes_inputs = set()
            for n in nodes:
                if is_keras_model:
                    # Update tensor to node name for keras model because node name + ':0'
                    # could be different with tensor name, while in pb model they are same.
                    for out_tensor_name, _ in n['output']:
                        if out_tensor_name == n['name']:
                            continue
                        if out_tensor_name in graph._attr['input_names']:
                            index = graph._attr['input_names'].index(out_tensor_name)
                            graph._attr['input_names'][index] = n['name']
                            if out_tensor_name in params['input_shapes']:
                                params['input_shapes'][n['name']] = params['input_shapes'].pop(out_tensor_name)
                        if out_tensor_name in graph._attr['output_names']:
                            index = graph._attr['output_names'].index(out_tensor_name)
                            graph._attr['output_names'][index] = n['name']
                if n['name'] in graph._attr['input_names']:
                    if n.get('input', []):
                        nodes_inputs.update([src_name for src_name, _, _ in n['input']])
                    continue
                try:
                    for in_port, (src_name, src_out_port, is_control) in enumerate(n.get('input', [])):
                        nodes_inputs.add(src_name)
                        if not is_control:
                            tensor_name, tensor_shape, tensor_value = '', tuple(), None
                            if src_name in nodes_outputs \
                                    and len(nodes_outputs[src_name]) > src_out_port \
                                    and len(nodes_outputs[src_name][src_out_port]) == 2:
                                tensor_name, tensor_shape = nodes_outputs[src_name][src_out_port]
                                if tensor_name in np_tensors:
                                    tensor_value = np_tensors[tensor_name]
                                    if type(tensor_value).__name__ == 'bytes':
                                        tensor_value = np.array(
                                            tensor_value, '<U0')
                                    tensor_shape = tensor_value.shape if tensor_value is not None else list(
                                        tensor_shape)
                                elif tensor_name in tensors:
                                    t = tensors[tensor_name]
                                    tensor_shape = tensor_value.shape if tensor_value is not None else list(
                                        tensor_shape)
                                    if tensor_value is None:
                                        is_valid_shape = True
                                        if is_keras_model:
                                            if src_name in params['input_shapes']:
                                                tensor_shape = params['input_shapes'][src_name]
                                            elif tensor_name in params['input_shapes']:
                                                tensor_shape = params['input_shapes'][tensor_name]
                                                params['input_shapes'][src_name] = \
                                                    params['input_shapes'].pop(tensor_name)
                                            elif t.op.type not in ('KerasInputLayer', 'Placeholder'):
                                                is_valid_shape = False
                                        else:
                                            if trim_tensor_name(tensor_name) in params['input_shapes']:
                                                tensor_shape = params['input_shapes'][trim_tensor_name(tensor_name)]
                                            elif t.op.type != 'Placeholder':
                                                is_valid_shape = False
                                        if all([s is not None for s in tensor_shape[:]]) and is_valid_shape:
                                            tensor_type = t.dtype.name
                                            tensor_value = np.random.randint(0, 1, size=tensor_shape,
                                                                             dtype=np.dtype(tensor_type)) \
                                                if re.search(r'int', str(tensor_type)) \
                                                else np.random.ranf(tensor_shape).astype(np.dtype(tensor_type))
                            edge_tensor = Tensor(
                                name=tensor_name, shape=tensor_shape, value=tensor_value)
                            graph.add_edge(src_name,
                                           n['name'],
                                           **{'src_out_port': src_out_port, 'dst_in_port': in_port,
                                              'tensor': edge_tensor}
                                           )
                            if src_name in params['input_shapes']:
                                graph._attr['input_tensors'].update(
                                    {src_name: edge_tensor})
                        else:
                            DEBUG(
                                '[Parser]: Meets control edge from Node(%s) in convert_tf_to_graph!' % src_name)
                except Exception as e:
                    ERROR(
                        '[Parser]: Meets error (%s) in reading TF nodes (%s) in convert_tf_to_graph!',
                        (str(e), n['name']))

            if not graph._attr['output_names']:
                # Try to find out output nodes
                output_names = [n['name']
                                for n in nodes if n['name'] not in nodes_inputs]
                graph._attr['output_names'] = output_names
            else:
                for out_name in graph._attr['output_names']:
                    if not graph.has_node(out_name):
                        FATAL(
                            '[Parser]: Graph does not contain a output Node(%s) in convert_tf_to_graph.' % out_name)
                        meta_ret = False

            if not graph._attr['output_names']:
                FATAL('[Parser]: Graph does not contain any output Node!')

            if meta_ret:
                unusual_placeholders, unusual_others = [], []
                for n_name in nodes_dict:
                    if graph.has_node(n_name) \
                            and nodes_dict[n_name]['type'] in ('Placeholder', 'KerasInputLayer') \
                            and n_name not in graph._attr['input_tensors']:
                        unusual_placeholders.append(n_name)
                    if graph.has_node(n_name) \
                            and nodes_dict[n_name]['type'] not in ('Placeholder', 'KerasInputLayer') \
                            and n_name in graph._attr['input_tensors']:
                        unusual_others.append(n_name)
                        nodes_dict[n_name]['type'] = 'Placeholder'
                        try:
                            t_shape = nodes_dict[n_name]['output'][0][1]
                            if None in t_shape:
                                try:
                                    t_shape = list(np_tensors[nodes_dict[n_name]
                                                              ['output'][0][0]].shape)
                                except:
                                    t_shape = input_shapes[n_name]
                        except:
                            t_shape = []
                        try:
                            dtype = str(
                                np_tensors[nodes_dict[n_name]['output'][0][0]].dtype)
                        except:
                            dtype = 'float32'
                        nodes_dict[n_name]['attr']['shape'] = {
                            'dim': np.array(t_shape, np.int64), 'unkwown_rank': False}
                        nodes_dict[n_name]['attr']['dtype'] = dtype

                removing_nodes = set()
                for u_p, u_o in itertools.product(unusual_placeholders, unusual_others):
                    if has_path(graph, u_p, u_o):
                        simple_paths = list(
                            all_simple_paths(graph, u_p, u_o))
                        simple_paths = extend_lists(simple_paths)
                        simple_paths_set = set(simple_paths)
                        simple_paths_set = simple_paths_set.difference([
                            u_o])
                        for sp in simple_paths_set:
                            invalid = False
                            for out_name in graph._attr['output_names']:
                                sp_out_paths = list(
                                    all_simple_paths(graph, sp, out_name))
                                for meta_path in sp_out_paths:
                                    if u_o not in meta_path:
                                        invalid = True
                                        break
                                if invalid:
                                    break
                            if not invalid:
                                removing_nodes.update([sp])
                graph.remove_nodes_from(list(removing_nodes))

                for u_o in unusual_others:
                    if graph.has_node(u_o):
                        in_edges = graph.sorted_in_edges(u_o)
                        graph.remove_edges_from(in_edges)

                if not graph._attr['input_tensors'] and unusual_placeholders:
                    found_valid = True
                    for placeholder in unusual_placeholders:
                        try:
                            t_shape = nodes_dict[placeholder]['output'][0][1]
                            if all([d is not None for d in t_shape]):
                                tensor_name = placeholder + ':0'
                                if tensor_name in np_tensors:
                                    tensor_value = np_tensors[tensor_name]
                                else:
                                    dtype = nodes_dict[placeholder]['attr']['dtype']
                                    tensor_value = np.random.randint(0, 1, size=t_shape, dtype=np.dtype(dtype)) \
                                        if re.search(r'int', str(dtype)) \
                                        else np.random.ranf(t_shape).astype(np.dtype(dtype))
                                tensor = Tensor(
                                    value=tensor_value, shape=t_shape, name=tensor_name)
                                graph._attr['input_tensors'].update(
                                    {placeholder: tensor})
                            else:
                                found_valid = False
                                break
                        except:
                            found_valid = False
                            break
                    if found_valid:
                        unusual_placeholders = []
                    else:
                        graph._attr['input_tensors'].clear()

                if not graph._attr['input_tensors']:
                    FATAL('[Parser]: Graph does not contain any input Node!')

                for placeholder in unusual_placeholders:
                    if graph.has_node(placeholder):
                        for out_name in graph._attr['output_names']:
                            if has_path(graph, placeholder, out_name):
                                FATAL('[Parser]: Output_name(%s) has path from invalid Input (%s)! '
                                      'Please check config file.' % (out_name, placeholder))

                for n in graph.nodes:
                    node_type = nodes_dict[n]['type']
                    if node_type not in PARSER_OP_DICT:
                        attr_dict = convert_attr_to_onnx(
                            nodes_dict[n].get('attr', {}), node_type.startswith('Keras'))
                    else:
                        attr_dict = nodes_dict[n].get('attr', {})
                    attr_dict.update({'name': n,
                                      'opcode_version': nodes_dict[n].get('opcode_version', 1)})
                    NodeWrap(graph, n).replace_obj(
                        'Tf' + node_type, attr_dict)

                for out_name in graph._attr['output_names']:
                    for out_port, out_info in enumerate(nodes_dict[out_name].get('output', [])):
                        out_op_name = get_valid_node_name(
                            graph, out_name + '_out_' + str(out_port))
                        try:
                            t_name = out_info[0]
                        except:
                            t_name = out_op_name
                        try:
                            t_value = np_tensors[out_info[0]]
                        except:
                            t_value = None
                        if t_value is not None:
                            t_shape = t_value.shape
                        else:
                            try:
                                t_shape = tuple(out_info[1])
                            except:
                                t_shape = tuple()
                        edge_tensor = Tensor(
                            name=t_name, shape=t_shape, value=t_value)
                        edge_attr = {'src_out_port': out_port,
                                     'dst_in_port': 0, 'tensor': edge_tensor}
                        graph.add_edge(out_name, out_op_name, **edge_attr)
                        NodeWrap(graph, out_op_name).replace_obj(
                            'Out', {'name': out_op_name})

        except Exception as e:
            WARN('[Parser]: Reading pb/saved_model/h5 file (%s) meets error (%s)!' %
                 (model_path, str(e)))
            meta_ret = False

    else:
        WARN('[Parser]: Onnx version is too low, needs updating!')
    return graph
