# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from ..tf.load import convert_tf_to_graph


def convert_tf2_to_graph(graph, model_path, params):
    '''Reuse convert_tf_to_graph from tf.load for now.'''

    return convert_tf_to_graph(graph, model_path, params)
