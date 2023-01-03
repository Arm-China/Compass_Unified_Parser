# Copyright © 2022 Arm Technology (China) Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from ..tf.load import convert_tf_to_graph


def convert_tf2_to_graph(model_path, params):
    '''Reuse convert_tf_to_graph from tf.load for now.'''

    return convert_tf_to_graph(model_path, params)
