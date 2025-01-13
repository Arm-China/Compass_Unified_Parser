# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import numpy as np
import onnx
from onnx import TensorProto, helper
from utils.run import run_parser


def create_resize_model(onnx_path, input_shape, output_shape, axes, keep_aspect_ratio_policy,
                        mode, coordinate_transformation_mode,
                        antialias, version=19):
    ''' Create onnx model for resize op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_shape)

    input_nodes = []
    tensor_names = []
    for input_name in ('roi', 'scales', 'sizes'):
        if input_name == 'sizes':
            tensor_value_shape = [len(axes)]
            data_type = TensorProto.INT64
            tensor_value = [output_shape[axis] for axis in axes]
        else:
            tensor_value_shape = [0]
            data_type = TensorProto.FLOAT
            tensor_value = []
        const_tensor = helper.make_tensor_value_info(input_name, data_type, tensor_value_shape)
        const_node = helper.make_node(
            'Constant',
            [],
            [input_name],
            value=helper.make_tensor(
                name=input_name + '_value',
                data_type=data_type,
                dims=tensor_value_shape,
                vals=tensor_value,
            )
        )
        input_nodes.append(const_node)
        tensor_names.append(input_name)
    resize = helper.make_node(
        'Resize', ['X'] + tensor_names,
        ['Y'],
        mode=mode,
        axes=axes,
        keep_aspect_ratio_policy=keep_aspect_ratio_policy,
        coordinate_transformation_mode=coordinate_transformation_mode,
        antialias=antialias,
    )
    graph_def = helper.make_graph(
        input_nodes + [resize],  # nodes
        'resize-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name='resize-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


TEST_NAME = 'resize'
input_shapes = [[3, 4, 5, 16], ]  # [3, 4, 11, 12],
output_shapes = [[3, 4, 16, 11], ]  # [3, 4, 5, 6],

feed_dict = {}
# failed_tests = []
for input_shape, output_shape in zip(input_shapes, output_shapes):
    # Generate input data
    feed_dict['X'] = (np.random.randint(-100, 200, input_shape).astype(np.float32) * 1e-1)
    # np.save('input', feed_dict)
    for axes in ([2, 3], [-1], ):
        for keep_aspect_ratio_policy in ('stretch', 'not_larger', 'not_smaller', ):
            if axes == [-1]:
                if keep_aspect_ratio_policy == 'stretch':
                    exp_output_shape = input_shape[:-1] + [output_shape[-1]]
                else:
                    exp_output_shape = [3, 4, 5, 11]
            else:
                if keep_aspect_ratio_policy == 'stretch':
                    exp_output_shape = output_shape
                elif keep_aspect_ratio_policy == 'not_larger':
                    exp_output_shape = [3, 4, 3, 11]
                else:
                    exp_output_shape = [3, 4, 16, 51]
            exp_shape_in_nhwc = [exp_output_shape[0]] + exp_output_shape[2:] + [exp_output_shape[1]]
            exp_shape_str = str(exp_shape_in_nhwc).replace(' ', '')
            for antialias in (True, False, ):
                for mode in ('linear', 'nearest', ):  # 'cubic'
                    if mode == 'nearest' and antialias:
                        continue
                    for coordinate_transformation_mode in ('align_corners', 'half_pixel', 'half_pixel_symmetric',
                                                           'pytorch_half_pixel', 'asymmetric'):
                        # if mode == 'cubic' and coordinate_transformation_mode == 'align_corners':
                        #     continue
                        model_name = '-'.join([TEST_NAME, str(len(axes)), mode, coordinate_transformation_mode,
                                               str(keep_aspect_ratio_policy), str(antialias)])

                        onnx_model_path = model_name + '.onnx'
                        # Create onnx model
                        create_resize_model(onnx_model_path, input_shape, output_shape,
                                            axes, keep_aspect_ratio_policy,
                                            mode, coordinate_transformation_mode,
                                            antialias)

                        # FIXME: onnxruntime fails for this case. Disable verify before it's fixed.
                        # verify = False if len(axes) == 1 else True
                        # FIXME: Enable verify after opt fixes the similarity issue
                        exit_status = run_parser(
                            onnx_model_path, feed_dict, model_type='onnx', verify=False,
                            expected_keywords=['layer_type=Resize',
                                               'method=' + ('BILINEAR' if mode == 'linear' else 'NEAREST'),
                                               'size=' + exp_shape_str,
                                               'mode=' + coordinate_transformation_mode.upper(),
                                               'antialias=' + str(antialias).lower(),
                                               'exclude_outside=false'])
                        assert exit_status
                        # if exit_status:
                        #     print('%s is passed!' % model_name)
                        # else:
                        #     print('%s is failed!' % model_name)
                        # failed_tests.append(model_name)

# if len(failed_tests) > 0:
#     print('Failed tests: %s' % str(failed_tests))
