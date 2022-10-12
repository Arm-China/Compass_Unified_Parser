import numpy as np
import onnx
# from AIPUBuilder.Parser.tool_utils.run import run_parser
from AIPUBuilder.Parser.tool_utils.forward import onnx_forward
from AIPUBuilder.Parser.tool_utils.compare import compare_data_dict
from onnx import TensorProto, helper


def create_conv_transpose_model(onnx_path, input_size, output_size, weight, version=1,
                                stride_h=1, stride_w=1,
                                outpad_h=0, outpad_w=0,
                                dilation_h=1, dilation_w=1,
                                group=1,
                                pad_top=0,
                                pad_bottom=0,
                                pad_left=0,
                                pad_right=0):
    ''' Create onnx model for conv_transpose op.
    '''
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_size)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_size)

    W_tensor = helper.make_tensor_value_info('W', TensorProto.FLOAT, weight.shape)
    const_W = helper.make_node(
        'Constant',
        [],
        ['W'],
        value=helper.make_tensor(
            name='weight_value',
            data_type=TensorProto.FLOAT,
            dims=weight.shape,
            vals=weight,
        )
    )
    conv_transpose = helper.make_node(
        OP_NAME,
        inputs=['X', 'W'],
        outputs=['Y'],
        kernel_shape=[weight.shape[-2], weight.shape[-1]],
        output_padding=[outpad_h, outpad_w],
        output_shape=[output_size[-2], output_size[-1]],
        strides=[stride_h, stride_w],
    )
    graph_def = helper.make_graph(
        [const_W, conv_transpose],  # nodes, sequences matters
        OP_NAME + '-model',  # name
        [X],  # inputs
        [Y],  # outputs
    )
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model')
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


def run_torch(inp, weight, bias,
              stride_h=1, stride_w=1,
              outpad_h=0, outpad_w=0,
              dilation_h=1, dilation_w=1,
              group=1,
              pad_top=0,
              pad_bottom=0,
              pad_left=0,
              pad_right=0):
    import torch

    # inp = np.pad(inp, [[0, 0], [0, 0], [0, 0], [0, 2]], 'constant', constant_values=0)
    # print('padded_inp shape: %s' % str(inp.shape))
    # inp: NCHW layout
    inp = torch.tensor(inp)
    weight = torch.tensor(weight)
    bias = torch.tensor(bias)
    conv_res = torch.nn.functional.conv_transpose2d(inp,
                                                    weight,
                                                    bias,
                                                    stride=(stride_h, stride_w),
                                                    padding=(0, 0),
                                                    output_padding=(outpad_h, outpad_w),
                                                    groups=group,
                                                    dilation=(dilation_h, dilation_w)
                                                    )
    print('conv_res shape: %s' % str(conv_res.shape))
    h = conv_res.shape[-2]
    w = conv_res.shape[-1]
    crop_torch_res = conv_res[..., pad_top:(h-pad_bottom), pad_left:(w-pad_right)]
    res = crop_torch_res.cpu().detach().numpy()
    print('torch_res shape: %s' % str(res.shape))
    print(res[0, 0, 0, :])  # [0. 1. 3. 2.]
    # padded_res = np.pad(res, [[0, 0], [0, 0], [0, 4], [0, 2]], 'constant', constant_values=0)
    # print('padded_res shape: %s' % str(padded_res.shape))

    res_dict = {'res': res}
    np.save('torch', res_dict)
    return res_dict


OP_NAME = 'ConvTranspose'
input_shape = [1, 1, 2, 3]
# output_shape = [1, 1, 3, 4] # similarity is 1
output_shape = [1, 1, 4, 5]  # torch has different shape, data is different
# output_shape = [1, 1, 5, 6]
weight_shape = [1, 1, 2, 2]

# Generate input data
feed_dict = dict()
input_data = np.reshape(np.arange(np.prod(input_shape), dtype=np.float32), input_shape)
# input_data = np.ones(input_shape).astype(np.float32)
feed_dict['X'] = input_data
input_data_path = 'input.npy'
np.save(input_data_path, feed_dict)

# weight = np.reshape(np.arange(np.prod(weight_shape), dtype=np.float32), weight_shape)
weight = np.ones(weight_shape, dtype=np.float32)
bias = np.zeros(weight_shape[1]).astype(np.float32)

for version in (11, ):  # 1,
    model_name = '-'.join([OP_NAME, str(version)])
    model_path = model_name + '.onnx'
    # Create model
    create_conv_transpose_model(
        model_path, input_shape, output_shape, weight, version)
    onnx_res_dict = onnx_forward(model_path, feed_dict, output_names=None, save_output=True)
    print(onnx_res_dict['Y'][0, 0, 0, :])  # [0. 1. 3. 5. 3.]
    torch_res_dict = run_torch(input_data, weight, bias)
    compare_data_dict(onnx_res_dict, torch_res_dict)
