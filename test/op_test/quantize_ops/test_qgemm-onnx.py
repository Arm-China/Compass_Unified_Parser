import os
import numpy as np
import onnx
from utils.run import run_parser
from onnx import TensorProto, helper


def create_QGemm_model(onnx_path, inp_a_shape, inp_b_shape, inp_c_shape, output_shape,
                       const_BC=False, return_float=False, version=16):
    ''' Create onnx model for QGemm op.
    '''
    A = helper.make_tensor_value_info('A', TensorProto.INT8, inp_a_shape)
    B = helper.make_tensor_value_info('B', TensorProto.INT8, inp_b_shape)
    C = helper.make_tensor_value_info('C', TensorProto.INT32, inp_c_shape)
    if const_BC:
        B = helper.make_node('Constant', [], ['B'], value=helper.make_tensor(
            name='B_value',
            data_type=TensorProto.INT8,
            dims=inp_b_shape,
            vals=np.random.randint(-2, 3, inp_b_shape).astype(np.int8)))
        C = helper.make_node('Constant', [], ['C'], value=helper.make_tensor(
            name='C_value',
            data_type=TensorProto.INT32,
            dims=inp_c_shape,
            vals=np.random.randint(-2, 3, inp_c_shape).astype(np.int32)))
        inputs = [A]
        const_nodes = [B, C]
    else:
        inputs = [A, B, C]
        const_nodes = []
    if return_float:
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_shape)
    else:
        Y = helper.make_tensor_value_info('Y', TensorProto.INT8, output_shape)

    a_scale = helper.make_node('Constant', [], ['a_scale'], value_float=0.5)
    a_zero_point = helper.make_node('Constant', [], ['a_zp'], value_int=3)
    a_cast = helper.make_node('Cast', ['a_zp'], ['a_zp_int8'], to=TensorProto.INT8)
    b_scale = helper.make_node('Constant', [], ['b_scale'], value_float=1.15)
    b_zero_point = helper.make_node('Constant', [], ['b_zp'], value_int=2)
    b_cast = helper.make_node('Cast', ['b_zp'], ['b_zp_int8'], to=TensorProto.INT8)
    input_tensors = ['A', 'a_scale', 'a_zp_int8', 'B', 'b_scale', 'b_zp_int8', 'C']
    nodes = [a_scale, a_zero_point, a_cast, b_scale, b_zero_point, b_cast]
    if not return_float:
        y_scale = helper.make_node('Constant', [], ['y_scale'], value_float=2.05)
        y_zero_point = helper.make_node('Constant', [], ['y_zp'], value_int=1)
        y_cast = helper.make_node('Cast', ['y_zp'], ['y_zp_int8'], to=TensorProto.INT8)
        input_tensors.extend(['y_scale', 'y_zp_int8'])
        nodes.extend([y_scale, y_zero_point, y_cast])

    QGemm = helper.make_node(
        OP_NAME,
        inputs=input_tensors,
        outputs=['Y'],
        alpha=1.2,
        transA=1,
    )
    QGemm.domain = 'com.microsoft'
    graph_def = helper.make_graph(
        const_nodes + nodes + [QGemm],  # nodes
        OP_NAME + '-model',  # name
        inputs,  # inputs
        [Y],  # outputs
    )
    # add ms domain
    onnxdomain = onnx.OperatorSetIdProto()
    onnxdomain.version = version
    onnxdomain.domain = ''
    msdomain = onnx.OperatorSetIdProto()
    msdomain.version = 1
    msdomain.domain = 'com.microsoft'
    model_def = helper.make_model(graph_def, producer_name=OP_NAME+'-model', opset_imports=[onnxdomain, msdomain])
    model_def.opset_import[0].version = version
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnx_path)
    return onnx_path


OP_NAME = 'QGemm'
input_a_shape = [12, 10]
input_b_shape = [12, 14]
output_shape = [10, 14]
input_c_shape = output_shape

# Generate input data
input_a = np.random.randint(-2, 6, input_a_shape).astype(np.int8)

for const_BC in (False, True, ):
    feed_dict = {'A': input_a}
    if not const_BC:
        feed_dict['B'] = np.random.randint(-6, 4, input_b_shape).astype(np.int8)
        feed_dict['C'] = np.random.randint(-10, 6, input_c_shape).astype(np.int32)
    np.save('input', feed_dict)
    for return_float in (False, True, ):
        model_path = '-'.join([OP_NAME, str(const_BC), str(return_float)]) + '.onnx'
        # Create model
        create_QGemm_model(
            model_path, input_a_shape, input_b_shape, input_c_shape, output_shape,
            const_BC=const_BC, return_float=return_float)

        # Run tests with parser and compare result with runtime
        exit_status = run_parser(model_path, feed_dict, force_float_ir=True,
                                 expected_keywords=['compat_quantized_model=false'],
                                 unexpected_keywords=['layer_top_scale', 'layer_top_zp'], verify=True)
        assert exit_status

        # Set verify to False because need qtlib to generate IR for opt firstly
        exit_status = run_parser(model_path, feed_dict, force_float_ir=False,
                                 expected_keywords=['compat_quantized_model=true', 'layer_top_scale', 'layer_top_zp'],
                                 unexpected_keywords=['Round'], verify=False)
        assert exit_status
