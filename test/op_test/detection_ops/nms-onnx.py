# import numpy as np
# import onnx
# from utils.run import run_parser
# from onnx import TensorProto, helper


# def create_nms(nms_path, center_point_box, boxes, scores, max_output_boxes_per_class=0, iou_threshold=0., score_threshold=0.):
#     ret = ''
#     if nms_path:
#         max_output_boxes_per_class_value = np.array(
#             max_output_boxes_per_class, np.int64)
#         iou_threshold_value = np.array(iou_threshold, np.float32)
#         score_threshold_value = np.array(score_threshold, np.float32)

#         boxes_tensor = helper.make_tensor_value_info(
#             'boxes', TensorProto.FLOAT, list(boxes.shape))
#         scores_tensor = helper.make_tensor_value_info(
#             'scores', TensorProto.FLOAT, list(scores.shape))
#         selected_indices_tensor = helper.make_tensor_value_info(
#             'selected_indices', TensorProto.INT64, [max_output_boxes_per_class, 3])

#         max_output_boxes_per_class_const = helper.make_node('Constant',
#                                                             [],
#                                                             ['max_output_boxes_per_class'],
#                                                             value=onnx.helper.make_tensor(name='max_output_boxes_per_class',
#                                                                                           data_type=onnx.TensorProto.INT64,
#                                                                                           dims=max_output_boxes_per_class_value.shape,
#                                                                                           vals=max_output_boxes_per_class_value.flatten()
#                                                                                           )
#                                                             )
#         iou_threshold_const = helper.make_node('Constant',
#                                                [],
#                                                ['iou_threshold'],
#                                                value=onnx.helper.make_tensor(name='iou_threshold',
#                                                                              data_type=onnx.TensorProto.FLOAT,
#                                                                              dims=iou_threshold_value.shape,
#                                                                              vals=iou_threshold_value.flatten()
#                                                                              )
#                                                )
#         score_threshold_const = helper.make_node('Constant',
#                                                  [],
#                                                  ['score_threshold'],
#                                                  value=onnx.helper.make_tensor(name='score_threshold',
#                                                                                data_type=onnx.TensorProto.FLOAT,
#                                                                                dims=score_threshold_value.shape,
#                                                                                vals=score_threshold_value.flatten()
#                                                                                )
#                                                  )

#         nms_def = helper.make_node(
#             'NonMaxSuppression',  # name
#             ['boxes', 'scores', 'max_output_boxes_per_class',
#                 'iou_threshold', 'score_threshold'],  # inputs
#             ['selected_indices'],  # outputs
#             center_point_box=center_point_box,  # attributes
#         )

#         graph_def = helper.make_graph(
#             [max_output_boxes_per_class_const, iou_threshold_const,
#                 score_threshold_const, nms_def],  # nodes
#             'nms',  # name
#             [boxes_tensor, scores_tensor],  # inputs
#             [selected_indices_tensor],  # outputs
#         )

#         model_def = helper.make_model(graph_def, producer_name='onnx')
#         model_def.opset_import[0].version = 15
#         onnx.save(model_def, nms_path)
#         onnx.checker.check_model(model_def)

#         ret = nms_path
#     else:
#         print('invalid path')
#     return ret


# OP_NAME = 'NMS_onnx'
# #################################
# # only support onnx output box >= batch(inp1.shape[0])*box_num(inp1.shape[1])*class_num(inp2.shape[1])
# # input_shape = [2, 6, 4]
# # input_shape2 = [2, 3, 6]
# # center_point_box = 1
# # max_output_boxes_per_class = 100
# # iou_threshold = 0.5
# # score_threshold = 0.
# ###############################
# # input_shape = [1,6,4]
# # input_shape2 = [1,1,6]
# # center_point_box = 0
# # max_output_boxes_per_class = 100
# # iou_threshold = 0.7
# # score_threshold = 0.4
# ###############################
# # input_shape = [1,6,4]
# # input_shape2 = [1,1,6]
# # center_point_box = 0
# # max_output_boxes_per_class = 3
# # iou_threshold = 0.5
# # score_threshold = 0
# ######################
# # input_shape = [1, 6, 4]
# # input_shape2 = [1, 2, 6]
# # center_point_box = 0
# # max_output_boxes_per_class = 2
# # iou_threshold = 0.5
# # score_threshold = 0.
# #############################
# # input_shape = [1,6,4]
# # input_shape2 = [1,1,6]
# # center_point_box = 1
# # max_output_boxes_per_class = 3
# # iou_threshold = 0.5
# # score_threshold = 0
# #####################
# # input_shape = [1,10,4]
# # input_shape2 = [1,1,10]
# # center_point_box = 0
# # max_output_boxes_per_class = 3
# # iou_threshold = 0.5
# # score_threshold = 0
# #####################
# # input_shape = [1,1,4]
# # input_shape2 = [1,1,1]
# # center_point_box = 0
# # max_output_boxes_per_class = 3
# # iou_threshold = 0.5
# # score_threshold = 0
# ##################
# # input_shape = [1,6,4]
# # input_shape2 = [1,1,6]
# # center_point_box = 0
# # max_output_boxes_per_class = 3
# # iou_threshold = 0.5
# # score_threshold = 0.4
# ########################
# # input_shape = [2,6,4]
# # input_shape2 = [2,1,6]
# # center_point_box = 0
# # max_output_boxes_per_class = 2
# # iou_threshold = 0.5
# # score_threshold = 0.


# # Generate input data
# feed_dict = dict()
# feed_dict['boxes'] = np.random.ranf(input_shape).astype(np.float32) * 100
# feed_dict['scores'] = np.random.ranf(input_shape2).astype(np.float32) * 100
# input_data_path = 'input.npy'
# np.save(input_data_path, feed_dict)


# # boxes = np.random.ranf((2, 6, 4)).astype(np.float32)
# # scores = np.random.ranf((2, 3, 6)).astype(np.float32)

# model_name = '-'.join([OP_NAME, str(int(1))])
# model_path = model_name + '.onnx'


# create_nms(model_path,
#            center_point_box,
#            feed_dict['boxes'],
#            feed_dict['scores'],
#            max_output_boxes_per_class=max_output_boxes_per_class,
#            iou_threshold=iou_threshold,
#            score_threshold=score_threshold)


# # Run tests with parser and compare result with runtime
# exit_status = run_parser(
#     model_path, feed_dict, model_type=None, save_output=True, verify=True)
# assert exit_status
