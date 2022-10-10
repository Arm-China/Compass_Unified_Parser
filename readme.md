# Compass Unified Parser(CUP)

Compass Unified Parser is designed for converting mutil-framework models to a float Intermediate Representation (IR), which aligns with the standard IR definition of Arm China Zhouyi AIPU Neural Network (NN) compiler.


## CUP process flow and design philosophy
The main objective of CUP is to convert a trained model to a float IR feeding to OPT(optimizer). Below is the process flow of the parser.

1. A model will be feeded in by an unified configuration file.
2. The entry point: Configuration reader will parse the config file, and dispatch the job to a supported reader.
3. One of supported reader will take over the input model. There are two steps for reading the model:
    * parse the model file(for example, protobuf/flattenbuf/json or private format) and build a raw graph for the model
    * convert the raw graph nodes to unified nodes, for example:
        * merging several tensorflow nodes to one GRUSeq node
        * converting caffe `detectionoutput` node to `detectbox` and `nms`
4. The reader will generate a unified graph, then pass to front-end optimizer
5. The front-end optimizer will operate on the unified graph. It will merge or eliminate some nodes for OPT, for example:
    * merge `conv` and `add` to one node
    * merge `conv/fc` and `batchnorm`
    * eliminate useless node: a `transpose` node with permutation in order

6. After optimization, we will do once shape inference, for getting all tensor shape.

7. Do some additional passes, for example:
    * add post-process nodes for some models

8. Serialize to file.

![](../images/parser_arch.svg)

### `Graph` and `Node` design

In this parser, we use `Graph` and `Node` to represent models just like common framework. We use linked list type to represent the graph.

The `Graph` only keeps all nodes, and the topology info will be stored in `Node` by linking other `Node`.

`Node` represents the IR's layer, which can be serialized simply by `serialize` method.


### More About the design of CUP

* CUP only supports fixed shape graph i.e. static graph, and it will do multi times shape inference.
* After each graph operation, such as merge, convert, eliminate, a shape inference is preferred, except you are sure that the shape is correct.
    * This is because any graph operation may change the topology of the graph, and the shape may be changed as well.
    * If some parameters can be only known after known shape, then put the parameters process in shape inference stage.
* Optimization passes only support unified graph, so put all framework dependent passes in the model reader part, because these passes can not be used for other frameworks.


## Quick Start

### Installation instructions

The CUP parser is a part of Compass Build Tool. You can follow the instruments of Compass Build Tool to install the buildtool, and the CUP will be available.

Beside, the CUP can also run independently. Before running the `main.py`, please make sure the following requirements are installed.

### Requirements
* python (3.8 or higher)
* numpy
* onnx (> 12)
* protobuf
* flatbuffers
* tensorflow (> 1.13, 2.6 is preferred)
* torch

### Run the CUP

CUP uses a config file(.ini) as input. You can directly run the `main.py` with your configure file as below:
```bash
python3 main.py -c my_config.ini
```

### Config file format
All options are under a `Common` section:
* `input_shape` [__required__]

    the input shape(s) of model. Usually it is a single tensor shape, for example: `input_shape=[1,224,224,3]`

    if you have several inputs, please use comma the separate them, for example: `input_shape=[1,224,224,3],[1,112,112,3]`
* `model_name` [__required__]

    the name for the input model

* `model_domain`  [__required__]

    the domain of the model, for example:
    * `image_classification`
    * `object_detection`
    * `keyword_spotting`
    * `speech_recognition`

* `detection_postprocess`  [__required__ when `model_domain` is  `object_detection`]

    if your model_domain is `object_detection`, and if you are using the official detection model, please specify your detection post process. Now it only supports two types of post process:

    * `yolo`
    * `ssd`

* `input_model`  [__required__]

    file path of the input 3rd party model. Current tensorflow frozen pb, tflite, caffe and onnx models are supported.

* `input`  [__required__]

    the input(s) node(s)' name of the model. If you have several inputs, use `,` to separate each one.

* `output`  [__required__]

    the output(s) node(s)' name of the model. If you have several outputs, use `,` to separate each one.

### example
```ini
[Common]
input_shape = [1,224,224,3]
model_name = resnet50
model_domain = image_classification
detection_postprocess =
input_model = resnet50/frozen.pb
input = Placeholder
output = resnet_v1_50/predictions/Reshape
```

For more examples, please refer to [examples](examples).