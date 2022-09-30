# Compass Unified Parser(CUP) Architecture

* Overview design and concept
* Parser process flow and design philosophy


## Overview design and concept
The CUP parser is design for converting mutil-framework models to our internal representation. 


## Parser process flow and design philosophy
The main objective of parser is to convert a trained model to a internal representation (IR) feeding to OPT(optimizer). Below is the parser architecture:  

1. a model will be feeded in by a unified configuration file
2. the entry point: configuration reader will parse the config file, and dispatch the job to a supported reader
3. one of supported reader will take over the input model,  there are two steps for read the model:
    * parse the model file(for example: protobuf/flattenbuf/json or private format) and build a raw graph for the model
    * converting the raw graph nodes to unified nodes, for example: 
        * merging several tensorflow nodes to one GRUSeq node
        * converting caffe `detectionoutput` node to `detectbox` and `nms`
        * etc.
4. the reader will generate a unified graph, then will pass to front-end optimizer
5. the front-end optimizer will operate on unified graph, it will merge or eliminate some nodes for OPT, for example:
    * merge conv and add to one node
    * merge conv/fc and bn
    * eliminate useless node: a transpose node with permutation in order, etc.

6. after optimization, then we will do once shape inference, for getting all tensor shape.

7. do some additional passes: add post-process nodes for some models

8. serialization to file.

![](../images/parser_arch.svg)





In this design and process flow, we keep some ideas in mind:

* we only support fixed shape graph i.e. static graph, and  we will do multi times shape inference.
* after each graph operation, such as merge, convert, eliminate, a shape inference is preferred, except you are make sure the shape is correct.
    * this is because of any graph operations may change the topology of the graph, and the shape also may be changed.
    * if some parameters can be only known after known shape, that put the parameters process in shape inference stage.
* optimization passes only support unified graph, so put all framework depended passes in the model reader part, because these passes can not be used for other frameworks.

## `Graph` and `Node` design

In this parser, we use (Graph,Node) to represent models just like common framework. We use linked list type to represent the graph. The `Graph` only keep all nodes, and the topology info will store in `Node` by linking other `Node`, and the `Node` is represent the IR's layer, it can be serialized simply by `serialize` method.

## Quick Start

### Installation instructions

The CUP parser is a part of Compass Build tool. You can follow the instruments of Compass Build Tool to install the buildtool, and the CUP will be available.

Beside, the CUP is also can run independently.  Before running the `main.py`, we need install some requirements.

### Requirements
* python 3.8 or higher
* numpy
* onnx >12
* protobuf
* flatbuffers
* tensorflow (>1.13 2.6 is preferred)
* torch

### Run the CUP

the CUP is use a config file(.ini) as input, you can directly run the `main.py` with your configure file as below:
```bash
python3 main.py -c my_config.ini
```

### Config file format
All options is a `Common` section:
* `input_shape` [__required__ ]

    the input shape(s) of model, usually it is a single tensor shape, for example: `input_shape=[1,224,224,3]`

    if you have several inputs, please use comma the separate them, for example: `input_shape=[1,224,224,3],[1,112,112,3]`
* `model_name` [__required__ ]

    the name for the input model, must be a string, it use to identify the model.

* `model_domain`  [__required__ ]

    the domain of the model, there are only 4 available options:
    * `image_classification`
    * `object_detection`
    * `keyword_spotting`
    * `speech_recognition`
    * ~~`image_segmentation`~~

* `detection_postprocess`  [__required__ when `model_domain` is  `object_detection`]

    if your model_domain is `object_detection`, and if you a using the official detection model, please specify your detection post process. Now it only support tow types of post process:

    * `yolo`
    * `ssd`

* `input_model`  [__required__ ]

    file path of the input 3rd party model, current only support tensorflow frozen pb.

* `input`  [__required__ ]

    the input(s) node(s)' name of the model. if you have several inputs, use `,` to separate each one.

* `output`  [__required__ ]
    the output(s) node(s)' name of the model. if you have several outputs, use `,` to separate each one.

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

For more examples, please ref [examples](/examples/).