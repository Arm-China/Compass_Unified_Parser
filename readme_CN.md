# Compass Unified Parser

Compass Unified Parser 是为将多种不同框架的模型转化成一种浮点中间表示（IR）而设计的, 这种IR是由安谋中国设计的一种标准的IR,用于周易系列的神经网络编译器。

## Parser 的处理流程和设计理念

Parser 的主要目标是将一个训练好的模型转换成浮点IR喂给OPT（优化器）。如下是Parser的主要处理流程：

1. 一个模型通过统一的配置文件传给Parser。
2. 配置解析器解析配置文件，并且根据不同的配置将提交一个任务给相应的模型读取器。
3. 支持的模型读取器会接管输入模型，完成模型的读取：
    * 解析模型文件（例如 protobuf/flattenbuf/json 或一些私有格式）然后建立一个原始图表征。
    * 将原始图表征里面的节点转换为内部的统一节点， 例如：
        * 合并若干的TensorFlow　节点到一个GRUSeq节点。
        * 将caffe的 `detectionoutput` 节点转换为 `detectbox` 和 `nms` 节点。
4. 模型读取器会生成一个内部统一图表征，然后交给前端优化器。
5. 前端优化器主要操作对象是统一图表征。它将会合并或者消除一些节点， 例如：
    * 合并`conv` 和 `add` 到一个节点。
    * 合并 `conv/fc` 节点和 `batchnorm` 节点 到一个节点。
    * 消除一些无用的节点，例如：一个交换维度未变的`transpose`节点。
6. 优化之后，Parser至少会进行一次形状推导来获取所有张量的形状。
7. 进行一些额外的处理，例如：
    * 为一些模型添加一些后处理节点。
8. 序列化为IR文件。

![](doc/images/parser_arch.svg)

### `Graph` 和 `Node` 的设计

在Parser里面，和其他框架类似，我们使用 `Graph` 和 `Node` 来表征一个模型，使用一个链表来表征一个图。
`Graph` 只保存所有节点，节点间的拓扑关系是保存在一个  `Node` 和另一个 `Node`连接上。
`Node` 表征 IR里的层概念（layer）, `Node` 可以通过调用 `serialize` 方法来序列化成一个字串。

###  关于Parser设计

* Parser 只支持一个固定形状的图（静态图），在整个解析转换过程中，会进行若干次的形状推导。
* 每一次图操作之后，例如合并、转换、消除节点之后，我们都希望进行一次形状推导，除非你清楚并且保证所有形状是无误的。
    * 进行形状推导是因为图操作可能会改变图拓扑，也可能会导致形状的变化。
    * 如果某些参数依赖于形状，那么请将这些参数的处理放到形状推导之后或在推导阶段。
* 优化的处理只指出统一图表征，不支持原始图表征。因此，所有框架的模型都能受益于这些优化处理。

## 快速入门

### 安装向导

Parser 是 Compass AIPUBuilder(NN-Compiler) 编译器的一部分。 你可以参考如下Compass AIPUBuilder的指引来安装AIPUBuilder。 完成AIPUBuilder的安装后, Parser 也会被安装并且可以直接使用。

你也可以通过[Compass_Integration](https://github.com/Arm-China/Compass_Integration)中的指引来编译一个包含Parser的AIPUBuilder。关于AIPUBuilder的使用说明，请参考[MiniPkg](https://aijishu.com/a/1060000000215443)里面的说明书：Zhouyi_Compass_Software_Programming_Guide_61010011_0205_01_en.pdf。

初除此之外，Parser可以单独运行。只要满足如下的依赖，就可以直接运行 `main.py`文件来运行Parser。
### 安装依赖
* python (3.8 or higher)
* numpy
* onnx (> 12)
* protobuf
* flatbuffers
* tensorflow (== 2.6)
* torch

### 运行Parser

Parser是以配置文件为输入驱动的，你可以使用如下实例来运行Parser
```bash
python3 main.py -c my_config.ini
```

### 配置文件格式

所有的选项必须在 `Common` 段里面:
* `input_shape` [__required__]
    
    输入张量的形状。常间的模型只有一个输入张量，如：`input_shape=[1,224,224,3]`
    如果你有多个输入张量，使用英文逗号分隔，如： `input_shape=[1,224,224,3],[1,112,112,3]`

* `model_name` [__required__]

    输入模型的名称

* `model_type` [__optional__]

   输入模型的框架，默认是tensorflow，目前支持:
    * `tensorflow`
    * `tflite`
    * `onnx`
    * `caffe`

* `model_domain`  [__required__]

   模型的分类，例如:
    * `image_classification`
    * `object_detection`
    * `keyword_spotting`
    * `speech_recognition`

* `detection_postprocess`  [__required__ 当 `model_domain` 是  `object_detection`]

    如果你的模型是`object_detection`，并且你使用的是官方的模型，你可以选择如下两种后处理方式，我们将在结束出添加相应的后处理节点：
    * `caffe_fasterrcnn`
    * `ssd`
    * `ssd_resnet`
    * `yolo2`
    * `yolo3_tiny`
    * `yolo3_full`

* `input_model`  [__required__]

    输入模型的文件路径，当前支持tensorflow frozen pb, tflite, caffe and onnx 格式。

* `input`  [__required__]

    输入节点（或张量）的名称，如果有多个输入，使用英文逗号`,`分隔。

* `output`  [__required__]

    输出节点（或张量）名称，如果有多个输出，使用英文逗号`,`分隔。

### 配置文件示例
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
更多示例请参考 [examples](examples)。

### 运行示例

首先，你需要下载相应的原始模型。你可以通过[examples](examples)下面的download_model.sh脚本来下载。
```bash
sh examples/tensorflow/download_model.sh
```
然后配置example.cfg文件里的相应的输入输出
```ini
[Common]
model_type = tensorflow
model_name = gru_l
model_domain = image_classification
input_model = ./GRU_L.pb
input = Mfcc:0
input_shape = [1, 49, 10]
output = labels_softmax:0
output_dir = ./
```

运行 run_example.py
* `--framework` [__optional__]

    指定相应的示例，默认是tensorflow。
* `--input_data` [__optional__]

    指定相应的输入数据，如果没有指定将使用随机数据。

```bash
python3 run_example.py --framework [specify example] --input_data [specify feed data]
```

### [贡献指引](doc/Contributing.md)