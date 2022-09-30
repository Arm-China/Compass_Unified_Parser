# How to write a pass

* What is a pass
* How to write a pass by example
* Some tips for pass writing

## What is a pass

The pass is an important concept in compiler system, the pass is the core part of a compiler. Here, the build-tool is a kind of compiler, it compiler a trained model to an executable file. The Parser is one part of the build-tool, so pass also plays an important role in Parser. Passes perform the transformations and optimizations.

## How to write a pass by example
In this section, we will show you two examples for what a actually working pass in Parser is.

### Example 1: merging relu

Below is the merge relu pass code, the objective of this pass is the merge a fully connected node with a relu-like node, as below:

![](../images/fc_relu.svg)


The code is short and simple

* first all, we define several patterns for finding all matcher pattern. here the pattern is simple: fc+relu
* after matched, we can get the  fc node and relu node. we just apply the graph method `g.remove_inplace_node(relu)`, it will remove the relu node, and connect the relu's children to fc node, as well as their outputs. 
* then we set the relu parameter to the fc node.

```python
@graph_pass
def merge_relu(g):
    patterns=[
        ([("0", "FullyConnected"), ("1", "Relu6")], [("0", "1")]),
        ([("0", "FullyConnected"), ("1", "LeakyRelu")], [("0", "1")]),
        ([("0", "FullyConnected"), ("1", "Relu")], [("0", "1")]),
    ]
    for p in patterns:
        matcher=GraphMatcher(*p)
        for m in matcher.match(g):
            node, relu = m[:2]
            g.remove_inplace_node(relu)
            node._params["with_relu"] = True
            node._params.update(relu._params)
            if relu.type=="Relu6":
                node._params["relu_min"] = 0
                node._params["relu_max"] = 6
```

### Example 2: merge batchnorm
In Tensorflow framework, BatchNorm(BN) operator may be implement in RAW, so we need to merge it as one op.

below is the pass process:
![](../images/bn.svg)

Here we need to:
* build a pattern for batchnorm
* find all matched subgraphs
* merge subgraph as one node by:
    * graph.from_nodes
    * graph.as_node()

* update parameters for merged op

Below is a pice of code for do that.
```python
def merge_batchnorm(g):
    pattern = GraphMatcher(
        [("0", "ANY"), ("1", "Add"), ("2", "Rsqrt"), ("3", "Mul"),
         ("4", "Mul"), ("5", "Sub"), ("6", "Add"), ("7", "Mul")],
        [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"),
         ("5", "6"), ("3", "7"), ("7", "6"), ("0", "7")]
    )
    matched = pattern.match(g)
    for m in matched:
        p, add, sqrt, mul3, mul4, sub, add6, mul7 = m
        gamma = [i for i in mul3.inputs if i.np is not None][0]
        mean = [i for i in mul4.inputs if i.np is not None][0]
        beta = [i for i in sub.inputs if i.np is not None][0]
        variance = add.inputs[0]
        epsilon = add.inputs[1]
        if len(variance.np) == 1:
            variance, epsilon = epsilon, variance
        epsilon = float(epsilon.np)
        # remove parent
        m.remove(p)
        for i in m:
            g.nodes.pop(i)
        sg = PGraph("SubGraph")
        sg.from_nodes(m)
        node = sg.as_node()
        node.type = "BatchNorm"

        node._params["epsilon"] = epsilon
        node.np_weights["gamma"] = gamma
        node.np_weights["beta"] = beta
        node.np_weights["mean"] = mean
        node.np_weights["variance"] = variance
        node.inputs=[]
        for inp in p.outputs:
            node.inputs.append(inp)
        node.outputs=add6.outputs
        g.nodes[node] = node
        dumps=[i for i in g.dumps]
        for dump in dumps:
            if dump.name  in m:
                g.dumps.replace(dump, node)
        pass
```