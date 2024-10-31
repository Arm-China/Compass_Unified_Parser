include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"




def Compass_Dialect : Dialect {
    let summary = "A short one line description of compass dialect.";
    let description = [{
        compass dialect is a very important dialect. This section contains a much more
        detailed description that documents all of the important pieces of information
        to know about the document.
    }];
  
    let name = "compass";

    /// The C++ namespace that the dialect, and its sub-components, get placed in.
    let cppNamespace = "::mlir::compass";

    let dependentDialects = [
    ];

    let isExtensible = 1;

}


class Compass_Op<string mnemonic, list<Trait> traits = []> :
Op<Compass_Dialect, mnemonic, traits> ;


def InputOp:Compass_Op<"Input"> {
    let summary = "input";
    let results = (outs F32Tensor);
}


def TransposeOp:Compass_Op<"Transpose"> {
    let summary = "transpose";
    let description = [{
      The "transpose" operation performs tranpose operation of input tensors.
    }];

    let arguments = (ins F32Tensor:$input, I32ArrayAttr:$perm);
    let results = (outs F32Tensor);

    // Invoke a static verify method to verify this transpose operation.
    let hasVerifier = 1;
}

