include "/home/ik/sdk/llvm/llvm-for-mlir/install/include/mlir/IR/DialectBase.td"
include "/home/ik/sdk/llvm/llvm-for-mlir/install/include/mlir/IR/OpBase.td"
///include "/home/ik/sdk/llvm/llvm-for-mlir/install/include/mlir/IR/BytecodeBase.td"

def Compass_Dialect : Dialect {
    let summary = "A short one line description of compass dialect.";
    let description = [{
        compass dialect is a very important dialect. This section contains a much more
        detailed description that documents all of the important pieces of information
        to know about the document.
    }];
  
    /// of the dialect, such as operations ("my_dialect.foo").
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
}

