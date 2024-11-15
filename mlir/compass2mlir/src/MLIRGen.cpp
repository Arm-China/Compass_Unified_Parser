#include "MLIRGen.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace compass {

mlir::ModuleOp MLIRGenImpl::gen(compass::IrAST &moduleAST) {
  _module = mlir::ModuleOp::create(_builder.getUnknownLoc());

  //    auto mainFunc = mlir::func::FuncOp::create(UnknownLoc(),
  //                                              "main_graph",
  //                                              _builder.getFunctionType({},
  //                                              {}),
  //                                              {});

  //    // Create an MLIR function for the given prototype.
  //    _builder.setInsertionPointToEnd(_module.getBody());

  for (BlockAST &b : moduleAST) {
    gen(b);
  }
  return _module;
}

mlir::Value MLIRGenImpl::gen(compass::BlockAST &blockAST) {
  _builder.setInsertionPointToEnd(_module.getBody());

  return nullptr;
}

mlir::OwningOpRef<mlir::ModuleOp> genMlir(mlir::MLIRContext &context,
                                          compass::IrAST &moduleAST) {
  MLIRGenImpl impl(context);
  return impl.gen(moduleAST);
}

} // namespace compass
