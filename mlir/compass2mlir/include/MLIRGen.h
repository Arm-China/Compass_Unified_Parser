#ifndef MLIRGEN_H
#define MLIRGEN_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace compass {

class MLIRGenImpl {

public:
  explicit MLIRGenImpl(mlir::MLIRContext &context)
      : _context(context), _builder(&context) {
    _module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  }

  mlir::ModuleOp gen();

private:
  mlir::MLIRContext &_context;

  mlir::ModuleOp _module;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder _builder;
};

class IrAST;
mlir::OwningOpRef<mlir::ModuleOp>
genMlir(mlir::MLIRContext &context, std::unique_ptr<compass::IrAST> &moduleAST);
} // namespace compass

#endif // MLIRGEN_H
