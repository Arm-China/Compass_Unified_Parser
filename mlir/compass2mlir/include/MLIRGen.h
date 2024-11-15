#ifndef MLIRGEN_H
#define MLIRGEN_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "AST.h"

namespace compass {

class MLIRGenImpl {

public:
  explicit MLIRGenImpl(mlir::MLIRContext &context)
      : _context(context), _builder(&context) {
    _module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  }

  mlir::ModuleOp gen(compass::IrAST &moduleAST);

  mlir::Location UnknownLoc() const { return mlir::UnknownLoc::get(&_context); }

private:
  mlir::MLIRContext &_context;

  mlir::ModuleOp _module;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder _builder;

  mlir::Value gen(compass::BlockAST &blockAST);
};

class IrAST;
mlir::OwningOpRef<mlir::ModuleOp> genMlir(mlir::MLIRContext &context,
                                          compass::IrAST &moduleAST);
} // namespace compass

#endif // MLIRGEN_H
