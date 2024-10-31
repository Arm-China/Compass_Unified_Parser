#include "MLIRGen.h"

namespace compass {

mlir::ModuleOp MLIRGenImpl::gen() { return _module; }

mlir::OwningOpRef<mlir::ModuleOp>
genMlir(mlir::MLIRContext &context,
        std::unique_ptr<compass::IrAST> &moduleAST) {
  MLIRGenImpl impl(context);
  return impl.gen();
}

} // namespace compass
