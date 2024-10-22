/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: compass.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::compass::InputOp
#endif // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

    //===----------------------------------------------------------------------===//
    // Local Utility Method Definitions
    //===----------------------------------------------------------------------===//

    namespace mlir {
  namespace compass {} // namespace compass
} // namespace mlir
namespace mlir {
namespace compass {

//===----------------------------------------------------------------------===//
// ::mlir::compass::InputOp definitions
//===----------------------------------------------------------------------===//

namespace detail {} // namespace detail
InputOpAdaptor::InputOpAdaptor(InputOp op)
    : InputOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult InputOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void InputOp::build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState) {}

void InputOp::build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InputOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult InputOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::llvm::LogicalResult InputOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace compass
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::compass::InputOp)

#endif // GET_OP_CLASSES
