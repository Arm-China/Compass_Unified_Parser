/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: compass.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace compass {

class CompassDialect : public ::mlir::ExtensibleDialect {
  explicit CompassDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;

public:
  ~CompassDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("compass");
  }
};
} // namespace compass
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::compass::CompassDialect)
