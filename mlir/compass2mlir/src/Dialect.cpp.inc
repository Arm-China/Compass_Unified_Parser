/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: compass.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::compass::CompassDialect)
namespace mlir {
namespace compass {

CompassDialect::CompassDialect(::mlir::MLIRContext *context)
    : ::mlir::ExtensibleDialect(getDialectNamespace(), context,
                                ::mlir::TypeID::get<CompassDialect>())

{

  initialize();
}

CompassDialect::~CompassDialect() = default;

} // namespace compass
} // namespace mlir
