
#include "Dialect.h"
#include "Dialect.cpp.inc"

namespace mlir {
namespace compass {
void CompassDialect::initialize() { addOperations<>(); }
} // namespace compass
} // namespace mlir

#define GET_OP_CLASSES
#include "Op.cpp.inc"
