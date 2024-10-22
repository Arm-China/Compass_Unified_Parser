#ifndef DIALECT_H
#define DIALECT_H

#include "mlir/IR/ExtensibleDialect.h"

#include "Dialect.h.inc"

#define GET_OP_CLASSES
#include "Op.h.inc"

#endif // DIALECT_H
