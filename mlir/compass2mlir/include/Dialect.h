#ifndef DIALECT_H
#define DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ExtensibleDialect.h"

#include "Dialect.h.inc"

#define GET_OP_CLASSES
#include "Op.h.inc"

#endif // DIALECT_H
