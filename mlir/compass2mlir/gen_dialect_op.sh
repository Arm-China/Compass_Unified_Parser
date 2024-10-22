mlir-tblgen -gen-dialect-decls include/compass.td -I /home/ik/sdk/llvm/llvm-for-mlir/mlir/include -o include/Dialect.h.inc
mlir-tblgen -gen-dialect-defs include/compass.td -I /home/ik/sdk/llvm/llvm-for-mlir/mlir/include -o src/Dialect.cpp.inc

mlir-tblgen -gen-op-decls include/compass.td -I /home/ik/sdk/llvm/llvm-for-mlir/mlir/include -o include/Op.h.inc
mlir-tblgen -gen-op-defs include/compass.td -I /home/ik/sdk/llvm/llvm-for-mlir/mlir/include -o src/Op.cpp.inc


