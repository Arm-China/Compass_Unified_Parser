mlir-tblgen -gen-dialect-decls include/compass.td -I /home/ik/sdk/llvm/llvm-project-mlir/install/include -o include/Dialect.h.inc --dialect compass
mlir-tblgen -gen-dialect-defs include/compass.td -I /home/ik/sdk/llvm/llvm-project-mlir/install/include -o src/Dialect.cpp.inc --dialect compass

mlir-tblgen -gen-op-decls include/compass.td -I /home/ik/sdk/llvm/llvm-project-mlir/install/include -o include/Op.h.inc --dialect compass
mlir-tblgen -gen-op-defs include/compass.td -I /home/ik/sdk/llvm/llvm-project-mlir/install/include -o src/Op.cpp.inc --dialect compass


