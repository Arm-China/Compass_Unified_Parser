TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt



SOURCES += main.cpp \
    src/Compass.cpp \
    src/Dialect.cpp \
    src/MLIRGen.cpp

HEADERS += \
    include/AST.h \
    include/Dialect.h \
    include/Lexer.h \
    include/MLIRGen.h \
    include/Parser.h


INCLUDEPATH += /home/ik/sdk/llvm/llvm-project-mlir/install/include
INCLUDEPATH += /home/ik/code/c++/mlir/compass2mlir/include


LIBS += /home/ik/sdk/llvm/llvm-project-mlir/install/lib/libLLVMSupport.so
LIBS += /home/ik/sdk/llvm/llvm-project-mlir/install/lib/libMLIRIR.so
LIBS += /home/ik/sdk/llvm/llvm-project-mlir/install/lib/libMLIRSupport.so
LIBS += /home/ik/sdk/llvm/llvm-project-mlir/install/lib/libMLIRFuncDialect.so

DISTFILES += \
    gen_dialect_op.sh \
    include/Dialect.h.inc \
    include/Op.h.inc \
    include/compass.td \
    src/Dialect.cpp.inc \
    src/Op.cpp.inc \
    mobilenet_v2.bin \
    mobilenet_v2.txt
