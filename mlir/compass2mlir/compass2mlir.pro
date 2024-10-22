TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt



SOURCES += main.cpp \
    src/Compass.cpp \
    src/Dialect.cpp

HEADERS += \
    include/AST.h \
    include/Dialect.h \
    include/Lexer.h \
    include/Parser.h


INCLUDEPATH += /home/ik/sdk/llvm/llvm-for-mlir/install/include
INCLUDEPATH += /home/ik/code/c++/mlir/compass2mlir/include


LIBS += /home/ik/sdk/llvm/llvm-for-mlir/install/lib/libLLVMSupport.so
LIBS += /home/ik/sdk/llvm/llvm-for-mlir/install/lib/libMLIRIR.so
LIBS += /home/ik/sdk/llvm/llvm-for-mlir/install/lib/libMLIRSupport.so

DISTFILES += \
    include/Dialect.h.inc \
    include/Op.h.inc \
    include/compass.td \
    src/Dialect.cpp.inc \
    src/Op.cpp.inc \
    mobilenet_v2.bin \
    mobilenet_v2.txt
