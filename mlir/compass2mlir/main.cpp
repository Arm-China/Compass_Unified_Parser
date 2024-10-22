#include "Dialect.h"
#include "Parser.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CommandLine.h"
#include <string>

enum Action { None, DumpAST, DumpMLIR };

int main(int argc, char **argv) {

  mlir::MLIRContext context;

  context.getOrLoadDialect<mlir::compass::CompassDialect>();

  llvm::cl::opt<enum Action> emitAction(
      "emit", llvm::cl::desc("Select the kind of output desired"),
      llvm::cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
      llvm::cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

  llvm::cl::opt<std::string> txtName("t", llvm::cl::desc("Specify input txt"),
                                     llvm::cl::value_desc("txt file"));
  llvm::cl::opt<std::string> binName("b", llvm::cl::desc("Specify input bin"),
                                     llvm::cl::value_desc("bin file"));

  llvm::cl::ParseCommandLineOptions(argc, argv, "Compass IR compiler\n");

  auto module = compass::parseCompassTxtBin(txtName, binName);

  return 0;
}
