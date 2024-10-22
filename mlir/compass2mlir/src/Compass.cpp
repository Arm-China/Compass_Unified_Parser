
#include "AST.h"
#include "Lexer.h"
#include "Parser.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace compass {

std::unique_ptr<IrAST> parseCompassTxtBin(llvm::StringRef txtName,
                                          llvm::StringRef binName) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> txtOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(txtName);
  if (std::error_code ec = txtOrErr.getError()) {
    llvm::errs() << "Could not open txt file: " << ec.message() << "\n";
    return nullptr;
  }

  auto buffer = txtOrErr.get()->getBuffer();

  LexerBuffer lexer(std::string(txtName), buffer.begin(), buffer.end());
  Parser parser(lexer);
  return parser.parseIR();
}
} // namespace compass
