#ifndef PARSER_H
#define PARSER_H

#include "AST.h"
#include "Lexer.h"
#include "llvm/ADT/StringRef.h"
#include <iostream>

namespace compass {

class Parser {
public:
  /// Create a Parser for the supplied lexer.
  Parser(Lexer &lexer) : _lexer(lexer) {}

  std::unique_ptr<IrAST> parseIR() {
    _lexer.getNextToken();

    // Parse blocks one at a time and accumulate in this vector.
    std::vector<BlockAST> blocks;

    while (auto block = parseBlock()) {
      // block type is std::unique_ptr<ExprASTList>, same as args of BlockAST
      // constructor
      blocks.push_back(
          std::move(*std::make_unique<BlockAST>(std::move(block))));
      if (_lexer.getCurToken() == tok_eof) {
        break;
      }
    }

    // If we didn't reach EOF, there was an error during parsing
    if (_lexer.getCurToken() != tok_eof) {
      return parseError<IrAST>("nothing", "at end of module");
    }

    return std::make_unique<IrAST>(std::move(blocks));
  }

private:
  /// block ::= line
  ///           line
  ///           ...
  std::unique_ptr<ExprASTList> parseBlock() {

    if (_lexer.getCurToken() != tok_attribute) {
      return parseError<ExprASTList>("tok_attribute", "to begin block");
    }

    auto exprList = std::make_unique<ExprASTList>();

    while (_lexer.getCurToken() != tok_block_end &&
           _lexer.getCurToken() != tok_eof) {
      auto l = parseLine();
      if (!l) {
        return parseError<ExprASTList>("read line", "nullptr");
      }
      exprList->push_back(std::move(l));

      if (_lexer.getCurToken() == tok_block_end ||
          _lexer.getCurToken() == tok_eof) {
        break;
      }

      // std::cout << tok << std::endl;
      // _lexer.getNextToken();
    }

    if (_lexer.getCurToken() != tok_block_end &&
        _lexer.getCurToken() != tok_eof) {
      return parseError<ExprASTList>("No block end", "to close block");
    }

    if (_lexer.getCurToken() != tok_eof) {
      while (_lexer.getCurToken() == tok_block_end) {
        _lexer.consume(tok_block_end);
      }
    }
    return exprList;
  }

  std::unique_ptr<LineExprAST> parseLine() {
    if (_lexer.getCurToken() != tok_attribute) {
      return parseError<LineExprAST>("tok_attribute", "to begin line");
    }

    std::string name{_lexer.getAttribute()};
    Location loc{_lexer.getLastLocation()};

    std::cout << "attribute name is:" << name.c_str() << " at line:" << loc.line
              << std::endl;
    _lexer.consume(tok_attribute);
    _lexer.consume(tok_assign);

    bool layer_token = name.find("layer_") != std::string::npos;
    bool id_token = name.find("layer_id") != std::string::npos;
    bool shape_token = name.find("_shape") != std::string::npos;
    bool offset_token = name.find("_offset") != std::string::npos;
    bool size_token = name.find("_size") != std::string::npos;
    auto lineValue = parsePrimary(layer_token, id_token, shape_token,
                                  offset_token, size_token);
    if (!lineValue) {
      return parseError<LineExprAST>("parsePrimary", "to line value");
    }

    auto line = std::make_unique<LineExprAST>(loc, name, std::move(lineValue));
    return line;
  }

  /// primary
  ///   ::= identifierExpr
  ///   ::= listExpr
  ///   ::= expressionExpr
  std::unique_ptr<ExprAST> parsePrimary(bool has_layer_token = false,
                                        bool has_id_token = false,
                                        bool has_shape_token = false,
                                        bool has_offset_token = false,
                                        bool has_size_token = false) {
    bool has_number =
        has_id_token || has_shape_token || has_offset_token || has_size_token;
    switch (_lexer.getCurToken()) {
    default:
      llvm::errs() << "unknown token '" << _lexer.getCurToken()
                   << "' when expecting an expression\n";
      return nullptr;
    case tok_identifier:
      return parseExpression(has_number);
    case tok_sbracket_open:
      return parseList(has_layer_token, has_number);
    case tok_sbracket_close:
      return nullptr;
    }
  }

  std::unique_ptr<ExprAST> parseExpression(bool has_number = false) {
    if (_lexer.getCurToken() != tok_identifier) {
      return parseError<ExprAST>("tok_identifier", "to begin expression");
    }

    auto exprList = std::make_unique<ExprASTList>();
    auto loc = _lexer.getLastLocation();
    std::string literal(_lexer.getIdentifier());
    while (auto i = parseIdentifier(has_number)) {
      //                    std::cout << i->getLiteral() << std::endl;
      exprList->push_back(std::move(i));
      if (_lexer.getCurToken() == tok_comma) {
        _lexer.consume(tok_comma);
      }
      if (_lexer.getCurToken() == tok_eof) {
        break;
      }
      if (_lexer.getCurToken() != tok_identifier) {
        break;
      }
    }

    if (exprList->size() == 1) {
      return std::move(exprList->at(0));
    } else {
      return std::make_unique<ExpressionExprAST>(std::move(loc),
                                                 std::move(exprList));
    }
  }

  std::unique_ptr<ExprAST> parseIdentifier(bool has_number = false) {
    if (_lexer.getCurToken() != tok_identifier) {
      return parseError<IdentifierExprAST>("tok_identifier",
                                           "to begin identifier");
    }
    auto loc = _lexer.getLastLocation();
    auto result = std::unique_ptr<ExprAST>();
    if (!has_number) {
      result = std::make_unique<IdentifierExprAST>(std::move(loc),
                                                   _lexer.getIdentifier());
      //                    std::cout<< std::boolalpha
      //                    <<IdentifierExprAST::classof(&(*result))<<std::endl;
    } else {
      result = std::make_unique<NumberExprAST>(std::move(loc),
                                               _lexer.getIdentifier());
      //                    std::cout<< std::boolalpha
      //                    <<NumberExprAST::classof(&(*result))<<std::endl;
    }
    _lexer.consume(tok_identifier);
    return result;
  }

  std::unique_ptr<ExprAST> parseList(bool has_layer_token = false,
                                     bool has_number = false) {
    auto loc = _lexer.getLastLocation();
    _lexer.consume(tok_sbracket_open);

    // Hold the list of values at this nesting level.
    ExprASTList values;

    do {
      if (_lexer.getCurToken() == tok_sbracket_open) {
        values.push_back(parseList(has_layer_token, has_number));
        if (!values.back()) {
          return nullptr; // parse error in the nested array.
        }
      } else if (_lexer.getCurToken() == tok_identifier) {
        values.push_back(parseExpression(has_number));
      } else if (_lexer.getCurToken() == tok_comma) {
        _lexer.consume(tok_comma);
      } else {
      }

      // End of this list on ']'
      if (_lexer.getCurToken() == tok_sbracket_close) {
        break;
      }

    } while (true);

    _lexer.getNextToken();

    return std::make_unique<ListExprAST>(
        std::move(loc), std::make_unique<ExprASTList>(std::move(values)));
  }

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
    auto curToken = _lexer.getCurToken();
    llvm::errs() << "Parse error (" << _lexer.getLastLocation().line << ", "
                 << _lexer.getLastLocation().col << "): expected '" << expected
                 << "' " << context << " but has Token " << curToken;
    if (isprint(curToken)) {
      llvm::errs() << " '" << (char)curToken << "'";
    }
    llvm::errs() << "\n";
    return nullptr;
  }

private:
  Lexer &_lexer;
};

std::unique_ptr<IrAST> parseCompassTxtBin(llvm::StringRef, llvm::StringRef);

} // namespace compass

#endif
