
#ifndef AST_H
#define AST_H

#include "Lexer.h"

namespace compass {

class ExprAST {
public:
  enum ExprASTKind {
    Expr_None,
    Expr_Attribute,
    Expr_Identifier,
    Expr_Number,
    Expr_Expression, // separated by comma
    Expr_List,
    Expr_Line
  };

  ExprAST(ExprASTKind kind, Location location)
      : _kind(kind), _location(std::move(location)) {}

  ExprAST() : _kind(Expr_None) {}

  virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return _kind; }

  const Location &loc() { return _location; }

private:
  const ExprASTKind _kind = Expr_None;
  Location _location;
};

/// A block-list of expressions.
using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

/// Expression class for referencing an attribute, like "layer_name".
class AttributeExprAST : public ExprAST {
public:
  /// LLVM style RTTI
  static bool classof(const ExprAST *c) {
    return c->getKind() == Expr_Attribute;
  }

public:
  AttributeExprAST(Location loc, const std::string &name)
      : ExprAST(Expr_Attribute, std::move(loc)), _name(name) {}

  const std::string &getName() const { return _name; }

private:
  std::string _name;
};

/// Expression class for referencing an Identifier.
class IdentifierExprAST : public ExprAST {
public:
  /// LLVM style RTTI
  static bool classof(const ExprAST *c) {
    return c->getKind() == Expr_Identifier;
  }

  IdentifierExprAST(Location loc, const std::string &literal)
      : ExprAST(Expr_Identifier, std::move(loc)), _literal(literal) {}

  const std::string &getLiteral() const { return _literal; }

protected:
  std::string _literal;
};

class NumberExprAST : public ExprAST {
public:
  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Number; }

  NumberExprAST(Location loc, const std::string &literal)
      : ExprAST(Expr_Number, std::move(loc)), _int(std::stoi(literal)) {}

  int getValue() const { return _int; }

private:
  int _int;
};

/// Expression class for referencing an Expression (Identifier + ',' +
/// Identifier + ...)
class ExpressionExprAST : public ExprAST {
public:
  /// LLVM style RTTI
  static bool classof(const ExprAST *c) {
    return c->getKind() == Expr_Expression;
  }

  ExpressionExprAST(Location loc, std::unique_ptr<ExprASTList> list)
      : ExprAST(Expr_Expression, std::move(loc)), _expression(std::move(list)) {
  }

private:
  std::unique_ptr<ExprASTList> _expression;
};

/// Expression class for referencing a List or nested List.
class ListExprAST : public ExprAST {
public:
  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_List; }

  ListExprAST(Location loc, std::unique_ptr<ExprASTList> list)
      : ExprAST(Expr_List, std::move(loc)), _list(std::move(list)) {}

  bool is_empty() const { return _list->size() == 0; }

private:
  std::unique_ptr<ExprASTList> _list;
};

class LineExprAST : public ExprAST {
public:
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Line; }

  LineExprAST(Location loc, const std::string &name,
              std::unique_ptr<ExprAST> value)
      : ExprAST(Expr_Line, std::move(loc)), _attribute(name),
        _value(std::move(value)) {}

  const std::string &getName() const { return _attribute; }

  ExprAST *getValue() { return _value.get(); }

private:
  std::string _attribute;
  std::unique_ptr<ExprAST> _value;
};

class BlockAST {
public:
  BlockAST(std::unique_ptr<ExprASTList> body) : _body(std::move(body)) {}

  ExprASTList *getBody() { return _body.get(); }

private:
  std::unique_ptr<ExprASTList> _body;
};

class IrAST {
public:
  IrAST(std::vector<BlockAST> blocks) : _blocks(std::move(blocks)) {}

  auto begin() { return _blocks.begin(); }
  auto end() { return _blocks.end(); }

private:
  std::vector<BlockAST> _blocks;
};
} // namespace compass

#endif
