#ifndef LEXER_H
#define LEXER_H

#include "llvm/ADT/StringRef.h"
#include <cctype>
#include <iostream>
#include <memory>

namespace compass {

/// Structure definition a location in a file.
struct Location {
  std::shared_ptr<std::string> file; ///< filename.
  int line;                          ///< line number.
  int col;                           ///< column number.
};

// List of Token returned by the lexer.
enum Token : int {
  tok_comma = ',',
  tok_assign = '=',
  tok_sbracket_open = '[',
  tok_sbracket_close = ']',

  tok_eof = -1,

  // primary
  tok_attribute = -2,
  tok_identifier = -3,
  tok_block_end = -4
};

class Lexer {
public:
  static int is_attribute_char(int ch) {
    if (std::islower(ch) || ch == '_') {
      return 1;
    } else {
      return 0;
    }
  }

  static int is_identifier_char(int ch) {
    if (std::isalnum(ch)) {
      return 1;
    }
    // ispunct:  !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    if (std::ispunct(ch) && ch != ',' && ch != '=' && ch != '[' && ch != ']') {
      return 1;
    }
    return 0;
  }

  Lexer(std::string filename)
      : _lastLocation(
            {std::make_shared<std::string>(std::move(filename)), 0, 0}) {}

  virtual ~Lexer() = default;

  /// Look at the current token in the stream.
  Token getCurToken() const { return _curTok; }

  /// Move to the next token in the stream and return it.
  Token getNextToken() { return _curTok = getTok(); }

  /// Move to the next token in the stream, asserting on the current token
  /// matching the expectation.
  void consume(Token tok) {
    assert(tok == _curTok && "consume Token mismatch expectation");
    getNextToken();
  }

  /// Return the location for the beginning of the current token.
  Location getLastLocation() { return _lastLocation; }

  // Return the current line in the file.
  int getLine() { return _curLineNum; }

  // Return the current column in the file.
  int getCol() { return _curCol; }

  /// Return the current attribute (prereq: getCurToken() == tok_attribute)
  const std::string getAttribute() const {
    assert(_curTok == tok_attribute);
    return _attributeStr;
  }

  /// Return the current identifier (prereq: getCurToken() == tok_identifier)
  const std::string getIdentifier() const {
    assert(_curTok == tok_identifier);
    return _identifierStr;
  }

private:
  virtual llvm::StringRef readNextLine() = 0;

  /// Return the next character from the stream. This manages the buffer for the
  /// current line and request the next line buffer to the derived class as
  /// needed.
  int getNextChar() {
    // The current line buffer should not be empty unless it is the end of file.
    if (_curLineBuffer.empty()) {
      return EOF;
    }

    ++_curCol;
    auto nextchar = _curLineBuffer.front();
    _curLineBuffer = _curLineBuffer.drop_front();
    if (_curLineBuffer.empty()) {
      _curLineBuffer = readNextLine();
    }
    if (nextchar == '\n') {
      ++_curLineNum;
      _curCol = 0;
    }
    return nextchar;
  }

  ///  Return the next token from standard input.
  Token getTok() {
    // Skip any whitespace.
    //                 while(isspace(_lastChar)) {
    //                     _lastChar = Token(getNextChar());
    //                 }

    Token charBeforeNewline = Token(' ');
    while (isblank(_lastChar) || _lastChar == '\n') {
      if (_lastChar == '\n') {
        charBeforeNewline = Token('\n');
      }
      _lastChar = Token(getNextChar());

      if (_lastChar == '\n' && charBeforeNewline == '\n') {
        _lastLocation.line = _curLineNum - 1;
        _lastLocation.col = _curCol;
        return tok_block_end;
      }
    }

    // Save the current location before reading the token characters.
    _lastLocation.line = _curLineNum;
    _lastLocation.col = _curCol;

    // Attribute: [a-z_]
    if (_curCol == 1 && std::islower(_lastChar)) {
      _attributeStr = (char)_lastChar;
      while (is_attribute_char(_lastChar = Token(getNextChar()))) {
        _attributeStr += (char)_lastChar;
      }
      return tok_attribute;
    }

    // Identifier
    if (_curCol >= 3 && is_identifier_char(_lastChar)) {
      _identifierStr = (char)_lastChar;
      while (is_identifier_char(_lastChar = Token(getNextChar()))) {
        _identifierStr += (char)_lastChar;
      }
      return tok_identifier;
    }

    // Check for end of file.  Don't eat the EOF.
    if (_lastChar == EOF) {
      return tok_eof;
    }

    // Otherwise, just return the character as its ascii value.
    Token thisChar = Token(_lastChar);
    _lastChar = Token(getNextChar());
    return thisChar;
  }

private:
  /// The last token read from the input.
  Token _curTok = tok_eof;

  /// Location for `curTok`.
  Location _lastLocation;

  /// If the current Token is an attribute, this string contains the value.
  std::string _attributeStr;

  /// If the current Token is an identifier, this string contains the value.
  std::string _identifierStr;

  /// The last value returned by getNextChar(). We need to keep it around as we
  /// always need to read ahead one character to decide when to end a token and
  /// we can't put it back in the stream after reading from it.
  Token _lastChar = Token(' ');

  /// Keep track of the current line number in the input stream
  int _curLineNum = 0;

  /// Keep track of the current column number in the input stream
  int _curCol = 0;

  /// Buffer supplied by the derived class on calls to `readNextLine()`
  llvm::StringRef _curLineBuffer = "\n";
};

class LexerBuffer final : public Lexer {
public:
  LexerBuffer(std::string filename, const char *begin, const char *end)
      : Lexer(std::move(filename)), _current(begin), _end(end) {}

  /// Provide one line at a time to the Lexer, return an empty string when
  // reaching the end of the buffer.
  llvm::StringRef readNextLine() {
    auto *begin = _current;
    while (_current <= _end && *_current && *_current != '\n') {
      ++_current;
    }

    if (_current <= _end && *_current) {
      ++_current;
    }

    llvm::StringRef result{begin, static_cast<size_t>(_current - begin)};
    return result;
  }

private:
  const char *_current;
  const char *_end;
};

} // namespace compass

#endif
