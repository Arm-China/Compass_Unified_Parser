/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: compass.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace compass {
class InputOp;
} // namespace compass
} // namespace mlir
namespace mlir {
namespace compass {
class TransposeOp;
} // namespace compass
} // namespace mlir
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace compass {

//===----------------------------------------------------------------------===//
// ::mlir::compass::InputOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class InputOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;

public:
  InputOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {},
                            const ::mlir::EmptyProperties &properties = {},
                            ::mlir::RegionRange regions = {})
      : odsAttrs(attrs), odsRegions(regions) {
    if (odsAttrs)
      odsOpName.emplace("compass.Input", odsAttrs.getContext());
  }

  InputOpGenericAdaptorBase(::mlir::Operation *op)
      : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()),
        odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned>
  getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() { return odsAttrs; }
};
} // namespace detail
template <typename RangeT>
class InputOpGenericAdaptor : public detail::InputOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::InputOpGenericAdaptorBase;

public:
  InputOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {},
                        const ::mlir::EmptyProperties &properties = {},
                        ::mlir::RegionRange regions = {})
      : Base(attrs, properties, regions), odsOperands(values) {}

  InputOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs,
                        ::mlir::OpaqueProperties properties,
                        ::mlir::RegionRange regions = {})
      : InputOpGenericAdaptor(values, attrs,
                              (properties
                                   ? *properties.as<::mlir::EmptyProperties *>()
                                   : ::mlir::EmptyProperties{}),
                              regions) {}

  template <typename LateInst = InputOp,
            typename = std::enable_if_t<std::is_same_v<LateInst, InputOp>>>
  InputOpGenericAdaptor(RangeT values, LateInst op)
      : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {
        std::next(odsOperands.begin(), valueRange.first),
        std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() { return odsOperands; }

private:
  RangeT odsOperands;
};
class InputOpAdaptor : public InputOpGenericAdaptor<::mlir::ValueRange> {
public:
  using InputOpGenericAdaptor::InputOpGenericAdaptor;
  InputOpAdaptor(InputOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class InputOp
    : public ::mlir::Op<
          InputOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult,
          ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl,
          ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands,
          ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InputOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = InputOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("compass.Input");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
            std::next(getOperation()->operand_begin(),
                      valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
            std::next(getOperation()->result_begin(),
                      valueRange.first + valueRange.second)};
  }

  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Type resultType0);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();

public:
};
} // namespace compass
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::compass::InputOp)

namespace mlir {
namespace compass {

//===----------------------------------------------------------------------===//
// ::mlir::compass::TransposeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TransposeOpGenericAdaptorBase {
public:
  struct Properties {
    using permTy = ::mlir::ArrayAttr;
    permTy perm;

    auto getPerm() {
      auto &propStorage = this->perm;
      return ::llvm::cast<::mlir::ArrayAttr>(propStorage);
    }
    void setPerm(const ::mlir::ArrayAttr &propValue) { this->perm = propValue; }
    bool operator==(const Properties &rhs) const {
      return rhs.perm == this->perm && true;
    }
    bool operator!=(const Properties &rhs) const { return !(*this == rhs); }
  };

protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;

public:
  TransposeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs,
                                const Properties &properties,
                                ::mlir::RegionRange regions = {})
      : odsAttrs(attrs), properties(properties), odsRegions(regions) {
    if (odsAttrs)
      odsOpName.emplace("compass.Transpose", odsAttrs.getContext());
  }

  TransposeOpGenericAdaptorBase(TransposeOp op);

  std::pair<unsigned, unsigned>
  getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() { return properties; }

  ::mlir::DictionaryAttr getAttributes() { return odsAttrs; }

  ::mlir::ArrayAttr getPermAttr() {
    auto attr = ::llvm::cast<::mlir::ArrayAttr>(getProperties().perm);
    return attr;
  }

  ::mlir::ArrayAttr getPerm();
};
} // namespace detail
template <typename RangeT>
class TransposeOpGenericAdaptor : public detail::TransposeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TransposeOpGenericAdaptorBase;

public:
  TransposeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs,
                            const Properties &properties,
                            ::mlir::RegionRange regions = {})
      : Base(attrs, properties, regions), odsOperands(values) {}

  TransposeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs,
                            ::mlir::OpaqueProperties properties,
                            ::mlir::RegionRange regions = {})
      : TransposeOpGenericAdaptor(
            values, attrs,
            (properties ? *properties.as<Properties *>() : Properties{}),
            regions) {}

  TransposeOpGenericAdaptor(RangeT values,
                            ::mlir::DictionaryAttr attrs = nullptr)
      : TransposeOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = TransposeOp,
            typename = std::enable_if_t<std::is_same_v<LateInst, TransposeOp>>>
  TransposeOpGenericAdaptor(RangeT values, LateInst op)
      : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {
        std::next(odsOperands.begin(), valueRange.first),
        std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() { return (*getODSOperands(0).begin()); }

  RangeT getOperands() { return odsOperands; }

private:
  RangeT odsOperands;
};
class TransposeOpAdaptor
    : public TransposeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TransposeOpGenericAdaptor::TransposeOpGenericAdaptor;
  TransposeOpAdaptor(TransposeOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TransposeOp
    : public ::mlir::Op<
          TransposeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult,
          ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl,
          ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand,
          ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TransposeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TransposeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("perm")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getPermAttrName() { return getAttributeNameForIndex(0); }

  static ::mlir::StringAttr getPermAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("compass.Transpose");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
            std::next(getOperation()->operand_begin(),
                      valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(
        *getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
            std::next(getOperation()->result_begin(),
                      valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(
      Properties &prop, ::mlir::Attribute attr,
      ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx,
                                               const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute>
  getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop,
                  llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name,
                              mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx,
                                    const Properties &prop,
                                    ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(
      ::mlir::OperationName opName, ::mlir::NamedAttrList &attrs,
      llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult
  readProperties(::mlir::DialectBytecodeReader &reader,
                 ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::ArrayAttr getPermAttr() {
    return ::llvm::cast<::mlir::ArrayAttr>(getProperties().perm);
  }

  ::mlir::ArrayAttr getPerm();
  void setPermAttr(::mlir::ArrayAttr attr) { getProperties().perm = attr; }

  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Type resultType0,
                    ::mlir::Value input, ::mlir::ArrayAttr perm);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::Value input,
                    ::mlir::ArrayAttr perm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();

private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name,
                                                     unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() &&
           "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
                                  "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace compass
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::compass::TransposeOp)

#endif // GET_OP_CLASSES
