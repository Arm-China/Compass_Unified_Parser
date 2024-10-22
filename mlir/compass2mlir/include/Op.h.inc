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
          InputOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults,
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
                    ::mlir::OperationState &odsState);
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

#endif // GET_OP_CLASSES
