/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: compass.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::compass::InputOp, ::mlir::compass::TransposeOp
#endif // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

    //===----------------------------------------------------------------------===//
    // Local Utility Method Definitions
    //===----------------------------------------------------------------------===//

    namespace mlir {
  namespace compass {

  static ::llvm::LogicalResult __mlir_ods_local_type_constraint_compass0(
      ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
      unsigned valueIndex) {
    if (!(((::llvm::isa<::mlir::TensorType>(type))) &&
          ([](::mlir::Type elementType) {
            return (elementType.isF32());
          }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
      return op->emitOpError(valueKind)
             << " #" << valueIndex
             << " must be tensor of 32-bit float values, but got " << type;
    }
    return ::mlir::success();
  }

  static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_compass0(
      ::mlir::Attribute attr, ::llvm::StringRef attrName,
      llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) &&
                  (::llvm::all_of(
                      ::llvm::cast<::mlir::ArrayAttr>(attr),
                      [&](::mlir::Attribute attr) {
                        return attr &&
                               (((::llvm::isa<::mlir::IntegerAttr>(attr))) &&
                                ((::llvm::cast<::mlir::IntegerAttr>(attr)
                                      .getType()
                                      .isSignlessInteger(32))));
                      }))))
      return emitError() << "attribute '" << attrName
                         << "' failed to satisfy constraint: 32-bit integer "
                            "array attribute";
    return ::mlir::success();
  }
  static ::llvm::LogicalResult
  __mlir_ods_local_attr_constraint_compass0(::mlir::Operation *op,
                                            ::mlir::Attribute attr,
                                            ::llvm::StringRef attrName) {
    return __mlir_ods_local_attr_constraint_compass0(
        attr, attrName, [op]() { return op->emitOpError(); });
  }
  } // namespace compass
} // namespace mlir
namespace mlir {
namespace compass {

//===----------------------------------------------------------------------===//
// ::mlir::compass::InputOp definitions
//===----------------------------------------------------------------------===//

namespace detail {} // namespace detail
InputOpAdaptor::InputOpAdaptor(InputOp op)
    : InputOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult InputOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void InputOp::build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::Type resultType0) {
  odsState.addTypes(resultType0);
}

void InputOp::build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InputOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult InputOp::verifyInvariantsImpl() {
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_compass0(
              *this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult InputOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace compass
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::compass::InputOp)

namespace mlir {
namespace compass {

//===----------------------------------------------------------------------===//
// ::mlir::compass::TransposeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TransposeOpGenericAdaptorBase::TransposeOpGenericAdaptorBase(TransposeOp op)
    : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()),
      properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::ArrayAttr TransposeOpGenericAdaptorBase::getPerm() {
  auto attr = getPermAttr();
  return attr;
}

} // namespace detail
TransposeOpAdaptor::TransposeOpAdaptor(TransposeOp op)
    : TransposeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TransposeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_perm = getProperties().perm;
  (void)tblgen_perm;
  if (!tblgen_perm)
    return emitError(loc, "'compass.Transpose' op "
                          "requires attribute 'perm'");

  if (tblgen_perm &&
      !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_perm))) &&
        (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_perm),
                        [&](::mlir::Attribute attr) {
                          return attr &&
                                 (((::llvm::isa<::mlir::IntegerAttr>(attr))) &&
                                  ((::llvm::cast<::mlir::IntegerAttr>(attr)
                                        .getType()
                                        .isSignlessInteger(32))));
                        }))))
    return emitError(loc, "'compass.Transpose' op "
                          "attribute 'perm' failed to satisfy constraint: "
                          "32-bit integer array attribute");
  return ::mlir::success();
}

::llvm::LogicalResult TransposeOp::setPropertiesFromAttr(
    Properties &prop, ::mlir::Attribute attr,
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.perm;
    auto attr = dict.get("perm");
    if (attr) {
      auto convertedAttr =
          ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(
              attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `perm` in property conversion: "
                    << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TransposeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx,
                                                   const Properties &prop) {
  ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
  ::mlir::Builder odsBuilder{ctx};

  {
    const auto &propStorage = prop.perm;
    if (propStorage)
      attrs.push_back(odsBuilder.getNamedAttr("perm", propStorage));
  }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TransposeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(llvm::hash_value(prop.perm.getAsOpaquePointer()));
}

std::optional<mlir::Attribute>
TransposeOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop,
                             llvm::StringRef name) {
  if (name == "perm")
    return prop.perm;
  return std::nullopt;
}

void TransposeOp::setInherentAttr(Properties &prop, llvm::StringRef name,
                                  mlir::Attribute value) {
  if (name == "perm") {
    prop.perm =
        ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.perm)>>(
            value);
    return;
  }
}

void TransposeOp::populateInherentAttrs(::mlir::MLIRContext *ctx,
                                        const Properties &prop,
                                        ::mlir::NamedAttrList &attrs) {
  if (prop.perm)
    attrs.append("perm", prop.perm);
}

::llvm::LogicalResult TransposeOp::verifyInherentAttrs(
    ::mlir::OperationName opName, ::mlir::NamedAttrList &attrs,
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  {
    ::mlir::Attribute attr = attrs.get(getPermAttrName(opName));
    if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_compass0(
                    attr, "perm", emitError)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

::llvm::LogicalResult
TransposeOp::readProperties(::mlir::DialectBytecodeReader &reader,
                            ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>();
  (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.perm)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TransposeOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties();
  (void)prop;
  writer.writeAttribute(prop.perm);
}

::mlir::ArrayAttr TransposeOp::getPerm() {
  auto attr = getPermAttr();
  return attr;
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder,
                        ::mlir::OperationState &odsState,
                        ::mlir::Type resultType0, ::mlir::Value input,
                        ::mlir::ArrayAttr perm) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().perm = perm;
  odsState.addTypes(resultType0);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder,
                        ::mlir::OperationState &odsState,
                        ::mlir::TypeRange resultTypes, ::mlir::Value input,
                        ::mlir::ArrayAttr perm) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().perm = perm;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                        ::mlir::TypeRange resultTypes,
                        ::mlir::ValueRange operands,
                        ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
        &odsState.getOrAddProperties<TransposeOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
        odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(
            odsState.name, properties,
            odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult TransposeOp::verifyInvariantsImpl() {
  auto tblgen_perm = getProperties().perm;
  (void)tblgen_perm;
  if (!tblgen_perm)
    return emitOpError("requires attribute 'perm'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_compass0(
          *this, tblgen_perm, "perm")))
    return ::mlir::failure();
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_compass0(
              *this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_compass0(
              *this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TransposeOp::verifyInvariants() {
  if (::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace compass
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::compass::TransposeOp)

#endif // GET_OP_CLASSES
