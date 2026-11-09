goal: Add end-to-end support in hip-ep for ordinary (non-fused) ONNX operators (ONNX → HIP → LLVM → runtime/kernel)

# When to use this workflow

The ONNX graph contains a **native `onnx.{OP}`** or **`onnx.Custom`** (domains such as `com.microsoft`) that must be offloaded to hip-ep.

**Reuse first.** If compile time (OnnxToHip / HipToLLVM) can handle it, do not push work into a kernel.

**Out of scope (use another path):**

- `QuantizeLinear` / `DequantizeLinear` wrapping a compute op that should fuse into `hip.q*` → `.cursor/workflows/add_qdq_op.md`
- Weight-quantized LLM linear (fp16 activations × int4/int8 weights) → existing `hip.matmul_nbits` path

---

# Design goals and trade-offs

## Goals

Implement the ONNX operator correctly in hip-ep so the graph can fully offload to GPU. Value comes from:

1. **Correctness:** semantics aligned with the ONNX spec (broadcast, optional inputs, dtype, edge cases)
2. **Fewer paths:** map / decompose / extend an existing family instead of adding a new dialect + ABI
3. **Less runtime work:** constant folding, shapes, and attribute normalization should be finished at compile time whenever possible

## Trade-off rules

| Dimension | Requirement |
|-----------|-------------|
| Semantics | **Aligned** with the ONNX definition. NaN, overflow, rounding, and missing optionals must state whether they are covered |
| Accuracy | Floating-point impls may differ from ORT CPU by dtype-level error; integer / compare ops should match. If you cannot write a threshold → the formula is not finished |
| Performance | **Pass bar:** not worse than a reasonable baseline (existing similar kernel / decomposition result). If decomposition is clearly slower, confirm with the user first |
| Reuse | Same formula shape and same ABI → one `wrap_*` + `kind`; do not copy a full lowering/runtime stack per op |

**Do not:** specialize around one model. Stay faithful to the ONNX definition.

**Criterion:** "slightly slower decomposition that is correct and adds zero new ABI" is allowed; **"wrong semantics that happen to pass on one model" is not**.

## Accuracy budget (write this down before implementing)

- Acceptable error under the target dtype
- Error sources (accumulation order, approximate formula, extra rounding from decomposition)
- How to verify, and the threshold (numeric test L2 / max abs error)

If you cannot write an accuracy budget → the formula/decomposition is incomplete; go back to Stage 1.

---

# Variable definitions

| Variable | Meaning |
|----------|---------|
| `{ROOT}` | hip-ep repo root (all paths in this doc are relative to it; do not use absolute paths) |
| `{OP}` | ONNX operator name, keep original casing (`Softmax`, `MatMul`, `Greater`) |
| `{op}` | HIP assembly name / filename, snake_case (`softmax`, `matmul`, `greater`) |
| `{FAMILY}` | Shared runtime/kernel family name (`elementwise`, `unary`, `reduce`, …); for a standalone ABI this equals `{op}` |
| `{STRATEGY}` | `extend` / `map_hip` / `decompose` / `reuse_lowering` / `native` / `control_flow` |
| `{RUNTIME_KIND}` | `miopen` / `hipblaslt` / `custom_kernel` / `reuse` / `none` |

---

# Preconditions

- Working tree is clean (`git status`). If not → stop and ask to commit or stash first
- An `op_definition` already exists: opset, inputs/outputs/attributes, type constraints, compute semantics. If missing → stop

---

# Step overview

| Step | Content | extend | map_hip | decompose | reuse_lowering | native | control_flow |
|------|---------|:------:|:-------:|:---------:|:--------------:|:------:|:------------:|
| 1.1 | Read ONNX spec, derive Op contract | v | v | v | v | v | v |
| 1.2 | Choose strategy and kernel/ABI family | v | v | v | v | v | v |
| 1.3 | Formula / decomposition / accuracy budget | v | v | v | v | v | v |
| 2.1 | Define `hip.{op}` in `HipOps.td` | - | - | - | skip if already present | v | v |
| 2.2 | DPS / verify in `HipDialect.cpp` | - | - | - | skip if already present | v | v |
| 3.1 | ONNX→HIP C++ pattern | v | v | v | v | v | v(CF) |
| 3.2 | onnx-to-hip LIT | v | v | v | v | v | v |
| 3.3 | Register in `HipBufferize.h` | - | - | - | when adding a new Hip Op | v | v |
| 4.1 | HIP→LLVM lowering | as needed | - | - | register existing | v | v |
| 4.2 | hip-to-llvm LIT | as needed | - | - | register existing | v | v |
| 5.1 | Runtime wrapper + device kernel | as needed | - | - | v | v | v(CF) |
| 5.2 | mock stub + CMake bitcode | as needed | - | - | v | v | v |
| 6.1 | Accuracy and performance validation | v | v | v | v | v | as needed |
| 7.1 | Docs | ask | ask | ask | ask | ask | ask |

The seven stages:

1. Formula derivation and strategy: lock the direction first
2. HIP op definition (only when a new `hip.{op}` is needed)
3. Convert ONNX op to HIP op
4. Convert HIP op to LLVM
5. Runtime + kernel
6. Tests
7. Docs

Entering Stage 2 before Stage 1 is finalized will cause rework on every later layer.

---

# Stage 1: Formula derivation and direction

This stage writes no implementation. The deliverable is a reviewable design: Op contract table, `{STRATEGY}` / `{RUNTIME_KIND}`, formula or decomposition, accuracy budget.

---

## 1.1 Read the ONNX spec and derive the Op contract

**Highest priority: the `hip.{op}` signature is determined by the ONNX operator spec, not by copying some existing HIP op.**

For `com.microsoft` ops, follow that Custom op's documentation and record `function_name` + `domain_name`.

### 1.1.1 Copy these fields from the spec

| Field | Source | Impact |
|-------|--------|--------|
| Input names / order / count | ONNX op spec `Inputs` | Op operand names and order |
| Output names / count | ONNX op spec `Outputs` | DPS init names |
| Optional inputs | `(optional)` in spec | `Optional<...>` |
| Variadic inputs | `(variadic)` in spec | `Variadic<...>` + `AttrSizedOperandSegments` |
| Attributes | spec `Attributes` | `*Attr`, including defaults |
| Type constraints | spec `Type Constraints` | Supported element-type set |
| Broadcast semantics | whether spec declares multidirectional broadcasting | Whether a broadcast base class / 4D ABI is needed |
| Compute semantics | spec description | Input to formula / decomposition |
| Data-dependent shape | output dims depend on runtime scalars (k, depth, etc.) | reify + host readback |
| Control flow | If / Loop | `{STRATEGY}=control_flow` |

### 1.1.2 Naming rules (important)

**Operand / output names should follow ONNX names**, not invented `lhs`/`rhs`/`input`/`output`:

| ONNX op | Input names | Output names | Suggested HIP operands |
|---------|-------------|--------------|------------------------|
| Add / Sub / Mul / Div | `A`, `B` | `C` | `$A`, `$B`, `$C` |
| MatMul | `A`, `B` | `Y` | `$A`, `$B`, `$Y` |
| Gemm | `A`, `B`, `C` | `Y` | `$A`, `$B`, `$C`, `$Y` |
| Conv | `X`, `W`, `B` | `Y` | `$X`, `$W`, `$B`, `$Y` |
| Relu / Sigmoid / Softmax | `X` | `Y` | `$X`, `$Y` |
| ReduceMean / ReduceSum | `data`, `axes` | `reduced` | `$data`, `$axes`, `$reduced` |
| Min / Max | `data_0` (variadic) | `min` / `max` | `$data_0`, `$min` / `$max` |
| Concat | `inputs` (variadic) | `concat_result` | `$inputs`, `$concat_result` |
| Gather | `data`, `indices` | `output` | `$data`, `$indices`, `$output` |

**Exceptions and priority:**

1. ONNX spec names **win**
2. If joining an existing shared ABI family that already uses other names → **keep the family consistent**; do not rename existing ops as a drive-by in the same PR
3. DPS init operand accessor names must match `Hip_DpsOp`'s `outsAccessor` (default `"Output"` maps to `$output`; `$Y` uses `"Y"`, `$C` uses `"C"`)

### 1.1.3 Output

One contract table (include it in the PR / commit message) listing operands, type constraints, attributes, optional/variadic, broadcast, and shape rules. Every later layer follows this table.

---

## 1.2 Choose strategy and kernel / ABI family

**Answer these two questions before writing any code:**

1. Is this path map / decompose / extend / a new Hip Op?
2. If runtime is needed: which other ops will share this kernel and ABI?

Once an ABI lands, the whole stack must change together, so design by **family**, not by a single op. If useful, look up whether similar ops are isomorphic (treat web results as reference only if confidence is below 80%).

### 1.2.1 Strategy priority (top-down when `impl_strategy=auto`)

| Priority | `{STRATEGY}` | When | `{RUNTIME_KIND}` | Skip |
|----------|--------------|------|------------------|------|
| 0 | `extend` | Conversion/Runtime already exist; only add a case (Conv 1D+group, Reduce new dtype) | keep existing | do not create a new Hip Op |
| 1 | `map_hip` | Semantically equivalent to an existing `hip.*` (`onnx.RMSNormalization` → `hip.rms_norm`) | `reuse` | Stages 2 / 4 / 5 |
| 2 | `decompose` | Can be expressed equivalently with already-supported ops | `none` | Stages 2 / 4 / 5 |
| 3 | `reuse_lowering` | New `hip.{op}`, but lowering/runtime use an existing unary/reduce/elementwise family | per backend | do not create a new lowering file |
| 4 | `native` | Needs a new Hip Op + new lowering + new runtime (MIOpen / hipBLASLt / custom kernel) | per backend | none |
| 5 | `control_flow` | `onnx.If` / Loop-class | dedicated runtime | see 3.1-CF |

Forced `decompose` that cannot actually decompose → stop and error. Forced `native` skips 0–2.

### 1.2.2 Decomposition discipline (`decompose`)

Inspect `populate*ConversionPatterns` already registered in `convertComputeOps` in `lib/Conversion/OnnxToHip/OnnxToHip.cpp`.

| Decomposition target | When it applies | Notes |
|----------------------|-----------------|-------|
| Existing `onnx.*` | The decomposed ops are already registered in OnnxToHip | **Forbidden as the default.** `convert-onnx-to-hip` uses `GreedyRewriteStrictness::ExistingOps`; newly created `onnx.*` in the same round **will not be revisited** |
| Existing `hip.*` | Equivalent to an existing Hip op (e.g. `Greater` → `hip.less(B, A)`) | **Must emit `hip.*` directly inside the same pattern**; see `GreaterConversion.cpp` |

If decomposition causes a clear performance hit → confirm with the user before continuing.

### 1.2.3 Family classification reference

| ONNX op class | Existing impl | Recommendation for a new op |
|---------------|---------------|-----------------------------|
| Binary elementwise (Add/Sub/Mul/Div/Min/Max/Pow/And/Or) | `ElementwiseLowering.cpp`, `real/elementwise.cpp`, `elementwise_binary_kernel.hip` | Extend `kind` or existing binary lowering |
| Unary elementwise (Abs/Neg/Relu/Sigmoid/Ceil/Log/…) | `UnaryElementwiseLowering.cpp`, `elementwise_unary_kernel.hip` | `{STRATEGY}=reuse_lowering` |
| Reduce (Sum/Mean/Max/Min/Prod/L2) | `ReduceLowering.cpp`, `real/reduce_*.cpp` | Register into the reduce family |
| MatMul / Gemm / Conv | hipBLASLt / MIOpen | Evaluate independently; do not force the elementwise ABI |
| Compare (Less/Greater/Equal) | existing `hip.less` etc. + possible decompose | Prefer map/decompose |
| Control flow | `IfOutline` / `IfLowering` / `runtime_if` | `control_flow` |

### 1.2.4 Reuse decision rules

| Situation | Action |
|-----------|--------|
| Semantically equivalent to an existing `hip.*` | `map_hip`; Conversion only |
| Same formula shape, same ABI parameter set | **One** `wrap_{FAMILY}` + `kind`; lowering uses a template + `kind` specialization; new ops add one registration line |
| Same formula shape, different parameter set (extra axes/keepdims/alpha) | Same kernel file, same `wrap_*`, ABI reserves that parameter; other ops pass defaults |
| Different formula shape | Independent `kind` branch in the same kernel file; ABI field meaning is interpreted by `kind`, documented in the file-header comment |
| Completely different (MatMul vs elementwise) | Independent `wrap_*` |

### 1.2.5 `kind` enum discipline

Introduce `kind` only when ≥2 ops truly share the same kernel. After introducing it:

- Compiler-side enum (`HipToLLVMUtils.h`) and runtime-side macros (`lib/Runtime/hipdnn_ep_runtime.h`) **must have matching values**, with "Must match ..." comments on both sides
- Unsupported `kind` in the kernel must return an error; do not silently take a default branch

### 1.2.6 ABI extension discipline

When changing an existing `wrap_*` signature, the following must stay in sync in the **same PR**:

`HipToLLVM lowering` → `lib/Runtime/hipdnn_ep_runtime.h` → `lib/Runtime/real/*.cpp` → `lib/Runtime/mock/mock_gpu.cpp` → `lib/Runtime/Kernels/include/hip_custom_kernels.h` → kernel → `test/lit/Conversion/hip-to-llvm/test_*.mlir`

---

## 1.3 Formula derivation and compile-time plan

**Finish the paper derivation before touching code.** Put the result as comments at the top of the conversion or runtime source file (see "Comment style").

### 1.3.1 General derivation flow

1. Write ONNX semantics: `Y = Op(A, B, ..., attrs)`
2. Mark which quantities are compile-time constants (attributes, static shapes, constant inputs)
3. Decide: map / decompose into existing `hip.*` / native kernel
4. Check: does decomposition change broadcast timing, NaN, integer overflow, or optional defaults

### 1.3.2 Deliverable shape by strategy

**`map_hip`:** Write the equivalence, e.g. `RMSNormalization(X, scale) ≡ hip.rms_norm(...)`, and list how attributes map.

**`decompose`:** Step-by-step formula; every step's result must be existing `hip.*` (default):

```
onnx.Greater(A, B) ≡ hip.less(B, A)
```

State known limits (e.g. NaN vs IEEE total-order differences).

**`native` / `reuse_lowering`:** Write the formula the kernel will execute; mark coefficients that can be folded in lowering as compile-time.

### 1.3.3 Folding-location discipline

- Constants, default attributes, and static axis normalization that can be computed at **compile time** must be done in OnnxToHip / HipToLLVM
- Illegal attributes (e.g. statically out-of-range `axis`, constant divisor of 0) → `notifyMatchFailure`; do not leave a silent runtime failure
- Host reads of GPU scalars must use a synchronized HIP readback (`readbackScalarToHost`); do not use a bare `tensor.extract` / `memref.load` of device data

---

# Stage 2: HIP op definition

The Stage 1 contract table is the only input. Skip this stage for `map_hip` / `decompose` / `extend` (no new Hip Op).

At the end of this stage, `hip.{op}` should parse with `hip-mlir-opt` and print back identically.

---

## 2.1 Define `hip.{op}` in `HipOps.td`

**File:** `include/hip/Dialect/IR/HipOps.td`

### Base-class choice (from the ONNX spec, not by copying an existing op)

| ONNX semantics | Base class |
|----------------|------------|
| Declares multidirectional broadcasting | `Hip_DpsOp_Broadcast` (includes broadcast shape reification) |
| Output shape == some input shape | `Hip_DpsOp` + `autoReify=1` |
| Reduce-class | `Hip_DpsOp_Reduction` (if applicable) |
| Shape needs custom inference (MatMul, data-dependent dims, etc.) | `Hip_DpsOp` + `autoReify=0` + `HipReifyResultShapesImpl.cpp` |
| Non-compute / resource management | `Hip_Op` (see `Hip_AllocOp`) |

Reference templates: DPS with no attributes → `Hip_MatmulOp`; attributes + optional inputs → `Hip_ConvOp`; multi-output DPS → `TopKOp`; control flow → `IfOp`.

### Skeleton

```tablegen
def Hip_ReluOp : Hip_DpsOp<"relu", /*traits=*/[], /*outsAccessor=*/"Y"> {
  let summary = "ONNX Relu";
  let description = [{
    Y = max(X, 0). Operand names follow the ONNX Relu spec (X -> Y).
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    Hip_TensorOrMemRef:$X,
    Hip_TensorOrMemRef:$Y
  );

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $X `:` type($X) `)`
    `outs` `(` $Y `:` type($Y) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}
```

**Constraints:**

- First operand is always `Hip_ContextType:$ctx`
- Use declarative `assemblyFormat`; do not set `hasCustomAssemblyFormat = 1`
- ONNX optional inputs → `Optional<Hip_TensorOrMemRef>`; variadic → `Variadic<...>` + `AttrSizedOperandSegments`
- Integer array attributes `I64ArrayAttr`, single integer `I64Attr`, float attributes `F64Attr` / `F32Attr` (align with the existing family ABI)
- If output shape cannot be fully determined at compile time, attach `ReifyRankedShapedTypeOpInterface` (already declared on `Hip_DpsOp`; write the body by hand when `autoReify=0`)

---

## 2.2 DPS interface and on-demand verify / reify

**File:** `lib/Dialect/IR/HipDialect.cpp`

```cpp
//===----------------------------------------------------------------------===//
// ReluOp: ins(X), outs(Y)
//===----------------------------------------------------------------------===//

MutableOperandRange ReluOp::getDpsInitsMutable() { return getYMutable(); }

void ReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  emitDpsMemoryEffects(getDpsInputOperands(), getDpsInitsMutable(), effects);
}
```

Insert with the same section-banner style as existing ops, in the same order as definitions in `HipOps.td`. For multiple outputs use the corresponding `getValuesMutable()` / `getOutputsMutable()`.

### Shape reification (as needed)

Output has data-dependent dims (OneHot depth, TopK k, Compress length) or needs dynamic-shape refinement:

1. `autoReify=0`
2. Implement `{OP}Op::reifyResultShapes` in `lib/Dialect/IR/HipReifyResultShapesImpl.cpp`
3. Add a case in `test/lit/Dialect/hip-infer-shapes.mlir`, RUN `--hip-infer-shapes`

References: `OneHotOp`, `GatherElementsOp`, `TopKOp`.

### verify / inferReturnTypes (as needed)

For control-flow or complex ops, implement `{OP}Op::verify()` / `verifySymbolUses()` / `inferReturnTypes()` in `HipDialect.cpp`. See `IfOp`.

---

# Stage 3: Convert ONNX ops to HIP ops

At the end of this stage, `hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip` should eliminate the target `onnx.{OP}` (or `onnx.Custom`), and any new Hip Op must be bufferizable.

**Do not** write a parallel PDLL fusion for ordinary ops; this path uses C++ `RewritePattern`.

---

## 3.1 C++ Conversion pattern

**Create or edit:** `lib/Conversion/OnnxToHip/{OP}Conversion.cpp`  
**Declare:** `lib/Conversion/OnnxToHip/OnnxToHipUtils.h` → `populate{OP}ConversionPatterns`  
**Register:** `convertComputeOps` in `lib/Conversion/OnnxToHip/OnnxToHip.cpp`  
**Build:** `lib/Conversion/OnnxToHip/CMakeLists.txt`

`map_hip` / `extend` may edit an existing file (e.g. `NormConversion.cpp`); do not force a new file.

### Match entry

| Op source | Pattern match |
|-----------|---------------|
| Standard ONNX | `"onnx.{OP}"` |
| `com.microsoft` etc. | `"onnx.Custom"` + check `function_name` + `domain_name` |

### 3.1-A `decompose`

When decomposing to `hip.*`, must `getContextArg` and create inits.

```cpp
struct GreaterDecompose : public mlir::RewritePattern {
  GreaterDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Greater", /*benefit=*/1, ctx) {}
  // matchAndRewrite: getContextArg, create init, emit hip.less(B, A)
};
```

Put the equivalence formula and known limits in the file header (see `GreaterConversion.cpp`).

### 3.1-B `map_hip`

Map a new ONNX variant onto an existing `hip.*` without a new Hip Op / Runtime. See `RMSNormalizationToHip` in `NormConversion.cpp`.

### 3.1-C `native` / `reuse_lowering`

```cpp
struct ReluToHip : public mlir::RewritePattern {
  ReluToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Relu", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    // extract operands/attrs -> createEmptyTensor / createBroadcastEmptyTensor
    // -> ReluOp::create -> replaceOp
  }
};
```

### Init tensor

| Scenario | Helper |
|----------|--------|
| Output shape = some input shape | `createEmptyTensor(rewriter, loc, resultType, source)` |
| ONNX broadcast binary op | `createBroadcastEmptyTensor(rewriter, loc, resultType, {a, b})` |
| MatMul-style inference | Manual `tensor::DimOp` + `tensor::EmptyOp` |
| Data-dependent scalar (k, depth) | `readbackScalarToHost` (`ReadbackScalar.h`) |
| Multiple outputs | Create an init for each output |

### `com.microsoft` Custom

```cpp
struct BiasGeluToHip : public mlir::RewritePattern {
  BiasGeluToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op, ...) const override {
    auto fn = op->getAttrOfType<StringAttr>("function_name");
    auto domain = op->getAttrOfType<StringAttr>("domain_name");
    if (!fn || fn != "BiasGelu" || !domain || domain != "com.microsoft")
      return failure();
    // ...
  }
};
```

### 3.1-CF Control flow

Do not use ordinary linear Conversion in 3.1. Extra pieces:

| Component | File |
|-----------|------|
| Outline Pass | `lib/Conversion/OnnxToHip/IfOutline.cpp` |
| Pass declare / register | `include/hip/Conversion/OnnxToHip/Passes.h`, `include/hip/InitAllPasses.h` |
| Pipeline | `lib/Dialect/Transforms/Pipelines.cpp` (**before** `--convert-onnx-to-hip`) |
| Hip Op + verify | `HipOps.td` + `HipDialect.cpp` |
| Lowering / Runtime | `IfLowering.cpp`, `hipdnn_ep_runtime_if.cpp` + bitcode |
| Lit | outline / lowering / e2e in separate files |

---

## 3.2 onnx-to-hip LIT

**File:** `test/lit/Conversion/onnx-to-hip/test_{op}.mlir`

```mlir
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s
```

**CHECK rules:**

- `CHECK-NOT: onnx.{OP}` (Custom: also `CHECK-NOT: onnx.Custom`)
- `native` / `reuse_lowering`: have `hip.{op}` and `tensor.empty`
- `decompose`: CHECK the decomposition products (`hip.*`); do not only write `CHECK-NOT`
- `map_hip`: CHECK the target existing `hip.*`
- Include `@main_graph`
- **dynamic shape** (native / not pure metadata): at least one `tensor<?x...>` case, CHECK `tensor.dim` + `tensor.empty(%dim...)`
- Cover forms declared by the spec: missing optional inputs, broadcast, main dtypes
- File-header comment describing the conversion (passthrough / decompose formula / map target)
- Usually 1–2 examples; do not add too many
- Use `CHECK-NEXT` to lock adjacent lines

Control flow: separate files such as `test_if_outline.mlir`; do not dump into ordinary `test_{op}.mlir`.

---

## 3.3 Bufferization registration

Only when this PR **creates** a new Hip DPS Op.

**File:** `include/hip/Dialect/IR/HipBufferize.h`

```cpp
ReluOp::attachInterface<HipDstBufferizableModel<ReluOp>>(*ctx);
```

Missing this → `error: op was not bufferized`.

`hip-mlir-opt.cpp` already references this uniformly via `registerHipBufferizableOpInterfaceModels`; **do not** edit the tool file as well.

---

# Stage 4: Convert HIP ops to LLVM

Skip for `map_hip` / `decompose` (products already have lowering). For `extend`, change only if ABI or dispatch changes.

---

## 4.1 HIP → LLVM lowering

**First check whether it can register into an existing file; do not create `{OP}Lowering.cpp` without need.**

| Op type | Registration site |
|---------|-------------------|
| Unary elementwise | `UnaryElementwiseLowering.cpp` |
| Reduce | `ReduceLowering.cpp` |
| Existing similar binary | corresponding `*Lowering.cpp` |

**New family:** create `lib/Conversion/HipToLLVM/{OP}Lowering.cpp`  
**Also edit:** `HipToLLVMUtils.h` (`kWrap*`, `kind` enum, populate declaration), `HipToLLVM.cpp`, and `CMakeLists.txt` when adding a new file

### Templated reuse

When formulas in a family are isomorphic, use a template + `kind` specialization; a new op is only a thin shell:

```cpp
template <typename OpTy, HipdnnUnaryKind Kind>
struct UnaryElementwiseLowering : public ConvertOpToLLVMPattern<OpTy> { /* ... */ };
```

Do not force a template when formulas are not isomorphic; split an independent pattern and share the kernel's `kind` branch.

### Required items

- `ConvertOpToLLVMPattern<Hip::{OP}Op>`
- Fold compile-time constants per 1.3; illegal attributes / unsupported element types → `notifyMatchFailure`
- Dynamic shape: static dims via `LLVM::ConstantOp`, dynamic dims via `MemRefDescriptor::size`
- Pointers passed to runtime use `extractContiguousMemRefPtr`; non-contiguous inputs must be materialized first
- Runtime returns `i32`
- Binary ops that need ONNX broadcast: pass **4D shape** (N,C,H,W; rank 1–3 left-padded); if runtime shapes differ, `hip_expand` then compute. See `OrLowering.cpp` + `real/or.cpp`

---

## 4.2 hip-to-llvm LIT

**File:** `test/lit/Conversion/hip-to-llvm/test_{op}.mlir`

```mlir
// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s
```

**CHECK rules:**

- `llvm.call @wrap_{FAMILY}` (or `@wrap_{op}`) with a **full parameter type list** matching the `hipdnn_ep_runtime.h` declaration field-by-field
- If lowering folded constants: CHECK the expected values and comment the formula
- memrefs use device address space 1
- Dynamic-shape case: CHECK `llvm.extractvalue` for sizes
- 1–2 examples are enough
- Use `CHECK-NEXT`

`reuse_lowering` may fold the case into the family's existing `test_*.mlir`; a separate file per unary is not required. If merged, CHECKs must distinguish the new `kind` / new symbol.

---

# Stage 5: Runtime + Kernel

Skip for `map_hip` / `decompose` (fully reuse existing `wrap_*`).

ABI is the contract of this stage: the declaration in `hipdnn_ep_runtime.h`, the lowering call site, the `real/` impl, the mock stub, and the kernel declaration must match **verbatim**.

---

## 5.1 Runtime wrapper + device kernel

### 5.1.1 ABI declaration

**File:** `lib/Runtime/hipdnn_ep_runtime.h`

- `extern "C"` declaration of `wrap_{FAMILY}` (or `wrap_{op}`)
- `kind` macros (if the family uses them) with a "Must match ... in HipToLLVMUtils.h" comment
- Parameter order matches lowering construction order **verbatim**
- First parameter is `RuntimeState* state`; return `int` (0 = success)

### 5.1.2 Host wrapper

**File:** `lib/Runtime/real/{FAMILY}.cpp` (or `{op}.cpp`)

Backend choice: MIOpen / hipBLASLt see `real/matmul.cpp`, `real/activation.cpp`; Custom Kernel below.

- File header contains the formula (see "Comment style")
- Validate inputs: null pointers, rank/shape consistency, unsupported dtype/kind → return a negative value and `fprintf(stderr, ...)`
- Map dtype from `HIPDNN_EP_DATATYPE_*` to `HIP_DTYPE_*`
- Debug branch (`hipdnn_ep_debug_enabled()`) prints shapes and key coefficients
- Freeable resources are initialized to `nullptr` at the top of the function, with a unified `goto cleanup`; do not destroy in a block then goto

### 5.1.3 Device kernel (`{RUNTIME_KIND}` = `custom_kernel`)

**File:** `lib/Runtime/Kernels/hip/{FAMILY}_kernel.hip`  
**Declare:** add `HIP_KERNEL_API int hip_{FAMILY}(...)` in `lib/Runtime/Kernels/include/hip_custom_kernels.h`  
**Build:** `_kernel_sources` in `lib/Runtime/Kernels/CMakeLists.txt`

- Cover dtypes from ONNX type constraints via template + dtype switch dispatch
- Saturation / bounds use an explicit `__device__` helper (HIP compile flags can make `std::numeric_limits` members host-only)
- Comment why edge-case specializations exist
- Unsupported `kind`/dtype return an error code; no silent fallback
- Clear error state before launch; return `hipGetLastError()` after launch; **do not** unconditionally return `hipSuccess`
- Multiple files may include via `.inc.hip`

`real/{FAMILY}.cpp` includes `hip_custom_kernels.h` and calls `hip_{FAMILY}(stream, ...)`.

References: `bias_gelu_kernel.hip` + `real/bias_gelu.cpp`.

---

## 5.2 mock stub + CMake bitcode

### 5.2.1 Mock stub

**File:** `lib/Runtime/mock/mock_gpu.cpp`

A stub with the **exact same signature** as `hipdnn_ep_runtime.h`; `MOCK_PRINT` key parameters then return 0. A signature mismatch will fail to link under the mock build.

### 5.2.2 Bitcode build

**File:** `lib/Runtime/CMakeLists.txt`

- `compile_to_bitcode(real/{FAMILY}.cpp runtime_{FAMILY}.bc)` and add it to the bitcode module list
- Newly included runtime headers are added to bitcode build dependencies

Control-flow example: `compile_to_bitcode(hipdnn_ep_runtime_if.cpp runtime_if.bc)`.

---

# Stage 6: Tests

> Follow:
> **Do not compile proactively**; ask first when a build/run is needed.
> **Working path:** pick one directory as the working path, and confirm the parent is a dedicated test directory; if not, get approval before creating it.
> Save all files produced during testing under that working path so they can be deleted afterward without leaving junk.

---

## 6.1 Accuracy and performance validation

### 6.1.1 Accuracy

- Generate test scripts, data, and models in the working directory (do not dump one-off scripts into `test/numeric/`): ORT CPU vs HIP EP, with thresholds from the Stage 1.3 accuracy budget
- Cover main dtypes from ONNX type constraints, missing optional inputs, broadcast (if applicable), dynamic shape (if LIT covered it)
- Use `HIPDNN_EP_STRICT=1` to confirm the graph fully offloaded to GPU, not CPU fallback compared against CPU
- Regression tests for landing go in `test/numeric/tests/test_{op}.py` (framework in `test/numeric/README.md`); see `test_bias_gelu.py`, `test_topk.py`

### 6.1.2 Performance (decides whether to land)

- Run serially; disable `HIPDNN_EP_PERF` / `HIPDNN_EP_DEBUG`
- After changing runtime/kernel, invalidate the model cache before measuring
- Decomposition strategy: if clearly slower than a native expectation, put the conclusion in the PR and confirm the user still wants to land
- Put conclusions in the PR: performance delta + accuracy delta

### 6.1.3 E2E LIT (as needed)

```mlir
// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s
```

- `@main_graph` + `onnx.EntryPoint`
- CHECK `inference_init` / `compute` / `cleanup` / `get_metadata_json`
- native: CHECK `llvm.func @wrap_{...}`
- `CHECK-NOT: onnx.{OP}`

---

# Stage 7: Docs

**Ask the user whether docs need updating**; do not do this by default.

---

## 7.1 Docs

- Append or edit one row in the README Supported Operations table:

| Implementation | Backend column |
|----------------|----------------|
| MIOpen | `MIOpen` |
| hipBLASLt | `hipBLASLt` |
| Custom Kernel | `Custom HIP Kernel` |
| Decomposition | `Decomposed` or the equivalent formula |
| Mapped to existing Hip Op | actual backend |
| `com.microsoft` | append `(com.microsoft)` after the name |

- Complex ops get extra notes in `docs/design/`
- PR body: strategy, formula/decomposition, accuracy budget and measured results

---

# Files typically touched

Stage 1 produces design conclusions only; no code changes. The table below is for a **full `native` + custom kernel** path; other strategies drop rows per the overview table.

| Stage | Layer | Path | Required |
|:-----:|-------|------|:--------:|
| 2.1 | Op definition | `include/hip/Dialect/IR/HipOps.td` | new Hip Op |
| 2.1 | Shape reification | `lib/Dialect/IR/HipReifyResultShapesImpl.cpp` | custom shape |
| 2.2 | DPS interface | `lib/Dialect/IR/HipDialect.cpp` | new Hip_DpsOp |
| 3.1 | Conversion | `lib/Conversion/OnnxToHip/{OP}Conversion.cpp` | v |
| 3.1 | Conversion declare | `lib/Conversion/OnnxToHip/OnnxToHipUtils.h` | v |
| 3.1 | Conversion register | `lib/Conversion/OnnxToHip/OnnxToHip.cpp` | v |
| 3.1 | Conversion build | `lib/Conversion/OnnxToHip/CMakeLists.txt` | when adding a file |
| 3.2 | Conversion LIT | `test/lit/Conversion/onnx-to-hip/test_{op}.mlir` | v |
| 3.3 | Bufferize register | `include/hip/Dialect/IR/HipBufferize.h` | new Hip Op |
| 4.1 | Lowering | `lib/Conversion/HipToLLVM/{OP}Lowering.cpp` or existing family file | not map/decompose |
| 4.1 | Lowering utils/constants/kind | `lib/Conversion/HipToLLVM/HipToLLVMUtils.h` | not map/decompose |
| 4.1 | Lowering register | `lib/Conversion/HipToLLVM/HipToLLVM.cpp` | not map/decompose |
| 4.1 | Lowering build | `lib/Conversion/HipToLLVM/CMakeLists.txt` | when adding a file |
| 4.2 | Lowering LIT | `test/lit/Conversion/hip-to-llvm/test_{op}.mlir` | not map/decompose |
| 5.1 | Runtime ABI | `lib/Runtime/hipdnn_ep_runtime.h` | new wrap |
| 5.1 | Runtime impl | `lib/Runtime/real/{FAMILY}.cpp` | new wrap |
| 5.1 | Kernel | `lib/Runtime/Kernels/hip/{FAMILY}_kernel.hip` | custom_kernel |
| 5.1 | Kernel declare | `lib/Runtime/Kernels/include/hip_custom_kernels.h` | custom_kernel |
| 5.1 | Kernel build | `lib/Runtime/Kernels/CMakeLists.txt` | when adding a file |
| 5.2 | Mock | `lib/Runtime/mock/mock_gpu.cpp` | new wrap |
| 5.2 | Bitcode build | `lib/Runtime/CMakeLists.txt` | new wrap |
| 6.1 | E2E / numeric | `test/lit/e2e/`, `test/numeric/tests/` | as needed |
| 7.1 | README | `README.md` | as needed |
| 7.2 | Doc | `docs/supported-operations.md` | as needed |

`control_flow` also edits Passes / InitAllPasses / Pipelines / outline.

---

# Code style

## Naming

| Object | Convention | Example |
|--------|------------|---------|
| TableGen def | `Hip_{OP}Op` | `Hip_ReluOp` |
| Assembly name | ONNX name in snake_case | `relu`, `reduce_mean` |
| operand / attribute | ONNX spec name (see 1.1.2) | `$X`, `$Y`, `$alpha` |
| Conversion file | `{OP}Conversion.cpp` | `ReluConversion.cpp` |
| Symbol constant | `kWrap{Family}` | `kWrapUnaryElementwise` |
| runtime entry | `wrap_{family}` | `wrap_elementwise` |
| kernel entry | `hip_{family}` | `hip_elementwise_unary` |
| kernel file | `{family}_kernel.hip` | `elementwise_unary_kernel.hip` |
| kind enum | compiler `k{Family}{Op}`, runtime `HIPDNN_EP_{FAMILY}_{OP}` | match the existing family |

## General

- C++17; format C++ with clang-format, Python with Ruff
- New files get an AMD MIT license header
- **Code and comments are English only**
- Runtime functions called from generated code must be `extern "C"`; keep declaration and definition export attributes consistent
- Prefer `llvm::seq` / `llvm::seq_inclusive` for compile-time integer ranges
- Match ONNX ops through MLIR's generic `Operation` API; do not add an onnx-mlir dependency
- Adding an op inside a family should look like "one thin shell + one kind", not a copy-paste of a whole stack
- If an experiment fails, revert it completely before trying another approach; do not leave half-finished branch logic

---

# Comment style

## Runtime / kernel / decompose conversion file headers: write the formula, not the process

```cpp
/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// onnx.Greater(A, B) ≡ hip.less(B, A)
// Identity holds for a total order; NaN-bearing float models are not covered.
```

The derivation must be independently verifiable by a reader.

## Reduce comments

1. When the function name fully expresses the purpose
2. Comments that apply to every op in the family
3. Overly long comments
4. Explaining a very simple function

## Comments that are needed

1. For long functions, short step comments by content **inside** the function, not at the top
2. Comments that significantly speed up understanding

## Lowering / Pass: write IR transforms and invariants

- Non-obvious rewrites get a minimal `Before:` / `After:` IR example; update it when the transform changes
- Explain **why** folding/decomposition is done this way, and what goes wrong if it is not

## LIT

Use `CHECK-NEXT`

## Forbidden

- Explaining obvious mechanics (`// add one to i`)
- Hardware model names, model names, or investigation history in generic compiler/dialect/pass comments; put those in tests, design docs, issues, or commit messages
- Using comments to tell a reviewer "what I changed this time"; comments are for the next reader
- Non-English comments
---

# Completion checklist

## Stage 1: Formula derivation and direction

- [ ] Operand / output / attribute names come from the ONNX spec (or a reason is given for keeping existing family names)
- [ ] Optional / variadic inputs are modeled per the spec
- [ ] Supported element types match ONNX type constraints
- [ ] `{STRATEGY}` / `{RUNTIME_KIND}` are chosen with reasons
- [ ] Other candidate ops that can share this kernel/ABI are listed (when runtime is needed)
- [ ] Existing `wrap_*` / `hip.*` is reused, or why a new one is required is stated
- [ ] Formula or decomposition is finished; compile-time work is marked
- [ ] Accuracy budget is written down

## Stage 2: HIP op definition

- [ ] Skipped if no new Hip Op is needed
- [ ] `HipOps.td` definition + declarative `assemblyFormat`
- [ ] Broadcast base class used only when the ONNX spec declares broadcast
- [ ] DPS interface in `HipDialect.cpp` (+ reify / verify as needed)

## Stage 3: Convert ONNX ops to HIP ops

- [ ] C++ pattern registered; `com.microsoft` checks `function_name` + `domain_name`
- [ ] `decompose` emits `hip.*` directly in the same pattern (does not rely on a second match of newly created `onnx.*`)
- [ ] onnx-to-hip LIT: `CHECK-NOT: onnx.{OP}`; native has `hip.{op}`; dynamic shape as needed
- [ ] New Hip Op registered in `HipBufferize.h`

## Stage 4: Convert HIP ops to LLVM

- [ ] Existing lowering files reused first
- [ ] Illegal attributes and unsupported types use `notifyMatchFailure`
- [ ] Dynamic shape / broadcast ABI is correct
- [ ] hip-to-llvm LIT: full ABI signature (+ folded constants if any)

## Stage 5: Runtime + Kernel

- [ ] Skipped if no new wrap is needed
- [ ] Derived formula is in the runtime source file header
- [ ] `kind` enum values match on compiler and runtime sides, with cross-comments
- [ ] ABI matches in five places (lowering / runtime header / real / mock / kernel declaration)
- [ ] Kernel launch returns `hipGetLastError()`
- [ ] Mock stub signature matches; CMake bitcode and header dependencies are registered

## Stage 6: Tests

- [ ] GPU execution is proven, not CPU-fallback compared against CPU
- [ ] Thresholds match the Stage 1 accuracy budget
- [ ] One-off artifacts in the working directory did not pollute the repo

## General

- [ ] All new files have an AMD license header
- [ ] No non-English code/comments
- [ ] Style matches existing ops at the same layer

---

# Agent execution constraints

- **Do not compile proactively** (unless the user explicitly asks)
- **Do not commit / push proactively** (unless the user explicitly asks)
- Follow repo `CLAUDE.md`: tests must prove GPU execution; invalidate the model cache after runtime/kernel changes
- Prefer solving problems at compile time (OnnxToHip / HipToLLVM); kernels are second
- If this conflicts with the QDQ fusion flow: ordinary ops follow this workflow; QDQ fusion follows `add_qdq_op.md`
- Do not start editing `HipOps.td` / Conversion / lowering / runtime until the user confirms Stage 1

---

# Prompt template

```
Please add an ordinary ONNX operator to hip-ep following add_op.md.

Target operator:
- ONNX op: <e.g. Softmax / Greater / BiasGelu>
- Source: <native onnx.{OP} | com.microsoft onnx.Custom function_name=...>
- Implementation preference: <auto | extend | map_hip | decompose | native>
- Target dtype: <f32/f16/bf16/...>

Requirements:
1. First produce the Op contract table (from the ONNX spec), strategy choice, and formula/decomposition; wait for my confirmation before writing code
2. First answer whether this can map/decompose/extend an existing family; if runtime is needed, then answer which ops can share this kernel/ABI
3. Operand and attribute names follow the ONNX spec; if reusing existing family names, explain why
4. Do not push compile-time work into the kernel; decompose must emit hip.* in the same pattern
5. Provide an accuracy budget
6. Do not compile or commit unless I explicitly ask
7. When done, output the list of changed files and a test plan
```
