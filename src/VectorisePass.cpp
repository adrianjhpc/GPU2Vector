#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCFOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct CudaToVectorPattern : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Configuration: Vector width (e.g., 8 for AVX-256 f32)
    const int64_t vWidth = 8;
    Type f32Type = rewriter.getF32Type();
    VectorType vType = VectorType::get({vWidth}, f32Type);
    VectorType maskType = VectorType::get({vWidth}, rewriter.getI1Type());

    // Setup Loop: Change step to vWidth
    if (op.getStep().empty()) return failure();
    Value ub = op.getUpperBound()[0];
    Value iv = op.getInductionVars()[0];
    
    rewriter.modifyOpInPlace(op, [&]() {
      Value newStep = rewriter.create<arith::ConstantIndexOp>(loc, vWidth);
      op.getStepMutable().assign(newStep);
    });

    // Create Mask: (index + lane) < UpperBound
    // This handles the "Tail Loop" (remainder) automatically
    rewriter.setInsertionPointToStart(op.getBody());
    Value diff = rewriter.create<arith::SubIOp>(loc, ub, iv);
    Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, diff);
    Value zeroPadding = rewriter.create<arith::ConstantOp>(loc, f32Type, rewriter.getF32FloatAttr(0.0f));

    // Transform Body Operations
    for (Operation &innerOp : llvm::make_early_inc_range(*op.getBody())) {
      
      // LOAD -> Vector Transfer Read (Masked)
      if (auto load = dyn_cast<memref::LoadOp>(innerOp)) {
        rewriter.setInsertionPoint(load);
        auto vRead = rewriter.create<vector::TransferReadOp>(
            loc, vType, load.getMemRef(), load.getIndices(), mask, zeroPadding);
        rewriter.replaceOp(load, vRead.getResult());
      }

      // ARITH -> Handle potential scalar-vector mix (Broadcasting)
      else if (auto addf = dyn_cast<arith::AddFOp>(innerOp)) {
        rewriter.setInsertionPoint(addf);
        Value lhs = addf.getLhs();
        Value rhs = addf.getRhs();
        if (lhs.getType() != vType) lhs = rewriter.create<vector::BroadcastOp>(loc, vType, lhs);
        if (rhs.getType() != vType) rhs = rewriter.create<vector::BroadcastOp>(loc, vType, rhs);
        auto vAdd = rewriter.create<arith::AddFOp>(loc, lhs, rhs);
        rewriter.replaceOp(addf, vAdd.getResult());
      }

      // ATOMIC ADD -> Vector Reduction
      else if (auto atomic = dyn_cast<memref::GenericAtomicRMWOp>(innerOp)) {
        rewriter.setInsertionPoint(atomic);
        // Reduce the vector to a scalar sum
        auto reduction = rewriter.create<vector::ReductionOp>(
            loc, vector::CombiningKind::ADD, atomic.getValue());
        // Apply the scalar reduction to the original memory
        rewriter.create<memref::GenericAtomicRMWOp>(
            loc, reduction.getResult(), atomic.getMemref(), atomic.getIndices());
        rewriter.eraseOp(atomic);
      }

      // STORE -> Vector Transfer Write (Masked)
      else if (auto store = dyn_cast<memref::StoreOp>(innerOp)) {
        rewriter.setInsertionPoint(store);
        Value val = store.getValueToStore();
        if (val.getType() != vType) val = rewriter.create<vector::BroadcastOp>(loc, vType, val);
        rewriter.create<vector::TransferWriteOp>(
            loc, val, store.getMemRef(), store.getIndices(), mask);
        rewriter.eraseOp(store);
      }

      // SHARED MEMORY BARRIER -> Remove (CPU vectors are synchronous)
      else if (isa<gpu::BarrierOp>(innerOp)) {
        rewriter.eraseOp(&innerOp);
      }
    }
    return success();
  }
};

struct CudaToVectorPass : public PassWrapper<CudaToVectorPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CudaToVectorPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Shared Memory -> Stack Allocation (Alloca)
    module.walk([&](gpu::AllocOp alloc) {
      if (alloc.getType().getMemorySpaceAsInt() == 3) { // 3 is CUDA Shared
          OpBuilder b(alloc);
          auto stackMem = b.create<memref::AllocaOp>(alloc.getLoc(), alloc.getType().cast<MemRefType>());
          alloc.replaceAllUsesWith(stackMem.getResult());
          alloc.erase();
      }
    });

    // Memory Alignment Assumptions (Speedup)
    module.walk([&](func::FuncOp func) {
      OpBuilder b(&func.getBody().front(), func.getBody().front().begin());
      for (auto arg : func.getArguments()) {
        if (arg.getType().isa<MemRefType>())
          b.create<memref::AssumeAlignmentOp>(func.getLoc(), arg, 64);
      }
    });

    // Apply Vectorisation Patterns
    RewritePatternSet patterns(ctx);
    patterns.add<CudaToVectorPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createCudaToVectorPass() {
  return std::make_unique<CudaToVectorPass>();
}
