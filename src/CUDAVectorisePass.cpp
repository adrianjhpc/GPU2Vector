#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct CUDAToVectorPattern : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    // Setup Types and Constants
    const int64_t vWidth = 8;
    Type f32Type = rewriter.getF32Type();
    VectorType vType = VectorType::get({vWidth}, f32Type);
    VectorType maskType = VectorType::get({vWidth}, rewriter.getI1Type());

    if (op.getStep().empty()) return failure();
    Value ub = op.getUpperBound()[0];
    Value iv = op.getInductionVars()[0];
    
    // Adjust Loop Step
    rewriter.modifyOpInPlace(op, [&]() {
      Value newStep = rewriter.create<arith::ConstantIndexOp>(loc, vWidth);
      op.getStepMutable().assign(newStep);
    });

    // Setup Masking logic
    rewriter.setInsertionPointToStart(op.getBody());
    Value diff = rewriter.create<arith::SubIOp>(loc, ub, iv);
    Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, diff);
    Value zeroPadding = rewriter.create<arith::ConstantOp>(loc, f32Type, rewriter.getF32FloatAttr(0.0f));

    // Setup Attributes
    auto mapAttr = AffineMapAttr::get(rewriter.getMultiDimIdentityMap(1));
    auto inBoundsAttr = rewriter.getBoolArrayAttr({false});

    // Transform Body
    for (Operation &innerOp : llvm::make_early_inc_range(*op.getBody())) {
      
      // --- LOAD -> vector.transfer_read ---
      if (auto load = dyn_cast<memref::LoadOp>(innerOp)) {
        rewriter.setInsertionPoint(load);
        
        SmallVector<Value, 4> operands;
        operands.push_back(load.getMemref());
        for (Value idx : load.getIndices()) operands.push_back(idx);
        operands.push_back(zeroPadding);
        operands.push_back(mask);

        auto vRead = rewriter.create<vector::TransferReadOp>(
            loc, 
            vType, 
            operands, 
            ArrayRef<NamedAttribute>{
                rewriter.getNamedAttr("permutation_map", mapAttr),
                rewriter.getNamedAttr("in_bounds", inBoundsAttr)
            }
        );
        rewriter.replaceOp(load, vRead.getResult());
      }

      // --- ARITHMETIC (AddF) ---
      else if (auto addf = dyn_cast<arith::AddFOp>(innerOp)) {
        rewriter.setInsertionPoint(addf);
        Value lhs = addf.getLhs();
        Value rhs = addf.getRhs();
        if (lhs.getType() != vType && lhs.getType().isF32()) 
            lhs = rewriter.create<vector::BroadcastOp>(loc, vType, lhs);
        if (rhs.getType() != vType && rhs.getType().isF32()) 
            rhs = rewriter.create<vector::BroadcastOp>(loc, vType, rhs);
        
        if (lhs.getType() == vType && rhs.getType() == vType) {
            auto vAdd = rewriter.create<arith::AddFOp>(loc, lhs, rhs);
            rewriter.replaceOp(addf, vAdd.getResult());
        }
      }

      // --- ATOMIC ADD ---
      else if (auto atomic = dyn_cast<memref::AtomicRMWOp>(innerOp)) {
        Value atomicVal = atomic.getValue();
        if (atomicVal.getType() == vType) {
            rewriter.setInsertionPoint(atomic);
            auto reduction = rewriter.create<vector::ReductionOp>(
                loc, vector::CombiningKind::ADD, atomicVal);
            rewriter.create<memref::AtomicRMWOp>(
                loc, arith::AtomicRMWKind::addf, reduction.getResult(), 
                atomic.getMemref(), atomic.getIndices());
            rewriter.eraseOp(atomic);
        }
      }

      // --- STORE -> vector.transfer_write ---
      else if (auto store = dyn_cast<memref::StoreOp>(innerOp)) {
        rewriter.setInsertionPoint(store);
        Value val = store.getValue(); 
        if (val.getType() != vType && val.getType().isF32()) 
            val = rewriter.create<vector::BroadcastOp>(loc, vType, val);
        
        if (val.getType() == vType) {
            SmallVector<Value, 4> operands;
            operands.push_back(val);
            operands.push_back(store.getMemref());
            for (Value idx : store.getIndices()) operands.push_back(idx);
            operands.push_back(mask);

            rewriter.create<vector::TransferWriteOp>(
                loc, 
                TypeRange{}, /* No result type for Store */
                operands, 
                ArrayRef<NamedAttribute>{
                    rewriter.getNamedAttr("permutation_map", mapAttr),
                    rewriter.getNamedAttr("in_bounds", inBoundsAttr)
                }
            );
            rewriter.eraseOp(store);
        }
      }

      // --- BARRIER ---
      else if (isa<gpu::BarrierOp>(innerOp)) {
        rewriter.eraseOp(&innerOp);
      }
    }
    return success();
  }
};

struct CUDAToVectorPass : public PassWrapper<CUDAToVectorPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CUDAToVectorPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Shared memory to stack conversion
    module.walk([&](gpu::AllocOp alloc) {
      auto memSpace = alloc.getType().getMemorySpace();
      if (memSpace && mlir::isa<IntegerAttr>(memSpace) && 
	  mlir::cast<IntegerAttr>(memSpace).getInt() == 3){
          OpBuilder b(alloc);
          auto stackMem = b.create<memref::AllocaOp>(alloc.getLoc(), mlir::cast<MemRefType>(alloc.getType()));
	  alloc->getResult(0).replaceAllUsesWith(stackMem->getResult(0));

          alloc.erase();
      }
    });

    // Alignment Assumptions
    module.walk([&](func::FuncOp func) {
      if (func.isExternal() || func.getFunctionBody().empty()) return;
      OpBuilder b(&func.getFunctionBody().front(), func.getFunctionBody().front().begin());
      auto alignAttr = b.getI64IntegerAttr(64);
      for (auto arg : func.getArguments()) {
        if (mlir::isa<MemRefType>(arg.getType()))
          b.create<memref::AssumeAlignmentOp>(func.getLoc(), arg, alignAttr);
      }
    });

    // Run Patterns
    RewritePatternSet patterns(ctx);
    patterns.add<CUDAToVectorPattern>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createCUDAToVectorPass() {
  return std::make_unique<CUDAToVectorPass>();
}
