#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <iostream>

using namespace mlir;

namespace {

// struct CUDAToVectorPattern : public OpRewritePattern<scf::ParallelOp> {
//   using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
//     auto loc = op.getLoc();
    
//     // Setup Types and Constants
//     const int64_t vWidth = 8;
//     Type f32Type = rewriter.getF32Type();
//     VectorType vType = VectorType::get({vWidth}, f32Type);
//     VectorType maskType = VectorType::get({vWidth}, rewriter.getI1Type());

//     if (op.getStep().empty()) return failure();
//     Value ub = op.getUpperBound()[0];
//     Value iv = op.getInductionVars()[0];
    
//     // Adjust Loop Step
//     rewriter.modifyOpInPlace(op, [&]() {
//       Value newStep = rewriter.create<arith::ConstantIndexOp>(loc, vWidth);
//       op.getStepMutable().assign(newStep);
//     });

//     // Setup Masking logic
//     rewriter.setInsertionPointToStart(op.getBody());
//     Value diff = rewriter.create<arith::SubIOp>(loc, ub, iv);
//     Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, diff);
//     Value zeroPadding = rewriter.create<arith::ConstantOp>(loc, f32Type, rewriter.getF32FloatAttr(0.0f));

//     // Setup Attributes
//     auto mapAttr = AffineMapAttr::get(rewriter.getMultiDimIdentityMap(1));
//     auto inBoundsAttr = rewriter.getBoolArrayAttr({false});

//     // Transform Body
//     for (Operation &innerOp : llvm::make_early_inc_range(*op.getBody())) {
      
//       // --- LOAD -> vector.transfer_read ---
//       if (auto load = dyn_cast<memref::LoadOp>(innerOp)) {
//         rewriter.setInsertionPoint(load);
        
//         SmallVector<Value, 4> operands;
//         operands.push_back(load.getMemref());
//         for (Value idx : load.getIndices()) operands.push_back(idx);
//         operands.push_back(zeroPadding);
//         operands.push_back(mask);

//         auto vRead = rewriter.create<vector::TransferReadOp>(
//             loc, 
//             vType, 
//             operands, 
//             ArrayRef<NamedAttribute>{
//                 rewriter.getNamedAttr("permutation_map", mapAttr),
//                 rewriter.getNamedAttr("in_bounds", inBoundsAttr)
//             }
//         );
//         rewriter.replaceOp(load, vRead.getResult());
//       }

//       // --- ARITHMETIC (AddF) ---
//       else if (auto addf = dyn_cast<arith::AddFOp>(innerOp)) {
//         rewriter.setInsertionPoint(addf);
//         Value lhs = addf.getLhs();
//         Value rhs = addf.getRhs();
//         if (lhs.getType() != vType && lhs.getType().isF32()) 
//             lhs = rewriter.create<vector::BroadcastOp>(loc, vType, lhs);
//         if (rhs.getType() != vType && rhs.getType().isF32()) 
//             rhs = rewriter.create<vector::BroadcastOp>(loc, vType, rhs);
        
//         if (lhs.getType() == vType && rhs.getType() == vType) {
//             auto vAdd = rewriter.create<arith::AddFOp>(loc, lhs, rhs);
//             rewriter.replaceOp(addf, vAdd.getResult());
//         }
//       }

//       // --- ATOMIC ADD ---
//       else if (auto atomic = dyn_cast<memref::AtomicRMWOp>(innerOp)) {
//         Value atomicVal = atomic.getValue();
//         if (atomicVal.getType() == vType) {
//             rewriter.setInsertionPoint(atomic);
//             auto reduction = rewriter.create<vector::ReductionOp>(
//                 loc, vector::CombiningKind::ADD, atomicVal);
//             rewriter.create<memref::AtomicRMWOp>(
//                 loc, arith::AtomicRMWKind::addf, reduction.getResult(), 
//                 atomic.getMemref(), atomic.getIndices());
//             rewriter.eraseOp(atomic);
//         }
//       }

//       // --- STORE -> vector.transfer_write ---
//       else if (auto store = dyn_cast<memref::StoreOp>(innerOp)) {
//         rewriter.setInsertionPoint(store);
//         Value val = store.getValue(); 
//         if (val.getType() != vType && val.getType().isF32()) 
//             val = rewriter.create<vector::BroadcastOp>(loc, vType, val);
        
//         if (val.getType() == vType) {
//             SmallVector<Value, 4> operands;
//             operands.push_back(val);
//             operands.push_back(store.getMemref());
//             for (Value idx : store.getIndices()) operands.push_back(idx);
//             operands.push_back(mask);

//             rewriter.create<vector::TransferWriteOp>(
//                 loc, 
//                 TypeRange{}, /* No result type for Store */
//                 operands, 
//                 ArrayRef<NamedAttribute>{
//                     rewriter.getNamedAttr("permutation_map", mapAttr),
//                     rewriter.getNamedAttr("in_bounds", inBoundsAttr)
//                 }
//             );
//             rewriter.eraseOp(store);
//         }
//       }

//       // --- BARRIER ---
//       else if (isa<gpu::BarrierOp>(innerOp)) {
//         rewriter.eraseOp(&innerOp);
//       }
//     }
//     return success();
//   }
// };

struct CUDAToVectorPattern : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    
    // Setup types and constants
    const int64_t vWidth = 8;
    Type f32Type = rewriter.getF32Type();
    VectorType vType = VectorType::get({vWidth}, f32Type);
    VectorType maskType = VectorType::get({vWidth}, rewriter.getI1Type());
    
    if (op.getStep().empty()) return failure();
    Value scalarIv = op.getInductionVars()[0];
    Value ub = op.getUpperBound()[0];
    
    // Update loop step to vWidth
    rewriter.modifyOpInPlace(op, [&]() {
      Value newStep = rewriter.create<arith::ConstantIndexOp>(loc, vWidth);
      op.getStepMutable().assign(newStep);
    });
    
    // Setup masking and thread id vectorisation
    rewriter.setInsertionPointToStart(op.getBody());
    
    // Create a thread id vector (i.e. [i, i+1, ..., i+7])
    SmallVector<float, 8> offsets;
    for (int64_t i = 0; i < vWidth; ++i) offsets.push_back(static_cast<float>(i));
    auto offsetsConst = rewriter.create<arith::ConstantOp>(loc, vType, rewriter.getF32VectorAttr(offsets));
    
    auto floatIv = rewriter.create<arith::IndexCastOp>(loc, f32Type, scalarIv);
    auto splatIv = rewriter.create<vector::SplatOp>(loc, vType, floatIv);
    Value vectorisedThreadId = rewriter.create<arith::AddFOp>(loc, splatIv, offsetsConst);
    
    // This map tracks scalar values to their vectorised equivalents
    DenseMap<Value, Value> vectorisedMapping;
    vectorisedMapping[scalarIv] = vectorisedThreadId;
    
    // Standard attributes for TransferRead/Write
    auto mapAttr = AffineMapAttr::get(rewriter.getMultiDimIdentityMap(1));
    auto inBoundsAttr = rewriter.getBoolArrayAttr({false});

    //    Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, rewriter.create<arith::SubIOp>(loc, ub, scalarIv));

    Value diff = rewriter.create<arith::SubIOp>(loc, ub, scalarIv).getResult();

    Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, diff);
    Value zeroPadding = rewriter.create<arith::ConstantOp>(loc, f32Type, rewriter.getF32FloatAttr(0.0f));
    
    // Transform the body
    for (Operation &innerOp : llvm::make_early_inc_range(*op.getBody())) {
      
      // Helper to get vectorised operands for the current operation
      auto getVecOp = [&](Value scalarVal) -> Value {
	if (vectorisedMapping.count(scalarVal))
	  return vectorisedMapping[scalarVal];
	// If not in map and is a float, broadcast it (Symmetric/Uniform value)
	if (scalarVal.getType().isF32())
	  return rewriter.create<vector::BroadcastOp>(loc, vType, scalarVal);
	return scalarVal;
      };
      
      // --- LOAD ---
      if (auto load = dyn_cast<memref::LoadOp>(innerOp)) {
	rewriter.setInsertionPoint(load);
	SmallVector<Value, 4> indices(load.getIndices());
	auto vRead = rewriter.create<vector::TransferReadOp>(
							     loc, vType, load.getMemref(), indices, mapAttr, zeroPadding, mask, inBoundsAttr);
	
	vectorisedMapping[load.getResult()] = vRead.getResult(); // Record result
	rewriter.replaceOp(load, vRead.getResult());
      }
      
      // --- ARITHMETIC (AddF) ---
      else if (auto addf = dyn_cast<arith::AddFOp>(innerOp)) {
	rewriter.setInsertionPoint(addf);
	Value vLhs = getVecOp(addf.getLhs());
	Value vRhs = getVecOp(addf.getRhs());
        
	auto vAdd = rewriter.create<arith::AddFOp>(loc, vLhs, vRhs);
	vectorisedMapping[addf.getResult()] = vAdd.getResult(); // Record result
	rewriter.replaceOp(addf, vAdd.getResult());
      }
      
      // --- MATH OPERATIONS ---
      else if (mlir::isa<math::ExpOp, math::SqrtOp, math::TanhOp, math::SinOp, math::CosOp>(innerOp)) {
	
	rewriter.setInsertionPoint(&innerOp);
	
	// Get the vectorised version of the input operand
	Value scalarInput = innerOp.getOperand(0);
	Value vInput = getVecOp(scalarInput);
	
	Operation *vMath = nullptr;
	
	// Create the vector-version of the math op
	if (mlir::isa<math::ExpOp>(innerOp))       vMath = rewriter.create<math::ExpOp>(loc, vInput);
	else if (mlir::isa<math::SinOp>(innerOp))  vMath = rewriter.create<math::SinOp>(loc, vInput);
	else if (mlir::isa<math::CosOp>(innerOp))  vMath = rewriter.create<math::CosOp>(loc, vInput);
	else if (mlir::isa<math::TanhOp>(innerOp)) vMath = rewriter.create<math::TanhOp>(loc, vInput);
	else if (mlir::isa<math::LogOp>(innerOp))  vMath = rewriter.create<math::LogOp>(loc, vInput);
	else if (mlir::isa<math::SqrtOp>(innerOp)) vMath = rewriter.create<math::SqrtOp>(loc, vInput);
	
	if (vMath) {
	  // Register the result so subsequent ops can use it
	  vectorisedMapping[innerOp.getResult(0)] = vMath->getResult(0);
	  rewriter.replaceOp(&innerOp, vMath->getResults());
	}
      }
      
      // --- SHUFFLE ---
      else if (auto shfl = dyn_cast<gpu::ShuffleOp>(innerOp)) {
	rewriter.setInsertionPoint(shfl);
	Value vInput = getVecOp(shfl.getValue());
	uint32_t delta = 0;
	if (auto constOp = shfl.getOffset().getDefiningOp<arith::ConstantIntOp>())
	  delta = constOp.value();
	
	if (shfl.getMode() == gpu::ShuffleMode::DOWN) {
	  SmallVector<int64_t> shuffleMask;
	  for (int64_t i = 0; i < vWidth; ++i)
	    shuffleMask.push_back(std::min<int64_t>(i + delta, vWidth - 1));
	  
	  auto vShfl = rewriter.create<vector::ShuffleOp>(loc, vInput, vInput, shuffleMask);
	  vectorisedMapping[shfl.getResult(0)] = vShfl.getResult();
	  rewriter.replaceOp(shfl, vShfl.getResult());
	}
      }
      
      // --- STORE ---
      else if (auto store = dyn_cast<memref::StoreOp>(innerOp)) {
	rewriter.setInsertionPoint(store);
	Value vVal = getVecOp(store.getValueToStore());
	SmallVector<Value, 4> indices(store.getIndices());
        
	rewriter.create<vector::TransferWriteOp>(
						 loc, vVal, store.getMemref(), indices, mapAttr, mask, inBoundsAttr);
	rewriter.eraseOp(store);
      }
      
      // --- ATOMIC ADD ---
      else if (auto atomic = dyn_cast<memref::AtomicRMWOp>(innerOp)) {
	Value vVal = getVecOp(atomic.getValue());
	rewriter.setInsertionPoint(atomic);
	auto reduction = rewriter.create<vector::ReductionOp>(loc, vector::CombiningKind::ADD, vVal);
	rewriter.create<memref::AtomicRMWOp>(
					     loc, arith::AtomicRMWKind::addf, reduction.getResult(), atomic.getMemref(), atomic.getIndices());
	rewriter.eraseOp(atomic);
      }
      
      // --- BARRIER ---
      else if (isa<gpu::BarrierOp>(innerOp)) {
	rewriter.eraseOp(&innerOp);
      }
      
      else if (auto call = dyn_cast<func::CallOp>(innerOp)) {
	auto callee = call.getCallee();
	if (callee == "sinf" || callee == "__nv_sinf") {
	  rewriter.setInsertionPoint(call);
	  Value vInput = getVecOp(call.getOperand(0));
	  auto vSin = rewriter.create<math::SinOp>(loc, vInput);
	  vectorisedMapping[call.getResult(0)] = vSin.getResult();
	  rewriter.replaceOp(call, vSin.getResult());
	}else{
	  std::cout << "Error 'function call' identified but not handled: " << callee.str() << std::endl;
	}
      }
      
    }
    
    return success();
  }

};


struct CUDAToHierarchicalParallelPass : public PassWrapper<CUDAToHierarchicalParallelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CUDAToHierarchicalParallelPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Move shared memory (CUDA) to stack (Alloca)
    // We place it at the top of the block parallel loop (grid loop body)
    module.walk([&](gpu::AllocOp alloc) {
      if (auto intAttr = mlir::dyn_cast_or_null<IntegerAttr>(alloc.getType().getMemorySpace())) {
        if (intAttr.getInt() == 3) {
          OpBuilder b(alloc);
          auto stackMem = b.create<memref::AllocaOp>(alloc.getLoc(), mlir::cast<MemRefType>(alloc.getType()));
          alloc->getResult(0).replaceAllUsesWith(stackMem->getResult(0));
          alloc.erase();
        }
      }
    });

    // Apply vectorisation to innermost (thread) loops
    RewritePatternSet patterns(ctx);
    patterns.add<CUDAToVectorPattern>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};
  
  
  /*struct CUDAToVectorPass : public PassWrapper<CUDAToVectorPass, OperationPass<ModuleOp>> {
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
  };*/

} // namespace

//std::unique_ptr<Pass> createCUDAToVectorPass() {
//  return std::make_unique<CUDAToVectorPass>();
//}

std::unique_ptr<Pass> createCUDAToHierarchicalParallelPass() {
  return std::make_unique<CUDAToHierarchicalParallelPass>();
}

