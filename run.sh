#!/bin/bash
set -e # Exit on error

# --- Configuration ---
VECTOR_WIDTH=8
ARCH_FLAGS="-march=x86-64 -mattr=+avx2" # Change to +avx512f for newer CPUs
PASS_LIB="./libCUDA2VectorPass.so"

echo "[1/4] Parsing CUDA with Polygeist..."
# --cuda-lower: Converts GPU blocks/threads to scf.parallel loops
# -S: Output as text (MLIR)
cgeist kernel.cu --cuda-lower -S > 1_polygeist.mlir

echo "[2/4] Vectorising and Lowering MLIR..."
mlir-opt 1_polygeist.mlir \
    -load-pass-plugin=$PASS_LIB \
    --cuda-to-vector \
    --convert-vector-to-scf \
    --convert-scf-to-cf \
    --convert-vector-to-llvm \
    --finalize-memref-to-llvm \
    --convert-func-to-llvm="emit-c-wrappers" \
    --reconcile-unrealized-casts > 2_llvm_dialect.mlir

echo "[3/4] Translating to LLVM IR and Compiling to Object File..."
mlir-translate --mlir-to-llvmir 2_llvm_dialect.mlir | \
clang++ -x ir - -c -o kernel.o $ARCH_FLAGS -O3

echo "[4/4] Linking with Host Driver..."
# Compile the host C++ file
clang++ main.cpp -c -o main.o -O3
# Link both objects into a final executable
clang++ main.o kernel.o -o vectorised_app

echo "Success! Execute './vectorised_app' to run your vectorised kernel."
