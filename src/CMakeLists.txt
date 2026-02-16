cmake_minimum_required(VERSION 3.18)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

add_library(CUDA2VectorPass SHARED CUDAVectorisePass.cpp)
target_link_libraries(CUDA2VectorPass 
    MLIRVectorDialect 
    MLIRSCFDialect 
    MLIRArithDialect 
    MLIRMemRefDialect 
    MLIRGPUDialect 
    MLIRFuncDialect
    MLIRPass
)
