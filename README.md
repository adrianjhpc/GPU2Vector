# GPU2Vector
This code base is designed to convert code that has been written using GPU specific languages, such as CUDA and HIP, to vectorised code that can run directly on CPUs.

The code development approach for CUDA or HIP restricts code functionality to something that can match CPU vectorisation approaches closely enough to give high performance. However, such codes cannot run directly on CPUs because of the embedded CUDA/HIP functionality. Therefore, we look to automatically generate vectorised executables from such code to enable high performance CPU versions to be used.

## Structure of the repository
The source code for the MLIR functionality that will covert CUDA or HIP code into vectorised instructions is in the `src/` directory.

The `example/` directory has example applications to test the functoinality.

`run.sh` is an example end to end compilation script for a single CUDA file assuming you have already built the MLIR code in the src directory. Instructions on how to build that are below.

## Prerequisites
This tool depends on the LLVM compiler infrastructure, in particular the `clang` compiler(s). 

We also use the [Polygeist](https://github.com/llvm/Polygeist) LLVM front-end for converting CUDA and HIP code into MLIR intermediate representation.

We use the [CMake](https://cmake.org/) build system to create the MLIR plugin that generates the vector instructions.

## Building
To build our MLIR plugin create a build directory (for instance `mkdir build`, enter that directory (i.e. `cd build`) and then run cmake (`cmake ../; make`).

