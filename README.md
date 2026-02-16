#GPU2Vector
This code base is designed to convert code that has been written using GPU specific languages, such as CUDA and HIP, to vectorised code that can run directly on CPUs.

The code development approach for CUDA or HIP restricts code functionality to something that can match CPU vectorisation approaches closely enough to give high performance. However, such codes cannot run directly on CPUs because of the embedded CUDA/HIP functionality. Therefore, we look to automatically generate vectorised executables from such code to enable high performance CPU versions to be used.
