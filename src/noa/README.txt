namespaces
==========

- noa       :   Contains the "common" functionalities and types, e.g. files, strings, vectors and utilities
                accessible and used by the backends. This code is mostly meant to be used by the host threads,
                however, some types and functions call be called and used from the device. See "noa/common/Math.h"
                or "noa/common/types/Int2.h" for examples.

                To do so, macros are used to tag (member) functions with attributes to let the compiler know
                whether the function should be compiled for the host and/or the device. These macros are defined
                in "noa/common/Definitions.h". Note however that this mechanism is only useful when the compilation
                is steered by the device compiler (we only support nvcc atm) and doesn't prevent the host compilers,
                i.e. GCC or Clang, to compile device functions.

                To prevent the host compiler to compile a device function, pre-processor fences are usually used.
                For instance with CUDA, __CUDA_ARCH__ is only defined on device code. __CUDACC__ can be used to
                know whether the compiler is steered by nvcc.

- noa::cpu  :   CPU backend.
                Contains the main functionalities of the library. This code is meant to be called by the host only.
                The CPU backend is meant to support "CPU execution policies" and to define multiple CPU implementations
                to allow for multithreading and SIMD support, for instance via the OpenMP and/or oneTBB libraries.
                This is currently not in development but is the direction we would like to take.

- noa::cuda :   CUDA backend.
                Contains the same functionalities available in ::noa::cpu, but with CUDA implementations. This code is
                meant to be mostly called by the host, however host/device or device only functions are also available.

Note: All headers can be included in C++ (compiled with GCC or Clang) or CUDA C++ source files (compiled with nvcc).
Note: Backends are independent of each other, meaning that the library can be built without the CPU or CUDA backend
      for instance. Note however that in order to test the CUDA backend via "noa_tests", the CPU backend must be
      included.


directory hierarchy
===================

These are defined for each backend:

- filter        :   Filtering/masking functions, e.g. convolutions, median filters, geometric shapes, etc.
- fourier       :   Fourier specific functions, e.g. plans, Fourier transforms, bandpass filters, etc.
- math          :   Math functions for arrays, e.g. arithmetics, reductions, etc.
- memory        :   Scoped bound memory resources, memory manipulations, etc.
- recons        :   Reconstruction methods, e.g. backward and forward projections, 3D reconstructions, etc.
- transform     :   Linear and affine transforms, symmetries, phase shifts, etc.

Note: The directory hierarchy is the same as the namespace hierarchy. This might result in quite long signatures,
      e.g. "::noa::cpu::transform::rotate2D()" but is often easier, specially for newcomers, to understand where
      functions are declared and defined. Moreover, the length of the signature can always be simplified by including
      the namespace at the beginning of the scope, e.g. "using namespace ::noa::cpu".
