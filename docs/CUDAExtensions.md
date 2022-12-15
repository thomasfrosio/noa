The library doesn't support Just In Time compilation yet, but users can still create their own functors and types to be
supported by the library. There are three step to follow:

1. The functor should be marked with the `__device__` attribute. One can use the macro `NOA_DEVICE` so that the 
   attribute is only added when the compilation is steered by the CUDA compiler. To add support for a type that the 
   library doesn't support, make sure the operator and functions used by the element-wise operator() are marked with 
   the `__device__` attribute.
2. The plugin function should be explicitly instantiated in a `.cu` file and compiled by the same compiler that 

```c++
// Usually in a .h (or .cuh if CUDA only)
struct my_special_division {
    float epsilon = 1e-3;
    NOA_HD constexpr double operator(float lhs, float rhs) const noexcept { return lhs / (rhs + epsilon); }
};

namespace noa::cuda::ext {
    template<> struct proclaim_is_valid_ewise_binary<float, float, double, my_special_division>: std::false_type {};
}
```

We don't have a JIT yet, so you'll have to explicitly instantiate the function and the linker will do the rest.
Fortunately, if you have a CUDA compiler set up (which is a public dependency of the library), this step is really
easy with the tools provided by the library (TODO)

```c++
// In a .cu file
#include "my_header_with_my_division.h" // your header with the types/functors you want to add

#include <noa/gpu/cuda/Extentions.cuh> // Provides all the supported extensions and instantiation macros
NOA_CUDA_EXT_EWISE_BINARY(float, float, double, my_division);
// The above fails if proclaim_is_valid_ewise_binary is not set for these types.
// This adds CUDA support for the following overloads:
// math::ewise(const Array<float>& lhs, const Array<float>& rhs, const Array<double>& output, my_division op);
// math::ewise(const Array<float>& lhs, float rhs,               const Array<double>& output, my_division op);
// math::ewise(float lhs,               const Array<float>& rhs, const Array<double>& output, my_division op);
```
