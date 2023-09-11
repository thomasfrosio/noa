## `cryoEM`


- __High-order aberrations__ and optics in general... Look at Warp and RELION's code. I know nothing about this...


- Complex CTFs, for Russo/Henderson's EWS correction?


- __IO__: Refactor IO. Remove polymorphism and use something more modern and less coupled, like `std::variant`? Add compression and other file formats. Also, better support for 3dmod with complex types: IMOD excepts the logical shape but always excepts the non-redundant data. ImageFile treats the shape as the physical shape.


## `General`

- __Refactor CPU backend__.
  
  Move most of the code to header-only. The scale issue is still manageable on the CPU side, but the library is becoming too big, so we want to compile only what is needed (thanks to template). The explicit template instantiations are too expensive, and they do limit the flexibility, as well as prevent inline optimizations (although LTO can in theory fix that).


- __Refactor CUDA backend__.

  This is where the scaling issue is not manageable anymore. There are too many things to compile, and applications usually need a fraction of that. This is problematic because it makes everything slow to build, binaries become too big, and CUDA cannot fully strip unused code. Moreover, this greatly limit the flexibility of the library, which always has been an issue. Fortunately, there's a solution to this: runtime compilation! I plan to move most of the CUDA source files to runtime compilation using jitify2. This has huge benefits other than solving the scaling issue. Now, everything becomes valid C++ and we can use templates to instantiate and call kernels directly from user's translation units (launching the kernels from C++ was always possible using the driver though).

  - This allows users to pass their own types to the GPU backend. Indeed, we can easily ask them (at build type) what headers should be included in the runtime compilation and these can be included when we compile a type that is not one of ours. This makes everything smooth, and other than this extra step in the config, the code is as clean as it can be: we pass a type to noa, jitify2 reflects on it and compile/link everything (including the user header(s)) and that's it.
  - This also allows us to refactor our backends and unified interface. The unified interface can have the "core" functions (ewise, iwise, reduce, etc.) calling the backend-specific core functions (which are templates) with the input/output types. The other functions in the unified interface are now backend agnostic for the most part and can do the preprocessing and construct the algo operators directly (atm this is done by the backends). As such, the backends could only have core functions, some utilities and special functions that are not that generic (e.g. FFTs). This is a huge simplification and I don't see any disadvantages to it.
  - Adding AMD support would be a significant task, but HIP also has a runtime compilation (HIPRTC), the only thing is that jitfify is a CUDA only wrapper (but it isn't too complicated, and we can hipify it for the most part).

  In practice, this also removes the need for nvcc (finally). We would only need nvrtc and cufft/curand/cublas (there's a world were we could lazy load these too, to remove them from the build (it's what Pytorch does), so CUDA only becomes a runtime dependency).
  
  Links:
  [arrayfire implementation](https://github.com/arrayfire/arrayfire/blob/master/src/backend/cuda/compile_module.cpp),
  [cuda-api-wrappers](https://github.com/eyalroz/cuda-api-wrappers/tree/master/src/cuda/nvrtc),
  [jitify](https://github.com/NVIDIA/jitify).


- __algorithms__. Move them in `core`. These would be "ops" (operators).


- Refactor the `Vec`, `Shape`, `Strides` and `Complex` to be aggregates, ensuring its trivially copyable, moveable, assignable, constructible, and destructible. This could/should in better code generation by the compiler, allowing to be passed in registers, and to be serialized and deserialized via memcpy (although I would expect the optimizer to do that). This can be done, mostly by moving the constructors (and rely on the aggregate initialization) to factory functions, e.g. `Vec<f32>::from_pointer()`. This is more explicit too!


- Move the __core types__ to __noa::types__. The goal is that this namespace should be included in noa, so we can still refer to these types from `noa::`. But, user code can use `noa::types` without having to use the entire `noa::`. Maybe inline namespaces could help? These core types would be our type aliases (`i32`, `i64`, `f32`, etc.), `Shape`, `Strides`, `Vec`, etc.


- __simplify namespaces__. Remove `math`, `io` and `memory` namespaces. Use `linalg` for BLAS/LAPACK. Remove the nested `fft` namespaces in `signal` and `geometry`, and instead prefix functions with `fourier_` when it is needed. For instance `noa::signal::lowpass` is unambiguous, we don't need `noa::signal::fft::lowpass` or `noa::signal::fouier_lowpass`. However, `noa::geometry::fourier_extract_3d` is better than `noa::geometry::extract_3d` (which is to ambiguous). At the end, we would have:
  - `noa::`: contains everything that is currently in memory and io, as well as the core cmath functions, reductions and random numbers.
  - `noa::geometry`: everything geometry related (so unchanged). Remove the nested `fft` and prefix functions with `fourier_` when necessary.
  - `noa::signal`: same but for signal processing stuff.
  - `noa::linalg`: linear algebra. idk if we should include `dot` and `matmul` here or keep these in `noa::`.
  - `noa::fft`: unchanged
  - `noa::traits`: unchanged
  - `noa::indexing`: idk about this. We mostly use this internally, but there's `broadcast` for instance, which could be in `noa` directly... I certainly wouldn't want to have the `at` functions in `noa::`.


- __Windows support__. It should not be that complicated. One thing to look at is OpenMP support.


- __SIMD__ Trying explicit SIMD in the CPU-"core" functions (start with ewise_*?). [xsimd](https://xsimd.readthedocs.io/en/latest/index.html) seems interesting (their "abstract batch" at least).


- __CUDA constant memory.__
  
  Test the benefits of constant memory. The issue with this is the (host) thread-safety. If multiple host threads use the same device but are on different streams, we need to lock guard the resource and kernel launch since the resource is global to the device. Idk how this works with nvrtc... is the ressource private to a context, or a module, or a translation unit?


- Use the __CUDA driver to handle the context__. That way, the library can keep track of its own context and reset it
  without affecting then entire application. This could be attached to the current Session.
  [Pytorch DeviceThreadHandles](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/DeviceThreadHandles.h)


- Add __vkFFT__ support? Also look at FFTW CPU port for Intel-MKL and AMD.
  This is only for zero-padding, which can bring up to 2x increase of performance for 2D, 3x for 3D. Convolution can be added as a call- back but there's no benchmark for that. It could be useful to contact the author about some of our applications, e.g. convolution with small (compared to input) template.


- __C++20/23__
  - Add concepts
  - Add more constexpr (specially string stuff)
  - Add mdspan. This could replace our Accessor in the backends' API (not in the kernel). We don't have to wait to a new standard and can take the Kokkos implementation (it works on device code too).
  - The fft remap template parameter could be a string: `my_function<fft::H2H>(...)` -> `my_function<"h2h">(...)`.
  - using enum


- __C++26__
  - [std::execution-P2300](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2300r7.html) seems to be targeted for C++26. This has the potential to change C++, hopefully in a good way. The paper is promising and the talks from Bryce look ambitious. This could be amazing, but of course requires vendors to support it. NVIDIA is of course pushing for this, but hopefully it will have broader (GPU-)support. Note that this sort of changes the paradigm a bit, now the backends would be whatever the compiler supports, so if users want CUDA support for instance, they would need to compile their code with a compiler that has this backend (e.g. nvc++). `std::execution` is lazy, and is designed to allow operator-fusion.
