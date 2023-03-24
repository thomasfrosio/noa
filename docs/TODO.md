## `cryoEM`


- __CTF__: Add CTF class gathering all the necessary variables (voltage, Cs, etc.) and basic utility functions,
  e.g. compute value at a given frequency. Single side-band algorithm support (complex CTFs, cones).


- __High-order aberrations__ and optics in general... Look at Warp and RELION's code. I know very little about this...


- __Binning__: real space binning is a nice to have, although Fourier cropping is often preferred. Note that efficient
  binning on the GPU requires some thinking.


- Add center of mass, radial grid.


- __IO__: Add compression. Also, better support for 3dmod with complex types: IMOD excepts the logical shape but always
  excepts the non-redundant data. ImageFile treats the shape as the physical shape...


## `General`

- __Windows__. It should not be that complicated. One thing to look at is OpenMP support.


- Session: Not sure what `Session` should be. It only holds the main `Logger` and the number of internal threads.
  Maybe we should link all global data to the session as well, but I don't think that's a good idea. Instead,
  maybe remove `Session` and create a `GlobalLogger` and a free function keeping track of the thread number or
  something like that.


- __Test CUDA LTO__ and unused kernels support from version 11.5.


- __CUDA constant memory.__ Test the benefits of constant memory. The issue with this is the (host) thread-safety.
  If multiple host threads use the same device but are on different streams, we need to lock guard the resource
  and kernel launch since the resource is global to the device.


- Use the __CUDA driver to handle the context__. That way, the library can keep track of its own context and reset it
  without affecting then entire application. This could be attached to the current Session.
  [Pytorch DeviceThreadHandles](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/DeviceThreadHandles.h)


- __lazy evaluation and kernel fusion vs transform/map operators__.
  I much prefer the simplicity of `std::transform` or range-like APIs, where we define __lambdas/functors as transform 
  operators__. It is much more flexible, versatile, cleaner, and it is where the C++ STL is going anyway (including 
  NVIDIA) with `std::execution` and context policies. The issue with this approach though: 1) it  cannot easily be 
  used from another language, 2) CUDA is not really possible from a .cpp file, although it looks like 
  this is likely to change in the coming years with Executors.

  __Lazy evaluation and template expressions ala xtensor__.
  Again, I don't want to go down that path but FYI:
  `ArrayFire` is quite good at fusing kernels, the implementation seems relatively clean. I think `Cupy` is similar 
  but CUDA only. `Pytorch` should be the same but the codebase is huge, and it is sometimes difficult to follow.
  Also `Eigen` and `xtensor` have lazy evaluation. `xtensor` has a series of articles on how they implemented
  lazy-evaluation (and kernel fusion at the same time), but these are CPU only. Adding GPU support should be relatively
  simple though, with a string conversion on the operators and something like `jitify` from nvidia. With C++20, that
  can be `constexpr`.
  Links:
  [arrayfire JIT](https://arrayfire.com/performance-of-arrayfire-jit-code-generation/)
  [xtensors series](https://johan-mabille.medium.com/how-we-wrote-xtensor-9365952372d9).


- Add __vkFFT__ support? Also look at FFTW CPU port for Intel-MKL and AMD.
  Zero-padding can bring up to 2x increase of performance for 2D, 3x for 3D. Convolution can be added as a call-
  back but there's no benchmark for that. It could be useful to contact the author about some of our applications,
  e.g. convolution with small (compared to input) template. FastFFT from Ben Himes is also promising since it is
  really fitted for cryoEM applications, however it is CUDA only and still in development.


- __AMD GPU backend__. `romc/hip` from AMD
  seems relatively easy to add using their hipify tool from CUDA code.

- __nvrtc and jitify__. Add runtime CUDA kernel compilation.
  [arrayfire implementation](https://github.com/arrayfire/arrayfire/blob/master/src/backend/cuda/compile_module.cpp),
  [cuda-api-wrappers](https://github.com/eyalroz/cuda-api-wrappers/tree/master/src/cuda/nvrtc),
  [jitify](https://github.com/NVIDIA/jitify).


- __Add nvc++__. One big change is the `if target` construct and no need for the `__device__` attribute.
  `Kokkos` uses a trick to define and compile CUDA kernels from .cpp files. They trick the user by treating EVERY
  .cpp file to a CUDA file, so everything is compiled with nvcc or nvc++. This is far from an ideal solution IMHO,
  one reason being that nvcc warnings are almost none and nvcc is not as robust as gcc/clang (there's
  two examples that come to me from the top of my head, where nvcc fail to compile standard C++ code). So yeah,
  `Kokkos` is not a solution for my problem (but `SYCL` might be).


- __Future framework__. `SYCL`. Ideally, for most operations, we would like to rely on the compiler to generate device
  specific code while staying in standard C++, which seems to be exactly what `SYCL` does.
  __HOWEVER__, in the future it seems that the `C++ Standard Parallelism` could simplify everything, especially with
  Senders/Receivers and Executions context policies, it is targeted to C++26, but who knows...
  [std::execution](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2300r5.html).
  The talks on YouTube from Bryce Adelstein Lelbach are quite good. I REALLY hope we'll be able to right device-agnostic
  code with std::ranges, std::execution, std::mdspan and stay in C++ for everything!
