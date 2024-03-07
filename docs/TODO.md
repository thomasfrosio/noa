## `cryoEM`


- __High-order aberrations__ and optics in general... Look at Warp and RELION's code. I know nothing about this...


- Complex CTFs, for Russo/Henderson's EWS correction?


- __IO__: Refactor IO. Remove polymorphism and use something more modern and less coupled, like `std::variant`? Add compression and other file formats. Also, better support for 3dmod with complex types: IMOD excepts the logical shape but always excepts the non-redundant data. ImageFile treats the shape as the physical shape.


## `General`




- `noa::linalg`: linear algebra. idk if we should include `dot` and `matmul` here or keep these in `noa::`.
- `noa::indexing`: idk about this. We mostly use this internally, but there's `broadcast` for instance, which could be in `noa` directly... I certainly wouldn't want to have the `at` functions in `noa::`.


- __Windows support__. It should not be that complicated. One thing to look at is OpenMP support.


- __SIMD__ Trying explicit SIMD in the CPU-"core" functions (start with ewise_*?). [xsimd](https://xsimd.readthedocs.io/en/latest/index.html) seems interesting (their "abstract batch" at least).


- __CUDA constant memory.__
  
  Test the benefits of constant memory. The issue with this is the (host) thread-safety. If multiple host threads use the same device but are on different streams, we need to lock guard the resource and kernel launch since the resource is global to the device. Idk how this works with nvrtc... is the ressource private to a context, or a module, or a translation unit?


- Use the __CUDA driver to handle the context__. That way, the library can keep track of its own context and reset it
  without affecting then entire application. This could be attached to the current Session.
  [Pytorch DeviceThreadHandles](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/DeviceThreadHandles.h)


- SYCL backend!


- Add __vkFFT__ support? Also look at FFTW CPU port for Intel-MKL and AMD.
  This is only for zero-padding, which can bring up to 2x increase of performance for 2D, 3x for 3D. Convolution can be added as a call- back but there's no benchmark for that. It could be useful to contact the author about some of our applications, e.g. convolution with small (compared to input) template.


- __C++20/23__
  - Better use of concepts
  - Add more constexpr (specially string stuff)
  - Add mdspan to unified API: Array::mdspan(), Array::span()
  - using enum


- __C++26__
  - [std::execution-P2300](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2300r7.html) seems to be targeted for C++26. This has the potential to change C++, hopefully in a good way. The paper is promising and the talks from Bryce look ambitious. This could be amazing, but of course requires vendors to support it. NVIDIA is of course pushing for this, but hopefully it will have broader (GPU-)support. Note that this sort of changes the paradigm a bit, now the backends would be whatever the compiler supports, so if users want CUDA support for instance, they would need to compile their code with a compiler that has this backend (e.g. nvc++). `std::execution` is lazy, and is designed to allow operator-fusion.
