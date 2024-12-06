## `cryoEM`


- __High-order aberrations__ and optics in general... Look at Warp and RELION's code. I know little to nothing about this...


- Complex CTFs, for Russo/Henderson's EWS correction?


## `General`

- __Windows support__. It should not be that complicated. One thing to look at is OpenMP support.


- __as\<T\>()__ functions. We use these conversion member functions for various types: `Vec`, `Shape`, `Strides`, `Mat`, `Quaternion`, `CTFIsotropic`, `CTFAnisotropic`, and `Span`. These functions return by value. Should we return conditionally by reference if no conversion happens (`T == value_type`)? Performance wise, for these types, it should make any difference with optimizations on. From what I've seen it shouldn't break any code too. Maybe for C++23 when explicit object parameter is available, because right now it would mean to add at least 3 extra overloads for `const&`, `&`, and `&&`.


- __SIMD__ Trying explicit SIMD in the CPU-"core" functions (start with ewise_*?). [xsimd](https://xsimd.readthedocs.io/en/latest/index.html) seems interesting (their "abstract batch" at least).


- __CUDA constant memory.__ Test the benefits of constant memory. The issue with this is the (host) thread-safety. If multiple host threads use the same device but are on different streams, we need to lock guard the resource and kernel launch since the resource is global to the device. Idk how this works with nvrtc... is the ressource private to a context, or a module, or a translation unit?


- Use the __CUDA driver to handle the context__. That way, the library can keep track of its own context and reset it without affecting then entire application. This could be attached to the current Session. [Pytorch DeviceThreadHandles](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/DeviceThreadHandles.h)


- SYCL backend? OpenMP GPU? OpenACC? clang-cuda?


- Add __vkFFT__ support? Also look at FFTW CPU port for Intel-MKL and AMD. This is only for zero-padding, which can bring up to 2x increase of performance for 2D, 3x for 3D. Convolution can be added as a call- back but there's no benchmark for that.
