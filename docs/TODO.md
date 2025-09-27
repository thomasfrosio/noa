## `cryoEM`


- __High-order aberrations__ and optics in general... Look at Warp and RELION's code. I know little to nothing about this...


- Complex CTFs, for Russo/Henderson's EWS correction?


- Do a quick review on the function taking fftfreq_range. Replace by a Linspace? Do we need one for the input and one for the output? Move FSC to geometry since it's a transformation, like rotational_average.



## `General`

- The core should be extracted into a separate utility library. Tuple, Vec, Shape/Strides, Span, etc. Same for io.


- __Windows support__. It should not be that complicated. One thing to look at is OpenMP support.


- __as\<T\>()__ functions. We use these conversion member functions for various types: `Vec`, `Shape`, `Strides`, `Mat`, `Quaternion`, `CTFIsotropic`, `CTFAnisotropic`, and `Span`. These functions return by value. Should we return conditionally by reference if no conversion happens (`T == value_type`)? Performance wise, for these types, it shouldn't make any difference with optimizations on. From what I've seen it shouldn't break any code too. Maybe for C++23 when explicit object parameter is available, because right now it would mean to add at least 3 extra overloads for `const&`, `&`, and `&&`.


- Frontend functions are templates, thus implicitly inline. I've noticed that sometimes it puts too much code into the caller block and may prevent some optimizations from being triggered. I think it could be useful to start using `[[gnu::noinline]]` in these frontend functions and/or in the core functions.


- __SIMD__ Trying explicit SIMD in the CPU-"core" functions (start with ewise_*?). [xsimd](https://xsimd.readthedocs.io/en/latest/index.html) seems interesting (their "abstract batch" at least).


- __CUDA constant memory.__ Test the benefits of constant memory. The issue with this is the (host) thread-safety. If multiple host threads use the same device but are on different streams, we need to lock guard the resource and kernel launch since the resource is global to the device. Idk how this works with nvrtc... is the ressource private to a context, or a module, or a translation unit?


- Use the __CUDA driver to handle the context__. That way, the library can keep track of its own context and reset it without affecting then entire application. This could be attached to the current Session. [Pytorch DeviceThreadHandles](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/DeviceThreadHandles.h)


- SYCL backend (AdaptiveCpp looks great!)? OpenMP GPU? OpenACC? clang-cuda?


- Add __vkFFT__ support? Also look at FFTW CPU port for Intel-MKL and AMD. This is only for zero-padding, which can bring up to 2x increase of performance for 2D, 3x for 3D. Convolution can be added as a call- back but there's no benchmark for that.


- Replace libtiff with something else. I hate it.
  Then entire IO/Encoders is still not great, but I'll wait until I add compression to think about this. I think some of it could be in the frontend since we may want to compress on the GPU!

