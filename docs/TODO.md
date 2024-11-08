## `cryoEM`


- __High-order aberrations__ and optics in general... Look at Warp and RELION's code. I know little to nothing about this...


- Complex CTFs, for Russo/Henderson's EWS correction?


- __IO__: Refactor IO. Remove polymorphism and use something more modern and less coupled, like `std::variant`? Add compression and other file formats. Also, better support for 3dmod with complex types: IMOD excepts the logical shape but always excepts the non-redundant data. ImageFile treats the shape as the physical shape.


## `General`

- __Windows support__. It should not be that complicated. One thing to look at is OpenMP support.


- __SIMD__ Trying explicit SIMD in the CPU-"core" functions (start with ewise_*?). [xsimd](https://xsimd.readthedocs.io/en/latest/index.html) seems interesting (their "abstract batch" at least).


- __CUDA constant memory.__ Test the benefits of constant memory. The issue with this is the (host) thread-safety. If multiple host threads use the same device but are on different streams, we need to lock guard the resource and kernel launch since the resource is global to the device. Idk how this works with nvrtc... is the ressource private to a context, or a module, or a translation unit?


- Use the __CUDA driver to handle the context__. That way, the library can keep track of its own context and reset it without affecting then entire application. This could be attached to the current Session. [Pytorch DeviceThreadHandles](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/DeviceThreadHandles.h)


- SYCL backend? OpenMP GPU? OpenACC? clang-cuda?


- Add __vkFFT__ support? Also look at FFTW CPU port for Intel-MKL and AMD. This is only for zero-padding, which can bring up to 2x increase of performance for 2D, 3x for 3D. Convolution can be added as a call- back but there's no benchmark for that.
