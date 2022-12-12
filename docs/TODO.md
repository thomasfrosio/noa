## `cryoEM`


- __CTF__: Add CTF class gathering all the necessary variables (voltage, Cs, etc.) and basic utility functions,
  e.g. compute value at a given frequency. Single side-band algorithm support (complex CTFs, cones).


- __High-order aberrations__ and optics in general... Look at Warp and RELION's code. I know very little about this...


- __FSC__: Conical `FSC` and spectral whitening: these two are very similar in terms of implementation. One way, which
  would be more correct is to use `geometry::fft::cartesian2polar` to extract the lines. Having them in a polar grid
  makes the normalization extremely easy. One other solution is to round the frequency and store it in its bin, which is
  what cisTEM/RELION are doing I think. In this case, we need to keep track of the number of components added to
  each bins (lower bins have fewer components than higher bins).


- __Binning__: real space binning is a nice to have, although Fourier cropping is often preferred. Note that efficient
  binning on the GPU requires some thinking.


- Add center of mass, radial grid.


- __IO__: Add compression. Also, better support for 3dmod with complex types: IMOD excepts the logical shape but always
  excepts the non-redundant data. ImageFile treats the shape as the physical shape...


## `General`

- Refactor API/backend to enforce input and output being on the same device. This only concerns `geometry` and
  `signal::fft`, so it's quite simple. For `geometry`, the matrices should be on the output device, so we can
  remove temporaries/copies. Also, this goes with the fact that for affine transformations, we should not use textures
  for the overloads that take an array as input. This relies on a temporary texture, which isn't great, but would also
  allow to support all BorderMode and InterpMode, as well as double precision. To use textures, use the overloads
  taking textures. The API provides a device-agnostic way of using textures, so that's ok IMO.


- __Windows__. It should not be that complicated. One thing to look at is OpenMP support.


- __Generic CPU reductions__: Move the CPU reduction kernels in a details header and add a transform operator. This is
  especially useful when we want to apply a transform operator (e.g. abs_t) for the high precision sum reductions.


- __Common code between backends__: Progressively move what can be moved (e.g. every element-wise and index-wise
  operation) to the `common` directory to have a single implementation dispatched to the backends.
  Add `iwise(...)`. The operator parameters could just be (i, j, k, l) and the operators could just hold the Accessors,
  which works in CUDA as well. This could actually trigger be a huge refactoring to remove code duplication and
  centralize most CPU and CUDA to a few index-wise functions. Then we just need to provide functors/lambdas,
  which can be shared by all the backends... I really think this is where the library should go.

  For the `geometry` namespace, move the main kernels to the `common` directory as well and encapsulate textures,
  so they have the same interface as the CPU interpolators.


- __Vector N__: Not having a generic static Vector<N, T> is becoming more and more problematic. I should create such 
  vector and progressively replace the FloatN and IntN alternatives. This will also go with making the library more
  type-safe and have strong types for Strides<N, T> and Shape<N, T>.


- Add `io::load_data()`, to return the file data section (the Array). And make `io::load()` return the Array and
  the pixel size.


- Move the backend tests to the unified API. While the backend are mostly well tested, the API isn't. Instead, use
  the unified API for the tests to test both the main API and the backend at the same time.


- Add easy way to create F-major arrays. Atm this is a bit annoying. Also, extend the dimension swapping to the
  `geometry` namespace. The `fft` namespace should be the only place where F-major arrays are not allowed.


- __JIT, lazy evaluation and kernel fusion VS transform operators__.
  I much prefer the simplicity of `std::transform` or
  range-like APIs, where we define __lambdas as transform operators__. It is much more flexible, versatile, cleaner, 
  and it is where the C++ STL is going anyway (including NVIDIA). The issue with this simpler approach though: 1) it 
  cannot easily be used from another language, 2) CUDA is not really possible from a .cpp file, although it looks like 
  this is likely to change in the coming years with Executors or even nvc++.

  __Lazy evaluation and template expressions ala xtensor__.
  Again, I don't want to go down that path but FYI:
  `ArrayFire` is quite good at fusing kernels, the implementation seems relatively clean. I think `Cupy` is similar 
  but CUDA only. `Pytorch` should be the same but the codebase is huge, and it's difficult to understand what is going on.
  Also `Eigen` and `xtensor` have lazy evaluation. `xtensor` has a series of articles on how they implemented
  lazy-evaluation (and kernel fusion at the same time), but these are CPU only. Adding GPU support should be relatively
  simple though, with a string conversion on the operators and something like `jitify` from nvidia.
  Links:
  [arrayfire JIT](https://arrayfire.com/performance-of-arrayfire-jit-code-generation/)
  [xtensors series](https://johan-mabille.medium.com/how-we-wrote-xtensor-9365952372d9).

  At the moment, the CPU backend for `ewise()` can take any unary/binary/trinary element-wise transformation
  operator, which is great. However, since we cannot include and compile CUDA kernels from .cpp files OR pass host
  lambdas to CUDA kernels, we have to limit ourselves to pre-instantiated operators, which is very annoying.
  Note that the kernels are already there and can take any operators like in the CPU backend, but they have to be
  compiled by .cu files. I really hope nvc++ will solve this issue.
  One simple solution for now would be to keep using `ewise()` (and even add `iwise()`), and add a way for projects
  to add and compile other operators, which should be relatively easy since we have the utils/Ewise###.cuh headers.
  We can explicitly instantiate for these new device functors and link against it. On the library side, we have to
  add a "proclaim_ewise" traits that can be "appended" by the user and accepted by the API. That way, we keep the
  user code device-agnostic (they can use NOA_HD-like macro) and they need to have an extra .cu file to compile
  the new kernels... So instead of lambdas, we must use functors, which is not as good, but still quite good.
  Another solution is to wrap the C++ compiler the way kokkos does with their nvcc_wrapper and treat EVERY
  USER TU as CUDA files. I'm not sure if I like that, but that's a "simple" trick and it just works (assuming
  user code doesn't break nvcc/nvc++). At least, this could be a build option, like `NOA_ENABLE_CUDA_IN_CPP`...


-  Remove Ptr* for tmps and use shared_ptr and Ptr*::alloc() instead.


-  Functions returning something by value, that currently requires to synchronize the stream: use std::future?


- Session: Not sure what `Session` should be. It only holds the main `Logger` and the number of internal threads.
  Maybe we should link all global data to the session as well, but I don't think that's a good idea. Instead,
  maybe remove `Session` and create a `GlobalLogger` and a free function keeping track of the thread number or
  something like that.


- Add __nvrtc__?.
  Atm, nvcc and the CUDA runtime are used to compile kernels. The idea is to use nvrtc to compile (some) kernels
  to cubin directory at runtime. The launch mechanism will be centralized and in .cpp files, the kernels in
  .cu files. Ultimately, this will make us use the driver API, which gets rid of the triple-chevron and offers
  better control over the parameter packing and kernel launch.
  One other major optimization allowed by runtime compilation is the ability to bring some runtime evaluation
  to compile time. For instance, in some cases, passing some dimensions or count as a template parameter
  could be beneficial. Of course this needs to be tuned since we don't want to recompile everytime the
  function is called because there's a different template parameter...
  [arrayfire](https://github.com/arrayfire/arrayfire/blob/master/src/backend/cuda/compile_module.cpp),
  [cuda-api-wrappers](https://github.com/eyalroz/cuda-api-wrappers/tree/master/src/cuda/nvrtc),
  [jitify](https://github.com/NVIDIA/jitify).


- __Test CUDA LTO__ and unused kernels support from version 11.5.


- __CUDA constant memory.__ Test the benefits of constant memory. The issue with this is the (host) thread-safety.
  If multiple host threads use the same device but are on different streams, we need to lock guard the resource
  and kernel launch since the resource is global to the device.


- Use the __CUDA driver to handle the context__. That way, the library can keep track of its own context and reset it
  without affecting then entire application. This could be attached to the current Session.
  [Pytorch DeviceThreadHandles](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/DeviceThreadHandles.h)


- Add __vkFFT__ support? Also look at FFTW CPU port for Intel-MKL and AMD.
  Zero-padding can bring up to 2x increase of performance for 2D, 3x for 3D. Convolution can be added as a call-
  back but there's no benchmark for that. It could be useful to contact the author about some of our applications,
  e.g. convolution with small (compared to input) template. FastFFT from Ben Himes is also promising since it is
  really fitted for cryoEM applications, however it is CUDA only and still in development.


- __Future framework__. Add Vulkan backend or modern approach like `SYCL`. Ideally, for most operations, we would
  like to relly on the compiler to generate device specific code, which seems to be exactly what `SYCL` does. 
  Vulkan also seems a good option unified GPU support, but that's another backend and Apple-Vulkan is sketchy...
  If Vulkan, GLSL for shading language and then glslangValidator or glslc to compile to SPIR-V? `romc/hip` from AMD
  seems relatively easy to add using their hipify tool from CUDA code.
  __HOWEVER__, in the future it seems that the `C++ Standard Parallelism` could simplify everything, especially with
  Senders/Receivers and Executions context policies, but it is targeted to C++26 so...
  [std::execution](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2300r5.html).
  The talks on YouTube from Bryce Adelstein Lelbach are quite good. I REALLY hope we'll be able to right device-agnostic
  code with std::ranges, std::execution, std::mdspan and stay in C++ for everything!
