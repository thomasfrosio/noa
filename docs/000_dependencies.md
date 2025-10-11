## `CMake`

The minimum version is `3.30`. Note that CMake has almost no dependencies and can be installed without root privileges.


## `C++ compiler`

The supported C++ compilers are `gcc >= 13` and `clang >= 16`. The C++ compiler can be specified to ensure that the correct host compiler is selected, or if the compiler is not installed in the default paths and is not found, by setting the `CMAKE_CXX_COMPILER` variable.


## `CUDA` and `CUDAToolkit`

To build the CUDA backend, the CUDA compiler `nvcc` is required. It comes with the [`CUDA toolkit`](https://docs.nvidia.com/cuda/index.html), which is also required. We require version `12.6` or newer.

The CUDA compiler can be specified to ensure that the correct compiler is selected, or if the compiler is not installed in the default paths and is not found, by setting the `CMAKE_CUDA_COMPILER` variable.

Once the CUDA compiler is set, the toolkit is usually found automatically, but the search can be guided by setting the CMake variable or environment variable `CUDAToolkit_ROOT` or by adding the path(s) to CMake variable `CMAKE_PREFIX_PATH`.

## `libtiff`

To support TIFF files, we require the `libtiff` shared library. CMake should be able to find it on your system easily, but the search can be guided using the CMake variable `CMAKE_PREFIX_PATH`.

## `Fetched dependencies`

The library comes with a few other public (like `{fmt}`) and private (like `Eigen`) dependencies. These are fetched from the internet during the CMake generation step. Private dependencies are not exposed to our headers, but public dependencies are.

The only public dependency that may cause an issue is `{fmt}`. By default, it is fetched from GitHub and will be installed with this library, so you have nothing to do. However, if you use `{fmt}` in your project, you should either use the version that came with `noa`, or force `noa` to use your libraries by including the `fmt::fmt` target before including `noa::noa`.
