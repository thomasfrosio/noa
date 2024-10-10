## `CMake`

The minimum version is `3.24`. Note that CMake has almost no dependencies and can be installed without root privileges.


## `C++ compiler`

The supported C++ compilers are `gcc >= 13` and `clang >= 16`. The C++ compiler can be specified to ensure that the correct host compiler is selected, or if the compiler is not installed in the default paths and is not found, by setting the `CMAKE_CXX_COMPILER` variable.


## `CUDA` and `CUDAToolkit`

To build the CUDA backend, the CUDA compiler `nvcc` is required. It comes with the [CUDA toolkit](https://docs.nvidia.com/cuda/index.html), which is also required. We require version 12.6 or newer.

The CUDA compiler can be specified to ensure that the correct compiler is selected, or if the compiler is not
installed in the default paths and is not found, by setting the `CMAKE_CUDA_COMPILER` variable.

Once the CUDA compiler is set, the CUDAToolkit is usually found automatically, but the search can be guided by setting the CMake variable and environment variable `CUDAToolkit_ROOT` or by adding the path(s) to CMake variable `CMAKE_PREFIX_PATH`.


## `TIFF`

The [TIFF library](https://gitlab.com/libtiff/libtiff) is necessary if support for the TIFF file format is desired. By default, the common installation directories are used but the search can be guided by setting the CMake variable and environment variable `TIFF_ROOT` or by adding the path(s) to CMake's `CMAKE_PREFIX_PATH`. These paths can also point to a `TIFFConfig.cmake` file.


## `fmt`

By default, it is fetched from GitHub and will be installed with noa, so you have nothing to do. However, if you use `fmt` in your project, you should either use the version that came with noa, or force noa to use your libraries by including the `fmt::fmt` target before including noa.


## `FFTW3`

The CPU backend requires the [FFTW3](http://fftw.org/) libraries. By default, these are fetched from GitHub and will be installed with noa, so you have nothing to do.

