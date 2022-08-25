The library can automatically fetch and install _some_ dependencies from the web. However, in most cases, dependencies
should be installed on the system. The library will search for these dependencies and users can guide this search as
detailed below. If the dependencies are installed in common locations, no input from the user should be required.


## `CMake`

The minimum version is `3.18`. Note that CMake has almost no dependencies and can be installed without root privileges.


## `C++ compiler`

The supported C++ compilers are `gcc >= 9.3` and `clang >= 10`. The C++ compiler can be
specified to ensure that the correct host compiler is selected, or if the compiler is not installed in the default
paths and is not found, by setting the `CMAKE_CXX_COMPILER` variable.


## `fmt` and `spdlog`

By default, it is fetched from GitHub and will be installed with noa, so you have nothing to do. However, if you use
these libraries in your project, you should either use the libraries that came with noa, or force noa to use your
libraries by including the `fmt::fmt` and `spdlog::spdlog` targets before including noa. If you cannot include them 
before including noa for some reason, there are two options to tell noa where to find these dependencies:

- We use [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html) to get `fmt` and `spdlog`.
  As such, previous calls to `FetchContent_Declare` for these libraries before will override what noa has set up and
  will use that instead.
- You can set the CMake variables `FETCHCONTENT_SOURCE_DIR_FMT` and `FETCHCONTENT_SOURCE_DIR_SPDLOG` to the
  source directory of these packages that you have in your system and noa will use that instead.


## `TIFF` _(optional)_

The TIFF library [TIFF library](https://gitlab.com/libtiff/libtiff) is necessary if support for the TIFF file format is
desired.
By default, the common installation directories are used but the search can be guided by setting the CMake variable
and environment variable `TIFF_ROOT` or by adding the path(s) to CMake variable `CMAKE_PREFIX_PATH`.
These paths can also point to a `TIFFConfig.cmake` file.

To search specifically for the static libraries, set the CMake variable `TIFF_STATIC` to `ON`.


## `FFTW3` _(CPU backend)_

The [FFTW3](http://fftw.org/) libraries are required by the CPU backend. We always ask for the single and double
precision libraries.
By default, the common installation directories are used but the search can be guided by setting the CMake variable
and environment variable `FFTW3_ROOT` or by adding the path(s) to CMake variable `CMAKE_PREFIX_PATH`.
These paths can also point to a `FFTW3Config.cmake` file.

To search specifically for the static libraries, set the CMake variable `FFTW3_STATIC` to `ON`.

To add the multi-threaded libraries, set the CMake variable `FFTW3_THREADS` to `ON`. To use the OpenMP version 
instead of the system threads, set the CMake variable `FFTW3_OPENMP` to `ON`. By default, `FFTW3_OPENMP` is set to 
`NOA_CPU_OPENMP`.


## `CBLAS` and `LAPACKE` _(CPU backend)_

The CBLAS and LAPACKE libraries are required for the CPU backend. The library searches for the BLAS library and makes
sure the CBLAS API is present. The same thing is done with LAPACK for the LAPACKE API, however, we do allow the LAPACKE
to be a different standalone library (which is often the case).
By default, the common installation directories are used but the search can be guided by setting the CMake variable
and environment variable `BLAS_ROOT` or `LAPACKE_ROOT` or by adding the path(s) to CMake variable `CMAKE_PREFIX_PATH`.

To search specifically for the static libraries, set the CMake variable `BLA_STATIC` to `ON`.
If the LAPACKE library is a different library, `LAPACKE_STATIC` should be used (it defaults to `BLA_STATIC`).

To search specifically for some vendors, see
[BLA_VENDOR](https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors).
If `OpenBLAS` is used, some additional diagnostic regarding the threading model used by the library will be printed 
during configuration. These are only for you to help select the threading model than you want. We recommend using
the OpenBLAS libraries built with OpenMP if `NOA_CPU_OPENMP` is `ON`.

## `CUDA` and `CUDAToolkit` (CUDA backend)

If the CUDA backend is built, the CUDA compiler `nvcc` is required. It comes with the
[CUDA toolkit](https://docs.nvidia.com/cuda/index.html), which is also required. We support any version starting from
11, although version 11.2 or any newer version is recommended.

The CUDA compiler can be specified to ensure that the correct compiler is selected, or if the compiler is not
installed in the default paths and is not found, by setting the `CMAKE_CUDA_COMPILER` variable.

Once the CUDA compiler is set, the CUDAToolkit is usually found automatically, but the search can be guided by setting
the CMake variable and environment variable `CUDAToolkit_ROOT` or by adding the path(s) to CMake variable
`CMAKE_PREFIX_PATH`.
