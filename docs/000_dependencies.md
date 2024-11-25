## `CMake`

The minimum version is `3.24`. Note that CMake has almost no dependencies and can be installed without root privileges.


## `C++ compiler`

The supported C++ compilers are `gcc >= 13` and `clang >= 16`. The C++ compiler can be specified to ensure that the correct host compiler is selected, or if the compiler is not installed in the default paths and is not found, by setting the `CMAKE_CXX_COMPILER` variable.


## `CUDA` and `CUDAToolkit`

To build the CUDA backend, the CUDA compiler `nvcc` is required. It comes with the [`CUDA toolkit`](https://docs.nvidia.com/cuda/index.html), which is also required. We require version `12.6` or newer.

The CUDA compiler can be specified to ensure that the correct compiler is selected, or if the compiler is not installed in the default paths and is not found, by setting the `CMAKE_CUDA_COMPILER` variable.

Once the CUDA compiler is set, the toolkit is usually found automatically, but the search can be guided by setting the CMake variable or environment variable `CUDAToolkit_ROOT` or by adding the path(s) to CMake variable `CMAKE_PREFIX_PATH`.

## `Automatically fetched dependencies`

### `fmt`

By default, it is fetched from GitHub and will be installed with noa, so you have nothing to do. However, if you use `fmt` in your project, you should either use the version that came with noa, or force noa to use your libraries by including the `fmt::fmt` target before including noa.


### `FFTW3`

The CPU backend requires the [`FFTW3`](http://fftw.org/) libraries. By default, these are fetched from GitHub and will be installed with noa, so you have nothing to do. Please note that we use a special CMake wrapper to facilitate the installation and to not collide with (possibly older) versions in the global `LIBRARY_PATH`. Your project can keep using its own `FFTW`, or you can also use the `fftw3::fftw3` and/or `fftw3::fftw3f` targets that are imported alongside noa. See the [repository](https://github.com/thomasfrosio/fftw3) for more details.
