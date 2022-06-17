## `Dependencies`

While some dependencies are directly fetched and install by the library, some are not.

- __CMake__. The minimum version is 3.18. Note that it has almost no dependencies and can be installed without root
  privileges.

- __C++__ compiler. The supported C++ compilers are `gcc >= 9.3` and `clang >= 10`. The C++ compiler path can be
  specified to ensure that the correct host compiler is selected, or if the compiler is not installed in the default
  paths and is not found, by setting the `CMAKE_CXX_COMPILER` variable.

- The [FFTW3](http://fftw.org/) libraries are required for the CPU backend. We ask for the single and double precision
  libraries. The multithreaded libraries will be used if found and if `NOA_FFTW_THREADS=1` (see below). By default, the
  common installation directories are used but the search can be guided by the following _environmental_ variables:
    - `NOA_ENV_FFTW_LIBRARIES`: If set and not empty, the libraries are exclusively searched under this path.
    - `NOA_ENV_FFTW_INCLUDE`: If set and not empty, the header `fftw3.h` is exclusively searched under this path.

- The [OpenBLAS](https://www.openblas.net/) libraries are required for the CPU backend. We recommend building the library from source since the
  build checks for the best implementation given the current architecture. For instance:
    ```shell
    mkdir OpenBLAS && cd OpenBLAS
    git clone --depth 1 --branch v0.3.20 https://github.com/xianyi/OpenBLAS.git
    mkdir _build && cd _build
    cmake -DCMAKE_INSTALL_PREFIX=../_install -DUSE_THREAD=1 -DUSE_OPENMP=1 ..
    cmake --build . --target install
    ```
  By default, the common installation directories are used but the search can be guided by the following _environmental_
  variables:
    - `NOA_ENV_OPENBLAS_LIBRARIES`: If set and not empty, the libraries are exclusively searched under this path.
    - `NOA_ENV_OPENBLAS_INCLUDE`: If set and not empty, the header `cblas.h` and `lapack.h` are exclusively searched
      under this path.

- If the CUDA backend is built (see below), the [CUDA toolkit](https://docs.nvidia.com/cuda/index.html) version 11 is
  required, although version 11.2 or any newer version is recommended. The CUDA compiler path can be specified to ensure
  that the correct CUDA compiler is selected, or if the compiler is not installed in the default paths and is not found,
  by setting the `CMAKE_CUDA_COMPILER` variable. Only __nvcc__ is currently supported.

- (Optional) The TIFF library [TIFF library](https://gitlab.com/libtiff/libtiff) is necessary if the TIFF file format is
  required. By default, the common installation directories are used but the search can be guided by the following _
  environmental_ variables:
    - `NOA_ENV_TIFF_LIBRARIES`: If set and not empty, the libraries are exclusively searched under this path.
    - `NOA_ENV_TIFF_INCLUDE`: If set and not empty, the header `tiffio.h` is exclusively searched under this path.

## `Build options`

"Options" are CACHE variables (i.e. they are not updated if already set), so they can be set from the
command line or the [cmake-gui](https://cmake.org/cmake/help/latest/manual/cmake-gui.1.html).
Options should be prefixed with `-D` when passed through the command line.

Here is the [list](../cmake/settings/ProjectOptions.cmake) of the project-specific available options. 
Note that the following CMake CACHE variables are often useful:
- `CMAKE_BUILD_TYPE`: The build type. Default: `Release`.
- `CMAKE_INSTALL_PREFIX`: Install directory used by CMake.
- `BUILD_SHARED_LIBS`: should be `OFF` as dynamic linking is not supported.

## `Build and Install`

To build and install the library the easiest is probably to use the command line:

```shell
mkdir noa && cd noa                             # (1)
git clone https://github.com/ffyr2w/noa.git     # (2)
mkdir _build && cd _build                       # (3)
cmake -DCMAKE_INSTALL_PREFIX=../_install ../noa # (4)
cmake --build . --target install                # (5)
```

1. Create a directory where to put the source code as well as the build and install directory. We'll refer to this
   directory as `{install_path}`.
2. Once in `{install_path}`, clone the repository.
3. Create and go into the build directory. This has to be outside the source directory that was just cloned.
4. Sets the environmental variables if the defaults are not satisfying and if it is not already done. Then configure and
   generate the project using _CMake_. This is also where the build options should be entered. It is usually useful to
   specify the installation directory using `CMAKE_INSTALL_PREFIX`. Note that the library has a few dependencies
   entirely managed by _CMake_ using
   [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html). Although it simplifies most workflows,
   this step requires a working internet connection to download the dependencies.
5. Once the generation is done, _CMake_ can build and install the project.

In this example, the `{install_path}/_install` directory will include:

- `lib/`, which contains the library. In debug mode, the library is postfixed with "d", e.g. `libnoad.a`.
- `lib/cmake/noa/`, which contains the _CMake_ project packaging files.
- `bin/`, which contains the `noa_tests` and `noa_benchmarks` executables.
- `include/`, which contains the headers. The directory hierarchy is identical to `src/` with the addition of some
  generated headers, like `include/noa/Version.h`.

_Note:_ Alternatively, one can use an IDE supporting _CMake_ projects, like _CLion_ and create a new project from an
existing source.
_Note:_ An `install_manifest.txt` is generated, so `make unistall` can be run to uninstall the library. However, if
`CMAKE_INSTALL_PREFIX` is used as shown above, deleting the directory is equivalent.

## `Import`

If the library is installed,
[find_package](https://cmake.org/cmake/help/latest/command/find_package.html?highlight=find_package)
can be used to import the library into a _CMake_ project. Otherwise,
[FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)
can also be used, which in its simplest form would look something like:

```cmake
include(FetchContent)
FetchContent_Declare(
        noa
        GIT_REPOSITORY https://github.com/ffyr2w/noa.git
        GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(noa)
```

The imported target are 1) the library, aliased to `noa::noa`, 2) the test executable, `noa::tests`, and 3) the
benchmark executable, `noa::benchmarks`.
