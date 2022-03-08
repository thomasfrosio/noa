## `Toolchain`

The build generator is `CMake >= 3.18`. Note that `CMake` has almost no dependencies and can be installed without root
privileges. If the CUDA backend is built (see below), the `CUDA toolkit >= 11.0` is required (`CUDA toolkit >= 11.2` is
very much recommended). The supported C++ compilers are `gcc >= 9.3` and `clang >= 10`.

## `Environmental variables`

- The `FFTW3` libraries are required. Currently, `noa` cannot install it for you but CMake will try to find your install
  in the default locations. These variables are also available for you to guide this search:
    - `NOA_FFTW_LIBRARIES`: If set and not empty, the libraries are exclusively searched under this path.
    - `NOA_FFTW_INCLUDE`: If set and not empty, the headers (i.e. `fftw3.h`) are exclusively searched under this path.

## `Build options`

"Options" are CACHE variables (i.e. they are not updated if already set), so they can be set from the
command line or the [cmake-gui](https://cmake.org/cmake/help/latest/manual/cmake-gui.1.html).
Options should be prefixed with `-D` when passed through the command line.

- `NOA_ENABLE_WARNINGS`: Enable host and device compiler warnings. Default: `ON`.
- `NOA_ENABLE_WARNINGS_AS_ERRORS`: Treat compiler warnings as errors. Default: `OFF`.
- `NOA_ENABLE_LTO`: Enable Interprocedural Optimization, aka Link Time Optimization (LTO). Default: `OFF`.
- `NOA_ENABLE_PCH`: Build using precompiled header to speed up compilation time in Debug mode. Default: `ON`.
- `NOA_ENABLE_PROFILER`: Build `noa::Profiler`. Default: `OFF`.
- `NOA_ENABLE_TIFF`: Enable support for the TIFF file format. Default: `ON`.

- `NOA_ENABLE_CPU`: Generate and build the CPU backend. Default: `ON`.
- `NOA_ENABLE_OPENMP`: Enable multithreading, using `OpenMP`, on the CPU backend. Default: `ON`.
- `NOA_FFTW_USE_STATIC`: Whether `FFTW3` should be statically linked against `noa`. Default: `OFF`.
- `NOA_FFTW_USE_THREADS`: Use a `FFTW3` multithreaded libraries. If `NOA_ENABLE_OPENMP` is true, `OpenMP` threads are
  used instead of the system threads. Default: `ON`.

- `NOA_ENABLE_CUDA`: Generate and build the CUDA GPU backend. Default: `ON`.
- `NOA_CUDA_ARCH`: List of architectures to generate device code for. Default: `61`.
- `NOA_CUDA_USE_CUFFT_STATIC`: Use the `cuFFT` static library instead of the shared ones. Default: `OFF`.

- `NOA_BUILD_TESTS`: Build the tests. The `noa::noa_tests` target will be available to the project. Default: `ON`*
- `NOA_BUILD_BENCHMARKS`: Build the benchmarks. The `noa::noa_benchmarks` target will be made available to the project.
  Default: `ON`*.

*: Defaults to `ON` when `noa` is the "main" project, i.e. when it is not being imported into another project.

## `CMake useful options`

- `CMAKE_BUILD_TYPE`: The build type. `noa` defaults to `Release`.
- `CMAKE_CXX_COMPILER`: The C++ compiler path can be specified to ensure that the correct host compiler is selected or
  if the compiler is not installed in the default paths and is not found by `CMake`.
- `CMAKE_CUDA_COMPILER`: The CUDA compiler path can be specified to ensure that the correct device compiler is selected
  or if the compiler is not installed in the default paths and is not found by `CMake`. `noa` only supports `nvcc`.
- `CMAKE_INSTALL_PREFIX`: Install directory used by CMake.

_Note:_  `BUILD_SHARED_LIBS` is not supported. `noa` is a statically linked library

## `Build and Install`

To build and install the library, as a project of its own, the easiest is probably to use the
command line:

```shell
mkdir noa && cd noa                                 # (1)
git clone https://github.com/ffyr2w/noa.git         # (2)
mkdir _build && cd _build                           # (3)
cmake -DCMAKE_INSTALL_PREFIX=../_install ../noa     # (4)
cmake --build . --target install                    # (5)
```

1. Create a directory where to put the source code as well as the build and install directory. We'll refer to this
   directory as `{installation home}`.
2. Once in `{installation home}`, clone the repository.
3. Create and go into the build directory. This has to be outside the source directory that was just cloned.
4. Sets the `noa` environmental variables if the defaults are not satisfying. Then configure and generate the project
   using `CMake`. It is usually useful to specify the install directory. This will set up `noa`, with its CUDA backend,
   as well as the tests and benchmarks, in `Release` mode. This behavior can be changed by passing the appropriate
   project options as specified above. `noa` has a few dependencies (see `ext/`), most of which are entirely managed by
   CMake using [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html). Although it simplifies most
   workflows, this step requires a working internet connection to download the dependencies.
5. Once the generation is done, CMake can build and install the project.

In this example, the `{installation home}/_install` directory will include:

- `lib` contains the library. In debug mode, the library is postfixed with `d`, e.g. `libnoad.a`.
- `lib/cmake/noa` contains the `CMake` project packaging files.
- `bin` contains the `noa_tests` and `noa_benchmarks` executables.
- `include` contains the headers. The directory hierarchy is identical to `src` with the addition of
  some generated headers, like `include/noa/Version.h`.

_Note:_ Alternatively, one can use an IDE supporting CMake projects, like CLion and create a new project from an
existing source.
_Note:_ An `install_manifest.txt` is generated, so `make unistall` can be run to uninstall the library.

## `Import`

If the library is installed,
[find_package](https://cmake.org/cmake/help/latest/command/find_package.html?highlight=find_package)
can be used to import the library into a project. Otherwise,
[FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)
is probably the easiest way to consume noa, which in its simplest form would look something like:

```cmake
# [...]

include(FetchContent)
FetchContent_Declare(
        noa
        GIT_REPOSITORY https://github.com/ffyr2w/noa.git
        GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(noa)

# [...]
```

The `CMake` project `noa` comes with three targets; the library, aliased to `noa::noa`, the test
executable, `noa::tests`, and the benchmark executable, `noa::banchmarks`.
