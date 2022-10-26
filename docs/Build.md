## `Dependencies`

The library as dependencies, most of which should be installed before starting to build the library.
Take a look the [dependencies](Dependencies.md) for more details.

TODO Add package managers, e.g. conan, vkpg.

## `Build options`

"Options" are cache variables (i.e. they are not updated if already set), so they can be set from the
command line or using a tool like [cmake-gui](https://cmake.org/cmake/help/latest/manual/cmake-gui.1.html).
Options should be prefixed with `-D` when entered on the command line.

Here is the [list](../cmake/util/ProjectOptions.cmake) of the project-specific options available.
Note that the following CMake cache variables are often useful:
- `CMAKE_BUILD_TYPE`: The build type. Default: `Release`.
- `CMAKE_INSTALL_PREFIX`: Directory where the library will be installed.

## `Build and Install`

To build and install the library the easiest is probably to use the command line. For instance:

```shell
git clone https://github.com/ffyr2w/noa.git     # (1)
cd noa && mkdir build && cd build               # (2)
cmake -DCMAKE_INSTALL_PREFIX=../install ../noa  # (3)
cmake --build . --target install -- -j 16       # (4)
```

1. Clone the repository.
2. Go inside the repository and create the build and install directory. Go in the build directory.
3. Configure and generate the project. This is also where the build options should be entered. It is usually useful to
   specify the installation directory using `CMAKE_INSTALL_PREFIX`. Note that this step can require a working internet 
   connection to download the dependencies (see [dependencies](Dependencies.md)).
4. Once the generation is done, we can build and install the project.

The installation directory will look something like this:

- `lib/`, which contains the library. In debug mode, the library is postfixed with "d", e.g. `libnoad.a`.
- `lib/cmake/`, which contains the CMake project config and packaging files.
- `bin/`, which contains the `noa_tests` and `noa_benchmarks` executables.
- `include/`, which contains the headers. The directory hierarchy is identical to the one in the source directory 
  inside `src/` with the addition of some generated headers, like `include/noa/Version.h`.

_Note:_ Alternatively, one can use an IDE supporting _CMake_ projects, like _CLion_ and create a new project from an
existing source.

_Note:_ If you are using `make`, an `install_manifest.txt` is generated during the installation. We don't generate the
`uninstall` target, but one can remove everything that was installed using ``xargs rm < install_manifest.txt``.
However, if `CMAKE_INSTALL_PREFIX` is used as shown above, deleting that same directory is probably best.

## `Import`

If the library is installed,
[find_package](https://cmake.org/cmake/help/latest/command/find_package.html?highlight=find_package)
can be used to import the library into another CMake project.
Suggested usage:
```cmake
find_package(noa)
# or
find_package(noa v0.1.0 EXACT CONFIG REQUIRED)
```
If you build the library from source, CMake will
know where to look for, but you can always add the installation directory to the `CMAKE_PREFIX_PATH`.

Otherwise, [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html) can be used instead, which in 
its simplest form would look something like:

```cmake
include(FetchContent)
FetchContent_Declare(noa GIT_REPOSITORY https://github.com/ffyr2w/noa.git GIT_TAG v0.1.0)
FetchContent_MakeAvailable(noa)
```

Either way, this will import in your CMake project the `noa::` namespace with the imported targets:
- `noa`: the library.
- `noa_tests`: the test executable (if `NOA_BUILD_TESTS` is `ON`).
- `noa_benchmarks`: the benchmark executable (if `NOA_BUILD_BENCHMARKS` is `ON`).
