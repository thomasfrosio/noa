## `Dependencies`

The library as very few dependencies, most of which should be installed before building the library. Take a look at the [`dependencies`](000_dependencies.md) for more details.

## `Build and Install`

_Without presets:_
```shell
git clone https://github.com/thomasfrosio/noa.git
cd noa && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..  # + config options
cmake --build . --target install -- -j 16
```

Additional configure options can be passed. Options should be prefixed with `-D` when entered via the command line. Here is the [`list`](../cmake/ProjectOptions.cmake) of the available options. Additionally, the following CMake options are often recommended:
- `CMAKE_BUILD_TYPE`: The build type. Default: `Release`.
- `CMAKE_INSTALL_PREFIX`: Directory where the library will be installed.
- `CMAKE_PREFIX_PATH`: Additional path where to search for dependencies (e.g. CUDA Toolkit).
- `CMAKE_C_COMPILER`: C compiler.
- `CMAKE_CXX_COMPILER`: C++ compiler.
- `CMAKE_CUDA_COMPILER`: CUDA compiler.

_With presets:_
TODO

_Note:_ Depending on the generator, an `install_manifest.txt` can generated during the installation. We don't generate the `uninstall` target, but one can remove everything that was installed using ``xargs rm < install_manifest.txt``. However, if `CMAKE_INSTALL_PREFIX` is used as shown above, deleting that same directory is probably the simplest.

## `Import`

### [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html?highlight=find_package)

If the library is installed, you can import it into your CMake project, such as:
```cmake
find_package(noa)
# or
find_package(noa EXACT CONFIG REQUIRED) # requires using noaConfig.cmake
```

If you built the library from source, CMake should know where to look for, but you can always add the installation directory to the `CMAKE_PREFIX_PATH` (this should correspond to the `CMAKE_INSTALL_PREFIX` used during the configuration) when configuring your project.

### [`FetchContent`](https://cmake.org/cmake/help/latest/module/FetchContent.html)

Otherwise, you can directly fetch the sources, configure and build noa alongside your project, such as:
```cmake
include(FetchContent)
FetchContent_Declare(noa GIT_REPOSITORY https://github.com/thomas.frosio/noa.git)
FetchContent_MakeAvailable(noa)
```

Either way, this will import in your CMake project the `noa::` namespace with the imported targets, most notably `noa::noa`.

### `CUDA specificities`
If CUDA is enabled, your targets linking against `noa::noa` should have their `CUDA_SEPARABLE_COMPILATION` property `ON` (if not, your application will immediately crash).
Furthermore, remember to set the `CUDA_ARCHITECTURES` property for your targets to have the best possible performance!
