## `Dependencies`

Most dependencies are automatically fetched during configuration, so you don't need to install them manually. Take a look at the [`dependencies`](000_dependencies.md) for more details.

## `Build & Install`

```shell
git clone https://github.com/thomasfrosio/noa.git
cd noa
cmake -B ./build -DCMAKE_INSTALL_PREFIX=./install -DNOA_ENABLE_CUDA=ON # optional GPU support
cmake --build ./build --parallel
cmake --install ./build
```

Additional configure options can be passed. Options should be prefixed with `-D` when entered via the command line. the library supports a few project specific options all described in [`ProjectOptions.cmake`](../cmake/ProjectOptions.cmake). Additionally, the following CMake options are often recommended:

- `CMAKE_BUILD_TYPE`: The build type. Default: `Release`.
- `CMAKE_INSTALL_PREFIX`: Directory where the library will be installed.
- `CMAKE_PREFIX_PATH`: Additional path where to search for dependencies (e.g. CUDA Toolkit).
- `CMAKE_C_COMPILER`: C compiler.
- `CMAKE_CXX_COMPILER`: C++ compiler.
- `CMAKE_CUDA_COMPILER`: CUDA compiler.
- `CMAKE_CUDA_ARCHITECTURES`: CUDA architecture. Default: `native`.

```shell
...
cmake -B ./build -DCMAKE_INSTALL_PREFIX=./install \
    -DNOA_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=native
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/unusual/path/to/dependency \
    -DCMAKE_C_COMPILER=/unusual/path/to/c \
    -DCMAKE_CXX_COMPILER=/unusual/path/to/c++ \
    -DCMAKE_CUDA_COMPILER=/unusual/path/to/cuda-toolkit
...
```

_Note:_ To uninstall, simply delete the installation directory specified by `CMAKE_INSTALL_PREFIX`.

## `CMake import`

### [`find_package`](https://cmake.org/cmake/help/latest/command/find_package.html?highlight=find_package)

If the library is installed, you can import it into your CMake project, such as:
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.30)
project(MyProject LANGUAGES CXX)

# Enable CUDA if noa was built with CUDA support.
enable_language(CUDA)

find_package(noa)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE noa::noa)

# Required for CUDA builds.
if (TARGET noa::noa AND NOA_ENABLE_CUDA)
    set_target_properties(myapp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
    )
endif()
```
```shell
cmake -B ./build -DCMAKE_PREFIX_PATH=/path/to/noa/install
```


### [`FetchContent`](https://cmake.org/cmake/help/latest/module/FetchContent.html)

You can also directly fetch the sources, configure and build alongside your project, such as:
```cmake
cmake_minimum_required(VERSION 3.30)
project(MyProject LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
    noa
    GIT_REPOSITORY https://github.com/thomasfrosio/noa.git
    GIT_TAG main
)


set(NOA_ENABLE_CUDA ON) # configure options before fetching
FetchContent_MakeAvailable(noa)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE noa::noa)
```

Either way, this will import in your CMake project the `noa::` namespace with the imported targets, most notably `noa::noa`.
