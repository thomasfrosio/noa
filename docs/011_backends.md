## `CPU backend`

The library currently has one CPU backend, which is simply the “native” architecture
(e.g. x86-64 or arm64). This backend is not optional to the build.
The entire backend is templated and almost entirely exposed in public
headers, so that the library internals can be inlined with the user code, and the entire
program can get optimized nicely without any link-time optimization (aka interprocedural
optimisation) involved. The CPU stream is synchronous by default, pushing all work to
the current thread without any runtime overhead. An asynchronous mode is also provided,
in which case tasks are wrapped in a std::function and sent to the thread managed
by the stream.

As of time of writing, no explicit SIMD vectorisation was implemented. Instead, code is
written to help the optimizer automatically generate these vectorisations. The element-wise
and reduction core functions are, for instance, checking for the 1d and contiguous cases and
explicitly generating code for these cases. This results in slightly longer compile time and
binary sizes since both the generic case and the 1d contiguous case have to be generated,
but given that library template functions are only instantiated on demand, this has not
appeared to be an issue so far.

Multithreading is done using OpenMP and each core function has a multithreaded implementation.
The backend implementations allow to turn multithreading off directly at the call site,
to keep the amount of generated code to a minimum. The current strategy is to only run the
multithreaded implementation for very large arrays, but this was not really explored and things
are may to change in the future.


## `CUDA backend`

As of time of writing, this is the only GPU backend.
CUDA-C++ is a DSL so similar to C++ that we can easily write code that can
be compiled to both C++ and CUDA-C++. We extensively use this feature and most library
operators have a single implementation which is used by both the CPU and the CUDA
backends. Unfortunately, while most code can be abstracted to native C++, CUDA kernels
cannot be compiled and launched from C++ source files. We currently see three different
strategies to deal with this problem.

The most popular strategy is to compile the CUDA backend (and its .cu files) when building
the library. As a result, the applications only need to link the library as part of their build,
and can still benefit from the GPU acceleration without having to compile any CUDA code
directly. This approach is one of the most common ways to use CUDA, especially for projects
that only use GPUs transitively in their programs. Unfortunately, this approach has a big
disadvantage: it does not scale well. Indeed, while CUDA kernels can be templated, in
this scenario the library needs to explicitly instantiate these templates with all the types
and type combinations it aims to support, as opposed to compiling the template kernels
on a per-application basis. For libraries that intend to support a wide range of types and
functions, this often means having to compile thousands of kernels, most of which will not
be used by the application. Compiling this much code quickly becomes impractical and leads
to unreasonable compile times (minutes to hours), large binary sizes (hundreds of Mbytes
to Gbytes) and slow loading times (multiple seconds). Maybe more importantly, this
early compilation stage prevents us from easily integrating user-defined types and operators.

Instead, the library and its CUDA backend is fully templated and (mostly) header-only, thereby
leaving the burden of compilation to the applications. By including the library, users 
thus also include the CUDA kernel definitions of the library. Unfortunately,
CUDA kernels cannot be compiled in C++ translation units, meaning that in order
to compile and launch CUDA kernels, the application source files need to be compiled as
CUDA (.cu) files using a CUDA-capable compiler (usually nvcc). This option has
the advantage that only the kernels necessary to the application would get compiled and
allow users to easily pass their own types and operators to the GPU.
The disadvantage of this approach is that __all C++ files calling the library need to be
compiled by the CUDA compiler, which is a big requirement for some projects__.

Here's an example:
```cmake

# -- CMakeLists.txt --
project(my_application LANGUAGES C++ CUDA)

# Fetch noa from github, build with CUDA backend
include(FetchContent)
FetchContent_Declare(noa
    GIT_REPOSITORY ${noa_github_repository}
    GIT_TAG ${noa_github_tag})
set(NOA_ENABLE_CUDA ON)
FetchContent_MakeAvailable(noa)

set(my_source_files my_application.cpp)

if (NOA_ENABLE_CUDA)
    # In this example, my_application.cpp uses the library. To use its CUDA backend,
    # we need to compile the application sources as CUDA files:
    set_source_files_properties(${my_source_files} PROPERTIES LANGUAGE CUDA)
endif()

# Then continue as normal...
add_executable(my_application ${my_source_files})
target_link_library(my_application PRIVATE noa::noa)
```

```c++
// -- my_application.cpp --

#include <noa/Array.hpp>
#include <noa/Ewise.hpp>

// Because CUDA-C++ is very similar to C++, we can easily write
// code that can be compiled by any C++ compiler and also compiled
// by the CUDA compiler.

// My special type.
struct my_type { double value; };

// My special operator. See core functions for more details on how
// to write operators.
template<typename T>
struct my_special_operator {
    T random{};
    constexpr void operator()(T value, my_type& out) const {
        out.value = value > 0 ? value : random;
    }
};

noa::Array array = noa::random(noa::Uniform{-1., 1.}, 1024, {.device="cuda"});
noa::Array output = noa::like<my_type>(array);
double random = noa::random_value(noa::Uniform{0., 1.});

noa::ewise(array, output, my_special_operator{random});
```
