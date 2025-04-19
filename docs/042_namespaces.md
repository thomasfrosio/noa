## `Namespaces`

```c++
namespace noa {
    // - core
    // - frontend / user API
    
    namespace fft {} // Fast Fourier Transform and related things
    namespace geometry {} // geometric transforms (affine, polar, projections, etc.)
    namespace signal {} // signal/image processing (convolution, filtering, CTF, etc.)
    namespace io {} // IO related things (ImageFile, read, write, etc.)
    namespace guts {} // implementation / private
    
    inline namespace types {
        //  - basic type aliases: i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, c16, c32, c64
        //  - core types: Vec, Shape, Strides, Tuple, Pair, Mat, Span, Accessor
        //  - frontend types: Session, Allocator, Device, ArrayOption, Stream, Event,
        //                    Array, View, Texture, ReduceAxes
    }
    
    namespace cpu {} // CPU backend
    namespace cuda {} // CUDA backend - only if NOA_ENABLE_CUDA=ON
    namespace gpu {} // alias of the GPU backend for this build
}
```

`noa::types` is an inline namespace used to "import" the main library types. Because it is inline, it can be entirely ignored, e.g. use `noa::Array` as opposed to `noa::types::Array`. However, if you want to include the library types and keep the library functions out of the way, `using namespace noa::types` is a nice option.
