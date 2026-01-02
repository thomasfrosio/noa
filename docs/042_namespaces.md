## `Namespaces`

User-facing namespaces:
```c++
namespace noa {
    // runtime
    
    namespace fft {} // Fast Fourier Transform and related things
    namespace xform {} // transformations (quaternion, affine matrices, polar, projections, etc.)
    namespace signal {} // signal/image processing (convolution, filtering, CTF, etc.)
    namespace io {} // IO related things (ImageFile, TextFile, read_image, write_text, etc.)
    
    inline namespace types {
        //  - basic type aliases: i8, i16, u32, u64, f16, f32, f64, c16, c32, c64, etc.
        //  - core types: Vec, Shape, Strides, Tuple, Pair, Span, etc.
        //  - frontend types: Session, Allocator, Device, Stream, Event, Array, View, etc.
    }
}
```

`noa::types` is an inline namespace used to "import" the main library types. Because it is inline, it can be entirely ignored, e.g., use `noa::Array` as opposed to `noa::types::Array`. However, if you want to include the library types and keep the library functions out of the way, `using namespace noa::types` is a nice option.
