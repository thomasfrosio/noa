## `Span`

`Array` and `View` provide an operator() taking 4d-indices, which can be used to access the underlying values of the nd-arrays. However, this is not the preferred way to access data. First, this may not be the most efficient way to access the data, for instance, 1d contiguous arrays can be accessed using a single index and a fixed stride of 1 (known at compile time). Second, depending on the device and memory resource, direct access from CPU threads may not be safe or even allowed.

`Span`, i.e. a view over multidimensional data, should be used instead. The library comes with its own Span ([Span.hpp](../src/noa/core/types/Span.hpp)), which is similar to C++23 `std::mdspan`.

```c++
template<typename T,        // type, can be const
         size_t N = 1,      // number of dimensions
         typename I = i64,  // integral type used for the strides and shape
         StridesTraits StridesTrait = StridesTraits::STRIDED>
class Span {
    T* pointer;
    Shape<I, N> shape;
    Strides<I, N - StridesTrait == StridesTraits::CONTIGUOUS> strides;
};
```

Strides are fully dynamic (one dynamic stride per dimension) by default, but the rightmost dimension can be marked contiguous. `Span` (as well as `Accessor` and the library internals) uses the rightmost convention, so that the innermost dimension is the rightmost dimension. As such, `StridesTraits::CONTIGUOUS` implies C-contiguous. F-contiguous layouts are only supported in `StridesTraits::STRIDED` mode.
With `StridesTraits::CONTIGUOUS`, the innermost/rightmost stride is fixed to 1 and is not stored, resulting in the strides being truncated by 1 (`Strides<I,N-1>`). In case of a 1d contiguous span, this means that the strides are empty (`Strides<I,0>`) and the indexing is equivalent to pointer/array indexing.

```c++
// Import i64, f64, Shape and Span
using namespace ::noa::types;

auto shape = Shape<i64, 4>{2, 3, 4, 5};
std::unique_ptr buffer = std::make_unique<f64[]>(shape.n_elements());

// Create a 4d span. Shape::strides() computes the rightmost
// strides of the shape. The C-contiguous order is the default,
// shape.strides() is equivalent to shape.strides<'C'>(), but
// F-contiguous order could also be created, using shape.strides<'F'>().
auto span_4d = Span<f64, 4, i64>(
    buffer.get(), shape.shape(), shape.strides());

for (i64 i{}; i < shape[0]; ++i) {
    // Reduce dimension using multidimensional C-style indexing.
    auto span_3d = span_4d[i].as_contiguous();
    for (i64 j{}; j < shape[1]; ++j)
        for (i64 k{}; k < shape[2]; ++k)
            for (i64 l{}; l < shape[3]; ++l)
                span_3d(j, k, l) = static_cast<f64>(j + k + l);
}

// Various type deductions are provided. For instance:
// Span(buffer.get(), 10) is deduced to Span<f64, 1, i32, StridesTraits::CONTIGUOUS>.
// In other words, if no strides are provided, we assume (at compile time) that
// a contiguous nd-range is passed.
// Of course, these deductions can be turned off by specifying the template parameters:
auto span_1d = Span<f64, 1>(buffer.get(), shape.n_elements());

// There's also a bunch of conversion functions:
auto s0 = span_1d.as<const f64, 1, i64, StridesTraits::CONTIGUOUS>();
// or equivalently:
auto s1 = span_1d.as_const().as_contiguous();

f64 sum{};
for (i64 i{}; i < shape.n_elements(); ++i)
    sum += s1(i);
```

`Span`, as well as `Array` and `View`, has many other utility functions making it easy to manipulate multidimensional data (e.g. `reshape`, `subregion`, `permute`, `filter`, `flat`, etc.). It is also straightforward to get a span from an `Array` or `View`.

```c++
using namespace ::noa::types; // f64, i64, i32, View, Array
using namespace ::noa::indexing; // Ellipsis and Slice

// This is similar to noa::fill<f64>({4, 1, 64, 64}, 3.);
const auto array = Array<f64>({4, 1, 64, 64});
for (f64& e: array.span_1d()) // or equivalently: array.span<f64, 1>()
    e = 3.;

// Extract the 32x32 center.
// This is be equivalent to array[..., 16:48, 16:48] in NumPy.
Array center = array.subregion(Ellipsis{}, Slice{16, 48}, Slice{16, 48});

// Array::span_1d() throws an error here, because "center"
// cannot be reshaped to a 1d view anymore.
// for (auto& e: center.span_1d())
//    e = 2.;

const auto span_3d = center.span().filter(0, 2, 3); // ignores dim 1
// We could have used center.span<f64, 3>() which collapses dimensions
// left to right. Since dim 1 is empty, this is equivalent to filtering
// out dim 1.
for (i64 b{}; i < span_3d.shape()[0]; ++b)
    for (i64 y{}; y < span_3d.shape()[1]; ++y)
        for (i64 x{}; x < span_3d.shape()[2]; ++x)
            span_3d(b, y, x) = 2.;

// We could also use the index-wise function (see the
// next section for more details). This should generate
// the same code as the above example.
noa::iwise(
    center.shape().filter(0, 2, 3), array.device(),
    [](i64 batch, i64 y, i64 x) {
        span_3d(batch, y, x) = 2.;
    });

// But since this example doesn't require indices,
// we could also (and should) use the element-wise core function.
// Something like:
noa::ewise({}, center, [](f64& e) { e = 2.; });

// Which is equivalent to:
noa::fill(center, 2.);
```


## `Accessor`

`Accessor` is similar to `Span` but does not store the shape. It contains the minimum required to iterate through the nd region: the pointer of the memory region and the strides. This type is intended to be used by the backends, and as such, it prioritizes performance above all else. While accessors are a good abstraction for backends, they are not the safest, nor they are the most practical.

```c++
template<typename T, size_t N, typename I,
         PointerTraits PointerTrait = PointerTraits::DEFAULT,
         StridesTraits StridesTrait = StridesTraits::STRIDED>
class Accessor {
    T* pointer;
    Strides<I, N - StridesTrait == StridesTraits::CONTIGUOUS> strides;
};
```

One other difference with `Span` is the `PointerTraits` template parameter. By default, the pointer is not marked with any attributes, but the ”restrict” traits can be added with `PointerTraits::RESTRICT`. This is useful to indicate that pointers don’t alias, which may help to generate better code.
