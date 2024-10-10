## `Accessor and Span`

`Array` and `View` provide an operator() taking 4d-indices, which can be used to access
the underlying values of the nd-arrays. However, this is not the preferred way to access
data from these arrays. First, this may not be the most efficient way to access the data, for
instance, 1d contiguous arrays can be accessed using a single index and uses a fixed stride
of 1 (known at compile time). Second, depending on the device and memory resource,
direct access may not be even allowed.

`Accessor` is a core-type used to abstract multidimensional element-accesses into a common
interface. This type is intended to be used by the backends, and as such, it prioritizes
performance. An accessor contains the strict necessity: the raw-pointer of the memory
region and the strides.

```c++
template<typename T, size_t N, typename I,
         PointerTraits PointerTrait = PointerTraits::DEFAULT,
         StridesTraits StridesTrait = StridesTraits::STRIDED>
class Accessor {
    T* pointer;
    Strides<I, N - StridesTrait == StridesTraits::CONTIGUOUS> strides;
};
```

As opposed to `Array` and `View`, the number of dimensions (`N`) is in the type definition, as
well as the integer type of the strides (`I`). Two additional template parameters are also available:
`PointerTraits` and `StridesTraits`. By default, the pointer is not marked with
any attributes, but the ”restrict” traits can be added with PointerTraits::RESTRICT.
This is useful to indicate that pointers don’t alias, which may help to generate better
code. By default, strides are fully dynamic (one dynamic stride per dimension). However,
the rightmost dimension can be marked contiguous using `StridesTraits::CONTIGUOUS`.
Accessors (and the library internals) use the rightmost convention, so that the innermost
dimension is the rightmost dimension. As such, contiguity here implies C-contiguous.
F-contiguous layouts are not supported by the accessors, as these layouts are expected
to be already reordered to C-contiguous before creating the contiguous accessor. With
`StridesTraits::CONTIGUOUS`, the innermost/rightmost stride is fixed to 1 and is not
stored, resulting in the strides being truncated by one dimension (`Strides<I,N-1>`).
In case of a 1d contiguous accessor, this means that the strides are empty (`Strides<I,0>`)
and the indexing is equivalent to pointer/array indexing.

```c++
// Import i64, f64, Shape and Accessor
using namespace ::noa::types;

auto shape = Shape<i64, 4>{2, 3, 4, 5};
std::unique_ptr buffer = std::make_unique<f64[]>(shape.n_elements());

// Create a 4d accessor. Shape::strides() computes the rightmost
// strides of the shape. The C-contiguous order is the default,
// shape.strides() is equivalent to shape.strides<'C'>(), but
// F-contiguous order could also be created, using shape.strides<'F'>().
auto accesor_4d = Accessor<f64, 4, i64>(buffer.get(), shape.strides());

for (i64 i{0}; i < shape[0]; ++i) {
    // Reduce dimension using multidimensional C-style indexing.
    // Note that accessor_3d is an AccessorReference, which
    // simply points to an existing Accessor.
    auto accessor_3d = accessor_4d[i]; 
    for (i64 j{0}; j < shape[1]; ++j)
        for (i64 k{0}; k < shape[2]; ++k)
            for (i64 l{0}; l < shape[3]; ++l)
                accessor_3d(j, k, l) = static_cast<f64>(j + k + l);
}

// Various type definitions are provided to simplify code.
// AccessorContiguousI64 defaults to I=i64 and StridesTrait=
// StridesTraits::CONTIGUOUS. Note here that the stride is
// not needed in the constructor, because the accessor is
// 1d and strides are contiguous, i.e. the strides are the
// Strides<I,0> empty type.
auto accessor_1d_contiguous = AccessorContiguousI64<const f64, 1>(buffer.get());

f64 sum{};
for (i64 i{}; i < shape.n_elements(); ++i) {
    sum += accessor_1d[i];
    // or equivalently accessor_1d(i) or buffer[i];
}
```

While accessors are a good abstraction for backends, they are not the safest, nor they are
the most practical. This mostly results from the fact that the size of each dimension is not
stored, so accessors cannot bound-check the indices against their dimension size. Sizes are
indeed not stored because they are not necessary to compute the memory offsets, and in a
lot of cases, the input and output arrays have the same shape, which would lead to storing
useless or redundant data in the operators.

In user code, (md)spans should be used instead. These are similar to `Accessor`, but keep
track of the size of each dimension. The library comes with its own Span, which is similar to
C++23 `std::mdspan`.

```c++
template<typename T, size_t N = 1, typename I = i64,
         StridesTraits StridesTrait = StridesTraits::STRIDED>
class Span {
    T* pointer;
    Shape<I, N> shape;
    Strides<I, N - StridesTrait == StridesTraits::CONTIGUOUS> strides;
};
```

```c++
using namespace ::noa::types; // f64, i64, i32, View, Array
using namespace ::noa::indexing; // Ellipsis and Slice

// This is similar to noa::filled<f64>({4, 1, 64, 64}, 3.);
const Array array = noa::empty<f64>({4, 1, 64, 64});
for (Span<f64, 1>& e: array.span_1d()) // or equivalently: array.span<f64, 1>()
    e = 3.;

// Extract the 32x32 centre.
// This is be equivalent to array[..., 16:48, 16:48] in NumPy.
Array centre = array.subregion(Ellipsis{}, Slice{16, 48}, Slice{16, 48});

// Array::span_1d() throws an error here, because "centre"
// cannot be reshaped to a 1d array anymore.
// for (auto& e: centre.span_1d())
//    e = 2.;

const auto span_3d = centre.span().filter(0, 2, 3); // ignores dim 1
for (i64 b{0}; i < span_3d.shape()[0]; ++b)
    for (i64 y{0}; y < span_3d.shape()[1]; ++y)
        for (i64 x{0}; x < span_3d.shape()[2]; ++x)
            span_3d(b, y, x) = 2.;

// We could also use the index-wise function (see the
// next section for more details). This should generate
// the same code as the above example.
noa::iwise(
    centre.shape().filter(0, 2, 3), array.device(),
    [span_3d = centre.span<f64, 3>()] // collapses dimensions left to right
    (i64 batch, i64 y, i64 x) {
        span_3d(batch, y, x) = 2.;
    });

// But since this example doesn't require indices,
// we could also (and should) use the element-wise core function.
// Something like:
noa::ewise({}, centre, [](f64& e) { e = 2.; });
// Which is equivalent to:
noa::fill(centre, 2.);

```
