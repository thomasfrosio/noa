# `Core functions`

The library heavily relies on pointwise operations and provides core functions for generic algorithms.

## `noa::iwise`

The index-wise core function, `noa::iwise`, provides a interface to nd-dimensional
for-loops, where n is between 1 and 4. This function takes a shape, a device and an index-wise
operator. The operator will be called once for each nd-indices in the nd-shape, in
no particular order. As one of the core functions, `noa::iwise` actually interacts with the
backends and dispatches work to the current-stream. Most cryoEM-specific functions in
the library rely on `noa::iwise` and never have to directly interact with the backends. For
instance, `noa::signal::bandpass(...)` is creating the `noa::signal::Bandpass<...>`
pointwise operator and passes it to the `noa::iwise`.

```c++
// The template-function declaration looks something like:
// template<typename T, size_t N, typename F, typename... A>
// void noa::iwise(
//     const noa::Shape<T, N>& shape,
//     const noa::Device& device,
//     F&& iwise_operator,
//     A&&... attachments
// );

// Iwise operators should either:
// - accept the nd-indices as a packed noa::Vec<T, N>. See (1).
// - or accept the indices one by one. See (2).
noa::iwise(
    array.shape(), array.device(),
    [span = array.span()](const Vec<i64, 4>& indices) { // (1)
        span(indices) = 2.;
    });

// Here's a more complex example:
// Shape::filter(0, 2, 3) returns a new shape, such as
// Shape<i64, 3>{shape[0], shape[2], shape[3]}.
// Shape::as<i32>() is here to emphasise that the
// indices have the integer type of the input shape.
noa::iwise(
    array.shape().filter(0, 2, 3).as<i32>(), array.device(),
    [span_3d = array.span<f64, 3, i32>()]
    (i32 batch, i32 k, i32 l) { // (2)
        span_3d(batch, k, l) = 2.;
    });
```


## `noa::ewise`

The element-wise core function, `noa::ewise`, can be seen as a wrapper over the index-wise
core function. This function is useful when the position of the element in the array(s)
is not required, e.g. copy, fill, for arithmetics or other maths functions like `sin`. This
takes the input and output arrays, and an element-wise operator. One advantage of this
function over the `noa::iwise` function is that it can analyze the input and output array(s)
to deduce the most efficient way to traverse them, by reordering the dimensions, collapsing
contiguous dimensions together (up to 1d), and can trigger vectorization by checking for
data contiguity and aliasing. For instance, `noa::ewise(input, output, noa::Copy{})`
can be optimized up to a memcpy CPU instruction, and similarly, `noa::ewise(input, output, noa::Fill{0})`
can be optimized up to a memset CPU instruction. On the GPU, vectorized load/store instructions can also be [used](031_gpu_vectorization.md).

The element-wise interface is defined as the following:
```c++
// Zero/One input - Zero/One output.

// The template-function declaration looks like:
// template<typename Inputs, typename Outputs, typename F>
// void noa::ewise(const Inputs&, const Outputs&, F&&);

// Arguments are passed to the operator as lvalue references,
// the constness is preserved (const outputs are not allowed).
// As such, any of the following would work.
View<const f32> a;
View<f64> b;
noa::ewise(a, b, [](const f32&, f64&) {});
noa::ewise(a, b, [](f32, f64&) {});
noa::ewise(b, b, [](f64&, f64&) {});

// Empty inputs or outputs are allowed:
noa::ewise(a, {}, [](f32) {});
noa::ewise({}, b, [](f64&) {});
```

```c++
// Multiple inputs and outputs.

// To pass multiple arguments, noa::wrap and/or noa::fuse should be used.
// These are simple types that behave (almost) as std::forward_as_tuple and
// that can be understood by the ewise function.

// noa::wrap says that the arguments should be passed as is, in the same order as specified.
// noa::fuse says that the arguments should be fused in a Tuple. As mentionned above,
// the arguments are passed as lvalue references. The Tuple itself is also passed as a
// lvalue reference.

View<const f32> a;
View<const i32> b;
View<f64> c;
View<i64> d;
noa::ewise(noa::wrap(a, b), noa::wrap(c, d), [](const f32&, const i32&, f64&, i64&) {});
noa::ewise(noa::wrap(a, b), noa::wrap(c, d), [](f32, i32, f64&, i64&) {}); // inputs by value
noa::ewise(noa::fuse(a, b), noa::fuse(c, d),
    [](const Tuple<const f32&, const i32&>&, const Tuple<f64&, i64&>&) {});

// Fuse is mostly useful when the operator is meant to handle a variable number of inputs:
struct EwiseSum {
    // If inputs is a parameter pack, output cannot be specified due to how C++ works...
    // constexpr void operator()(auto... inputs, f32 output) {}

    template<typename... T>
    constexpr void operator()(const Tuple<const T&...>& inputs, f64& output) { // ok
        output = [&]<size_t... I>(std::index_sequence<I...>) {
            return (static_cast<f64>(inputs[Tag<I>{}]) + ...);
        }(std::make_index_sequence<sizeof...(T)>{});
    }
};
noa::ewise(noa::fuse(a, b, d), c, EwiseSum{});
```

Here's a slightly more real example:
```c++
using namespace ::noa::types; // to imports i32, f32 and Tuple

const auto buffer = noa::arange<f32>(300).reshape({3,1,10,10});
const auto lhs = buffer.view().subregion(0); // shape=(1,1,10,10)
const auto mhs = buffer.view().subregion(1); // same shape
const auto rhs = buffer.view().subregion(2); // same shape

const auto output_0 = noa::empty<f32>({1, 1, 10, 10});
const auto output_1 = noa::like(output_1);

// All inputs and outputs should be on the same device and have compatible shapes.
// Also note that Array and View can be used interchangeably, as always.
noa::ewise(
    noa::wrap(lhs, mhs, rhs), noa::fuse(output_0, output_1),
    [](i32 l, i32 m, i32 r, const Tuple<f32&, f32&>& outputs) {
        auto& [o0, o1] = outputs;
        o0 = l + m * 2;
        o1 = noa::cos(m / l + r);
    });

// The (sort of) equivalent in NumPy would be:
// output_0 = lhs + mhs * 2
// output_1 = np.cos(mhs / lhs + rhs)
```


## `noa::reduce_iwise, noa::reduce_ewise`

The pointwise-reduction core functions are the final core functions that backends need to
implement. These aim to provide whole-array reductions or per-axis reductions. Each
reduction provides an element-wise or index-wise interface, with the same advantages and
disadvantages as the `noa::ewise` and `noa::iwise` functions. For brevity, the example
below presents the whole-array reduction API, but note that the per-axis reductions have a
similar API.

```c++
// The template-function declarations look like:
// template<typename Inputs, typename Reduced,
//          typename Outputs, typename F>
// void noa::reduce_ewise(
//     const Inputs&, const Reduced&,
//     const Outputs&, F&&);
//
// template<typename T, size_t N, typename Reduced,
//          typename Outputs, typename F>
// void noa::reduce_iwise(
//     const noa::Shape<T, N>&, const Device&,
//     const Reduced&, const Outputs&, F&&);

// Reduction operators should define (at most) 3 member functions:
// .init(inputs..., reduced...)
//  -> Main reduction step, to convert the input types to the
//     reduced types. The iwise reduction has init(Vec<T, N>,
//     reduced...) or init(indices..., reduced...). If not
//     defined, this defaults to:
//     (*this)(inputs..., reduced...);
// .join(reduced..., reduced...)
//  -> To join the reduced values together. This is necessary
//     for multithreaded/GPU reductions. If not defined, this
//     defaults to (*this)(reduced..., reduced...);
// .final(reduced..., outputs...)
//  -> Called as a postprocessing step to convert the
//     reduced types to the output types. If not defined,
//     this defaults to:
//     ((outputs = static_cast<Outputs>(reduced)), ...);

// These defaults greatly simplify the operators.
// For instance, computing the sum of an array can be done
// as simply as:
Array<f64> src;
f64 sum{};
noa::reduce_ewise(src, f64{}, sum, [](auto value, auto& sum) { sum += value; });

// Here's a more complex example.
const Array<f32> array; // an array we want to reduce
const Array<i32> mask; // a binary mask
f64 sum_inside_mask; // the wanted sum inside the mask...
f32 max_inside_mask; // and the maximum value inside the mask

// Using the element-wise approach:
struct ReduceMaskEwise {
    constexpr static void init(f32 value, i32 mask, f64& sum, f32& max) {
        if (mask > 0) {
            sum += static_cast<f64>(value);
            max = std::max(max, value);
        }
    }
    constexpr static void join(f64 isum, f32 imax, f64& osum, f32& omax) {
        osum += isum;
        omax = std::max(omax, imax);
    }
    // default .final()
};
noa::reduce_ewise(
    noa::wrap(array, mask),
    noa::wrap(f64{}, std::numeric_limits<f32>::lowest()),
    noa::wrap(sum_inside_mask, max_inside_mask),
    ReduceMaskEwise{});

// Or using the index-wise API:
struct ReduceMaskIwise {
    Span<const f32, 4> array;
    Span<const i32, 4> mask;

    constexpr void init(const Vec4<i32>& indices, f64& sum, f32& max) {
        if (mask(indices) > 0) {
            const auto& value = array(indices);
            sum += static_cast<f64>(value);
            max = std::max(max, value);
        }
    }
    // Same as ReduceMaskEwise.
    constexpr static void join(f64 isum, f32 imax, f64& osum, f32& omax) {
        osum += isum;
        omax = std::max(omax, imax);
    }
    // default .final()
};
noa::reduce_iwise(
    array.shape(), array.device(),
    noa::wrap(f64{}, std::numeric_limits<f32>::lowest()),
    noa::wrap(sum_inside_mask, max_inside_mask),
    ReduceMaskIwise{array.span(), mask.span()});
```
