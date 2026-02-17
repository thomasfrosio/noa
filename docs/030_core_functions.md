# `Compute functions`

The library is designed around pointwise operations (index-wise or element-wise) and exposes generic functions to efficiently compute these operations.


## `noa::iwise`

The index-wise function, `noa::iwise`, provides an interface to nd-dimensional for-loops, where n is between 1 and 4. This function takes a shape, a device and an index-wise operator. The operator will be called once for each nd-index in the nd-shape, _in no particular order_. As one of the core functions of the library, `noa::iwise` actually interacts with the backends and dispatches work to the current-stream of the device. Most cryoEM-specific functions in the library rely on `noa::iwise` and never have to directly interact with the backends. For instance, `noa::signal::bandpass(...)` is creating the `noa::signal::Bandpass<...>` operator and passes it to `noa::iwise`. The template-function declaration looks something like this:
```c++
// See runtime/Iwise.hpp for more details.
namespace noa {
    template<IwiseOptions = {}, typename T, usize N, typename Op, typename... Args>
    void iwise(const Shape<T, N>& shape, const Device& device, Op&& op, Args&&... attachments);
}
```

In its simplest form, the operator needs an `operator()` taking the nd-indices as argument. For instance:

```c++
auto array = Array<f64>(/*...*/);

// This sets all elements of the array to 2.
noa::iwise(
    array.shape(), array.device(),
    [span = array.span()](isize b, isize d, isize h, isize w) {
        span(b, d, h, w) = 2.;
    });

// Equivalently, the indices can be taken as a Vec.
noa::iwise(
    array.shape(), array.device(),
    [span = array.span()](const Vec<isize, 4>& indices) {
        span(indices) = 2.;
    });

// Here Shape::filter(0, 2, 3) returns a new shape, such as
// Shape{shape[0], shape[2], shape[3]} and Shape::as<i32>()
// is here to emphasise that the indices passed to the operator
// have the integer type of the input shape.
noa::iwise(
    array.shape().filter(0, 2, 3).as<i32>(), array.device(),
    [span_3d = array.span<f64, 3, i32>()]
    (i32 batch, i32 k, i32 l) {
        span_3d(batch, k, l) = 2.;
    });
```

On top of the required `operator()(indices...)`, the implementation looks for two member functions, `init()` and `deinit()`. These are defaulted to a no-op so they don't have to be always specified. However, if defined, each thread calls it once when the thread-block starts and ends, respectively. Since operators are per-thread, these functions can be used, for instance, to perform some (de)initialization of the operator.

The compute model we use is based on CUDA, so to traverse a nd-loop `iwise` launches a grid of thread-blocks. On the CPU, each block is made of one OpenMP thread, so the grid is simply the OpenMP threadpool (aka team). `iwise` can also allocate scratch space for each thread-block. This scratch maps to dynamic shared memory in CUDA. To access this scratch, `init`, `deinit` and `operators()` can take an additional first argument mapping to the concept of [`compute_handle`](../src/noa/runtime/core/Interfaces.hpp). This handle is an implementation-specific type that can be used to query various properties about the compute context. Here's full `iwise` interface:
```c++
// 1. Init.
// Defaulted to no-op. If defined, each per-block thread
// calls it once at the begining of the block.
-> op.init(handle) or
   op.init()

// 2. Call.
// Called once per nd-index.
-> op(handle, Vec{indices...}),
   op(handle, indices...),
   op(Vec{indices...}) or
   op(indices...)

// 3. Deinit.
// Defaulted to no-op. If defined, each per-block thread
// calls it once at the end of the block.
-> op.deinit(handle) or
   op.deinit()
```

To show how the compute handle can be used, we can try to implement a convolution between 2d images and a small 2d kernel. A naive implementation with `iwise` could look like this:

```c++
struct MyConvolution {
    // For simplicity, keep the kernel to a square shape of known size.
    static constexpr i32 KERNEL_SIZE = 11;
    static constexpr i32 PADDING = KERNEL_SIZE - 1;
    static constexpr i32 HALO = PADDING / 2;

    Span<f32, 3, i32> input_images; // (n,h,w)
    Span<f32, 3, i32> output_images; // (n,h,w)
    Span<f32, 2, i32> kernel; // (KERNEL_SIZE, KERNEL_SIZE)

    constexpr void operator()(i32 b, i32 i, i32 j) {
        f32 value{};
        for (i32 ki{}; ki < KERNEL_SIZE; ++ki) {
            for (i32 kj{}; kj < KERNEL_SIZE; ++kj) {
                const i32 wi = i - HALO + ki;
                const i32 wj = j - HALO + kj;
                if (noa::is_inbound(input_images.shape().filter(1, 2), wi, wj))
                    value += input_images(b, wi, wj) * kernel(ki, kj);
            }
        }
        output_images(b, i, j) = value;
    }
};

noa::iwise(shape, device, MyConvolution{
    .input_images = /*...input images...*/,
    .output_images = /*...output images...*/,
    .kernel = /*...convolution kernel...*/,
});
```

While this can serve as a CPU and GPU implementation, reading to global memory `KERNEL_SIZE` times per thread isn't very efficient on GPU. One common way to speed up this step is to preload the entire convolution window of the block to shared memory and then let each thread perform its local convolution from the input values saved in that shared memory. This kind of optimization makes little sense for the CPU implementation, so it's best to separate both implementations. Here's what it could look like:

```c++

struct MyConvolution {
    Span<f32, 3, i32> input_images; // (n,h,w)
    Span<f32, 3, i32> output_images; // (n,h,w)
    Span<f32, 2, i32> kernel;

    static constexpr i32 KERNEL_SIZE = 11;
    static constexpr i32 PADDING = KERNEL_SIZE - 1;
    static constexpr i32 HALO = PADDING / 2;

    constexpr void operator()(nt::compute_handle_cpu auto&, i32 b, i32 i, i32 j) {
        /*... same as above ...*/
    }

    static constexpr i32 GPU_BLOCK_SIZE_1D = noa::gpu::Constant::WARP_SIZE / 2;
    static constexpr i32 GPU_SCRATCH_SIZE_1D = GPU_BLOCK_SIZE_1D + PADDING;
    static constexpr i32 GPU_BLOCK_SIZE = GPU_BLOCK_SIZE_1D * GPU_BLOCK_SIZE_1D;
    static constexpr i32 GPU_SCRATCH_SIZE = GPU_SCRATCH_SIZE_1D * GPU_SCRATCH_SIZE_1D;
    
    constexpr void operator()(nt::compute_handle_gpu auto& handle, i32 b, i32 i, i32 j) {
        const auto image_shape = input_images.shape().filter(1, 2);
        const auto block = handle.block();
        const auto scratch = SpanContiguous(
            block.template scratch_pointer<f32>(),
            Shape{GPU_SCRATCH_SIZE_1D, GPU_SCRATCH_SIZE_1D}
        );

        // Write to the scratch the input values that are inside the block's convolution window.
        const auto tid = handle.thread().template id<2>();
        for (i32 ly = tid[0], gy = i; ly < GPU_SCRATCH_SIZE_1D; ly += GPU_BLOCK_SIZE_1D, gy += GPU_BLOCK_SIZE_1D) {
            for (i32 lx = tid[1], gx = j; lx < GPU_SCRATCH_SIZE_1D; lx += GPU_BLOCK_SIZE_1D, gx += GPU_BLOCK_SIZE_1D) {
                const i32 iy = gy - HALO;
                const i32 ix = gx - HALO;

                f32 value{};
                if (noa::is_inbound(image_shape, iy, ix))
                    value = input_images(b, iy, ix);
                scratch(ly, lx) = value;
            }
        }
        block.synchronize();

        // Convolve at location ij.
        if (i < image_shape[0] and j < image_shape[1]) {
            f32 result{};
            for (i32 y = 0; y < KERNEL_SIZE; ++y)
                for (i32 x = 0; x < KERNEL_SIZE; ++x)
                    result += scratch(tid[0] + y, tid[1] + x) * kernel(y, x);
            output_images(b, i, j) = result;
        }
    }
};

// Launch a multiple of the block size and let the operator check the bounds of the images.
auto shape = /*...image bhw shape...*/
if (device.is_gpu()) {
    shape[1] = noa::next_multiple_of(shape[1], MyConvolution::GPU_BLOCK_SIZE_1D);
    shape[2] = noa::next_multiple_of(shape[2], MyConvolution::GPU_BLOCK_SIZE_1D);
}
noa::iwise<{
    // Enforce a block size and allocate per-block scratch.
    .gpu_block_size = MyConvolution::GPU_BLOCK_SIZE,
    .gpu_scratch_size = MyConvolution::GPU_SCRATCH_SIZE,
}>(shape, device, MyConvolution{
    .input_images = /*...input images...*/,
    .output_images = /*...output images...*/,
    .kernel = /*...convolution kernel...*/,
});
```

Note that this is already quite advanced, and while most operators in the library are not nearly as involved, this shows that a lot can be done with a relatively simple interface.

## `noa::ewise`

The element-wise core function, `noa::ewise`, can be seen as a wrapper over `noa::iwise`. This function is useful when the operator does not require the positions of the elements in the array(s), e.g., copy, fill, for arithmetics or other maths-like functions. `noa::ewise` takes the input and output arrays, and an element-wise operator. One advantage of this function over the `noa::iwise` function is that it can analyze the input and output array(s) to deduce the most efficient way to traverse them, by reordering the dimensions, collapsing contiguous dimensions together (up to 1d), and can trigger vectorization by checking for data contiguity and aliasing. For instance, `noa::ewise(input, output, noa::Copy{})` can be optimized up to a memcpy call, and similarly, `noa::ewise(input, output, noa::Fill{0})` an be optimized up to a memset call. On the GPU, [vectorized load/store instructions](031_gpu_vectorization.md) can also be used.

The element-wise interface is defined as the following:
```c++
// Example with zero or one input, and zero or one output.

// The template-function declaration looks something like:
// template<typename Inputs, typename Outputs, typename F>
// void noa::ewise(const Inputs&, const Outputs&, F&&);

// Arguments are passed to the operator as lvalue references,
// the constness is preserved (const outputs are not allowed).
// As such, any of the following would work.
auto a = View<const f32>(/*...*/);
auto b = View<f64>(/*...*/);
noa::ewise(a, b, [](const f32&, f64&) {/*...*/});
noa::ewise(a, b, [](f32, f64&) {/*...*/});
noa::ewise(b, b, [](f64&, f64&) {/*...*/});

// No inputs or outputs are allowed:
noa::ewise(a, {}, [](f32) {/*...*/});
noa::ewise({}, b, [](f64&) {/*...*/});

// To pass multiple arguments, noa::wrap and/or noa::fuse should be used.
// These are simple types that behave (almost) as std::forward_as_tuple and
// that can be understood by the ewise function.

// noa::wrap encodes that the arguments should be passed as is, in the same order as specified.
// noa::fuse encodes that the arguments should be fused into a Tuple. As mentionned above,
// the arguments are passed as lvalue references. The Tuple itself is also passed as a
// lvalue reference.
auto a = View<const f32>(/*...*/);
auto b = View<const i32>(/*...*/);
auto c = View<f64>(/*...*/);
auto d = View<i64>(/*...*/);
noa::ewise(noa::wrap(a, b), noa::wrap(c, d), [](const f32&, const i32&, f64&, i64&) {/*...*/});
noa::ewise(noa::wrap(a, b), noa::wrap(c, d), [](f32, i32, f64&, i64&) {/*...*/}); // inputs by value
noa::ewise(noa::fuse(a, b), noa::fuse(c, d),
    [](const Tuple<const f32&, const i32&>&, const Tuple<f64&, i64&>&) {/*...*/});

// Fuse is mostly useful when the operator is meant to handle a variable number of inputs:
struct ReducePlus {
    // If inputs is a parameter pack, output cannot be specified due to how C++ works...
    // constexpr void operator()(auto... inputs, f32 output) {}
    // So to circumvent this:
    template<typename... T>
    constexpr void operator()(const Tuple<const T&...>& inputs, f64& output) { // ok
        // C++26 will remove the need for this lambda!
        output = [&]<size_t... I>(std::index_sequence<I...>) {
            return (static_cast<f64>(inputs[Tag<I>{}]) + ...);
        }(std::make_index_sequence<sizeof...(T)>{});
    }
};
noa::ewise(noa::fuse(a, b, d), c, ReducePlus{});
```

Here's a _slightly_ more real example:
```c++
using namespace ::noa::types; // to imports i32, f32, Tuple and Array

const auto buffer = noa::arange<f32>(300).reshape({3,1,10,10});
const auto lhs = buffer.view().subregion(0); // shape=(1,1,10,10)
const auto mhs = buffer.view().subregion(1); // same shape
const auto rhs = buffer.view().subregion(2); // same shape

const auto output_0 = Array<f32>({1, 1, 10, 10});
const auto output_1 = noa::like(output_1);

// All inputs and outputs should be on the same device and have compatible shapes.
// Also note that Array and View can be used interchangeably, as always.
noa::ewise(
    noa::wrap(lhs, mhs, rhs), noa::fuse(output_0, output_1),
    [](i32 l, i32 m, i32 r, const Tuple<f32&, f32&>& outputs) {
        auto& [o0, o1] = outputs;
        o0 = l + m * 2;
        o1 = std::cos(m / l + r);
    });

// The (sort of) equivalent in NumPy would be:
// output_0 = lhs + mhs * 2
// output_1 = np.cos(mhs / lhs + rhs)
```


## `noa::reduce_iwise, noa::reduce_axes_iwise, noa::reduce_ewise, noa::reduce_axes_ewise`

Four pointwise-reduction functions are available. These aim to provide whole-array reductions or per-axis reductions. While these are mostly intended for total reductions (reduction to one value per axis), partial reductions like histograms are also possible. Reductions can element-wise or index-wise interface, with the same advantages and disadvantages as the `noa::ewise` and `noa::iwise` template-functions. For brevity, we are only discussing the `reduce_axes_iwise` function since it's the most powerful one, but note that 1) iwise reductions (whole-array or per-axis) share the same operator interface, and similarly for the ewise reductions, and 2) all reductions functions have an almost identical API. Here's the actual code for every reduction: [`reduce_iwise`](../src/noa/runtime/ReduceIwise.hpp), [`reduce_axes_iwise`](../src/noa/runtime/ReduceAxesIwise.hpp), [`reduce_ewise`](../src/noa/runtime/ReduceEwise.hpp) and [`reduce_axes_ewise`](../src/noa/runtime/ReduceAxesEwise.hpp)).

Regarding `reduce_axes_iwise` specifically, the declaration looks something like this:

```c++
namespace noa {
    template<
        ReduceIwiseOptions OPTIONS = ReduceIwiseOptions{},
        typename Index, usize N,
        typename Reduced = nd::AdaptorUnzip<>,
        typename Outputs = nd::AdaptorUnzip<>,
        typename Operator, typename... Ts
    >
    void reduce_axes_iwise(
        const Shape<Index, N>& shape,
        Device device,
        Reduced&& reduced,
        Outputs&& outputs,
        Operator&& op,
        Ts&&... attachments
    )
}
```
And here's the full `reduce_(axes_)iwise`operator interface (for more details, look at the code documentation):

```c++
// Init.
// Defaulted to no-op.
// If defined, each thread calls it when the reduction (of each axis) starts.
->  op.init(handle, Vec{output-indices...}),
    op.init(handle, output-indices...),
    op.init(handle),
    op.init(Vec{output-indices...}),
    op.init(output-indices...) or
    op.init()

// Call.
// Main reduction step, called once per nd-index.
->  op(handle, Vec{input-indices...}, reduced&...),
    op(handle, input-indices..., reduced&...),
    op(Vec{input-indices...}, reduced&...) or
    op(input-indices..., reduced&...)

// Deinit.
// Defaulted to no-op.
// If defined, each thread calls it when its reduction (of each axis) ends.
->  op.deinit(handle, Vec{output-indices...}),
    op.deinit(handle, output-indices...),
    op.deinit(handle),
    op.deinit(Vec{output-indices...}),
    op.deinit(output-indices...) or
    op.deinit()

// Join.
// To join the reduced values together.
// This is necessary for multithreaded/GPU reductions, but may not always be called.
->  op.join(const reduced&..., reduced&...)

// Post.
// Defaulted to copy.
// If defined, it is called once per reduced axis.
->  op.post(const reduced&..., outputs&..., Vec{output-indices...}),
    op.post(const reduced&..., outputs&..., output-indices...) or
    op.post(const reduced&..., outputs&...)
```

Here's what a simple sum-reduction could look like:
```c++
auto src = Array<f64>(/*...input...*/);

f64 sum{};
struct MySum {
    Span<f64, 4> src;
    constexpr void operator()(const Vec<isize, 4>& indices, f64& sum) {
        sum += src(indices);
    }
    constexpr void join(f64 isum, f64& sum) {
        sum += isum;
    }
};
noa::reduce_iwise(src.shape(), array.device(), f64{}, sum, MySum{
    .src = src.span()
});

// Note that the join function can fall back to the operator(),
// for ewise reductions, so the above can be simplified to:
noa::reduce_ewise(src, f64{}, sum, [](auto value, auto& sum) { sum += value; });

// Here's a more complex example.
const Array<f32> array; // an array we want to reduce
const Array<i32> mask; // a binary mask
f64 sum_inside_mask; // the wanted sum inside the mask...
f32 max_inside_mask; // and the maximum value inside the mask

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

We can also implement partial reductions using one important feature of the reduction API: `reduced` and `output` can be ommited, leaving you (almost) full control on how the reduction should be implemented. Combined with the `compute_handle`, here's what an efficient batched implementation of a histogram could look like, for both CPU and GPU.

```c++
struct Histogram {
    Span<const f32, 2, i32> inputs; // (n,w)
    Span<i32, 2, i32> histograms; // (n,b)

    static constexpr void init(nt::compute_handle auto& handle) {
        // Zero-initialize the per-block histogram if it exists.
        const auto& block = handle.block();
        block.template zeroed_scratch<i32>();
        block.synchronize();
    }

    constexpr void operator()(nt::compute_handle auto& handle, i32 b, i32 i) const {
        // Compute the bin of the current value.
        // For simplicity, assume values are between [0,1].
        const auto n_bins = histograms.shape()[1];
        const auto value_scaled = inputs(b, i) * static_cast<f32>(n_bins);
        auto bin = static_cast<i32>(noa::round(value_scaled));
        bin = noa::clamp(bin, 0, n_bins - 1);

        // Increment the bin count.
        // If the block has its own histogram, increment it
        // instead of incrementing the global histogram.
        const auto& grid = handle.grid();
        const auto& block = handle.block();
        if (block.has_scratch()) {
            auto scratch = block.template scratch<i32>();
            grid.atomic_add(1, scratch, bin);
        } else {
            grid.atomic_add(1, histograms[b], bin);
        }
    }

    constexpr void deinit(nt::compute_handle auto& handle, i32 b) const {
        const auto& block = handle.block();
        const auto& thread = handle.thread();
        if (not block.has_scratch())
            return;

        // If the block has its own histogram, add it to the global histogram.
        block.synchronize();
        const auto& grid = handle.grid();
        auto scratch = block.template scratch<i32>();
        for (i32 i = thread.lid(); i < scratch.n_elements(); i += block.size())
            grid.atomic_add(scratch[i], histograms, b, i);
    }
};

constexpr isize HISTOGRAM_SIZE = 128;
auto inputs = Array<f32>(/*...B*WIDTH array*/);
auto histograms = Array<f32>(/*...B*HISTOGRAM_SIZE array*/);

constexpr auto OPTIONS = noa::ReduceIwiseOptions{
    .gpu_block_shape = {1, HISTOGRAM_SIZE * 4}, // 1d block
    .gpu_optimize_block_shape = false, // enforce the block shape
    .gpu_number_of_indices_per_threads = {1, 4}, // increase the value of the per-block histogram by working on it more
    .gpu_scratch_size = HISTOGRAM_SIZE * sizeof(i32), // per block histogram
};
noa::reduce_axes_iwise<OPTIONS>(shape, inputs_gpu.device(), {}, reduce_width, Histogram{
    .inputs = inputs.span<const f32, 2, i32>(),
    .histograms = histograms.span<i32, 2, i32>(),
});
```
