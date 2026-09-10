## `Strides and memory layout`

Each dimension of our nd-arrays is encoded by a runtime stride. The number of dimensions, however, is a compile time value and is part of the type, e.g., `Array<float, 3>`. Note that by default the stride type is `std::ssize_t`, but negative strides are not tested and will likely break things.

When allocating new arrays, the arrays are rightmost ordered, meaning that the innermost stride is on the right, and strides increase right-to-left. This is similar to what Numpy uses by default, expect that Numpy encodes strides in number of bytes, whereas we encode them in number of elements.

By default, the rightmost order is used. In the B..DHW convention, the rightmost order is synonym to row-major (the width/row is the fastest dimension to iterate through) and a rightmost contiguous array is said to be C-contiguous. For instance, a C-contiguous 4d array of shape `{1,2,3,4}` has its strides equal to `{24,12,4,1}`. Following the same convention, we define an F-contiguous array, aka column-major, of the same shape with strides equal to `{24,12,1,4}`. When referring to C- or F-ordering, it is important to note that this only affects the `height/column` and `width/row` of the array. This is different from what Numpy does, as it refers to F-contiguous as a synonym of leftmost order.

```c++
auto a = noa::Array<f32, 4>({1, 3, 4, 5});
a.shape(); // (1, 3, 4, 5)
a.strides(); // (60, 20, 5, 1)
assert(a.is_contiguous()); // equivalent to is_contiguous<'C'>();

noa::Array<f32> b = a.permute({0, 1, 3, 2}); // swap height and width
b.shape(); // (1, 3, 5, 4)
b.strides(); // (60, 20, 1, 5)
assert(b.is_contiguous<'C'>() == false);
assert(b.is_contiguous<'F'>() == true);
assert(b.copy().is_contiguous<'C'>() == true); // copy defaults to rightmost

// While .permute().copy() is possible, .permute_copy() offers a
// more efficient permutation to a new C-contiguous array.
noa::Array<f32> c = a.permute_copy({0, 1, 3, 2});
assert(c.is_contiguous<'C'>() == true); // copy defaults to rightmost
assert(c.is_contiguous<'F'>() == false);
```


## `Correctness and performance`

The library tries to make as few assumptions on the memory layout as possible. However, certain functions have some requirements regarding the layout of the input and output arrays. If these requirements are not met, a runtime error (or sometimes a compile-time error) will be thrown.

To ensure good performance, the library tries (whenever possible) to find the fastest way to iterate through arrays by looking at the shape and strides of the inputs and/or outputs. For instance, copying (which relies on the [`noa::ewise`](030_core_functions.md) core function) C or F arrays results in the same performance. In this case, F-arrays are automatically permuted to the rightmost order before calling the compute backend (which all expects rightmost layouts).

However, doing so is not always possible. For instance, [`noa::iwise`](030_core_functions.md) does not take arrays as inputs, thus cannot reorder dimensions. In these cases, the rightmost order is always assumed. Moreover, when input and/or output arrays have different layouts, it is not always easy or possible to find the overall best order, so we do recommend to keep with the default layout and use rightmost arrays whenever possible.

For instance, when importing column-major nd-data, we recommend to immediately reorder the data to row-major to prevent layout hiccups down the line.

```c++
// Eigen matrix, F-major by default.
int rows = 10, cols = 20;
auto eigen_matrix = Eigen::MatrixXd(rows, cols);

const auto pointer = eigen_matrix.data();
const auto outer_stride = eigen_matrix.outerStride();  // steps between columns
const auto inner_stride = eigen_matrix.innerStride();  // steps between rows, =1

// View the Eigen matrix as a noa::Array.
auto column_major_matrix = noa::Array(
    pointer,
    Shape{rows, columns}.as<isize>(),
    Strides{inner_stride, outer_stride}.as<isize>()
);

// When wrapping a raw pointer like above, the array is a view, it does not own the data.
// To import an owned array, without creating a copy, a std::shared_ptr must be passed.
static_assert(std::same_as<decltype(column_major_matrix), Array<double, 2, ArrayOwnership::VIEW>);

// Permute axes (no copy, still a view of the original Eigen data):
auto row_major_matrix = column_major_matrix.permute(1, 0);
// or equivalently: auto row_major_matrix = column_major_matrix.permute_to_rightmost();

assert(
    row_major_matrix.shape()[0] == cols and
    row_major_matrix.shape()[1] == rows and
    row_major_matrix.strides()[0] == outer_stride and
    row_major_matrix.strides()[1] == inner_stride
);
```


## `Broadcasting`

Empty dimensions (dimensions with a size of 1) can be broadcasted to any size. When broadcasting is always safe and correct, the library will automatically try to broadcast the input array(s) onto the output shape. If the shapes aren't compatible even after broadcasting, an error is thrown. Arrays can also be explicitly broadcasted using `noa::broadcast(array, desired_shape) -> array`.

Broadcasting is implemented by setting the stride to 0, effectively saying that the same element is repeated along that dimension. Broadcasting read-only arrays is always correct but some functions may reject broadcasted arrays because they break contiguity.

However, **broadcasting arrays can generate a data-race if values are written along the broadcast dimension**. This is because with a stride of 0, multiple indices can now refer to the same memory location. Except when explicitly documented otherwise, **there are no guarantees on the order of execution in element-wise or index-wise operations**. The library tries to check for this scenario when possible and may throw an error to preserve correctness.

```c++
noa::Array a = noa::empty<f32, 4>({1, 3, 4, 5});
noa::Array b = noa::broadcast(a, {10, 3, 4, 5});
b.shape(); // (10, 3, 10, 5)
b.strides(); // (0, 20, 5, 1)
assert(not b.is_contiguous()); // broadcasting breaks contiguity
assert(b.contiguity() == Vec{false, true, true, true});
```


## `Rank`

The library was designed using the

The library often names axes using the `Batch`-`Depth`-`Height`-`Width` (`BDHW` or `B..DHW`) order. 

Some functions need to assign logical meaning to axes. For instance, computing an FFT on an array with 3 dimensions is ambiguous: should we compute the transform only on the width and treat the last two dimensions as batch axes (1D arrays, aka `BBW`), or should it be on the height and width (2d arrays, aka `BHW`), or on all axes (a single 3d array, aka `DHW`)?

We refer to this problem as ranking the array, i.e., assigning the rank of an array. There are 4 possible scenarios:
- Some functions don't care about the rank, e.g., copy, permute, or any of the core functions.
- Some functions only support one rank, e.g., `noa::xform::transform_2d` or `noa::fft::remap_2d`. These are often inline functions of more generic functions (.e.g., `noa::fft::remap`).
- Some functions can deduce the rank of the arrays at compile time from other parameters. For instance, `noa::signal::phase_shift(input, output, shift)` takes the shifts as `Vec<T,R>`. If the shift is `Vec<T,2>`, the array rank is naturally `2`. Note that for clarity the overload `noa::signal::phase_shift_2d` also exists and only accepts `Vec<T,2>` shifts. 
- Some functions cannot deduce the rank at compile time and the ranking happens at runtime. In this case, the rank should preferably be specified by the user, e.g., `noa::fft::remap(remap, input, output, {.rank = 2})`. Otherwise, it is automatically deduced using the `B..DHW` order. This deduction is done by `Shape::rank_checked`: if the array as one, four or more axes, we use the `B..DHW` order (an array of shape `{10, 1, 64, 64}` has a rank 2, as opposed to `{1, 10, 64, 64}` which has a rank 3). Arrays with only 2 or 3 axes are ambiguous and if the rank is not specified directly an error is thrown. Relying on this automatic deduction is discouraged and clearer functions are usually available (`noa::fft::remap` vs `noa::fft::remap_2d` vs `noa::fft::remap_3d`).


## `Data-types and number of axes`

We currently limit the number of dimensions to 6. This number could be increase in the future, but this seems unnecessary. When compared to numpy, this limit may seem too low, but keep in mind that we support more complex data-types compared to NumPy (or PyTorch). For instance, to represent an array of 4-by-4 matrices in NumPy, one would need a floating-point array with at least 3 dimensions (e.g., shape=`[n,4,4]`). In our case, the 4-by-4 matrix is the data-type (e.g., `noa::Mat<f64, 4, 4>`) and the array has one dimension (e.g., shape=`[n]`).
