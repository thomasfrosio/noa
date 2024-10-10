## `Shape and strides`

The library uses shapes and strides to represent multidimensional array up to 4 dimensions. The dimension order is
always entered following the BDHW convention, where B:batches, D:depth/pages, H:height/rows and W:width/columns. In most
cases, the memory layout, encoded in the strides, is not fixed by the library so users might use any convention they
prefer, e.g. row-major or column-major, etc. However, some functions do expect a certain layout, e.g. C-contiguous, 
which will be documented in the function documentation.

__Rightmost order__: Rightmost ordering is when the innermost stride is on the right, and strides increase
right-to-left. This is similar to what Numpy uses by default, expect that Numpy encodes strides in number of bytes,
where we encode everything in number of elements. In the BDHW convention, the rightmost order is synonym to row-major
and a rightmost contiguous array is said to be C-contiguous. For instance, a C-contiguous 3D array of
shape `{1,30,64,128}` has its strides equal to `{245760,8192,128,1}`. Following the BDHW convention, a F-contiguous
array of the same shape has its strides equal to `{245760,8192,1,64}` and is therefore not in the rightmost order. When
referring to C- or F-ordering, it is important to note that this often only affects the Height and Width of the array,
so the last two dimensions in the BDHW convention. Similarly, for indexing, dimensions are entered left-to-right, like
in Numpy.

Empty dimensions are set to 1. A shape with a dimension of size 0 is referring to an empty array. Strides, which
contains the memory offset between consecutive indexes for each dimension, can be 0. In fact, broadcasting an array
along a particular dimension is achieved by setting a stride of 0 for that said dimension. Note that the library 
currently doesn't support negative strides.

As mentioned above, when the order of the dimensions has a meaning (e.g. we expect a 2D array), we always use the BDHW
convention. This results in the ability to differentiate between a stack of 2D images and a 3D volume or a stack of 3D
volumes. For instance, `{64,1,32,32}` is a set of 64 arrays each with 32x32 elements. On the other hand, `{1,64,32, 32}`
is a 3D array with 64x32x32 elements. This is quite important since some functions behave differently with 2D or 3D
array(s). For example, `geometry::transform2D` expects (batched) 2D arrays and will throw an error if a 3D array is
passed. Similarly, the shape `{1,1,1,5}` describes a row vector, while `{1,1,5,1}` describes a column vector.

For non-redundant Fourier transforms, the non-redundant dimension is always the width.

__Performance__: The library tries to find the fastest way to iterate through arrays by looking at the shape and strides
of the inputs and/or outputs. However, it is not always possible. In these cases, the rightmost order is always assumed.
As such, whenever possible, it is often much simpler to use rightmost arrays.

## `Broadcasting`

Empty dimensions (dimensions with a size of 1) can be broadcast to any size. When broadcasting is supported, the
library will automatically try to broadcast the input array(s) onto the output shape. If the shapes aren't compatible,
even after broadcasting, an error is thrown. Arrays can also be explicitly broadcast using `indexing::broadcast()`.

Broadcasting is implemented by setting the stride to 0, effectively saying that the same element is repeated along 
that dimension. Broadcasting arrays used in read-only contexts is always valid but not always allowed. When 
broadcasting isn't allowed, an error is thrown if the arrays' shapes aren't compatible. However, **broadcasting arrays
can generate a data-race if values are written along the broadcast dimension**. This is because with a stride of 0,
multiple indexes can now refer to the same memory location. Except when explicitly documented otherwise,
**there are no guarantees on the order of execution in element-wise or index-wise operations**.

## `Complex numbers`

The library never uses `std::complex`. Instead, it includes its own `Complex<T>` template type that can be used
interchangeably across all backends and can be reinterpreted to `std::complex`or `cuComplex`/`cuDoubleComplex`. It is a
simple struct with an array of two floats or two doubles. It is "well-defined" (it does not violate the stride aliasing
rule) to reinterpret this `Complex<T>` into a `T*`.

Since C++11, it is required for `std::complex` to have an array-oriented access. It is also defined behavior
to `reinterpret_cast` a `struct { float|double x, y; }` to a `float|double*`. As such, `noa::Complex<T>` can simply 
be reinterpreted to `std::complex<T>`, `cuComplex` or `cuDoubleComplex` whenever necessary.

Links: [std::complex](https://en.cppreference.com/w/cpp/numeric/complex),
[reinterpret_cast](https://en.cppreference.com/w/cpp/language/reinterpret_cast)

## `Exceptions`

The library uses exceptions to report errors. Exceptions messages are prefixed with the location of where they were
thrown. Usually exceptions messages have enough information regarding the error and exceptions are rarely nested.

Exceptions thrown by the library are always of type `noa::Exception`, which inherits from `std::exception`. However,
other exceptions, which all inherits from `std::exception`, can be thrown by the runtime, e.g. `std::bad_alloc`,
although these are very rare...

Exceptions thrown by the library are nested exceptions, which can be very useful if the user also uses nested 
exceptions. The library provides a way to unroll all nested exceptions via the `noa::Exception::backtrace()`
static function. Note that this function works with any `std::nested_exception`.

## `Cycle per pixel`

In the `fft` namespaces, frequencies are specified in fractional reciprocal lattice units from 0 to 0.5. Anything
outside this range is often still valid or clamped to this range. For instance, given a `{1,1,64,64}` image with a pixel
size of 1.4 A/pixel. To lowpass filter this image at a resolution of 8 A, the frequency cutoff should
be `1.4 / 8 = 0.175`. Note that multiplying this normalized value by the dimension of the image gives us the number of
oscillations in the real-space image at this frequency (or the resolution shell in Fourier space),
i.e. `0.175 * 64 = 22.4`. Naturally, the Nyquist frequency is at 0.5 in fractional reciprocal lattice units and, for
this example, at the 64th shell. One advantage of this notation is that it does not depend on the transform dimensions
and works for rectangular shapes.
