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


