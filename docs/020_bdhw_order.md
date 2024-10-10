## `BDHW order`

The library uses a fixed 4-dimensional (4d) model to manipulate multidimensional data. The
dimension order is always following the Batch-Depth-Height-Width (BDHW) order.

First, having a fixed number of dimensions allows us to encode the shape and strides in static
arrays, instead of dynamic arrays saved on the heap. These arrays can be easily created
and efficiently passed around, without involving dynamic allocations. The optimizations of
having a fixed sized array are also welcomed.

Second, having to deal with a fixed number of dimensions was originally simpler to code
on the library side, but is also simpler to reason about on the user side. Since the BDHW
convention is used throughout the library, users know what to expect and have to follow a
strict but simple convention. This is particularly relevant when comparing with NumPy-like
arrays and their dynamic number of dimensions, where a significant part of the code can be
dedicated to just rearranging dimensions.

Third, because the BDHW convention is enforced throughout the library, dimensions have
a logical meaning. For instance, we can naturally differentiate between a stack of 2d
images (shape=`[n,1,h,w]`) and a 3d volume (shape=`[1,d,h,w]`). Functions can have
strict checks regarding the shape of the input/output arrays and can quickly detect logical
incompatibilities. For instance, `noa::geometry::transform_2d` expects 2d arrays and will
throw an error if 3d arrays are passed.

While we only allow nd-arrays up to 4 dimensions, we do support more complex data-types
compared to NumPy (or PyTorch). For instance, to represent an array of 4-by-4 matrices in
NumPy, one would need a floating-point array with 3 dimensions (e.g. shape=`[n,4,4]`)
or maybe more to follow broadcasting rules. In our case, the 4-by-4 matrix is the data-type
(e.g. `noa::Mat<f64, 4, 4>`) resulting in having to use one single dimension to store an array of
matrices (e.g. shape=`[n,1,1,1]`).
