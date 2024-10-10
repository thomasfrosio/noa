## `Strides and memory layout`

Each dimension of our nd-arrays is encoded by a stride, which is defined at runtime. If the
array is contiguous, the innermost dimension, i.e. the dimension with the smallest stride,
has a stride of 1 element. Strides of 0 is for “broadcasted dimensions”, where a single
element in memory is used to represent an entire dimension. Note that while the library
technically supports negative strides, they are not tested yet and some things are likely to
break.

A memory layout is said to follow the rightmost order when the innermost stride is at the
right (the “last” dimension) and strides increase right to left. Similarly to NumPy, this
is what we default to. In the BDHW convention, the rightmost order is synonymous to
row-major and a rightmost contiguous array is said to be C-contiguous (as opposed to F-
contiguous). For instance, a C-contiguous 3d array of shape `[2,3,4,5]` has its strides equal
to `[60,20,5,1]`.

The library tries to make as few assumptions on the memory layout as possible. As a result,
to ensure good performance, it tries (whenever possible) to find the fastest way to iterate
through arrays by looking at the shape and strides of the inputs and/or outputs. However,
doing so is not always possible. For instance, `noa::iwise`, a function that abstracts a
nd-loop, does not take arrays as inputs, thus cannot reorder dimensions. In these cases,
the rightmost order is always assumed. Moreover, when input and/or output arrays have
different layouts, it is not always easy or possible to find the overall best order, so we do
recommend to keep with the default and use rightmost arrays.
