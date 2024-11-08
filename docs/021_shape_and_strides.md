## `Strides and memory layout`

Each dimension of our nd-arrays is encoded by a stride, which is defined at runtime. Note that while the library technically supports negative strides, they are not tested and some things are likely to break.

By default, the arrays are rightmost ordered, meaning that the innermost stride is on the right, and strides increase right-to-left. This is similar to what Numpy uses by default, expect that Numpy encodes strides in number of bytes, whereas we encode them in number of elements. In the BDHW convention, the rightmost order is synonym to row-major and a rightmost contiguous array is said to be C-contiguous. For instance, a C-contiguous 3d array of shape `{1,2,3,4}` has its strides equal to `{24,12,4,1}`. Following the BDHW convention, a F-contiguous array, aka column-major, of the same shape has its strides equal to `{24,12,1,4}` and is therefore not in the rightmost order. When referring to C- or F-ordering, it is important to note that this often only affects the `height/column` and `width/row` of the array, so the `batch` and `depth` dimensions not affected. This is different from what Numpy does!

```c++
noa::Array<f32> a = noa::empty<f32>({1, 3, 4, 5});
a.shape(); // (1, 3, 4, 5)
a.strides(); // (60, 20, 5, 1)
assert(a.are_contiguous()); // equivalent to are_contiguous<'C'>();

noa::Array<f32> b = a.permute({0, 1, 3, 2}); // swap height and width
b.shape(); // (1, 3, 5, 4)
b.strides(); // (60, 20, 1, 5)
assert(b.are_contiguous<'C'>() == false);
assert(b.are_contiguous<'F'>() == true);
assert(b.copy().are_contiguous<'C'>() == true); // copy defaults to rightmost

// While .permute().copy() is possible, .permute_copy() offers a
// more efficient permutation to a new C-contiguous array.
noa::Array<f32> c = a.permute_copy({0, 1, 3, 2});
assert(c.are_contiguous<'C'>() == true); // copy defaults to rightmost
assert(c.are_contiguous<'F'>() == false);
```

## `Broadcasting`

Empty dimensions (dimensions with a size of 1) can be broadcasted to any size. When broadcasting is supported, the library will automatically try to broadcast the input array(s) onto the output shape. If the shapes aren't compatible, even after broadcasting, an error is thrown. Arrays can also be explicitly broadcasted using `noa::broadcast()`.

Broadcasting is implemented by setting the stride to 0, effectively saying that the same element is repeated along that dimension. Broadcasting arrays used in read-only contexts is always valid. However, **broadcasting arrays can generate a data-race if values are written along the broadcast dimension**. This is because with a stride of 0, multiple indices can now refer to the same memory location. Except when explicitly documented otherwise, **there are no guarantees on the order of execution in element-wise or index-wise operations**.

```c++
noa::Array<f32> a = noa::empty<f32>({1, 3, 4, 5});
noa::Array<f32> b = noa::broadcast(a, {10, 3, 4, 5});
b.shape(); // (10, 3, 10, 5)
b.strides(); // (0, 20, 5, 1)
assert(b.are_contiguous()); // broadcasting breaks contiguity
assert(noa::all(b.is_contiguous(), Vec{false, true, true, true});
// "is_contiguous" tests each dimension, which can be useful
// to know where the contiguity is broken.
```

## `Correctness and performance`

The library tries to make as few assumptions on the memory layout as possible. However, certain functions have some requirements regarding the layout of the input and output arrays. If these requirements are not met, an error will be thrown.

To ensure good performance, the library tries (whenever possible) to find the fastest way to iterate through arrays by looking at the shape and strides of the inputs and/or outputs. For instance, copying (which relies on [`noa::ewise`](030_core_functions.md) core function) C or F arrays results in the same performance.

However, doing so is not always possible. For instance, [`noa::iwise`](030_core_functions.md) does not take arrays as inputs, thus cannot reorder dimensions. In these cases, the rightmost order is always assumed. Moreover, when input and/or output arrays have different layouts, it is not always easy or possible to find the overall best order, so we do recommend to keep with the default layout and use rightmost arrays whenever possible.
