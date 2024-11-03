## `GPU vectorized read/write`

By default, the element-wise interfaces (`ewise`, `reduce_ewise` and `reduce_axes_ewise`) allow the operators to write from the inputs and read from the outputs. While this can be useful for some operations, it may also constrain some backends in doing certain optimizations.

For instance, the CUDA backend supports "vectorized" reads/writes from/to global memory, which may help increase the data throughput on the GPU. However, to do so, we use temporary input/output values and pass these values to the operator instead of the values from the input/output arrays. For instance, in the case of `ewise`, we would do something like:
```c++
// noa::ewise(input: a, output: b, op) ->
a_buffer, b_buffer; // uninitialized
vectorized_read(a, a_buffer); // a_buffer is initialized with a
op(a_buffer..., b_buffer...); // op should initialze b_buffer
vectorized_write(b_buffer, b); // write to b
```

As a result, the operator should only read from the inputs (since modifying them would have no effect on the original arrays) and only write to the output values (since these values are passed uninitialized). Importantly, this also means that the operator should always initialize its output values since these are written back to the output arrays! Similarly, since the operator gets the values from this temporary buffer, the address of the input/output values will be different and any kind of aliasing is removed.

This behavior can be enabled automatically only if:
- `ewise`: there are no output arrays, and the input arrays are immutable (i.e. `View<const T>`).
- `reduce(_axes)_ewise`: the input arrays are all immutable.

Moreover, operators can opt in to this optimization by defining the optional type alias `allow_vectorization` to indicate that the operator supports it. Some library operators come with this flag whenever it seemed appropriate, so users should be careful when using them directly. For instance, `Zero` is defined such as:

```c++
struct Zero {
    using allow_vectorization = bool;

    template<typename... U>
    constexpr void operator()(U&... dst) const {
        ((dst = U{}), ...);
    }
};
static_assert(noa::traits::has_allow_vectorization_v<Zero>);
```

This means that doing `noa::ewise(a, {}, noa::Zero{})` on the GPU is likely to return `a` unchanged because `a` is passed as an input. Indeed, using the example above, this means that `a_buffer` is passed to `Zero`, which zeroes it, but `a_buffer` is not written back to `a`. `noa::ewise({}, a, noa::Zero{})` should be used instead.
