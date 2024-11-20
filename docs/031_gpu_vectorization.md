## `Operator interface`

By default, the element-wise interfaces (`ewise`, `reduce_ewise` and `reduce_axes_ewise`) allow the operators to write from the inputs and read from the outputs. While this can be useful for some operations, it may also constrain some backends in doing certain optimizations.


## `CUDA vectorized global memory accesses`

For instance, the CUDA backend supports "vectorized" reads/writes from/to global memory, which may help to increase the data throughput on the GPU. However, to do so, we use temporary input/output values and pass these values to the operator instead of the values from the input/output arrays. For instance, in the case of `ewise`, we would do something like:

```c++
// noa::ewise(input: a, output: b, op) ->
a_buffer, b_buffer; // uninitialized
vectorized_read(a, a_buffer); // a_buffer is initialized with a
op(a_buffer..., b_buffer...); // op should initialze b_buffer
vectorized_write(b_buffer, b); // write to b
```

As a result, the operator should only read from the inputs (since modifying them would have no effect on the original arrays) and only write to the output values (since these values are passed uninitialized). Importantly, this also means that the operator should always initialize its output values since these are written back to the output arrays! Finally, since the operator gets the values from this temporary buffer, the address of the input/output values will be different and any kind of aliasing is removed.


## `"enable_vectorization" flag`

Operators can opt in to these optimizations by defining the optional type alias `enable_vectorization` to indicate that the operator supports it. This flag creates a contract between the operator and the library saying that:

- the input values are read-only, and the output values are passed uninitialized and must be initialized by the operator.
- Temporary values can be passed, meaning that the memory addresses of the values passed to the operator don't have to match the addresses of the actual input and output arrays/values.

Some library operators come with this flag whenever it seemed appropriate to add it, so users should be careful when using them directly. For instance, `Zero` is defined such as:

```c++
struct Zero {
    using enable_vectorization = bool;

    template<typename... U>
    constexpr void operator()(U&... dst) const {
        ((dst = U{}), ...);
    }
};
static_assert(noa::traits::enable_vectorization_v<Zero>);
```

This means that doing `noa::ewise(a, {}, noa::Zero{})` on the GPU is likely to return `a` unchanged because `a` is passed as an input. Indeed, using the example above, this means that `a_buffer` is passed to `Zero`, which zeroes it, but `a_buffer` is not written back to `a`. `noa::ewise({}, a, noa::Zero{})` should be used instead.

Note: if the definition of the operator cannot be modified, template specialization can also be used:
```c++
template<> struct enable_vectorization<MyOperator> : std::true_type {};
```

## `CPU restrict pointers`

On the CPU side of things, the only optimization that can affect the operator is whether the input and output arrays alias each other. If they don't, pointers are marked `restrict`, which may help the optimizer later on. In this case, `enable_vectorization` is simply bypassing this runtime check and enforcing `restrict` on all inputs and outputs.

Note that `clang++` seems to ignore `restrict` attributes for member variables, which seem to be the reasons why `g++` perform so much better compared to `clang++` in certain scenarios...
