## `Array and View`

This library provides an owning 4d-array called `Array<T>` and a non-owning 4d-array called `View<T>`. The data-type of the array is defined at compile time. Having the data-type exposed in the type of the arrays gives more information to the type system to catch misuses and helps to enforce certain things at compile time. For instance, some (member)functions are only available for certain data-types, like `noa::signal::bandpass` which only accepts complex arrays.

Data-type reinterpretations are allowed, as long as the C++ strict type-aliasing rule is not broken. An array of any type can be reinterpreted to an array of bytes, thereby allowing to manipulate “type-erased” arrays. Reinterpreting between the complex data-type and the underlying floating-point type is also allowed. Reinterpretation is done via the `Array<T>::reinterpret_as<U>()` or `Array<T>::span<U>()` member functions.

`Array` allocates and owns its data, and uses `std::shared ptr` to reference-count resources. As a result, the data-type of `Array` cannot be const-qualified. On the other hand, `View` simply keeps track of a pointer (which can point to const data) and assumes the resource will stay valid throughout its lifetime. `View` is a lighter version of `Array`: it is 80 bytes in size, and satisfies the “trivial class” requirement, making it cheap to construct and move around.

```c++
template<typename T>
class Array {
    Shape<i64, 4> shape; // a static array of four i64 values
    Stride<i64, 4> strides; // same as above
    std::shared_ptr<T[]> buffer; // the 1d dynamic array
    ArrayOption option; // contains Device and Allocator
};

template<typename T>
class View {
    Shape<i64, 4> shape; // same as Array
    Stride<i64, 4> strides; // same as Array
    T* pointer; // points to an existing array
    ArrayOption option; // same as Array
};
```

`Array` and `View` can be used interchangeably in the library API, but in practice, we recommend to use `Array` to allocate and manage memory and use `View` to manipulate and pass data around. To read and write by indexing directly into the nd-array, we recommend using [`Span`](023_accessor_and_span.md).

While most of the library API is designed around free functions, `Array` and `View` have some useful member functions. See [View.hpp](../src/noa/unified/View.hpp) and [Array.hpp](../src/noa/unified/Array.hpp)
