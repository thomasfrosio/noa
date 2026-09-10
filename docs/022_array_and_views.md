## `Array and View`

This library provides a ndarray called `Array`, such as `Array<T, N, O>`, where `T` is the data-type, `N` the number of dimensions, and `O` is the ownership (`ArrayOwnership::RC` for owned/reference-counted, and `ArrayOwnership::VIEW` for not-owned/view).

Having the data-type exposed in the type of the arrays gives more information to the type system to catch misuses and helps to enforce certain things at compile time. For instance, some (member) functions are only available for certain data-types, like `noa::signal::bandpass` which only accepts real or complex arrays.

Data-type reinterpretations are allowed, as long as the C++ strict type-aliasing rule is not broken. An array of any type can be reinterpreted to an array of bytes, thereby allowing to manipulate “type-erased” arrays. Reinterpreting between the complex data-type and the underlying floating-point type is also allowed. Reinterpretation is done via the `noa::reinterpret_as<U>()` function.

An owning `Array` owns its data using `std::shared ptr` to reference-count resources. On the other hand, a viewing `Array` simply keeps track of a pointer and assumes the resource will stay valid throughout its lifetime. `Array<T, N, ArrayOwnership::VIEW>` is a slightly lighter version of `Array<T, N, ArrayOwnership::RC>` and satisfies the “trivial class” requirement, making it cheap to construct and move around.

In practice, we recommend to use an owning `Array` to allocate and manage memory and use views to manipulate and pass data around (similar to `std::string` and `std::string_view`). Both ownership cases can wrap existing data (wrapping a `std::shared_ptr` or a raw pointer), and converting from an owned array to a view is as simple as `array.view()`. The data-type of an array can be mutable or const.
