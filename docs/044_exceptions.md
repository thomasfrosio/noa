## `Exceptions`

By default, the library uses exceptions to report (non-recoverable/user) errors.

Exceptions thrown by the library are always of type `noa::Exception`, which inherits from `std::exception`. However, other exceptions, which all inherits from `std::exception`, can be thrown by the C++ runtime, e.g. `std::bad_alloc`, although these are very rare...

Exceptions thrown by the library are nested exceptions, which can be useful if the application also uses nested exceptions. The library provides a way to unroll all nested exceptions via the `noa::Exception::backtrace()` static function. Note that this function works with any `std::nested_exception`. Exception messages are prefixed with the location of where the exception was thrown. Usually exception messages have enough information regarding the error and exceptions are rarely nested.

## `Error policy`

Exceptions can be turned off using the CMAKE build option [`NOA_ERROR_POLICY`](../cmake/ProjectOptions.cmake). This is intended to reduce the amount of code generated due to nested exceptions. The `abort` mode is particularly lean since it generates a single instruction, and it can be used on GPU code, for instance to bound-check on the GPU in Release mode.
