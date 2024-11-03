## `Exceptions`

The library uses exceptions to report errors. Exception messages are prefixed with the location of where the exception was thrown. Usually exception messages have enough information regarding the error and exceptions are rarely nested.

Exceptions thrown by the library are always of type `noa::Exception`, which inherits from `std::exception`. However, other exceptions, which all inherits from `std::exception`, can be thrown by the runtime, e.g. `std::bad_alloc`, although these are very rare...

Exceptions thrown by the library are nested exceptions, which can be useful if the application also uses nested exceptions. The library provides a way to unroll all nested exceptions via the `noa::Exception::backtrace()` static function. Note that this function works with any `std::nested_exception`.
