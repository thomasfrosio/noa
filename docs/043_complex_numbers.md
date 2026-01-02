## `Complex numbers`

The library does not use `std::complex` (this was originally because of CUDA). Instead, it includes its own `Complex<T>` template type that can be used interchangeably across all backends and can be reinterpreted to `std::complex`or `cuComplex`/`cuDoubleComplex` when needed. It is a simple aggregate with two floating-point numbers.

```c++
template<typename T> // f16, f32, f64
struct alignas(sizeof(T) * 2) Complex {
    T real{};
    T imag{};
};
```

Note: Since C++11, it is required for `std::complex` to have an array-oriented access. It should also be well-defined to `reinterpret_cast` and access a `struct { float|double x, y; }` to a `float|double*`. As such, `noa::Complex<T>` should
be _safely_ reinterpretable to `std::complex<T>`, `cuComplex` or `cuDoubleComplex` whenever necessary.

Links: [std::complex](https://en.cppreference.com/w/cpp/numeric/complex),
[reinterpret_cast](https://en.cppreference.com/w/cpp/language/reinterpret_cast)
