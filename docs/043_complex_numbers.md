## `Complex numbers`

The library does not use `std::complex` (mostly because of CUDA). Instead, it includes its own `Complex<T>` template type that can be used interchangeably across all backends and can be reinterpreted to `std::complex`or `cuComplex`/`cuDoubleComplex`. It is a simple aggregate with two floating-point numbers.

```c++
template<typename T> // f16, f32, f64
struct alignas(sizeof(T) * 2) Complex {
    T real{};
    T imag{};
};

// note: f16 is noa::Half
```
Since C++11, it is required for `std::complex` to have an array-oriented access. It is also defined behavior
to `reinterpret_cast` a `struct { float|double x, y; }` to a `float|double*`. As such, `noa::Complex<T>` can simply
be reinterpreted to `std::complex<T>`, `cuComplex` or `cuDoubleComplex` whenever necessary.

Links: [std::complex](https://en.cppreference.com/w/cpp/numeric/complex),
[reinterpret_cast](https://en.cppreference.com/w/cpp/language/reinterpret_cast)
