# Conventions

## Size of dimensions

The code base often refers to `shapes` when dealing with sizes, especially with sizes of multidimensional arrays. They
have the same meaning and format as in `Numpy`, except that the innermost dimension comes first.

- Unless specified otherwise, shapes are always organized as such: `{x=fast, y=medium, z=slow}`. This directly refers to
  the memory layout of the data and is therefore less ambiguous than `{row|width, column|height, page|depth}`.
- Shapes should not have any zeros. An "empty" dimension is specified with 1. The entire API follows this convention
  (unless specified otherwise). See `ndim()` for more details. For instance, a 1D array is specified as {x}, {x, 1} or
  {x, 1, 1}. A 2D array is specified as {x, y} or {x, y, 1}. A 3D array is specified as {x, y, z}.

## Complex numbers and array-oriented access

Since C++11, it is required for `std::complex` to have an array-oriented access. It is also defined behavior
to `reinterpret_cast` a `struct { float|double x, y; }` to a `float|double*`. This does not violate the strict aliasing
rule. Also, `cuComplex`, `cuDoubleComplex` and `noa::Complex<>` have the same layout. As such, `std::complex<>`
or `noa::Complex<>` can simply be `reinterpret_cast<>` to `cuComplex` or `cuDoubleComplex` whenever necessary. Unittests
will make sure there's no weird padding and alignment is as expected so that array-oriented access is OK.

Links: [std::complex](https://en.cppreference.com/w/cpp/numeric/complex),
[reinterpret_cast](https://en.cppreference.com/w/cpp/language/reinterpret_cast)
