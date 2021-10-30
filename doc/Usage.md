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

## Lowpass, highpass and bandpass filters

These filters are all using a raised-cosine (Hann) window. The cutoffs and window width are specified in fractional
reciprocal lattice units from 0 to 0.5. Anything outside this range is still valid though.

For instance, given a 64x64 image with a pixel size of 1.4 A/pixel. To lowpass filter this image at a resolution
of 8 A, the frequency cutoff should be `1.4 / 8 = 0.175`. Note that multiplying this normalized value by the
dimension of the image gives us the number of oscillations in the real-space image at this frequency (or the
resolution shell in Fourier space), i.e. `0.175 * 64 = 22.4`. Naturally, the Nyquist frequency is at 0.5 in fractional
reciprocal lattice units and, for this example, at the 64th shell.
