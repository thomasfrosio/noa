## `Shape and strides`

The library uses rightmost shapes/strides, that is the innermost stride is on the right, and strides increase
right-to-left as the product of indexes. This is similar to what `Numpy` uses, expect that `Numpy` encodes strides as
number of bytes, where this library encodes everything in number of elements.

Empty dimensions are set to 1. Shapes with a dimension of size 0 are not valid. Strides, which contains for each
dimension the memory offset, in elements, between each logical element. Broadcasting an array along a particular
dimension can often be achieved by setting a stride of 0 for that said dimension.

Shape and strides are 4D, such as `{B,Z,Y,X}`, where `B` is the "batch" dimension. This results in the ability to
differentiate between a stack of 2D images (e.g. `{2,1,64,64}`), and a 3D volume or a stack of 3D volumes. Functions
that require 2D arrays, e.g. `geometry::transform2D`, will expect, and assert, the third-most dimensions to be empty.

## `Complex numbers`

The library never uses `std::complex`. Instead, it includes its own `Complex` type that can be used interchangeably
across all backends and can be reinterpreted to `std::complex`or `cuComplex`/`cuDoubleComplex`. It is a simple struct
with an array of two floats or two doubles. It is "well-defined" (it does not violate the stride aliasing rule) to
reinterpret this
`Complex<T>` into a pointer of `T`.

Since C++11, it is required for `std::complex` to have an array-oriented access. It is also defined behavior
to `reinterpret_cast` a `struct { float|double x, y; }` to a `float|double*`. As such, `std::complex<T>` or
`noa::Complex<T>` can simply be reinterpreted to `T*` or to `cuComplex` or `cuDoubleComplex` whenever necessary.

Links: [std::complex](https://en.cppreference.com/w/cpp/numeric/complex),
[reinterpret_cast](https://en.cppreference.com/w/cpp/language/reinterpret_cast)

## `Cycle per pixel`

In the `fft` namespaces, cutoffs and window widths are specified in fractional reciprocal lattice units from 0 to 0.5.
Anything outside this range is often still valid or clamped to this range.

For instance, given a `{1,1,64,64}` image with a pixel size of 1.4 A/pixel. To lowpass filter this image at a resolution
of 8 A, the frequency cutoff should be `1.4 / 8 = 0.175`. Note that multiplying this normalized value by the dimension
of the image gives us the number of oscillations in the real-space image at this frequency (or the resolution shell in
Fourier space), i.e. `0.175 * 64 = 22.4`. Naturally, the Nyquist frequency is at 0.5 in fractional reciprocal lattice
units and, for this example, at the 64th shell.

## `Geometry`

Conventions:

- Transformations are active (alibi), i.e. body rotates about the origin of the coordinate system.
- Transformations assume a right-handed coordinate system.
- Angles are given in radians by default.
- Positive angles specify a counter-clockwise rotation, when looking at the origin from a positive point.
- Positive translations specify a translation to the right, when looking at the origin from a positive point.
- Rotation/affine matrices pre-multiply rightmost (i.e. `{Z,Y,X}`) column vectors, to produce transformed column
  vectors: `M * v = v'`.

If the coordinates are the query of interpolation, we are often talking about an inverse transformation, i.e. we go from
the coordinates in the output reference frame to the coordinates of in the input reference frame. Instead of computing
the inverse of the rotation matrix (affine or not), we can simply:

- take the transpose of the `2x2` or `3x3` rotation matrix, which is equivalent to invert a pure rotation. Note that if
  the rotation matrix has a `determinant != 1`, i.e. it has a `scaling != 1`, then `transpose != inverse`.
- negate the translation, which is equivalent to invert a pure `3x3` or `4x4` (affine) translation matrix.
- invert the scaling values, i.e. `1/scalar`, which is equivalent to invert a pure `2x2` or `3x3` scaling matrix.

Since we pre-multiply column vectors, the order of the transformations goes from right to left, e.g. `A = T * R * S`,
scales, rotates then translates. However, as mentioned above, if we perform the inverse transformation, the inverse
matrix, i.e. `inverse(A)`, is needed. Since inverting a `3x3` or `4x4` affine matrix is "expensive", we can instead
invert the individual transformations and revert the order: `inverse(T * R * S)` is equivalent to
`inverse(S) * inverse(R) * inverse(T)`. Note that inverting pure transformations is trivial, as explained above. As
such, when matrices are passed directly, they are assumed to be already inverted, unless specified otherwise.

Right-most ordering denotes the C/C++ standard multidimensional array index mapping where the right-most index is stride
one and strides increase right-to-left as the product of indexes. This is often referred to as row major (as opposed to
column major). The library uses right-most indexes, and as such, our axes are always specified in the right-most order:
`{Z,Y,X}`, where `Z` is the outermost axis and `X` is the innermost one. This convention is uniformly applied across the
library for vectors (e.g. shapes, stride, shift, scale), including all matrices, which should be pre-multiplied by
rightmost `{Z,Y,X}` column vectors.

Links: [ambiguities](https://rock-learning.github.io/pytransform3d/transformation_ambiguities.html)

## `Euler angles`

By default, the library uses the `ZYZ` axes, with intrinsic and right-handed rotations. However, the `euler2matrix` 
and `matrix2euler` functions accepts all the other possible conventions. The library's API takes the inverse 
matrices as input, not the euler angles.

Links: [eulerangles](https://eulerangles.readthedocs.io/en/latest/usage/quick_start.html),
[some derivations](https://www.geometrictools.com/Documentation/EulerAngles.pdf)
