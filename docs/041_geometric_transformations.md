## `Geometric transformations - Conventions`

Geometric transformations can be quite confusing. Here's the convention used by this library:

- Transformations are active (alibi), i.e. body rotates about the origin of the coordinate system.
- Transformations assume a right-handed coordinate system.
- Angles are given in radians by default.
- Positive angles specify a counter-clockwise rotation, when looking at the origin from a positive point.
- Positive translations specify a translation to the right, when looking at the origin from a positive point.
- Rotation/affine matrices pre-multiply rightmost (i.e. `{Z,Y,X}`) column vectors, to produce transformed column
  vectors: `M * v = v'`.

If the coordinates are the query of interpolation, we are often talking about an inverse transformation, i.e. we go from the coordinates in the output reference frame to the coordinates of the input reference frame. Instead of computing the inverse of the rotation matrix (affine or not), we can simply:

- take the transpose of the `2x2` or `3x3` rotation matrix, which is equivalent to invert a pure rotation. Note that if
  the rotation matrix has a `determinant != 1`, i.e. it has a `scaling != 1`, then `transpose != inverse`.
- negate the translation, which is equivalent to invert a pure `3x3` or `4x4` (affine) translation matrix.
- invert the scaling values, i.e. `1/scalar`, which is equivalent to invert a pure `2x2` or `3x3` scaling matrix.

Since we pre-multiply column vectors, the order of the transformations goes from right to left, e.g. `A = T * R * S`,
scales, rotates then translates. However, as mentioned above, if we perform the inverse transformation, the inverse
matrix `inverse(A)` is needed. We can instead invert the individual transformations and revert the
order: `inverse(T * R * S)` is equivalent to `inverse(S) * inverse(R) * inverse(T)`. Note that inverting pure
transformations is trivial, as explained above. As such, when matrices are passed directly to the library, they are often assumed to be already inverted, unless specified otherwise.


## `Rightmost matrices`

The rightmost BDHW dimension order is used throughout the library. That includes multidimensional arrays, shape and strides, vectors (e.g. translation vectors, quaternions), but it also includes rotation and affine matrices.

Left-most matrices are admittedly more frequent, but note that: 1) the `noa::geometry` namespace provide tools to construct matrices from scaling factors, translation vectors, rotation angles and Euler angles, so users rarely have to construct matrices themselves and 2) leftmost matrices can easily be converted to right-most matrices.

```c++
using namespace ::noa::types;
namespace ng = ::noa::geometry;
namespace ni = ::noa::indexing;

// Example of a 4x4 affine matrix, in the leftmost
// and rightmost order:
// leftmost = [[Xx, Xy, Xz, Tx],
//             [Yx, Yy, Yz, Ty], 
//             [Zx, Zy, Zz, Tz],
//             [ 0,  0,  0,  1]]
//
// rightmost = [[Zz, Zy, Zx, Tz],
//              [Yz, Yy, Yx, Ty],
//              [Xz, Xy, Xx, Tx],
//              [ 0,  0,  0,  1]]

// The library is consistent with its dimension order, and since
// matrices are no exceptions, we use the rightmost dimension
// for matrices too. If the user comes with their own leftmost
// matrices, the library provides utilities to convert them to
// rightmost matrices:
Mat44 rightmost = ng::translate(Vec{0., 0., 1.}); // zyx
Mat44 leftmost = ni::reorder(rightmost_matrix, Vec{2, 1, 0, 3});

// Transform a vector.
const auto vector = Vec{0., 0., 1.}; // unit vector x
const auto rotation_z = ng::rotate_z(noa::deg2rad(45.));
const auto rotated_vector = rotation_z * vector;

// Here's a more complex example: rotating a 3d volume
// around z by 45 degrees.
i64 z = 64, y = 64, x = 64;
Shape<i64, 4> shape{1, z, y, x};
Vec<f64, 3> rotation_center = (shape.pop_front().vec / 2).as<f64>{};
// ... which is equivalent to Vec<f64, 3>::from_values(z/2, y/2, x/2)
Mat<f64, 3, 3> rotation_z = ng::rotate_z(noa::deg2rad(45.));
Mat<f32, 4, 4> inverse_transform = (
    ng::translate(rotation_center) *
    ng::linear2affine(rotation_z) *
    ng::translate(-rotation_center)
).inverse().as<f32>();

Array<f32> input, output; // ...initialise the arrays...

// Compute the affine transformation.
ng::transform_3d(input, output, inverse_transform);

// Internally, the index-wise operator looks something like:
// operator()(i64 batch, i64 z, i64 y, i64 x) {
//     auto coordinates = Vec<f32, 3>::from_values(z, y, x);
//     coordinates = inverse_transform * coordinates;
//     auto interpolated_value = interpolator.interpolate_at(coordinates, batch);
//     output_accessor(batch, z, y, x) = interpolated_value;
// }
```

Here's another example using the textures:
```c++
// Import f32, Mat33, Interp, Array and Texture.
using namespace ::noa::types;
Array<f32> input, output; // to initialise...
Array<Mat33<f32>> inverse_transforms; // to initialise...

// If output is on the GPU, a GPU texture is created from input.
// If output is on the CPU, Texture is a simple wrapper of the
// existing array.
auto texture = Texture<f32>(input, outputs.device(), Interp::LANCZOS6_FAST);

// Make sure the matrices are on the compute-device.
if (inverse_transforms.device() != outputs.device())
    inverse_transforms = inverse_transforms.to({output.device()});

// Compute the affine transformation. On the GPU, this uses
// hardware interpolation.
noa::geometry::transform_2d(texture, outputs, inverse_transforms);
```
