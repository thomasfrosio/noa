### :construction: This repository is in its initial phase of development :construction:

## `Objectives`

This is a `C++20` static library, aiming to provide basic signal and image processing tools for cryoEM software development. One goal is to retain reasonable efficiency so that it can be used in production code, while offering a relatively simple API.

## `Documentation`

- Config and build
  - [Dependencies](docs/000_dependencies.md)
  - [Build](docs/001_build.md)
  - [Tests](docs/002_running_tests.md)
  - [Backends](docs/011_backends.md)

- Getting started
  - [Execution model, `Stream`, `Allocator` and `Device`](docs/010_execution_model.md)
  - [BDHW order, `Shape` and `Strides`](docs/021_shape_and_strides.md)
  - [Multidimensional data: `Array` and `View`](docs/022_array_and_views.md)
  - [Accessing multidimensional data: `Span` and `Accessor`](docs/023_accessor_and_span.md)
  - [Core functions: `iwise`, `ewise`, `reduce(_axes)_iwise`, `reduce(_axes)_ewise`](docs/030_core_functions.md)

- Other topics
  - [Operator flags: enable_vectorization](docs/031_gpu_vectorization.md)
  - [FFT layouts: `Remap`](docs/040_fft_layouts.md)
  - [Geometric transformations](docs/041_geometric_transformations.md)
  - [Namespaces](docs/042_namespaces.md)
  - [Complex numbers: `Complex`](docs/043_complex_numbers.md)
  - [Exceptions](docs/044_exceptions.md)


## `Example`

This library provides high-level functionalities traditionally useful in cryoEM. For example, to transform 2d images using bilinear interpolation, one could do:

```c++
#include <noa/Array.hpp>
#include <noa/Geometry.hpp>

int main() {
    namespace ng = noa::geometry;

    // Create an array of uninitialized values of two 1024x1024 images on the GPU.
    noa::Array images = noa::empty<float>({2, 1, 1024, 1024}, {.device = "gpu"});
  
    // .. initialize the images, e.g. load from file ..
    // Note: reading from file can be as simple as:
    //       noa::Array images = noa::read_data<float>(path));

    // Create the affine matrix, rotating the images around their center by 45deg.
    const auto rotation_center = noa::Vec{512., 512.};
    const auto rotation_matrix = ng::rotate_z(noa::deg2rad(45.))

    noa::Mat<double, 3, 3> inverse_transform = (
        ng::translate(rotation_center) *
        ng::linear2affine(rotation_matrix) *
        ng::translate(-rotation_center)
    ).inverse(); // transform_2d expects the inverse transform

    // Compute the affine transformation.
    noa::Array<float> output = noa::like(images);
    ng::transform_2d(input, output, inverse_transform, {
        .interp = noa::Interp::LINEAR, // optional
        .border = noa::Border::ZERO,   // optional
    });
}
```

However, it is unreasonable to expect this library to implement everything. Instead, we provide a few [core functions](docs/030_core_functions.md), like `noa::iwise` (index-wise), so that users can implement their own operators. For instance, the same example above could be implemented like so:

```c++
struct MyAffineTransform {
    noa::Span<const float, 3> input;
    noa::Span<float, 3> output;
    noa::Mat<double, 3, 3> inverse_transform;

    // Index-wise operator called for every batch, y and x.
    constexpr void operator()(int batch, int y, int x) {
        auto coordinates = noa::Vec<float, 2>::from_values(y, x);
        coordinates = inverse_transform * coordinates;
        
        // Bilinear interpolation.
        // Note: we also provide an Interpolator that can do all of this.
        const auto floored = noa::floor(coordinate);
        const auto indices = floored.template as<int>();
        const auto fraction = coordinates - floored;
        const auto weights = noa::Vec{1 - fraction, fraction};

        float interpolated_value{};
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                auto pos = indices + noa::Vec{i, j};
                if (pos[0] >= 0 and pos[0] < input.shape()[1] and
                    pos[1] >= 0 and pos[1] < input.shape()[2]) {
                    auto value = input(batch, pos[0], pos[1]);
                    interpolated_value += value * weights[j][i];
                }
            }
        }

        // Save to the output.
        output(batch, y, x) = interpolated_value;
    }
}

void my_transform_2d(
    const noa::Array<float>& input,
    const noa::Array<float>& output,
    const noa::Mat<double, 3, 3>& inverse_transform
) {
    // A bit of sanity check. We all make mistakes.
    noa::check(not input.is_empty() and not output.is_empty());
    noa::check(input.device() == output.device());
    noa::check(input.shape()[1] == output.shape()[1] == 1,
               "Only 2d array are supported, but got shape {} and {}",
               input.shape(), output.shape());

    // Construct the operator.
    MyAffineTransform op{
        .input = input.span().filter(0, 2, 3),   // bdhw -> bhw
        .output = output.span().filter(0, 2, 3), // bdhw -> bhw
        .inverse_transform = inverse_transform,
    };

    // Call op from the output device, for every bhw indices.
    auto shape = output.shape().filter(0, 2, 3); // bdhw -> bhw
    noa::iwise(shape, output.device(), op);
}
```

## `Licence`

This should be moved to a more permissive license in the future.
GPL was originally used because of FFTW3, but now that FFTW3 is not exposed to the user and can be linked dynamically, we should be able to use something more permissive?
