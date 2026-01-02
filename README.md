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

This library comes with high-level abstractions traditionally useful in cryoEM. For example, to transform 2d images using bilinear interpolation on a GPU, one could do:

```c++
#include <noa/Runtime.hpp>
#include <noa/Xform.hpp>

namespace nx = noa::xform;

int main() {
    // Create an array of uninitialized values of two 1024x1024 images on the GPU.
    auto images = noa::Array<float>({2, 1, 1024, 1024}, {.device = "gpu"});

    // .. initialize the images... or for instance get the images from an MRC file:
    // auto images = noa::read_image<float>("image.mrc"));

    // Create the affine matrix, rotating the images around their center by 45deg.
    const auto rotation_center = noa::Vec{512., 512.};
    const auto rotation_matrix = nx::rotate(noa::deg2rad(45.));

    noa::Mat<double, 3, 3> inverse_transform = (
        nx::translate(rotation_center) *
        nx::affine(rotation_matrix) *
        nx::translate(-rotation_center)
    ).inverse(); // transform_2d expects the inverse transform

    // Compute the affine transformation.
    noa::Array<float> output = noa::like(images);
    nx::transform_2d(images, output, inverse_transform, {
        .interp = nx::Interp::LINEAR, // optional
        .border = noa::Border::ZERO,  // optional
    });
}
```

However, it is unreasonable to expect this library to implement everything. Instead, the runtime provides a few [core functions](docs/030_core_functions.md), like `noa::iwise` (for "index-wise", essentially an n-dimensional for-loop), so that we can implement our own operators directly. For instance, the example above could be implemented like so:

```c++
struct MyAffineTransform {
    noa::Span<const float, 3> input;
    noa::Span<float, 3> output;
    noa::Mat<double, 3, 3> inverse_transform;

    // Index-wise operator called for every batch, y and x.
    constexpr void operator()(int batch, int y, int x) {
        auto coordinates = noa::Vec<double, 2>::from_values(y, x);
        coordinates = (inverse_transform * coordinates.push_back(1)).pop_back();

        // Bilinear interpolation.
        // Note: we provide an Interpolator that can do all of this.
        const auto floored = floor(coordinates);
        const auto indices = floored.as<int>();
        const auto fraction = (coordinates - floored).as<float>();
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
};

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
    auto op = MyAffineTransform{
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
