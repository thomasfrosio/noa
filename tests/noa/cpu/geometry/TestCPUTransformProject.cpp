#include <noa/common/io/MRCFile.h>
#include <noa/common/geometry/Euler.h>
#include <noa/common/geometry/Transform.h>

#include <noa/cpu/math/Reduce.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Arange.h>
#include <noa/cpu/memory/Set.h>
#include <noa/cpu/geometry/Transform.h>
#include <noa/cpu/geometry/Interpolator.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

namespace {
    void transform3DProject_(const float* input, size4_t input_strides, size4_t input_shape,
                             float* output, size4_t output_strides, size4_t output_shape,
                             float44_t fwd_matrix, float* mask) {
        cpu::memory::set(output, output_strides, output_shape, 0.f);
        cpu::memory::set(mask, 257*257*257, 0.f);

        const auto inv_matrix = math::inverse(fwd_matrix);

        // Broadcast the input if it is not batched.
        const size3_t strides{0, input_strides[2], input_strides[3]};
        const size3_t shape{1, input_shape[2], input_shape[3]};
        const cpu::geometry::Interpolator3D<float> interp(input, strides, shape, 0.f);

        // Check YX range in output to not go through pixels that are OOB.

        // Z range depends on interpolation. INTERP_CUBIC would be -2 to 2.
        const int64_t start = -1;
        const int64_t end = 1;

        const float34_t fwd_matrix_{fwd_matrix};
        const float34_t inv_matrix_{inv_matrix};

            for (size_t y = 0; y < output_shape[2]; ++y) {
                for (size_t x = 0; x < output_shape[3]; ++x) {

                    // Find output ZYX coordinate by applying forward transform for the given zyx
                    const float3_t output_coords = fwd_matrix_ * float4_t{0, y, x, 1.f};

                    for (int64_t z = start; z < end; ++z) {
                        // Round coordinate to have the XYZ output index.
                        const float3_t output_idx{math::floor(output_coords[0]) + static_cast<float>(z),
                                                  math::floor(output_coords[1]),
                                                  math::floor(output_coords[2])};

                        mask[indexing::at(0, output_idx[0]+128, output_idx[1], output_idx[2], output_strides)] += 1;

                        // Apply inverse transform on the ZYX output index.
                        const float3_t input_coords =
                                inv_matrix_ * float4_t{output_idx[0], output_idx[1], output_idx[2], 1.f};

                        // Add the interpolated value at output index 0YZ
                        output[indexing::at(0, 0, output_idx[1], output_idx[2], output_strides)] +=
                                interp.get<INTERP_LINEAR, BORDER_ZERO>(input_coords);
                    }


                }
            }
    }
}

TEST_CASE("cpu::geometry::transform3D(), project2", "[.]") {
    const size4_t shape{1, 1, 257, 257};
    const size4_t strides = shape.strides();
    const size_t elements = shape.elements();

    const float3_t center{size3_t{shape.get() + 1} / 2};
    const float3_t euler{0, math::deg2rad(60.), 0};
    float44_t matrix{geometry::translate(center) *
                     float44_t{geometry::euler2matrix(euler)} * // ZYZ intrinsic right-handed
                     geometry::translate(-center)};

    cpu::Stream stream{cpu::Stream::DEFAULT};

    cpu::memory::PtrHost<float> input{elements};
    cpu::memory::PtrHost<float> output{elements};
    cpu::memory::arange(input.get(), strides, shape, 0.f, 1.f);

    io::MRCFile file{test::NOA_DATA_PATH / "geometry" / "test_xform_project2_input.mrc", io::WRITE};
    file.shape(shape);
    file.writeAll(input.get());

    cpu::memory::PtrHost<float> mask{elements * 257};
    transform3DProject_(input.get(), strides, shape, output.get(), strides, shape, matrix, mask.get());

    file.open(test::NOA_DATA_PATH / "geometry" / "test_xform_project2_output.mrc", io::WRITE);
    file.shape(shape);
    file.writeAll(output.get());

    file.open(test::NOA_DATA_PATH / "geometry" / "test_xform_project2_mask.mrc", io::WRITE);
    file.shape({1,257,257,257});
    file.writeAll(mask.get());
}

TEST_CASE("cpu::geometry::transform3D(), project", "[.]") {
    const size4_t shape{1, 257, 257, 257};
    const size4_t strides = shape.strides();
    const size_t elements = shape.elements();

    const float3_t center{size3_t{shape.get() + 1} / 2};
    const float3_t euler{0, math::deg2rad(45.), 0};
    float44_t matrix{geometry::translate(center) *
                     float44_t{geometry::euler2matrix(euler)} * // ZYZ intrinsic right-handed
                     geometry::translate(-center)};
    matrix = math::inverse(matrix);

    cpu::Stream stream{cpu::Stream::DEFAULT};

    cpu::memory::PtrHost<float> input{elements};
    cpu::memory::PtrHost<float> output{elements};
    cpu::memory::set(input.share(), strides, shape, 0.f, stream);
    cpu::memory::set(output.share(), strides, shape, 0.f, stream);
    cpu::memory::set(input.get() + 128 * strides[1], strides, size4_t{1, 1, 257, 257}, 1.f);

    io::MRCFile file{test::NOA_DATA_PATH / "geometry" / "test_xform_project_input.mrc", io::WRITE};
    file.shape(shape);
    file.writeAll(input.get());

    cpu::geometry::transform3D(input.share(), strides, shape, output.share(), strides, shape,
                               matrix, INTERP_LINEAR, BORDER_ZERO, 0.f, true, stream);

    file.open(test::NOA_DATA_PATH / "geometry" / "test_xform_project_output.mrc", io::WRITE);
    file.shape(shape);
    file.writeAll(output.get());

    cpu::math::sum(output.share(), strides, shape, input.share(), strides, {1, 1, 257, 257}, stream);
    file.open(test::NOA_DATA_PATH / "geometry" / "test_xform_project_output_sum.mrc", io::WRITE);
    file.shape({1, 1, 257, 257});
    file.writeAll(input.get());
}
