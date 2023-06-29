#include "noa/Geometry.hpp"
#include "noa/Signal.hpp"
#include "noa/Math.hpp"
#include "noa/Memory.hpp"
#include "noa/IO.hpp"
#include "noa/FFT.hpp"
#include "noa/core/signal/Windows.hpp"

#include <catch2/catch.hpp>

using namespace noa;

namespace {
    [[nodiscard]]
    inline auto patch_grid_1d(
            i64 grid_size,
            i64 patch_size,
            i64 patch_step
    ) {
        const auto max = grid_size - patch_size - 1;

        const auto arange = noa::math::ArangeOperator<i64>(0, patch_step);
        std::vector<i64> patch_origin;
        for (i64 i = 0;; ++i) {
            const auto value = arange(i);
            if (value >= max)
                break;
            patch_origin.push_back(value);
        }

        // Center:
        const i64 end = patch_origin.back() + patch_size;
        const i64 offset = (grid_size - end) / 2;
        for (auto& origin: patch_origin)
            origin += offset;

        return patch_origin;
    }

    [[nodiscard]]
    inline auto patch_grid_2d(
            Shape2<i64> grid_shape,
            Shape2<i64> patch_shape,
            Vec2<i64> patch_step
    ) {
        const std::vector origins_along_y = patch_grid_1d(grid_shape[0], patch_shape[0], patch_step[0]);
        const std::vector origins_along_x = patch_grid_1d(grid_shape[1], patch_shape[1], patch_step[1]);

        std::vector<Vec2<i64>> origins;
        origins.reserve(origins_along_y.size() * origins_along_x.size());
        for (auto y: origins_along_y)
            for (auto x: origins_along_x)
                origins.emplace_back(y, x);

        return origins;
    }

    void save_txt_1d(View<const f32> vector, const Path& filename) {
        Array<f32> buffer;
        if (!vector.is_dereferenceable()) {
            buffer = vector.to_cpu();
            vector = buffer.view();
        }
        vector.eval();

        std::string format;
        for (i64 i = 0; i < vector.shape()[0]; ++i) {
            const auto span = noa::Span(vector.subregion(i).flat().data(), vector.shape()[3]);
            format += fmt::format("{:.3f}\n", fmt::join(span, ","));
        }
        noa::io::save_text(format, filename);
    }

    auto patch_transformed_coordinate(
            Shape2<i64> slice_shape,
            Vec2<f64> slice_shifts,
            Vec3<f64> slice_angles,
            Vec2<f64> slice_sampling,
            Vec2<f64> patch_center
    ) -> Vec3<f64> {
        slice_angles = noa::math::deg2rad(slice_angles);

        // By convention, the rotation angle is the additional rotation of the image.
        // Subtracting it aligns the tilt-axis to the y-axis.
        slice_angles[0] *= -1;

        // Switch coordinates from pixels to micrometers.
        const auto scale = slice_sampling * 1e-4;
        const auto slice_center_3d = (slice_shape.vec().as<f64>() * scale).push_front(0) / 2;
        const auto slice_shifts_3d = (slice_shifts * scale).push_front(0);

        // Place the slice into a 3d volume, with the center of the slice at the origin of the volume.
        namespace ng = noa::geometry;
        const Double44 image2microscope_matrix =
                ng::linear2affine(ng::euler2matrix(slice_angles, /*axes=*/ "zyx", /*intrinsic=*/ false)) *
                ng::translate(- slice_center_3d - slice_shifts_3d);

        const auto patch_center_3d = (patch_center * scale).push_front(0).push_back(1);
        const Vec3<f64> patch_center_transformed = (image2microscope_matrix * patch_center_3d).pop_back();
        return patch_center_transformed;
    }

    void display_patches_z(
            View<f32> mask,
            Vec2<f64> slice_shifts,
            Vec3<f64> slice_angles,
            Vec2<f64> slice_sampling,
            Shape2<i64> patch_shape,
            Vec2<f32> delta_z_range_nanometers,
            std::vector<Vec2<i64>> patch_origins
    ) {
        const auto slice_shape = mask.shape().filter(2, 3);

        // Find the z (in nm) of every patch.
        std::vector<f32> patches_z;
        patches_z.reserve(patch_origins.size());
        for (auto patch_origin: patch_origins) {
            const auto patch_center = (patch_origin + patch_shape.vec() / 2).as<f64>();

            const auto patch_coordinates = patch_transformed_coordinate(
                    slice_shape, slice_shifts, slice_angles, slice_sampling, patch_center);
            const auto z_nanometers = static_cast<f32>(patch_coordinates[0] * 1e3); // micro -> nano
            patches_z.push_back(z_nanometers);
        }

        // Draw the patch centers with their z position.
        for (size_t i : noa::irange(patches_z.size())) {
            // Exclude patches that are outside the delta-z tolerance.
            f32 patch_z = patches_z[i];
            if (patch_z < delta_z_range_nanometers[0] ||
                patch_z > delta_z_range_nanometers[1])
                continue;

            // Visualisation.
            const auto patch_center = (patch_origins[i] + patch_shape.vec() / 2).as<f64>();
            patch_z /= delta_z_range_nanometers[1] - delta_z_range_nanometers[0];
            const auto radius = 16 + 32 * patch_z;
            noa::geometry::sphere(mask, mask, patch_center.as<f32>(), radius, 16, {}, noa::plus_t{});
        }
    }
}

TEST_CASE("example: tile extraction", "[.]") {
    const auto working_directory = Path("/home/thomas/Projects/noa-data/assets/test_patches");
    const auto image_filename = Path("/home/thomas/Projects/datasets/EMPIAR-10304/tilt2_slice19.mrc");
    auto image = noa::io::load_data<f32>(image_filename);
    const auto image_center = image.shape().vec().filter(2, 3).as<f32>() / 2;

    const auto patch_shape = Shape2<i64>{256, 256};
    const auto patch_step = Vec2<i64>{256, 256};
    const auto sampling = Vec2<f64>{1, 1}; // A/pix
    const auto delta_z_range_nanometers = Vec2<f32>{-200, 200}; // nm

    const std::vector<Vec2<i64>> patches_origins = patch_grid_2d(
            image.shape().filter(2, 3),
            patch_shape, patch_step
            );

    {
        // Visualise patches.
        const auto slice_shifts = Vec2<f64>{};
        std::vector<Vec3<f64>> stack_angles;
        for (i64 i = -60; i <= 60; i += 3)
            stack_angles.emplace_back(45, i, 0);

        const auto option = ArrayOption(Device("gpu"));
        auto stack_shape = Shape4<i64>{stack_angles.size(), 1, image.shape()[2], image.shape()[3]};
        const auto output_mask = noa::memory::zeros<f32>(stack_shape, option);

        for (size_t i = 0; i < stack_angles.size(); ++i) {
            auto mask = output_mask.view().subregion(i);
            display_patches_z(
                    mask, slice_shifts, stack_angles[i], sampling, patch_shape,
                    delta_z_range_nanometers,
                    patches_origins);
        }

        noa::io::save(output_mask, working_directory / "patch_centers.mrc");
    }

    // TODO Test with higher tilt image.
    //      - check if smaller patches are better for higher tilts.
    //      - check masking

    //
    auto patches_inv_matrices = noa::memory::empty<Float23>(static_cast<i64>(patches_origins.size()));
    auto patches_origins_4d = noa::memory::empty<Vec4<i32>>(patches_inv_matrices.elements());
    for (size_t i = 0; i < patches_origins.size(); ++i) {
        const auto& patch_origin = patches_origins[i];
        patches_inv_matrices(0, 0, 0, i) = noa::geometry::affine2truncated(
                noa::geometry::translate(patch_origin.as<f32>()
                ).inverse());

        patches_origins_4d(0, 0, 0, i) = {0, 0, patch_origin[0], patch_origin[1]};
    }

    // To GPU.
    const auto options = ArrayOption(Device("gpu"));
    image = image.to(options);
    patches_inv_matrices = patches_inv_matrices.to(options);
    patches_origins_4d = patches_origins_4d.to(options);

    noa::math::normalize(image, image);
    noa::geometry::rectangle(image, image, image_center, image_center - 128, 128);

    const auto image_ps = noa::ewise_unary(noa::fft::r2c(image, noa::fft::Norm::NONE), noa::abs_squared_t{});
    noa::io::save(noa::ewise_unary(image_ps, noa::abs_one_log_t{}), working_directory / "image_ps.mrc");

    const auto patches_shape = patch_shape.push_front(Vec2<i64>{patches_inv_matrices.elements(), 1});
    const auto patches = noa::memory::empty<f32>(patches_shape, image.options());
    noa::memory::extract_subregions(image, patches, patches_origins_4d);
    noa::math::normalize_per_batch(patches, patches);
    noa::io::save(patches, working_directory / "patches.mrc");

    const auto patches_rfft = noa::fft::r2c(patches, noa::fft::Norm::NONE);
    const auto patches_rfft_ps = noa::ewise_unary(patches_rfft, noa::abs_squared_t{});
    const auto patches_rfft_ps_mean = noa::math::mean(patches_rfft_ps, patches_rfft_ps.shape().set<0>(1));
    noa::io::save(noa::ewise_unary(patches_rfft_ps, noa::abs_one_log_t{}), working_directory / "patches_ps.mrc");
    noa::io::save(noa::ewise_unary(patches_rfft_ps_mean, noa::abs_one_log_t{}), working_directory / "patches_ps_mean.mrc");

    const auto rotational_average_size = noa::math::min(patches.shape().filter(2, 3)) / 2 + 1;
    const auto rotational_average_shape = Shape4<i64>{patches.shape()[0], 1, 1, rotational_average_size};
    const auto rotational_average = noa::memory::zeros<f32>(rotational_average_shape, image.options());
    const auto rotational_average_mean = noa::memory::zeros<f32>(rotational_average_size, image.options());
    noa::geometry::fft::rotational_average<fft::H2H>(
            patches_rfft_ps, patches.shape(),
            rotational_average);
    noa::geometry::fft::rotational_average<fft::H2H>(
            patches_rfft_ps_mean, patches.shape().set<0>(1),
            rotational_average_mean);

    save_txt_1d(rotational_average.view(), working_directory / "rotational_average.mrc");
    save_txt_1d(rotational_average_mean.view(), working_directory / "rotational_average_mean.mrc");

    auto rotational_average_mean_output = rotational_average_mean.to_cpu();
    noa::signal::convolve(
            rotational_average_mean.to_cpu(), rotational_average_mean_output,
            noa::signal::gaussian_window<f32>(7, 1.25, true));
    save_txt_1d(rotational_average_mean_output.view(), working_directory / "rotational_average_mean_smooth.mrc");
}
