#include <noa/Array.h>
#include <noa/FFT.h>
#include <noa/Geometry.h>
#include <noa/IO.h>
#include <noa/Math.h>
#include <noa/Memory.h>
#include <noa/Signal.h>
#include <noa/common/utils/Timer.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

namespace {
    using namespace ::noa;

    Array<float33_t> generateRotations_(size_t count) {
        constexpr float PI2 = math::Constants<float>::PI2;
        test::Randomizer<float> randomizer(-PI2, PI2);

        Array<float33_t> rotations(count);
        float33_t* rotations_ = rotations.get();
        for (size_t i = 0; i < count; ++i) {
            const float3_t euler_angles = {randomizer.get(), randomizer.get(), randomizer.get()};
            rotations_[i] = geometry::euler2matrix(euler_angles);
        }
        return rotations;
    }

    Array<float33_t> generateRotationsTiltSeries_() {
        constexpr float PI2 = math::Constants<float>::PI2;
        test::Randomizer<float> randomizer(-PI2, PI2);

        Array<float33_t> rotations(41);
        float33_t* rotations_ = rotations.get();
        float start = -60;
        for (size_t i = 0; i < 41; ++i) {
            if (i != 36)
                rotations_[i] = geometry::euler2matrix(float3_t{0, math::deg2rad(start), 0}, "ZYX");
            start += 3;
        }
        return rotations;
    }
}

TEST_CASE("unified::geometry::fft, bwd and fwd projection", "[.]") {
    using namespace ::noa;
    Timer timer;

    const size_t count = 41;
    const int32_t padding_factor = 2;
    const path_t input_path = "/home/thomas/Projects/data/ribo/emd_14454.mrc";
    const path_t output_dir = test::NOA_DATA_PATH / "geometry" / "fft" / "reconstruction";
    const Device device("cpu");

    fmt::print("problem size: count={}, shape={}, padding_factor={}\n",
               count, io::ImageFile(input_path, io::READ).shape(), padding_factor);

    const ArrayOption options(device, Allocator::DEFAULT_ASYNC);
    Array<float33_t> fwd_rotations = generateRotationsTiltSeries_();//generateRotations_(count);
    if (device.gpu())
        fwd_rotations = fwd_rotations.to(device);

    timer.start();
    Array volume = io::load<float>(input_path, false, options);
    volume.eval();
    fmt::print("loading file: {}ms\n", timer.elapsed());

    const dim4_t volume_shape = volume.shape();
    const dim_t volume_size_padded = std::max({volume_shape[1], volume_shape[2], volume_shape[3]}) * padding_factor;
    const dim4_t volume_shape_padded = {1, volume_size_padded, volume_size_padded, volume_size_padded};
    const float3_t volume_center = float3_t{volume_shape.get(1)} / 2;
    const float3_t volume_center_padded = float3_t{volume_shape_padded.get(1)} / 2;
    const dim4_t slice_shape{1, 1, volume_shape[2], volume_shape[3]};
    const dim4_t slices_shape{count, 1, volume_shape[2], volume_shape[3]};
    const dim4_t slices_shape_padded{count, 1, volume_shape_padded[2], volume_shape_padded[3]};
    const float2_t slices_center_padded(volume_center_padded.get(1));
    const float2_t slices_center = float2_t{slices_shape.get(2)} / 2;

    Array<float> slices;
    {
        // Zero pad:
        timer.start();
        auto [volume_padded, volume_padded_fft] = fft::empty<float>(volume_shape_padded, options);
        memory::resize(volume, volume_padded);
        volume_padded.eval();
        fmt::print("zero padding: {}ms\n", timer.elapsed());

        // Prepare the volume for the forward projection:
        timer.start();
        fft::r2c(volume_padded, volume_padded_fft);
        fft::remap(fft::H2HC, volume_padded_fft, volume_padded_fft, volume_shape_padded);
        signal::fft::shift3D<fft::HC2HC>(volume_padded_fft, volume_padded_fft,
                                         volume_shape_padded, -volume_center_padded);

        // Extract the slices:
        auto [slices_padded, slices_padded_fft] = fft::empty<float>(slices_shape_padded, options);
        geometry::fft::extract3D<fft::HC2H>(volume_padded_fft.release(), volume_shape_padded,
                                            slices_padded_fft, slices_shape_padded,
                                            float22_t{}, fwd_rotations);

        // Post-process the slices:
        signal::fft::shift2D<fft::H2H>(slices_padded_fft, slices_padded_fft,
                                       slices_shape_padded, slices_center_padded);
        fft::c2r(slices_padded_fft.release(), slices_padded);
        slices = memory::resize(slices_padded.release(), slices_shape);
        slices.eval();
        fmt::print("forward projection: {}ms\n", timer.elapsed());
        io::save(slices, output_dir / "extracted_slices.mrc");
    }

    {
        // Backward projection:
        timer.start();
        Array slices_fft = fft::r2c(slices);
        signal::fft::shift2D<fft::H2H>(slices_fft, slices_fft, slices_shape, -slices_center);
        fft::remap(fft::H2HC, slices_fft, slices_fft, slices_shape);

        for (size_t i = 0; i < fwd_rotations.size(); ++i)
            fwd_rotations[i] = fwd_rotations[i].transpose();

        // Backward projection of the data:
        Array volume_fft = memory::zeros<cfloat_t>(volume_shape.fft(), options);
        geometry::fft::insert3D<fft::HC2H>(slices_fft, slices_shape,
                                           volume_fft, volume_shape,
                                           float22_t{}, fwd_rotations, 0.0025f, 0.5f);
        signal::fft::shift3D<fft::H2H>(volume_fft, volume_fft, volume_shape, volume_center);

        // Backward projection of the weights:
        Array<float> weights_slices_fft = memory::ones<float>(slice_shape.fft(), options);
        weights_slices_fft = indexing::broadcast(weights_slices_fft, slices_shape.fft());
        Array<float> weights_volume_fft = memory::zeros<float>(volume_shape.fft(), options);
        geometry::fft::insert3D<fft::HC2H>(weights_slices_fft, slices_shape,
                                           weights_volume_fft, volume_shape,
                                           float22_t{}, fwd_rotations, 0.0025f, 0.5f);

        // Weighting:
        math::ewise(volume_fft, weights_volume_fft, 1e-3f, volume_fft, math::divide_epsilon_t{});
        volume = fft::alias(volume_fft, volume_shape);
        fft::c2r(volume_fft, volume);
        volume.eval();
        fmt::print("backward projection: {}ms\n", timer.elapsed());
        io::save(volume, output_dir / "reconstruction.mrc");
    }
}

namespace {
    class ProjectorRasterize {
    public:
        ProjectorRasterize(dim4_t grid_shape,
                  dim4_t slice_shape,
                  dim4_t target_shape = {},
                  ArrayOption options = {})
                : m_grid_data_fft(memory::zeros<cfloat_t>(grid_shape.fft(), options)),
                  m_grid_weights_fft(memory::zeros<float>(grid_shape.fft(), options)),
                  m_weights_ones_fft(memory::ones<float>(slice_shape.fft(), options)),
                  m_weights_extract_fft(memory::empty<float>(slice_shape.fft(), options)),
                  m_grid_shape(grid_shape),
                  m_slice_shape(slice_shape),
                  m_target_shape(target_shape) {}

        void backward(const Array<cfloat_t>& slice_fft,
                      float33_t fwd_rotation,
                      float2_t shift,
                      float22_t scaling = {},
                      float cutoff = 0.5f) const {
            noa::signal::fft::shift2D<fft::H2H>(
                    slice_fft, slice_fft, m_slice_shape, shift);
            noa::geometry::fft::insert3D<fft::H2HC>(
                    slice_fft, m_slice_shape,
                    m_grid_data_fft, m_grid_shape,
                    scaling, fwd_rotation, cutoff, m_target_shape);
            noa::geometry::fft::insert3D<fft::H2HC>(
                    m_weights_ones_fft, m_slice_shape,
                    m_grid_weights_fft, m_grid_shape,
                    scaling, fwd_rotation, cutoff, m_target_shape);
        }

        void forward(Array<cfloat_t>& slice_fft,
                     float33_t fwd_rotation,
                     float2_t shift,
                     float22_t scaling = {},
                     float cutoff = 0.5f) const {
            noa::geometry::fft::extract3D<fft::HC2H>(
                    m_grid_data_fft, m_grid_shape,
                    slice_fft, m_slice_shape,
                    scaling, fwd_rotation, cutoff, m_target_shape);
            noa::geometry::fft::extract3D<fft::HC2H>(
                    m_grid_weights_fft, m_grid_shape,
                    m_weights_extract_fft, m_slice_shape,
                    scaling, fwd_rotation, cutoff, m_target_shape);
            signal::fft::shift2D<fft::H2H>(slice_fft, slice_fft, m_slice_shape, shift);
            math::ewise(slice_fft, m_weights_extract_fft, 1e-3f, slice_fft,
                        math::divide_epsilon_t{});
        }

    public:
        Array<cfloat_t> m_grid_data_fft;
        Array<float> m_grid_weights_fft;
        Array<float> m_weights_ones_fft;
        Array<float> m_weights_extract_fft;
        dim4_t m_grid_shape;
        dim4_t m_slice_shape;
        dim4_t m_target_shape;
    };

    class ProjectorInterp {
    public:
        ProjectorInterp(float slice_z_radius,
                        dim4_t grid_shape,
                        dim4_t slice_shape,
                        dim4_t target_shape = {},
                        ArrayOption options = {})
                : m_grid_data_fft(memory::zeros<cfloat_t>(grid_shape.fft(), options)),
                  m_grid_weights_fft(memory::zeros<float>(grid_shape.fft(), options)),
                  m_weights_ones_fft(memory::ones<float>(slice_shape.fft(), options)),
                  m_weights_extract_fft(memory::empty<float>(slice_shape.fft(), options)),
                  m_grid_shape(grid_shape),
                  m_slice_shape(slice_shape),
                  m_target_shape(target_shape),
                  m_slice_z_radius(slice_z_radius) {}

        void backward(const Array<cfloat_t>& slice_fft,
                      float33_t fwd_rotation,
                      float2_t shift,
                      float22_t scaling = {},
                      float cutoff = 0.5f) const {
            noa::signal::fft::shift2D<fft::H2H>(
                    slice_fft, slice_fft, m_slice_shape, shift);
            noa::fft::remap(fft::H2HC, slice_fft, slice_fft, m_slice_shape);
            noa::geometry::fft::insert3D<fft::HC2HC>(
                    slice_fft, m_slice_shape,
                    m_grid_data_fft, m_grid_shape,
                    scaling, math::inverse(fwd_rotation), m_slice_z_radius, cutoff, m_target_shape);
            noa::geometry::fft::insert3D<fft::HC2HC>(
                    m_weights_ones_fft, m_slice_shape,
                    m_grid_weights_fft, m_grid_shape,
                    scaling, math::inverse(fwd_rotation), m_slice_z_radius, cutoff, m_target_shape);
        }

        void forward(Array<cfloat_t>& slice_fft,
                     float33_t fwd_rotation,
                     float2_t shift,
                     float22_t scaling = {},
                     float cutoff = 0.5f) const {
            noa::geometry::fft::extract3D<fft::HC2H>(
                    m_grid_data_fft, m_grid_shape,
                    slice_fft, m_slice_shape,
                    scaling, fwd_rotation, cutoff, m_target_shape);
            noa::geometry::fft::extract3D<fft::HC2H>(
                    m_grid_weights_fft, m_grid_shape,
                    m_weights_extract_fft, m_slice_shape,
                    scaling, fwd_rotation, cutoff, m_target_shape);
            signal::fft::shift2D<fft::H2H>(slice_fft, slice_fft, m_slice_shape, shift);
            math::ewise(slice_fft, m_weights_extract_fft, 1e-3f, slice_fft,
                        math::divide_epsilon_t{});
        }

    public:
        Array<cfloat_t> m_grid_data_fft;
        Array<float> m_grid_weights_fft;
        Array<float> m_weights_ones_fft;
        Array<float> m_weights_extract_fft;
        dim4_t m_grid_shape;
        dim4_t m_slice_shape;
        dim4_t m_target_shape;
        float m_slice_z_radius;
    };
}

TEST_CASE("unified::geometry::fft, transform with projections", "[.]") {
    const path_t input_path = "/home/thomas/Projects/data/ribo/tilt1/tilt1_cropped.mrc";
    const path_t output_dir = test::NOA_DATA_PATH / "geometry" / "fft" / "reconstruction";

    io::ImageFile file(input_path, io::READ);
    const auto stack_shape = file.shape();
    const auto slice_shape = dim4_t{1, 1, stack_shape[2], stack_shape[3]};
    const auto slice_shape_padded = dim4_t{1, 1, slice_shape[2] * 2, slice_shape[3] * 2};
    const auto slice_center = float2_t{stack_shape[2] / 2, stack_shape[3] / 2};
    const auto slice_center_padded = float2_t{dim2_t(slice_shape_padded.get(2)) / 2};

    // Prepare the input slice: center and taper edges.
    Array<float> slice = memory::empty<float>(slice_shape);
    file.readSlice(slice, 20);
    math::ewise(slice, math::mean(slice), slice, math::minus_t{});
    signal::rectangle(slice, slice, slice_center, slice_center - 100, 100);
    io::save(slice, output_dir / "test0_slice.mrc");

    // We will insert at a given orientation, and extract at another one.
    const float33_t fwd_matrix0 = geometry::euler2matrix(math::deg2rad(float3_t{0, 0, 0}), "ZYX", false);
    const float33_t fwd_matrix1 = geometry::euler2matrix(math::deg2rad(float3_t{0, 12, 0}), "ZYX", false);

    {
        // 1: Oversample during the projection.
        Array slice_fft = fft::r2c(slice);
        signal::fft::lowpass<fft::H2H>(slice_fft, slice_fft, slice_shape, 0.5f, 0.05f);

        const dim4_t target_shape{1, 1024, slice_shape_padded[2], slice_shape_padded[3]};
        const dim4_t grid_shape{1, 1024, slice_shape_padded[2], slice_shape_padded[3]};
        ProjectorRasterize projector(grid_shape, slice_shape, target_shape);
        projector.backward(slice_fft, fwd_matrix0, -slice_center);
        projector.forward(slice_fft, fwd_matrix1, slice_center);

        Array slice_fft_ps = memory::empty<float>(slice_shape.fft());
        math::ewise(slice_fft, slice_fft_ps, math::abs_one_log_t{});
        io::save(std::move(slice_fft_ps), output_dir / "test0_slice_fft_projected.mrc");

        io::save(fft::c2r(slice_fft, slice_shape), output_dir / "test0_slice_projected.mrc");
    }

    {
        // 2: Zero pad the real-space slice and then project without any oversampling.
        Array slice_padded = memory::resize(slice, slice_shape_padded);

        Array slice_padded_fft = fft::r2c(slice_padded);
        signal::fft::lowpass<fft::H2H>(slice_padded_fft, slice_padded_fft, slice_shape_padded, 0.5f, 0.05f);

        const dim4_t target_shape{1, 1024, slice_shape_padded[2], slice_shape_padded[3]};
        const dim4_t grid_shape{1, 1024, slice_shape_padded[2], slice_shape_padded[3]};
        ProjectorRasterize projector(grid_shape, slice_shape_padded, target_shape);
        projector.backward(slice_padded_fft, fwd_matrix0, -slice_center_padded);
        projector.forward(slice_padded_fft, fwd_matrix1, slice_center_padded);

        Array<float> slice_fft_ps(slice_shape_padded.fft());
        math::ewise(slice_padded_fft, slice_fft_ps, math::abs_one_log_t{});
        io::save(std::move(slice_fft_ps), output_dir / "test1_slice_fft_projected.mrc");

        fft::c2r(slice_padded_fft, slice_padded);
        Array output = memory::resize(slice_padded, slice_shape);
        io::save(output, output_dir / "test1_slice_projected.mrc");
    }

    {
        // 3: Use the new projection, skipping the grid. With zero-padding.
        Array slice_padded = memory::resize(slice, slice_shape_padded);
        Array slice_padded_fft = fft::r2c(slice_padded);
        signal::fft::lowpass<fft::H2H>(slice_padded_fft, slice_padded_fft, slice_shape_padded, 0.5f, 0.05f);

        noa::signal::fft::shift2D<fft::H2H>(
                slice_padded_fft, slice_padded_fft, slice_shape_padded, -slice_center_padded);
        noa::fft::remap(fft::H2HC, slice_padded_fft, slice_padded_fft, slice_shape_padded);

        auto [output_slice_padded, output_slice_padded_fft] = fft::zeros<float>(slice_shape_padded);
        Array slice_padded_fft_weights = memory::zeros<float>(slice_shape_padded.fft());
        noa::geometry::fft::extract3D<fft::HC2H>(
                slice_padded_fft, slice_shape_padded,
                output_slice_padded_fft, slice_shape_padded,
                float22_t{}, math::inverse(fwd_matrix0),
                float22_t{}, fwd_matrix1, 0.03f, 0.5f);
        noa::geometry::fft::extract3D<fft::HC2H>(
                memory::ones<float>(slice_shape_padded.fft()), slice_shape_padded,
                slice_padded_fft_weights, slice_shape_padded,
                float22_t{}, math::inverse(fwd_matrix0),
                float22_t{}, fwd_matrix1, 0.03f, 0.5f);

        signal::fft::shift2D<fft::H2H>(
                output_slice_padded_fft, output_slice_padded_fft, slice_shape_padded, slice_center_padded);
        math::ewise(output_slice_padded_fft, slice_padded_fft_weights, 1e-3f,
                    output_slice_padded_fft, math::divide_epsilon_t{});

        Array slice_fft_ps = memory::empty<float>(slice_shape_padded.fft());
        math::ewise(output_slice_padded_fft, slice_fft_ps, math::abs_one_log_t{});
        io::save(std::move(slice_fft_ps), output_dir / "test2_slice_fft_projected.mrc");

        fft::c2r(output_slice_padded_fft, output_slice_padded);
        Array output = memory::resize(output_slice_padded, slice_shape);
        io::save(output, output_dir / "test2_slice_projected.mrc");
    }

    // If the extract where at the same location we inserted the slice, oversampling using the projector and
    // oversampling using zero-padding look basically identical. When we start to extract with a tilt difference,
    // the zero-padding version look slightly less distorted perpendicular to the tilt axis (it's like if there
    // was a stretching factor on the other one, but I think it just that we have more info with the zero-padding).
    // HOWEVER, the oversampling on the projector is still better than nothing. Without zero-padding and without
    // oversampling, we do have a lot of aliasing at the edges. This is removed with either zero-padding or
    // oversampling on the projector.
    // I think this difference when extracting at a higher tilt is because we have very little information,
    // but in a SPA scenario, where the grid is filled with thousands of particles, I don't think that there's
    // a big difference between zero-padding and oversampling in the projector.
    // Zero-padding by 2 gives the FFT of the original image but interspaced with 0: F0, 0, F1, 0, F2, etc. where
    // FX are the Fourier component of the original image. This is what oversampling in the projector does though,
    // it normalizes the frequencies, does the transformation and scales the transformed frequencies back to the
    // oversampled size.
    // test2 is a bit better than test0, but test1 (zero-padding) still looks better.

//    {
//        // Transform
//        const float22_t matrix = geometry::rotate(math::deg2rad(45.f));
//        Array<cfloat_t> tmp(slice_shape_padded.fft());
//        signal::fft::shift2D<fft::H2HC>(slice_padded_fft, tmp, slice_shape_padded, -center_padded, 1.f);
//        geometry::fft::transform2D<fft::HC2H>(tmp, slice_padded_fft, slice_shape_padded, math::inverse(matrix), center_padded);
//
//        signal::fft::shift2D<fft::H2HC>(slice_padded_fft, tmp, slice_shape_padded, -center_padded, 1.f);
//        geometry::fft::transform2D<fft::HC2H>(tmp, slice_padded_fft, slice_shape_padded, matrix, center_padded);
//
//        fft::c2r(slice_padded_fft, slice_padded);
//        io::save(slice_padded, output_dir / "test0_slice_padded_transform2D.mrc");
//    }
}

TEST_CASE("unified::geometry::fft, project test5", "[.]") {
    const path_t output_dir = test::NOA_DATA_PATH / "geometry" / "fft" / "test5";
    const path_t input_path = output_dir / "tilt1_cropped_preali_bin2_mrcfile.mrc";

    // The goal here is to back-project the views at a given angle.
    const float yaw = 0.f;
    const dim_t target_index = 35;
    const float tilt_start = -60.f;
    const float tilt_increment = 3.f;
    const float target_tilt = tilt_start + target_index * tilt_increment;
    const float33_t fwd_matrix_target =
            geometry::euler2matrix(math::deg2rad(float3_t{0, target_tilt, 0}), "ZYX", false);
    fmt::print("target tilt: {}\n", target_tilt);

    // Prepare the input slice: center and taper edges.
    Array<float> stack = io::load<float>(input_path);
    stack = stack.reshape({stack.shape()[1], 1, stack.shape()[2], stack.shape()[3]});
    const auto slice_count = stack.shape()[0];
    const auto slice_shape = dim4_t{1, 1, stack.shape()[2], stack.shape()[3]};
    const auto slice_center = float2_t{stack.shape()[2] / 2, stack.shape()[3] / 2};
    const auto slice_size_padded = std::max(slice_shape[2], slice_shape[3]) * 2;
    const auto slice_shape_padded = dim4_t{1, 1, slice_size_padded, slice_size_padded};
    const auto slice_center_padded = float2_t{dim2_t(slice_shape_padded.get(2)) / 2};

    math::ewise(stack, math::mean(stack), stack, math::minus_t{});
    signal::rectangle(stack, stack, slice_center, slice_center - 50, 50);
    io::save(stack.subregion(target_index), output_dir / "test5_target.mrc");

    const dim4_t grid_shape{1, slice_size_padded, slice_size_padded, slice_size_padded};
    ProjectorInterp projector(0.0015f, grid_shape, slice_shape_padded);

    Array extract_slice_padded_fft = memory::zeros<cfloat_t>(slice_shape_padded.fft());
    Array extract_weight_padded_fft = memory::zeros<float>(slice_shape_padded.fft());
    Array extract_weights_padded_fft_ones = memory::ones<float>(slice_shape_padded.fft());

    float reference_tilt = tilt_start;
    for (dim_t i = 0; i < slice_count; ++i) {
        if (i == 0 || i == target_index) {
            reference_tilt += 3.f;
            continue;
        }

        fmt::print("back-project tilt: {}\n", reference_tilt);

        Array slice = stack.subregion(i);
        Array slice_padded = memory::resize(slice, slice_shape_padded);
        Array slice_padded_fft = fft::r2c(slice_padded);
        signal::fft::lowpass<fft::H2H>(slice_padded_fft, slice_padded_fft, slice_shape_padded, 0.5f, 0.05f);

        const float33_t fwd_matrix_reference =
                geometry::euler2matrix(math::deg2rad(float3_t{yaw, reference_tilt, 0}), "ZYX", false);
        projector.backward(slice_padded_fft, fwd_matrix_reference, -slice_center_padded);

        // Now with the new Fourier extraction.
        // The projector already shifts and centers the slice, so just insert.
        geometry::fft::extract3D<fft::HC2H>(
                slice_padded_fft, slice_shape_padded,
                extract_slice_padded_fft, slice_shape_padded,
                float22_t{}, fwd_matrix_reference.transpose(),
                float22_t{}, fwd_matrix_target,
                0.0015f, 0.5f);
        geometry::fft::extract3D<fft::HC2H>(
                extract_weights_padded_fft_ones, slice_shape_padded,
                extract_weight_padded_fft, slice_shape_padded,
                float22_t{}, fwd_matrix_reference.transpose(),
                float22_t{}, fwd_matrix_target,
                0.0015f, 0.5f);

        reference_tilt += 3.f;
    }

    // With the grid:
    io::save(projector.m_grid_weights_fft, output_dir / "test5_grid.mrc");

    Array reference_padded_fft = memory::empty<cfloat_t>(slice_shape_padded.fft());
    projector.forward(reference_padded_fft, fwd_matrix_target, slice_center_padded);

    Array tmp = memory::empty<float>(slice_shape_padded.fft());
    math::ewise(reference_padded_fft, tmp, math::abs_one_log_t{});
    io::save(tmp, output_dir / "test5_grid_reference_fft.mrc");

    Array reference_padded = fft::c2r(reference_padded_fft, slice_shape_padded);
    Array reference = memory::resize(reference_padded, slice_shape);
    io::save(reference, output_dir / "test5_grid_reference.mrc");

    // Without the grid:
    signal::fft::shift2D<fft::H2H>(
            extract_slice_padded_fft, extract_slice_padded_fft, slice_shape_padded, slice_center_padded);
    math::ewise(extract_slice_padded_fft, extract_weight_padded_fft, 1e-3f, extract_slice_padded_fft,
                math::divide_epsilon_t{});

    math::ewise(extract_slice_padded_fft, tmp, math::abs_one_log_t{});
    io::save(tmp, output_dir / "test5_nogrid_reference_fft.mrc");

    reference_padded = fft::c2r(extract_slice_padded_fft, slice_shape_padded);
    reference = memory::resize(reference_padded, slice_shape);
    io::save(reference, output_dir / "test5_nogrid_reference.mrc");

    // Notes:
    //  1.  The version with the grid includes a bit more signal, probably due to the grid sampling itself.
    //      As a result, the slice_z_radius should be a bit higher on the no-grid version.
    //  2.  The difference between neighbouring views is much bigger as the tilt increases. This is obvious,
    //      but we could maybe do something about it.
    //  3.  Increasing the slice_z_radius too much produces worst results. Signal from the other views starts
    //      to leak and the projected reference isn't correct.
}

//        {
//            // Cosine stretch
//            const float2_t cos_factor{1, math::cos(reference_tilt) / math::cos(target_tilt)};
//            const float33_t stretch_target_to_reference{
//                    noa::geometry::translate(slice_center) *
//                    float33_t{noa::geometry::scale(cos_factor)} *
//                    noa::geometry::translate(-slice_center)
//            };
//
//            // After this point, the target should "overlap" with the reference.
//            Array slice_cos_stretched = memory::like(slice);
//            noa::geometry::transform2D(slice, slice_cos_stretched, math::inverse(stretch_target_to_reference));
//            io::save(slice_cos_stretched, output_dir / "test5_slice_cos.mrc");
//        }
