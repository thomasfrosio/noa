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
}

TEST_CASE("unified::geometry::fft, bwd and fwd projection", "[.]") {
    using namespace ::noa;
    Timer timer;

    const size_t count = 5000;
    const path_t input_path = "/home/thomas/Projects/data/ribo/emd_14454.mrc";
    const path_t output_dir = test::NOA_DATA_PATH / "geometry" / "fft" / "reconstruction";
    const Array<float33_t> rotations = generateRotations_(count);

    io::ImageFile file(input_path, io::READ);
    const size4_t volume_shape = file.shape();
    fmt::print("problem size: count={}, shape={}\n", count, volume_shape);

    const Device device("cpu");
    const ArrayOption options(device, Allocator::DEFAULT_ASYNC);

    timer.start();
    Array<cfloat_t> volume_fft(volume_shape.fft(), options);
    Array<float> volume = fft::alias(volume_fft, volume_shape);
    file.read(volume);
    volume.eval();
    fmt::print("loading file: {}ms\n", timer.elapsed());

    // Prepare the volume for the forward projection:
    timer.start();
    fft::r2c(volume, volume_fft);
    Array volume_fft_centered = memory::like(volume_fft);
    fft::remap(fft::H2HC, volume_fft.release(), volume_fft_centered, volume_shape);
    const float3_t volume_center = float3_t{volume_shape.get(1)} / 2;
    signal::fft::shift3D<fft::HC2HC>(volume_fft_centered, volume_fft_centered, volume_shape, -volume_center);

    // Extract the slices:
    const size4_t slices_shape{count, 1, volume_shape[2], volume_shape[3]};
    Array<cfloat_t> slices_fft(slices_shape.fft(), options);
    geometry::fft::extract3D<fft::HC2H>(volume_fft_centered.release(), volume_shape,
                                        slices_fft, slices_shape, float22_t{}, rotations);

    // Post-process the slices:
    const float2_t slices_center(volume_center.get(1));
    signal::fft::shift2D<fft::H2H>(slices_fft, slices_fft, slices_shape, slices_center);
    Array slices = fft::alias(slices_fft, slices_shape);
    fft::c2r(slices_fft, slices);
    slices.eval();
    fmt::print("forward projection: {}ms\n", timer.elapsed());

    io::save(slices, output_dir / "extracted_slices.mrc");

    // Backward projection:
    timer.start();
    fft::r2c(slices, slices_fft);
    signal::fft::shift2D<fft::H2H>(slices_fft, slices_fft, slices_shape, -slices_center);
    volume_fft = memory::zeros<cfloat_t>(volume_shape.fft(), options);
    geometry::fft::insert3D<fft::H2H>(slices_fft, slices_shape,
                                      volume_fft, volume_shape,
                                      float22_t{}, rotations);
    signal::fft::shift3D<fft::H2H>(volume_fft, volume_fft, volume_shape, volume_center);

    // Backward projection on the weights:
    Array<float> weights_slices_fft = memory::ones<float>({1, 1, slices_shape[2], slices_shape[3] / 2 + 1}, options);
    weights_slices_fft = indexing::broadcast(weights_slices_fft, slices_shape.fft());
    Array<float> weights_volume_fft = memory::zeros<float>(volume_shape.fft(), options);
    geometry::fft::insert3D<fft::H2H>(weights_slices_fft.release(), slices_shape,
                                      weights_volume_fft, volume_shape,
                                      float22_t{}, rotations);

    // Weighting:
    weights_volume_fft += 1e-3f;
    volume_fft /= weights_volume_fft.release();

    volume = fft::alias(volume_fft, volume_shape);
    fft::c2r(volume_fft, volume);
    volume.eval();
    fmt::print("backward projection: {}ms\n", timer.elapsed());

    io::save(volume, output_dir / "reconstruction.mrc");
}

TEST_CASE("unified::geometry::insert3D, test for thickness", "[.]") {
    const path_t output_dir = test::NOA_DATA_PATH / "geometry" / "fft" / "reconstruction";

    const dim4_t target_shape{1, 512, 512, 512};
    const dim4_t grid_shape{1, 256, 512, 512};
    const dim4_t slice_shape{2, 1, 512, 512};

    Array slice = memory::ones<float>(slice_shape.fft());
    Array weights = memory::zeros<float>(grid_shape.fft());

    Array<float33_t> rotations(2);
    rotations[0] = geometry::euler2matrix(math::deg2rad(float3_t{0, 57, 0}), "ZYX", false);
    rotations[1] = geometry::euler2matrix(math::deg2rad(float3_t{0, 60, 0}), "ZYX", false);

    noa::geometry::fft::insert3D<fft::HC2HC>(
            slice, slice_shape,
            weights, grid_shape,
            float22_t{}, rotations[1], 0.5f, target_shape);

    noa::geometry::fft::extract3D<fft::HC2HC>(
            weights, grid_shape,
            slice, slice_shape,
            float22_t{}, rotations[0], 0.5f, target_shape);

    io::save(slice, output_dir / "test_slice_thickness.mrc");
}

namespace {
    class Projector {
    public:
        Projector(dim4_t grid_shape,
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
                      float33_t rotation,
                      float2_t shift,
                      float22_t scaling = {},
                      float cutoff = 0.5f) {
            noa::signal::fft::shift2D<fft::H2H>(
                    slice_fft, slice_fft, m_slice_shape, shift);
            noa::geometry::fft::insert3D<fft::H2HC>(
                    slice_fft, m_slice_shape,
                    m_grid_data_fft, m_grid_shape,
                    scaling, rotation, cutoff, m_target_shape);
            noa::geometry::fft::insert3D<fft::H2HC>(
                    m_weights_ones_fft, m_slice_shape,
                    m_grid_weights_fft, m_grid_shape,
                    scaling, rotation, cutoff, m_target_shape);
        }

        void forward(Array<cfloat_t>& slice_fft,
                     float33_t rotation,
                     float2_t shift,
                     float22_t scaling = {},
                     float cutoff = 0.5f) {
            noa::geometry::fft::extract3D<fft::HC2H>(
                    m_grid_data_fft, m_grid_shape,
                    slice_fft, m_slice_shape,
                    scaling, rotation, cutoff, m_target_shape);
            noa::geometry::fft::extract3D<fft::HC2H>(
                    m_grid_weights_fft, m_grid_shape,
                    m_weights_extract_fft, m_slice_shape,
                    scaling, rotation, cutoff, m_target_shape);
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
}

TEST_CASE("unified::geometry::fft, transform with projections", "[.]") {
    const path_t input_path = "/home/thomas/Projects/data/ribo/tilt1/tilt1_cropped.mrc";
    const path_t output_dir = test::NOA_DATA_PATH / "geometry" / "fft" / "reconstruction";

    io::ImageFile file(input_path, io::READ);
    const dim4_t stack_shape = file.shape();
    const dim4_t slice_shape{1, 1, stack_shape[2], stack_shape[3]};
    const float2_t center{stack_shape[2] / 2, stack_shape[3] / 2};

    Array<float> slice = memory::empty<float>(slice_shape);
    file.readSlice(slice, 20);
    slice -= math::mean(slice);
    signal::rectangle(slice, slice, center, center - 100, 100);
    io::save(slice, output_dir / "test0_slice.mrc");

    const dim4_t slice_shape_padded{1, 1, slice_shape[2] * 2, slice_shape[3] * 2};
    const float2_t center_padded{dim2_t(slice_shape_padded.get(2)) / 2};

    const float33_t rotm0 = geometry::euler2matrix(math::deg2rad(float3_t{-176, 20, 0}), "ZYX", false);
    const float33_t rotm1 = geometry::euler2matrix(math::deg2rad(float3_t{-176, 23, 0}), "ZYX", false);

    {
        // Project the original slice with an oversampling.
        Array slice_fft = fft::r2c(slice);
        signal::fft::lowpass<fft::H2H>(slice_fft, slice_fft, slice_shape, 0.5f, 0.05f);

        // Projector
        const dim4_t target_shape{1, 1024, slice_shape_padded[2], slice_shape_padded[3]};
        const dim4_t grid_shape{1, 256, slice_shape_padded[2], slice_shape_padded[3]};
        Projector projector(grid_shape, slice_shape, target_shape);

        projector.backward(slice_fft, rotm0, -center);
//        io::save(math::real(projector.m_grid_data_fft), output_dir / "test0_grid_data.mrc");
        projector.forward(slice_fft, rotm1, center);

        Array<float> tmp(slice_shape.fft());
        math::ewise(slice_fft, tmp, math::abs_one_log_t{});
        io::save(std::move(tmp), output_dir / "test0_slice_fft_projected.mrc");

        io::save(fft::c2r(slice_fft, slice_shape), output_dir / "test0_slice_projected.mrc");
    }

    {
        // Project the original slice with an oversampling, but forward project with zero-padding.
        Array slice_fft = fft::r2c(slice);
        signal::fft::lowpass<fft::H2H>(slice_fft, slice_fft, slice_shape, 0.5f, 0.05f);

        // Projector
        const dim4_t target_shape{1, 1024, slice_shape_padded[2], slice_shape_padded[3]};
        const dim4_t grid_shape{1, 256, slice_shape_padded[2], slice_shape_padded[3]};
        Projector projector(grid_shape, slice_shape, target_shape);

        projector.backward(slice_fft, rotm0, -center);

        Array<cfloat_t> slice_padded_fft(slice_shape_padded.fft());
        Array<float> weights_padded_fft(slice_shape_padded.fft());
        noa::geometry::fft::extract3D<fft::HC2H>(
                projector.m_grid_data_fft, projector.m_grid_shape,
                slice_padded_fft, slice_shape_padded,
                float22_t{}, rotm1, 0.5f, projector.m_target_shape);
        noa::geometry::fft::extract3D<fft::HC2H>(
                projector.m_grid_weights_fft, projector.m_grid_shape,
                weights_padded_fft, slice_shape_padded,
                float22_t{}, rotm1, 0.5f, projector.m_target_shape);
        signal::fft::shift2D<fft::H2H>(slice_padded_fft, slice_padded_fft, slice_shape_padded, center_padded);
        math::ewise(slice_padded_fft, weights_padded_fft, 1e-3f, slice_padded_fft,
                    math::divide_epsilon_t{});

        Array<float> tmp(slice_shape_padded.fft());
        math::ewise(slice_padded_fft, tmp, math::abs_one_log_t{});
        io::save(std::move(tmp), output_dir / "test2_slice_fft_projected.mrc");

        Array slice_padded = fft::c2r(slice_padded_fft, slice_shape_padded);
        Array output = memory::resize(slice_padded, slice_shape);
        io::save(output, output_dir / "test2_slice_projected.mrc");
    }

    {
        // Zero pad the real-space slice and then project without any oversampling.
        Array slice_padded = memory::resize(slice, slice_shape_padded);
        io::save(slice_padded, output_dir / "test1_slice_padded.mrc");

        Array slice_padded_fft = fft::r2c(slice_padded);
        signal::fft::lowpass<fft::H2H>(slice_padded_fft, slice_padded_fft, slice_shape_padded, 0.5f, 0.05f);

        const dim4_t target_shape{1, 1024, slice_shape_padded[2], slice_shape_padded[3]};
        const dim4_t grid_shape{1, 256, slice_shape_padded[2], slice_shape_padded[3]};
        Projector projector(grid_shape, slice_shape_padded, target_shape);

        projector.backward(slice_padded_fft, rotm0, -center_padded);
//        io::save(math::real(projector.m_grid_data_fft), output_dir / "test1_grid_data.mrc");
        projector.forward(slice_padded_fft, rotm1, center_padded);

        Array<float> tmp(slice_shape_padded.fft());
        math::ewise(slice_padded_fft, tmp, math::abs_one_log_t{});
        io::save(std::move(tmp), output_dir / "test1_slice_fft_projected.mrc");

        fft::c2r(slice_padded_fft, slice_padded);
        Array output = memory::resize(slice_padded, slice_shape);
        io::save(output, output_dir / "test1_slice_projected.mrc");
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
//        const float22_t rotm = geometry::rotate(math::deg2rad(45.f));
//        Array<cfloat_t> tmp(slice_shape_padded.fft());
//        signal::fft::shift2D<fft::H2HC>(slice_padded_fft, tmp, slice_shape_padded, -center_padded, 1.f);
//        geometry::fft::transform2D<fft::HC2H>(tmp, slice_padded_fft, slice_shape_padded, math::inverse(rotm), center_padded);
//
//        signal::fft::shift2D<fft::H2HC>(slice_padded_fft, tmp, slice_shape_padded, -center_padded, 1.f);
//        geometry::fft::transform2D<fft::HC2H>(tmp, slice_padded_fft, slice_shape_padded, rotm, center_padded);
//
//        fft::c2r(slice_padded_fft, slice_padded);
//        io::save(slice_padded, output_dir / "test0_slice_padded_transform2D.mrc");
//    }
}
