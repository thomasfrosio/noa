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

    const Device device("gpu");
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
