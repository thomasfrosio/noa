#include <noa/common/geometry/Euler.h>
#include <noa/common/geometry/Transform.h>
#include <noa/common/io/MRCFile.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>
#include <noa/cpu/geometry/fft/Project.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::geometry::fft::insert3D", "[.]") {
    const size4_t slices_shape{1, 1, 256, 256};
    const size4_t grid_shape{1, 256, 256, 256};
    const size4_t target_shape{0, 256, 256, 256};
    const size4_t slices_stride = slices_shape.fft().strides();
    const size4_t grid_stride = grid_shape.fft().strides();

    cpu::memory::PtrHost<float> slices(slices_shape.fft().elements());
    cpu::memory::PtrHost<float> grid(grid_shape.fft().elements());

    cpu::memory::set(slices.begin(), slices.end(), 1.f);
    cpu::memory::set(grid.begin(), grid.end(), 0.f);

    cpu::memory::PtrHost<float22_t> scaling_factors(slices_shape[0]);
    cpu::memory::PtrHost<float33_t> rotations(slices_shape[0]);

    for (uint i = 0; i < slices_shape[0]; ++i) {
        scaling_factors[i] = geometry::scale(float2_t{1, 1});
        rotations[i] = geometry::euler2matrix(math::deg2rad(float3_t{176, 20, 0}), "ZYX", false);
    }

    const bool do_ews = false;
    const float cutoff = 0.5f;
    const float wavelength = 0.01968761530923358f; // A
    const float2_t pixel_size{1, 1}; // A/pix
    const float2_t ews_radius = do_ews ? pixel_size / wavelength : float2_t{}; // 1/pix

    cpu::Stream stream(cpu::Stream::DEFAULT);
    stream.threads(1);

    for (int i = 0; i < (1 + do_ews) ; ++i) {
        const float sign = i == 0 ? 1 : -1;
        cpu::geometry::fft::insert3D<fft::HC2HC>(
                slices.share(), slices_stride, slices_shape,
                grid.share(), grid_stride, grid_shape,
                scaling_factors.share(), rotations.share(),
                cutoff, target_shape, sign * ews_radius, stream);

        cpu::geometry::fft::extract3D<fft::HC2HC>(
                grid.share(), grid_stride, grid_shape,
                slices.share(), slices_stride, slices_shape,
                scaling_factors.share(), rotations.share(),
                cutoff, target_shape, sign * ews_radius, stream);
    }

    stream.synchronize();
    io::MRCFile file(test::NOA_DATA_PATH / "geometry" / "fft" / "test_insert3D.mrc", io::WRITE);
    file.shape(grid_shape.fft());
    file.writeAll(grid.get(), false);

    file.open(test::NOA_DATA_PATH / "geometry" / "fft" / "test_extract3D.mrc", io::WRITE);
    file.shape(slices_shape.fft());
    file.writeAll(slices.get(), false);
}
