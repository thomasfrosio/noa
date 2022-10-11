#include <noa/common/geometry/Euler.h>
#include <noa/common/geometry/Transform.h>
#include <noa/common/io/MRCFile.h>

#include <noa/gpu/cuda/memory/PtrManaged.h>
#include <noa/gpu/cuda/memory/Set.h>
#include <noa/gpu/cuda/geometry/fft/Project.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::geometry::fft::insert3D", "[.]") {
    const size4_t slices_shape{1, 1, 256, 256};
    const size4_t grid_shape{1, 256, 256, 256};
    const size4_t target_shape{0, 256, 256, 256};
    const size4_t slices_stride = slices_shape.fft().strides();
    const size4_t grid_stride = grid_shape.fft().strides();

    cuda::Stream stream;

    cuda::memory::PtrManaged<float> slices(slices_shape.fft().elements(), stream);
    cuda::memory::PtrManaged<float> grid(grid_shape.fft().elements(), stream);

    cuda::memory::set(slices.share(), slices.elements(), 10.f, stream);
    cuda::memory::set(grid.share(), grid.elements(), 0.f, stream);

    cuda::memory::PtrManaged<float22_t> scaling_factors(slices_shape[0], stream);
    cuda::memory::PtrManaged<float33_t> rotations(slices_shape[0], stream);

    for (uint i = 0; i < slices_shape[0]; ++i) {
        scaling_factors[i] = geometry::scale(float2_t{1, 1});
        rotations[i] = geometry::euler2matrix(float3_t{0.54, 0, 0});
    }

    const bool do_ews = false;
    const float cutoff = 0.5f;
    const float wavelength = 0.01968761530923358f; // A
    const float2_t pixel_size{1, 1}; // A/pix
    const float2_t ews_radius = do_ews ? pixel_size / wavelength : float2_t{}; // 1/pix

    for (int i = 0; i < (1 + do_ews) ; ++i) {
        const float sign = i == 0 ? 1 : -1;
        cuda::geometry::fft::insert3D<fft::HC2HC>(
                slices.share(), slices_stride, slices_shape,
                grid.share(), grid_stride, grid_shape,
                scaling_factors.share(), rotations.share(),
                cutoff, target_shape, sign * ews_radius, stream);

        cuda::geometry::fft::extract3D<fft::HC2HC>(
                grid.share(), grid_stride, grid_shape,
                slices.share(), slices_stride, slices_shape,
                scaling_factors.share(), rotations.share(),
                cutoff, target_shape, sign * ews_radius, false, stream);
    }

    stream.synchronize();
    io::MRCFile file(test::NOA_DATA_PATH / "geometry" / "fft" / "test_insert3D_cuda.mrc", io::WRITE);
    file.shape(grid_shape.fft());
    file.writeAll(grid.get(), false);

    file.open(test::NOA_DATA_PATH / "geometry" / "fft" / "test_extract3D_cuda.mrc", io::WRITE);
    file.shape(slices_shape.fft());
    file.writeAll(slices.get(), false);
}
