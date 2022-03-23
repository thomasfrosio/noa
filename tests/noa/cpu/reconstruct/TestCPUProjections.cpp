#include <noa/common/geometry/Euler.h>
#include <noa/common/geometry/Transform.h>
#include <noa/common/io/ImageFile.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>
#include <noa/cpu/reconstruct/ProjectBackward.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

// WORK IN PROGRESS:
TEST_CASE("cpu::reconstruct::projectBackward", "[noa][cpu][reconstruct]") {
    const size4_t slices_shape{1, 1, 512, 512};
    const size4_t grid_shape{1, 512, 512, 512};

    cpu::memory::PtrHost<cfloat_t> slices(slices_shape.fft().elements());
    cpu::memory::PtrHost<cfloat_t> grid(grid_shape.fft().elements());
    cpu::memory::PtrHost<float> slices_weights(slices.size());
    cpu::memory::PtrHost<float> grid_weights(grid.size());

    cpu::memory::set(slices.begin(), slices.end(), cfloat_t{1, 1});
    cpu::memory::set(grid.begin(), grid.end(), cfloat_t{});
    cpu::memory::set(slices_weights.begin(), slices_weights.end(), 2.f);
    cpu::memory::set(grid_weights.begin(), grid_weights.end(), 0.f);

    cpu::memory::PtrHost<float2_t> shifts{slices_shape[0]};
    cpu::memory::PtrHost<float22_t> scales{slices_shape[0]};
    cpu::memory::PtrHost<float33_t> rotations(slices_shape[0]);

    for (uint i = 0; i < slices_shape[0]; ++i) {
        scales[i] = geometry::scale(float2_t{1, 1});
        rotations[i] = geometry::euler2matrix(float3_t{1.0, 0, 0});
    }

    const float cutoff = 0.5f;
    const float wavelength = 0.01968761530923358f; // A
    const float2_t pixel_size{1, 1}; // A/pix
    const float2_t ews_radius = pixel_size / wavelength; // 1/pix

    cpu::Stream stream(cpu::Stream::DEFAULT);
    cpu::reconstruct::fft::insert<fft::HC2HC>(View<const cfloat_t>{slices.get(), slices_shape},
                                              View<float>{slices_weights.get(), slices_shape},
                                              View<cfloat_t>{grid.get(), grid_shape},
                                              View<float>{grid_weights.get(), grid_shape},
                                              shifts.get(), scales.get(), rotations.get(),
                                              cutoff, ews_radius, stream);
    cpu::reconstruct::fft::insert<fft::HC2HC>(View<const cfloat_t>{slices.get(), slices_shape},
                                              View<float>{slices_weights.get(), slices_shape},
                                              View<cfloat_t>{grid.get(), grid_shape},
                                              View<float>{grid_weights.get(), grid_shape},
                                              shifts.get(), scales.get(), rotations.get(),
                                              cutoff, -ews_radius, stream);

    io::ImageFile file(test::NOA_DATA_PATH / "test.mrc", io::WRITE);
    file.shape(shapeFFT(vol_logical_shape));
    file.dataType(io::FLOAT32);
    file.writeAll(volume_weights.get(), false);
}
