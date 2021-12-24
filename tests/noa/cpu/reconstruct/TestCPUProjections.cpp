#include <noa/common/transform/Euler.h>
#include <noa/common/io/ImageFile.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>
#include <noa/cpu/reconstruct/ProjectBackward.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

// WORK IN PROGRESS:
TEST_CASE("cpu::reconstruct::projectBackward", "[noa][cpu][reconstruct]") {
    size_t proj_dim = 1024;
    size_t volume_dim = 1024;
    uint proj_count = 1;
    size3_t proj_logical_shape(proj_dim, proj_dim, proj_count);
    size3_t vol_logical_shape(volume_dim);

    cpu::memory::PtrHost<cfloat_t> projections(noa::elementsFFT(proj_logical_shape));
    cpu::memory::PtrHost<cfloat_t> volume(noa::elementsFFT(vol_logical_shape));
    cpu::memory::PtrHost<float> projections_weights(projections.size());
    cpu::memory::PtrHost<float> volume_weights(volume.size());

    cpu::memory::set(projections.begin(), projections.end(), cfloat_t{1, 1});
    cpu::memory::set(volume.begin(), volume.end(), cfloat_t{});
    cpu::memory::set(projections_weights.begin(), projections_weights.end(), 2.f);
    cpu::memory::set(volume_weights.begin(), volume_weights.end(), 0.f);

    cpu::memory::PtrHost<float2_t> shifts(proj_count);
    cpu::memory::PtrHost<float3_t> scales(proj_count);
    for (auto& scale: scales)
        scale = {1, 1, 0};
    cpu::memory::PtrHost<float33_t> rotations(proj_count);
    for (uint i = 0; i < proj_count; ++i)
        rotations[i] = transform::toMatrix(float3_t{1.0, 0, 0});
    float max_frequency = 0.5f;

    float wavelength = 0.01968761530923358f; // A
    float pixel_size = 1; // A/pix
    float ews_radius = pixel_size / wavelength; // 1/pix

    cpu::reconstruct::projectBackward(projections.get(), projections_weights.get(), proj_dim,
                                      volume.get(), volume_weights.get(), volume_dim,
                                      shifts.get(), scales.get(), rotations.get(), proj_count,
                                      max_frequency, ews_radius);
    cpu::reconstruct::projectBackward(projections.get(), projections_weights.get(), proj_dim,
                                      volume.get(), volume_weights.get(), volume_dim,
                                      shifts.get(), scales.get(), rotations.get(), proj_count,
                                      max_frequency, -ews_radius);

    io::ImageFile file(test::PATH_NOA_DATA / "test.mrc", io::WRITE);
    file.shape(shapeFFT(vol_logical_shape));
    file.dataType(io::FLOAT32);
    file.writeAll(volume_weights.get(), false);
}
