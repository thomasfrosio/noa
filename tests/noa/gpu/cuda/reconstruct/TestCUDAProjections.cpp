#include <noa/common/transform/Euler.h>
#include <noa/common/io/ImageFile.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>
#include <noa/cpu/reconstruct/ProjectBackward.h>

#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/memory/Set.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/reconstruct/ProjectBackward.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::reconstruct::projectBackward", "[noa][cuda][reconstruct]") {
    size_t proj_dim = 512;
    size_t volume_dim = 1024;
    uint proj_count = 1;
    size3_t proj_logical_shape(proj_dim, proj_dim, proj_count);
    size3_t vol_logical_shape(volume_dim);

    cuda::memory::PtrDevice<float> d_projections_weights(noa::elementsFFT(proj_logical_shape));
    cuda::memory::PtrDevice<float> d_volume_weights(noa::elementsFFT(vol_logical_shape));
    cuda::Stream stream;

    cuda::memory::set(d_projections_weights.get(), d_projections_weights.size(), 1.f, stream);
    cuda::memory::set(d_volume_weights.get(), d_volume_weights.size(), 0.f, stream);

    cpu::memory::PtrHost<float33_t> rotations(proj_count);
    for (uint i = 0; i < proj_count; ++i)
        rotations[i] = transform::toMatrix(float3_t{30, 0, 0});
    float max_frequency = 0.5f;
    float ews_radius = 0.f;

    cuda::reconstruct::projectBackward<false, true, float>(nullptr, d_projections_weights.get(), proj_dim/2+1, proj_dim,
                                                           nullptr, d_volume_weights.get(), volume_dim/2+1, volume_dim,
                                                           nullptr, nullptr, rotations.get(), proj_count,
                                                           max_frequency, ews_radius, stream);

    cpu::memory::PtrHost<float> volume_weights(d_volume_weights.size());
    cuda::memory::copy(d_volume_weights.get(), volume_weights.get(), d_volume_weights.size(), stream);
    stream.synchronize();

    io::ImageFile file(test::PATH_TEST_DATA / "test.mrc", io::WRITE);
    file.shape(shapeFFT(vol_logical_shape));
    file.dataType(io::FLOAT32);
    file.writeAll(volume_weights.get(), false);
}
