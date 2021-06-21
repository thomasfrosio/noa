#include <noa/gpu/cuda/math/Indexes.h>

#include <noa/cpu/math/Indexes.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("CUDA: Indexes", "[noa][cuda][math]") {
    uint batches = test::IntRandomizer<uint>(1, 20).get();
    size_t elements = test::IntRandomizer<size_t>(1, 65536).get();
    memory::PtrHost<int> data(elements * batches);
    memory::PtrHost<size_t> idx_results(batches);

    test::Randomizer<int> randomizer(-10., 10.);
    test::initDataRandom(data.get(), data.elements(), randomizer);
    cuda::memory::PtrDevice<int> d_data(elements * batches);
    cuda::memory::PtrDevice<size_t> d_idx_results(batches);
    memory::PtrHost<size_t> cuda_idx_results(batches);

    cuda::Stream stream(cuda::Stream::SERIAL);
    cuda::memory::copy(data.get(), d_data.get(), data.size(), stream);

    cuda::math::firstMin(d_data.get(), d_idx_results.get(), elements, batches, stream);
    cuda::memory::copy(d_idx_results.get(), cuda_idx_results.get(), d_idx_results.size(), stream);
    math::firstMin(data.get(), idx_results.get(), elements, batches);
    cuda::Stream::synchronize(stream);

    size_t diff = test::getDifference(cuda_idx_results.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);

    cuda::math::firstMax(d_data.get(), d_idx_results.get(), elements, batches, stream);
    cuda::memory::copy(d_idx_results.get(), cuda_idx_results.get(), d_idx_results.size(), stream);
    math::firstMax(data.get(), idx_results.get(), elements, batches);
    cuda::Stream::synchronize(stream);

    diff = test::getDifference(cuda_idx_results.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);

    cuda::math::lastMin(d_data.get(), d_idx_results.get(), elements, batches, stream);
    cuda::memory::copy(d_idx_results.get(), cuda_idx_results.get(), d_idx_results.size(), stream);
    math::lastMin(data.get(), idx_results.get(), elements, batches);
    cuda::Stream::synchronize(stream);

    diff = test::getDifference(cuda_idx_results.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);

    cuda::math::lastMax(d_data.get(), d_idx_results.get(), elements, batches, stream);
    cuda::memory::copy(d_idx_results.get(), cuda_idx_results.get(), d_idx_results.size(), stream);
    math::lastMax(data.get(), idx_results.get(), elements, batches);
    cuda::Stream::synchronize(stream);

    diff = test::getDifference(cuda_idx_results.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);
}
