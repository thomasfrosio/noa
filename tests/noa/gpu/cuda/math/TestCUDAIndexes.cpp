#include <noa/gpu/cuda/math/Indexes.h>

#include <noa/cpu/math/Indexes.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEST_CASE("CUDA: Indexes", "[noa][cuda][math]") {
    uint batches = Test::IntRandomizer<uint>(1, 20).get();
    size_t elements = Test::IntRandomizer<size_t>(1, 65536).get();
    Memory::PtrHost<int> data(elements * batches);
    Memory::PtrHost<size_t> idx_results(batches);

    Test::Randomizer<int> randomizer(-10., 10.);
    Test::initDataRandom(data.get(), data.elements(), randomizer);
    CUDA::Memory::PtrDevice<int> d_data(elements * batches);
    CUDA::Memory::PtrDevice<size_t> d_idx_results(batches);
    Memory::PtrHost<size_t> cuda_idx_results(batches);

    CUDA::Stream stream(CUDA::Stream::SERIAL);
    CUDA::Memory::copy(data.get(), d_data.get(), data.bytes(), stream);

    CUDA::Math::firstMin(d_data.get(), d_idx_results.get(), elements, batches, stream);
    CUDA::Memory::copy(d_idx_results.get(), cuda_idx_results.get(), d_idx_results.bytes(), stream);
    Math::firstMin(data.get(), idx_results.get(), elements, batches);
    CUDA::Stream::synchronize(stream);

    size_t diff = Test::getDifference(cuda_idx_results.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);

    CUDA::Math::firstMax(d_data.get(), d_idx_results.get(), elements, batches, stream);
    CUDA::Memory::copy(d_idx_results.get(), cuda_idx_results.get(), d_idx_results.bytes(), stream);
    Math::firstMax(data.get(), idx_results.get(), elements, batches);
    CUDA::Stream::synchronize(stream);

    diff = Test::getDifference(cuda_idx_results.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);

    CUDA::Math::lastMin(d_data.get(), d_idx_results.get(), elements, batches, stream);
    CUDA::Memory::copy(d_idx_results.get(), cuda_idx_results.get(), d_idx_results.bytes(), stream);
    Math::lastMin(data.get(), idx_results.get(), elements, batches);
    CUDA::Stream::synchronize(stream);

    diff = Test::getDifference(cuda_idx_results.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);

    CUDA::Math::lastMax(d_data.get(), d_idx_results.get(), elements, batches, stream);
    CUDA::Memory::copy(d_idx_results.get(), cuda_idx_results.get(), d_idx_results.bytes(), stream);
    Math::lastMax(data.get(), idx_results.get(), elements, batches);
    CUDA::Stream::synchronize(stream);

    diff = Test::getDifference(cuda_idx_results.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);
}
