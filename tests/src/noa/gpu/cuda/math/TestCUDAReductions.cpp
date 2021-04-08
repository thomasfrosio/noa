#include <noa/gpu/cuda/math/Reductions.h>

#include <noa/cpu/PtrHost.h>
#include <noa/cpu/math/Reductions.h>
#include <noa/gpu/cuda/PtrDevice.h>
#include <noa/gpu/cuda/PtrDevicePadded.h>
#include <noa/gpu/cuda/Memory.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA::Math: Reduction - contiguous", "[noa][cuda][math]",
                   float, double, cfloat_t, cdouble_t, int) {
    Test::Randomizer<TestType> randomizer(0., 255.);
    using value_t = Noa::Traits::value_type_t<TestType>;

    value_t abs_epsilon;
    if constexpr (Noa::Traits::is_float_v<value_t>)
        abs_epsilon = Math::Limits<value_t>::epsilon() * 10;

    AND_THEN("general cases") {
        uint batches = Test::IntRandomizer<uint>(1, 5).get();
        size_t elements = Test::IntRandomizer<size_t>(1, 262144).get();
        PtrHost<TestType> h_data(elements * batches);
        PtrHost<TestType> h_results(2 * batches);
        CUDA::PtrDevice<TestType> d_data(elements * batches);
        CUDA::PtrDevice<TestType> d_results(2 * batches);
        PtrHost<TestType> h_cuda_results(2 * batches);

        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        CUDA::Stream stream(CUDA::Stream::SERIAL);
        CUDA::Memory::copy(h_data.get(), d_data.get(), d_data.bytes(), stream);
        CUDA::Math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, elements, batches, stream);
        Math::sumMean(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
        CUDA::Stream::synchronize(stream);

        CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.bytes(), stream);
        TestType diff = Test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches * 2);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == 0);
        else
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), abs_epsilon));
    }


    AND_THEN("large batches") {
        uint batches = Test::IntRandomizer<uint>(500, 40000).get();
        size_t elements = Test::IntRandomizer<size_t>(1, 1024).get();
        PtrHost<TestType> h_data(elements * batches);
        PtrHost<TestType> h_results(2 * batches);
        CUDA::PtrDevice<TestType> d_data(elements * batches);
        CUDA::PtrDevice<TestType> d_results(2 * batches);
        PtrHost<TestType> h_cuda_results(2 * batches);

        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        CUDA::Stream stream(CUDA::Stream::SERIAL);
        CUDA::Memory::copy(h_data.get(), d_data.get(), d_data.bytes(), stream);
        CUDA::Math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, elements, batches, stream);
        Math::sumMean(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
        CUDA::Stream::synchronize(stream);

        CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.bytes(), stream);
        TestType diff = Test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches * 2);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == 0);
        else
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), abs_epsilon));
    }

    AND_THEN("multiple of 1024") {
        uint batches = Test::IntRandomizer<uint>(1, 3).get();
        size_t elements = 1024 * Test::IntRandomizer<size_t>(1, 20).get();
        PtrHost<TestType> h_data(elements * batches);
        PtrHost<TestType> h_results(2 * batches);
        CUDA::PtrDevice<TestType> d_data(elements * batches);
        CUDA::PtrDevice<TestType> d_results(2 * batches);
        PtrHost<TestType> h_cuda_results(2 * batches);

        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        CUDA::Stream stream(CUDA::Stream::SERIAL);
        CUDA::Memory::copy(h_data.get(), d_data.get(), d_data.bytes(), stream);
        CUDA::Math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, elements, batches, stream);
        Math::sumMean(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
        CUDA::Stream::synchronize(stream);

        CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.bytes(), stream);
        TestType diff = Test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches * 2);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == 0);
        else
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), abs_epsilon));
    }

    AND_THEN("mean only") {
        uint batches = Test::IntRandomizer<uint>(1, 5).get();
        size_t elements = Test::IntRandomizer<size_t>(1, 262144).get();
        PtrHost<TestType> h_data(elements * batches);
        PtrHost<TestType> h_results(batches);
        CUDA::PtrDevice<TestType> d_data(elements * batches);
        CUDA::PtrDevice<TestType> d_results(batches);
        PtrHost<TestType> h_cuda_results(batches);

        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        TestType* empty = nullptr;
        CUDA::Stream stream(CUDA::Stream::SERIAL);
        CUDA::Memory::copy(h_data.get(), d_data.get(), d_data.bytes(), stream);
        CUDA::Math::sumMean(d_data.get(), empty, d_results.get(), elements, batches, stream);
        Math::sumMean(h_data.get(), empty, h_results.get(), elements, batches);
        CUDA::Stream::synchronize(stream);

        CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.bytes(), stream);
        TestType diff = Test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == 0);
        else
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("CUDA::Math: Reduction - padded", "[noa][cuda][math]",
                   float, double, cfloat_t, cdouble_t, int) {
    Test::Randomizer<TestType> randomizer(0., 255.);
    using value_t = Noa::Traits::value_type_t<TestType>;

    value_t abs_epsilon;
    if constexpr (Noa::Traits::is_float_v<value_t>)
        abs_epsilon = Math::Limits<value_t>::epsilon() * 10;

    uint ndim = GENERATE(2U, 3U);

    AND_THEN("general cases") {
        uint batches = Test::IntRandomizer<uint>(1, 3).get();
        size3_t shape = Test::getRandomShape(ndim);
        size_t elements = getElements(shape);
        PtrHost<TestType> h_data(elements * batches);
        PtrHost<TestType> h_results(2 * batches);
        CUDA::PtrDevicePadded<TestType> d_data(size3_t(shape.x, shape.y, shape.z * batches));
        CUDA::PtrDevice<TestType> d_results(2 * batches);
        PtrHost<TestType> h_cuda_results(2 * batches);

        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        CUDA::Stream stream(CUDA::Stream::SERIAL);
        CUDA::Memory::copy(h_data.get(), shape.x * sizeof(TestType),
                           d_data.get(), d_data.pitch(), d_data.shape(), stream);
        CUDA::Math::sumMean(d_data.get(), d_data.pitchElements(), d_results.get(), d_results.get() + batches,
                            shape, batches, stream);
        Math::sumMean(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
        CUDA::Stream::synchronize(stream);

        CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.bytes(), stream);
        TestType diff = Test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches * 2);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == 0);
        else
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), abs_epsilon));
    }

    AND_THEN("row elements multiple of 64") {
        uint batches = Test::IntRandomizer<uint>(1, 3).get();
        size3_t shape(64 * Test::IntRandomizer<size_t>(1, 4).get(), 32, 1);
        size_t elements = getElements(shape);
        PtrHost<TestType> h_data(elements * batches);
        PtrHost<TestType> h_results(2 * batches);
        CUDA::PtrDevicePadded<TestType> d_data(size3_t(shape.x, shape.y, shape.z * batches));
        CUDA::PtrDevice<TestType> d_results(2 * batches);
        PtrHost<TestType> h_cuda_results(2 * batches);

        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        CUDA::Stream stream(CUDA::Stream::SERIAL);
        CUDA::Memory::copy(h_data.get(), shape.x * sizeof(TestType),
                           d_data.get(), d_data.pitch(), d_data.shape(), stream);
        CUDA::Math::sumMean(d_data.get(), d_data.pitchElements(), d_results.get(), d_results.get() + batches,
                            shape, batches, stream);
        Math::sumMean(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
        CUDA::Stream::synchronize(stream);

        CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.bytes(), stream);
        TestType diff = Test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches * 2);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == 0);
        else
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), abs_epsilon));
    }
}
