#include <noa/gpu/cuda/math/Reductions.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Reductions.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA::Math: Reductions - min & max - contiguous", "[noa][cuda][math]",
                   float, double, int) {
    uint batches = Test::IntRandomizer<uint>(1, 5).get();
    size_t elements = Test::IntRandomizer<size_t>(1, 262144).get();

    Memory::PtrHost<TestType> h_data(elements * batches);
    Memory::PtrHost<TestType> h_results(batches);
    CUDA::Memory::PtrDevice<TestType> d_data(elements * batches);
    CUDA::Memory::PtrDevice<TestType> d_results(batches);
    Memory::PtrHost<TestType> h_cuda_results(batches);

    CUDA::Stream stream(CUDA::STREAM_SERIAL);
    Test::Randomizer<TestType> randomizer(-100., 100.);
    Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    CUDA::Memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);

    CUDA::Math::min(d_data.get(), d_results.get(), elements, batches, stream);
    Math::min(h_data.get(), h_results.get(), elements, batches);
    CUDA::Stream::synchronize(stream);
    CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    TestType diff = Test::getDifference(h_results.get(), h_cuda_results.get(), batches);
    REQUIRE(diff == 0);

    CUDA::Math::max(d_data.get(), d_results.get(), elements, batches, stream);
    Math::max(h_data.get(), h_results.get(), elements, batches);
    CUDA::Stream::synchronize(stream);
    CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    diff = Test::getDifference(h_results.get(), h_cuda_results.get(), batches);
    REQUIRE(diff == 0);
}

TEMPLATE_TEST_CASE("CUDA::Math: Reductions - min & max - padded", "[noa][cuda][math]",
                   float, double, int) {
    uint batches = Test::IntRandomizer<uint>(1, 5).get();
    size3_t shape = Test::getRandomShape(3);
    size_t elements = getElements(shape);
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);

    Memory::PtrHost<TestType> h_data(elements * batches);
    Memory::PtrHost<TestType> h_results(batches);
    CUDA::Memory::PtrDevicePadded<TestType> d_data(shape_batched);
    CUDA::Memory::PtrDevice<TestType> d_results(batches);
    Memory::PtrHost<TestType> h_cuda_results(batches);

    CUDA::Stream stream(CUDA::STREAM_SERIAL);
    Test::Randomizer<TestType> randomizer(-100., 100.);
    Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    CUDA::Memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

    CUDA::Math::min(d_data.get(), d_data.pitch(), d_results.get(), shape, batches, stream);
    Math::min(h_data.get(), h_results.get(), elements, batches);
    CUDA::Stream::synchronize(stream);
    CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    TestType diff = Test::getDifference(h_results.get(), h_cuda_results.get(), batches);
    REQUIRE(diff == 0);

    CUDA::Math::max(d_data.get(), d_data.pitch(), d_results.get(), shape, batches, stream);
    Math::max(h_data.get(), h_results.get(), elements, batches);
    CUDA::Stream::synchronize(stream);
    CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    diff = Test::getDifference(h_results.get(), h_cuda_results.get(), batches);
    REQUIRE(diff == 0);
}

TEMPLATE_TEST_CASE("CUDA::Math: Reductions - minMax - contiguous", "[noa][cuda][math]",
                   float, double, int) {
    uint batches = Test::IntRandomizer<uint>(1, 5).get();
    size_t elements = Test::IntRandomizer<size_t>(1, 262144).get();

    Memory::PtrHost<TestType> h_data(elements * batches);
    Memory::PtrHost<TestType> h_results(batches * 2); // all mins, then all maxs.
    CUDA::Memory::PtrDevice<TestType> d_data(elements * batches);
    CUDA::Memory::PtrDevice<TestType> d_results(batches * 2);
    Memory::PtrHost<TestType> h_cuda_results(batches * 2);

    CUDA::Stream stream(CUDA::STREAM_SERIAL);
    Test::Randomizer<TestType> randomizer(-100., 100.);
    Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    CUDA::Memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);

    CUDA::Math::minMax(d_data.get(), d_results.get(), d_results.get() + batches, elements, batches, stream);
    Math::minMax(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
    CUDA::Stream::synchronize(stream);
    CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    TestType diff = Test::getDifference(h_results.get(), h_cuda_results.get(), batches * 2);
    REQUIRE(diff == 0);
}

TEMPLATE_TEST_CASE("CUDA::Math: Reductions - minMax - padded", "[noa][cuda][math]",
                   float, double, int) {
    uint batches = Test::IntRandomizer<uint>(1, 5).get();
    size3_t shape = Test::getRandomShape(3);
    size_t elements = getElements(shape);
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);

    Memory::PtrHost<TestType> h_data(elements * batches);
    Memory::PtrHost<TestType> h_results(batches * 2); // all mins, then all maxs.
    CUDA::Memory::PtrDevicePadded<TestType> d_data(shape_batched);
    CUDA::Memory::PtrDevice<TestType> d_results(batches * 2);
    Memory::PtrHost<TestType> h_cuda_results(batches * 2);

    CUDA::Stream stream(CUDA::STREAM_SERIAL);
    Test::Randomizer<TestType> randomizer(-100., 100.);
    Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    CUDA::Memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

    CUDA::Math::minMax(d_data.get(), d_data.pitch(), d_results.get(), d_results.get() + batches,
                       shape, batches, stream);
    Math::minMax(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
    CUDA::Stream::synchronize(stream);
    CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    TestType diff = Test::getDifference(h_results.get(), h_cuda_results.get(), batches * 2);
    REQUIRE(diff == 0);
}

TEMPLATE_TEST_CASE("CUDA::Math: Reductions - sumMean - contiguous", "[noa][cuda][math]",
                   float, double, cfloat_t, cdouble_t, int) {
    Test::Randomizer<TestType> randomizer(0., 255.);
    using value_t = Noa::Traits::value_type_t<TestType>;

    value_t abs_epsilon;
    if constexpr (Noa::Traits::is_float_v<value_t>)
        abs_epsilon = Math::Limits<value_t>::epsilon() * 10;

    AND_THEN("general cases") {
        uint batches = Test::IntRandomizer<uint>(1, 5).get();
        size_t elements = Test::IntRandomizer<size_t>(1, 262144).get();
        Memory::PtrHost<TestType> h_data(elements * batches);
        Memory::PtrHost<TestType> h_results(2 * batches);
        CUDA::Memory::PtrDevice<TestType> d_data(elements * batches);
        CUDA::Memory::PtrDevice<TestType> d_results(2 * batches);
        Memory::PtrHost<TestType> h_cuda_results(2 * batches);

        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        CUDA::Stream stream(CUDA::STREAM_SERIAL);
        CUDA::Memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);
        CUDA::Math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, elements, batches, stream);
        Math::sumMean(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
        CUDA::Stream::synchronize(stream);

        CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
        TestType diff = Test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches * 2);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == 0);
        else
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), abs_epsilon));
    }

    AND_THEN("large batches") {
        uint batches = Test::IntRandomizer<uint>(500, 40000).get();
        size_t elements = Test::IntRandomizer<size_t>(1, 1024).get();
        Memory::PtrHost<TestType> h_data(elements * batches);
        Memory::PtrHost<TestType> h_results(2 * batches);
        CUDA::Memory::PtrDevice<TestType> d_data(elements * batches);
        CUDA::Memory::PtrDevice<TestType> d_results(2 * batches);
        Memory::PtrHost<TestType> h_cuda_results(2 * batches);

        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        CUDA::Stream stream(CUDA::STREAM_SERIAL);
        CUDA::Memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);
        CUDA::Math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, elements, batches, stream);
        Math::sumMean(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
        CUDA::Stream::synchronize(stream);

        CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
        TestType diff = Test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches * 2);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == 0);
        else
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), abs_epsilon));
    }

    AND_THEN("multiple of 1024") {
        uint batches = Test::IntRandomizer<uint>(1, 3).get();
        size_t elements = 1024 * Test::IntRandomizer<size_t>(1, 20).get();
        Memory::PtrHost<TestType> h_data(elements * batches);
        Memory::PtrHost<TestType> h_results(2 * batches);
        CUDA::Memory::PtrDevice<TestType> d_data(elements * batches);
        CUDA::Memory::PtrDevice<TestType> d_results(2 * batches);
        Memory::PtrHost<TestType> h_cuda_results(2 * batches);

        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        CUDA::Stream stream(CUDA::STREAM_SERIAL);
        CUDA::Memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);
        CUDA::Math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, elements, batches, stream);
        Math::sumMean(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
        CUDA::Stream::synchronize(stream);

        CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
        TestType diff = Test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches * 2);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == 0);
        else
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), abs_epsilon));
    }

    AND_THEN("mean only") {
        uint batches = Test::IntRandomizer<uint>(1, 5).get();
        size_t elements = Test::IntRandomizer<size_t>(1, 262144).get();
        Memory::PtrHost<TestType> h_data(elements * batches);
        Memory::PtrHost<TestType> h_results(batches);
        CUDA::Memory::PtrDevice<TestType> d_data(elements * batches);
        CUDA::Memory::PtrDevice<TestType> d_results(batches);
        Memory::PtrHost<TestType> h_cuda_results(batches);

        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        TestType* empty = nullptr;
        CUDA::Stream stream(CUDA::STREAM_SERIAL);
        CUDA::Memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);
        CUDA::Math::sumMean(d_data.get(), empty, d_results.get(), elements, batches, stream);
        Math::sumMean(h_data.get(), empty, h_results.get(), elements, batches);
        CUDA::Stream::synchronize(stream);

        CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
        TestType diff = Test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == 0);
        else
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("CUDA::Math: Reduction - sumMean - padded", "[noa][cuda][math]",
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
        Memory::PtrHost<TestType> h_data(elements * batches);
        Memory::PtrHost<TestType> h_results(2 * batches);
        CUDA::Memory::PtrDevicePadded<TestType> d_data(size3_t(shape.x, shape.y, shape.z * batches));
        CUDA::Memory::PtrDevice<TestType> d_results(2 * batches);
        Memory::PtrHost<TestType> h_cuda_results(2 * batches);

        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        CUDA::Stream stream(CUDA::STREAM_SERIAL);
        CUDA::Memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), d_data.shape(), stream);
        CUDA::Math::sumMean(d_data.get(), d_data.pitch(), d_results.get(), d_results.get() + batches,
                            shape, batches, stream);
        Math::sumMean(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
        CUDA::Stream::synchronize(stream);

        CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
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
        Memory::PtrHost<TestType> h_data(elements * batches);
        Memory::PtrHost<TestType> h_results(2 * batches);
        CUDA::Memory::PtrDevicePadded<TestType> d_data(size3_t(shape.x, shape.y, shape.z * batches));
        CUDA::Memory::PtrDevice<TestType> d_results(2 * batches);
        Memory::PtrHost<TestType> h_cuda_results(2 * batches);

        Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        CUDA::Stream stream(CUDA::STREAM_SERIAL);
        CUDA::Memory::copy(h_data.get(), shape.x,
                           d_data.get(), d_data.pitch(), d_data.shape(), stream);
        CUDA::Math::sumMean(d_data.get(), d_data.pitch(), d_results.get(), d_results.get() + batches,
                            shape, batches, stream);
        Math::sumMean(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
        CUDA::Stream::synchronize(stream);

        CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
        TestType diff = Test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches * 2);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == 0);
        else
            REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("CUDA::Math: Reductions - minMaxSumMean - contiguous", "[noa][cuda][math]",
                   float, double, int) {
    uint batches = Test::IntRandomizer<uint>(1, 5).get();
    size_t elements = Test::IntRandomizer<size_t>(1, 262144).get();

    Memory::PtrHost<TestType> h_data(elements * batches);
    Memory::PtrHost<TestType> h_results(batches * 4); // all mins, all maxs, all sums, all means.
    CUDA::Memory::PtrDevice<TestType> d_data(elements * batches);
    CUDA::Memory::PtrDevice<TestType> d_results(batches * 4);
    Memory::PtrHost<TestType> h_cuda_results(batches * 4);

    CUDA::Stream stream(CUDA::STREAM_SERIAL);
    Test::Randomizer<TestType> randomizer(-100., 100.);
    Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    CUDA::Memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);

    CUDA::Math::minMaxSumMean(d_data.get(),
                              d_results.get(),
                              d_results.get() + batches,
                              d_results.get() + batches * 2,
                              d_results.get() + batches * 3,
                              elements, batches, stream);
    Math::minMaxSumMean(h_data.get(),
                        h_results.get(),
                        h_results.get() + batches,
                        h_results.get() + batches * 2,
                        h_results.get() + batches * 3,
                        elements, batches);
    CUDA::Stream::synchronize(stream);
    CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    TestType diff = Test::getDifference(h_results.get(), h_cuda_results.get(), batches * 2);
    REQUIRE(diff == 0);
    diff = Test::getAverageNormalizedDifference(h_results.get() + batches * 2,
                                                h_cuda_results.get() + batches * 2, batches * 2);
    if constexpr (std::is_floating_point_v<TestType>) {
        REQUIRE_THAT(diff, Test::isWithinAbs(0., Math::Limits<TestType>::epsilon() * 100));
    } else {
        REQUIRE(diff == 0);
    }
}

TEMPLATE_TEST_CASE("CUDA::Math: Reductions - minMaxSumMean - padded", "[noa][cuda][math]",
                   float, double, int) {
    uint batches = Test::IntRandomizer<uint>(1, 5).get();
    size3_t shape = Test::getRandomShape(3);
    size_t elements = getElements(shape);
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);

    Memory::PtrHost<TestType> h_data(elements * batches);
    Memory::PtrHost<TestType> h_results(batches * 4); // all mins, all maxs, all sums, all means.
    CUDA::Memory::PtrDevicePadded<TestType> d_data(shape_batched);
    CUDA::Memory::PtrDevice<TestType> d_results(batches * 4);
    Memory::PtrHost<TestType> h_cuda_results(batches * 4);

    CUDA::Stream stream(CUDA::STREAM_SERIAL);
    Test::Randomizer<TestType> randomizer(-100., 100.);
    Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    CUDA::Memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

    CUDA::Math::minMaxSumMean(d_data.get(), d_data.pitch(),
                              d_results.get(),
                              d_results.get() + batches,
                              d_results.get() + batches * 2,
                              d_results.get() + batches * 3,
                              shape, batches, stream);
    Math::minMaxSumMean(h_data.get(),
                        h_results.get(),
                        h_results.get() + batches,
                        h_results.get() + batches * 2,
                        h_results.get() + batches * 3,
                        elements, batches);
    CUDA::Stream::synchronize(stream);
    CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    TestType diff = Test::getDifference(h_results.get(), h_cuda_results.get(), batches * 2);
    REQUIRE(diff == 0);
    diff = Test::getAverageNormalizedDifference(h_results.get() + batches * 2,
                                                h_cuda_results.get() + batches * 2, batches * 2);
    if constexpr (std::is_floating_point_v<TestType>) {
        REQUIRE_THAT(diff, Test::isWithinAbs(0., Math::Limits<TestType>::epsilon() * 100));
    } else {
        REQUIRE(diff == 0);
    }
}

TEMPLATE_TEST_CASE("CUDA::Math: Reductions - statistics - contiguous", "[noa][cuda][math]", float, double) {
    uint batches = Test::IntRandomizer<uint>(1, 5).get();
    size_t elements = Test::IntRandomizer<size_t>(1, 262144).get();

    Memory::PtrHost<TestType> h_data(elements * batches);
    Memory::PtrHost<TestType> h_results(batches * 6); // all mins, all maxs, all sums, all means, all variances, all stddevs.
    CUDA::Memory::PtrDevice<TestType> d_data(elements * batches);
    CUDA::Memory::PtrDevice<TestType> d_results(batches * 6);
    Memory::PtrHost<TestType> h_cuda_results(batches * 6);

    CUDA::Stream stream(CUDA::STREAM_SERIAL);
    Test::Randomizer<TestType> randomizer(-100., 100.);
    Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    CUDA::Memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);

    CUDA::Math::statistics(d_data.get(),
                           d_results.get(),
                           d_results.get() + batches,
                           d_results.get() + batches * 2,
                           d_results.get() + batches * 3,
                           d_results.get() + batches * 4,
                           d_results.get() + batches * 5,
                           elements, batches, stream);
    Math::statistics(h_data.get(),
                     h_results.get(),
                     h_results.get() + batches,
                     h_results.get() + batches * 2,
                     h_results.get() + batches * 3,
                     h_results.get() + batches * 4,
                     h_results.get() + batches * 5,
                     elements, batches);
    CUDA::Stream::synchronize(stream);
    CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);

    TestType diff = Test::getDifference(h_results.get(), h_cuda_results.get(), batches * 2);
    REQUIRE(diff == 0);
    diff = Test::getAverageNormalizedDifference(h_results.get() + batches * 2,
                                                h_cuda_results.get() + batches * 2, batches * 4);
    if constexpr (std::is_floating_point_v<TestType>) {
        REQUIRE_THAT(diff, Test::isWithinAbs(0., Math::Limits<TestType>::epsilon() * 100));
    } else {
        REQUIRE(diff == 0);
    }
}

TEMPLATE_TEST_CASE("CUDA::Math: Reductions - statistics - padded", "[noa][cuda][math]", float, double) {
    uint batches = Test::IntRandomizer<uint>(1, 5).get();
    size3_t shape = Test::getRandomShape(3);
    size_t elements = getElements(shape);
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);

    Memory::PtrHost<TestType> h_data(elements * batches);
    Memory::PtrHost<TestType> h_results(batches * 6); // all mins, all maxs, all sums, all means, all variances, all stddevs.
    CUDA::Memory::PtrDevicePadded<TestType> d_data(shape_batched);
    CUDA::Memory::PtrDevice<TestType> d_results(batches * 6);
    Memory::PtrHost<TestType> h_cuda_results(batches * 6);

    CUDA::Stream stream(CUDA::STREAM_SERIAL);
    Test::Randomizer<TestType> randomizer(-100., 100.);
    Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    CUDA::Memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

    CUDA::Math::statistics(d_data.get(), d_data.pitch(),
                           d_results.get(),
                           d_results.get() + batches,
                           d_results.get() + batches * 2,
                           d_results.get() + batches * 3,
                           d_results.get() + batches * 4,
                           d_results.get() + batches * 5,
                           shape, batches, stream);
    Math::statistics(h_data.get(),
                     h_results.get(),
                     h_results.get() + batches,
                     h_results.get() + batches * 2,
                     h_results.get() + batches * 3,
                     h_results.get() + batches * 4,
                     h_results.get() + batches * 5,
                     elements, batches);
    CUDA::Stream::synchronize(stream);
    CUDA::Memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);

    TestType diff = Test::getDifference(h_results.get(), h_cuda_results.get(), batches * 2);
    REQUIRE(diff == 0);

    // sum & mean
    diff = Test::getAverageNormalizedDifference(h_results.get() + batches * 2,
                                                h_cuda_results.get() + batches * 2, batches * 2);
    if constexpr (std::is_floating_point_v<TestType>) {
        REQUIRE_THAT(diff, Test::isWithinAbs(0., Math::Limits<TestType>::epsilon() * 100));
    } else {
        REQUIRE(diff == 0);
    }

    // variance & stddev: expect slightly lower precision
    diff = Test::getAverageNormalizedDifference(h_results.get() + batches * 4,
                                                h_cuda_results.get() + batches * 4, batches * 2);
    if constexpr (std::is_floating_point_v<TestType>) {
        REQUIRE_THAT(diff, Test::isWithinAbs(0., Math::Limits<TestType>::epsilon() * 1000));
    } else {
        REQUIRE(diff == 0);
    }
}

TEMPLATE_TEST_CASE("CUDA::Math: Reductions - reduce* - contiguous", "[noa][cuda][math]",
                   int, float, double, cfloat_t, cdouble_t) {
    uint batches = Test::IntRandomizer<uint>(1, 5).get();
    uint vectors = Test::IntRandomizer<uint>(1, 5).get();
    size_t elements = Test::IntRandomizer<size_t>(1, 100000).get();

    Memory::PtrHost<TestType> h_vectors(elements * vectors * batches);
    Memory::PtrHost<TestType> h_reduced(elements * batches);

    Test::Randomizer<TestType> randomizer(-100., 100.);
    Test::initDataRandom(h_vectors.get(), h_vectors.elements(), randomizer);

    CUDA::Stream stream(CUDA::STREAM_SERIAL);
    CUDA::Memory::PtrDevice<TestType> d_vectors(h_vectors.elements());
    CUDA::Memory::PtrDevice<TestType> d_reduced(h_reduced.elements());
    Memory::PtrHost<TestType> h_cuda_reduced(h_reduced.elements());
    CUDA::Memory::copy(h_vectors.get(), d_vectors.get(), h_vectors.size(), stream);

    AND_THEN("reduceAdd") {
        CUDA::Math::reduceAdd(d_vectors.get(), d_reduced.get(), elements, vectors, batches, stream);
        CUDA::Memory::copy(d_reduced.get(), h_cuda_reduced.get(), d_reduced.size(), stream);
        Math::reduceAdd(h_vectors.get(), h_reduced.get(), elements, vectors, batches);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getAverageDifference(h_reduced.get(), h_cuda_reduced.get(), h_reduced.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), 1e-5));
    }

    AND_THEN("reduceMean") {
        CUDA::Math::reduceMean(d_vectors.get(), d_reduced.get(), elements, vectors, batches, stream);
        CUDA::Memory::copy(d_reduced.get(), h_cuda_reduced.get(), d_reduced.size(), stream);
        Math::reduceMean(h_vectors.get(), h_reduced.get(), elements, vectors, batches);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getAverageDifference(h_reduced.get(), h_cuda_reduced.get(), h_reduced.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), 1e-5));
    }

    AND_THEN("reduceMeanWeighted") {
        using real_t = Noa::Traits::value_type_t<TestType>;
        Memory::PtrHost<real_t> h_weights(elements * vectors * batches);
        CUDA::Memory::PtrDevice<real_t> d_weights(h_weights.elements());
        Test::Randomizer<real_t> randomizer_real(0., 10.);
        Test::initDataRandom(h_weights.get(), h_weights.elements(), randomizer_real);
        CUDA::Memory::copy(h_weights.get(), d_weights.get(), d_weights.size(), stream);

        CUDA::Math::reduceMeanWeighted(d_vectors.get(), d_weights.get(), d_reduced.get(),
                                       elements, vectors, batches, stream);
        CUDA::Memory::copy(d_reduced.get(), h_cuda_reduced.get(), d_reduced.size(), stream);
        Math::reduceMeanWeighted(h_vectors.get(), h_weights.get(), h_reduced.get(), elements, vectors, batches);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getAverageDifference(h_reduced.get(), h_cuda_reduced.get(), h_reduced.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), 1e-5));
    }
}

TEMPLATE_TEST_CASE("CUDA::Math: Reductions - reduce* - padded", "[noa][cuda][math]",
                   int, float, double, cfloat_t, cdouble_t) {
    uint batches = Test::IntRandomizer<uint>(1, 3).get();
    uint vectors = Test::IntRandomizer<uint>(1, 3).get();
    size3_t shape = Test::getRandomShape(2);
    size_t elements = getElements(shape);
    size3_t shape_batched(shape.x, shape.y * shape.z, vectors * batches);
    size3_t shape_reduced(shape.x, shape.y * shape.z, batches);

    Memory::PtrHost<TestType> h_vectors(elements * vectors * batches);
    Memory::PtrHost<TestType> h_reduced(elements * batches);

    Test::Randomizer<TestType> randomizer(-100., 100.);
    Test::initDataRandom(h_vectors.get(), h_vectors.elements(), randomizer);

    CUDA::Stream stream(CUDA::STREAM_SERIAL);
    CUDA::Memory::PtrDevicePadded<TestType> d_vectors(shape_batched);
    CUDA::Memory::PtrDevicePadded<TestType> d_reduced(shape_reduced);
    Memory::PtrHost<TestType> h_cuda_reduced(h_reduced.elements());
    CUDA::Memory::copy(h_vectors.get(), shape.x, d_vectors.get(), d_vectors.pitch(), shape_batched, stream);

    AND_THEN("reduceAdd") {
        CUDA::Math::reduceAdd(d_vectors.get(), d_vectors.pitch(), d_reduced.get(), d_reduced.pitch(),
                              shape, vectors, batches, stream);
        CUDA::Memory::copy(d_reduced.get(), d_reduced.pitch(), h_cuda_reduced.get(), shape.x,
                           shape_reduced, stream);
        Math::reduceAdd(h_vectors.get(), h_reduced.get(), elements, vectors, batches);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getAverageDifference(h_reduced.get(), h_cuda_reduced.get(), h_reduced.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), 1e-5));
    }

    AND_THEN("reduceMean") {
        CUDA::Math::reduceMean(d_vectors.get(), d_vectors.pitch(), d_reduced.get(), d_reduced.pitch(),
                              shape, vectors, batches, stream);
        CUDA::Memory::copy(d_reduced.get(), d_reduced.pitch(), h_cuda_reduced.get(), shape.x,
                           shape_reduced, stream);
        Math::reduceMean(h_vectors.get(), h_reduced.get(), elements, vectors, batches);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getAverageDifference(h_reduced.get(), h_cuda_reduced.get(), h_reduced.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), 1e-5));
    }

    AND_THEN("reduceMeanWeighted") {
        using real_t = Noa::Traits::value_type_t<TestType>;
        Memory::PtrHost<real_t> h_weights(elements * vectors * batches);
        CUDA::Memory::PtrDevicePadded<real_t> d_weights(size3_t(shape.x, shape.y * shape.z, vectors));
        Test::Randomizer<real_t> randomizer_real(0., 10.);
        Test::initDataRandom(h_weights.get(), h_weights.elements(), randomizer_real);
        CUDA::Memory::copy(h_weights.get(), shape.x,
                           d_weights.get(), d_weights.pitch(),
                           d_weights.shape(), stream);

        CUDA::Math::reduceMeanWeighted(d_vectors.get(), d_vectors.pitch(),
                                       d_weights.get(), d_weights.pitch(),
                                       d_reduced.get(), d_reduced.pitch(),
                                       shape, vectors, batches, stream);
        CUDA::Memory::copy(d_reduced.get(), d_reduced.pitch(), h_cuda_reduced.get(), shape.x,
                           shape_reduced, stream);
        Math::reduceMeanWeighted(h_vectors.get(), h_weights.get(), h_reduced.get(), elements, vectors, batches);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getAverageDifference(h_reduced.get(), h_cuda_reduced.get(), h_reduced.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0.), 1e-5));
    }
}
