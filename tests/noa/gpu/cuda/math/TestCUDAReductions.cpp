#include <noa/gpu/cuda/math/Reductions.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Reductions.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::math:: reductions, min & max, contiguous", "[noa][cuda][math]",
                   float, double, int) {
    size_t batches = test::IntRandomizer<size_t>(1, 5).get();
    size_t elements = test::IntRandomizer<size_t>(1, 262144).get();

    cpu::memory::PtrHost<TestType> h_data(elements * batches);
    cpu::memory::PtrHost<TestType> h_results(batches);
    cuda::memory::PtrDevice<TestType> d_data(elements * batches);
    cuda::memory::PtrDevice<TestType> d_results(batches);
    cpu::memory::PtrHost<TestType> h_cuda_results(batches);

    cuda::Stream stream(cuda::Stream::SERIAL);
    test::Randomizer<TestType> randomizer(-100., 100.);
    test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);

    cuda::math::min(d_data.get(), d_results.get(), elements, batches, stream);
    cpu::math::min(h_data.get(), h_results.get(), elements, batches);
    cuda::Stream::synchronize(stream);
    cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    TestType diff = test::getDifference(h_results.get(), h_cuda_results.get(), batches);
    REQUIRE(diff == 0);

    cuda::math::max(d_data.get(), d_results.get(), elements, batches, stream);
    cpu::math::max(h_data.get(), h_results.get(), elements, batches);
    cuda::Stream::synchronize(stream);
    cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    diff = test::getDifference(h_results.get(), h_cuda_results.get(), batches);
    REQUIRE(diff == 0);
}

TEMPLATE_TEST_CASE("cuda::math:: reductions, min & max, padded", "[noa][cuda][math]",
                   float, double, int) {
    size_t batches = test::IntRandomizer<size_t>(1, 5).get();
    size3_t shape = test::getRandomShape(3);
    size_t elements = noa::elements(shape);
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);

    cpu::memory::PtrHost<TestType> h_data(elements * batches);
    cpu::memory::PtrHost<TestType> h_results(batches);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
    cuda::memory::PtrDevice<TestType> d_results(batches);
    cpu::memory::PtrHost<TestType> h_cuda_results(batches);

    cuda::Stream stream(cuda::Stream::SERIAL);
    test::Randomizer<TestType> randomizer(-100., 100.);
    test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

    cuda::math::min(d_data.get(), d_data.pitch(), d_results.get(), shape, batches, stream);
    cpu::math::min(h_data.get(), h_results.get(), elements, batches);
    cuda::Stream::synchronize(stream);
    cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    TestType diff = test::getDifference(h_results.get(), h_cuda_results.get(), batches);
    REQUIRE(diff == 0);

    cuda::math::max(d_data.get(), d_data.pitch(), d_results.get(), shape, batches, stream);
    cpu::math::max(h_data.get(), h_results.get(), elements, batches);
    cuda::Stream::synchronize(stream);
    cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    diff = test::getDifference(h_results.get(), h_cuda_results.get(), batches);
    REQUIRE(diff == 0);
}

TEMPLATE_TEST_CASE("cuda::math:: reductions, minMax, contiguous", "[noa][cuda][math]",
                   float, double, int) {
    size_t batches = test::IntRandomizer<size_t>(1, 5).get();
    size_t elements = test::IntRandomizer<size_t>(1, 262144).get();

    cpu::memory::PtrHost<TestType> h_data(elements * batches);
    cpu::memory::PtrHost<TestType> h_results(batches * 2); // all mins, then all maxs.
    cuda::memory::PtrDevice<TestType> d_data(elements * batches);
    cuda::memory::PtrDevice<TestType> d_results(batches * 2);
    cpu::memory::PtrHost<TestType> h_cuda_results(batches * 2);

    cuda::Stream stream(cuda::Stream::SERIAL);
    test::Randomizer<TestType> randomizer(-100., 100.);
    test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);

    cuda::math::minMax(d_data.get(), d_results.get(), d_results.get() + batches, elements, batches, stream);
    cpu::math::minMax(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
    cuda::Stream::synchronize(stream);
    cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    TestType diff = test::getDifference(h_results.get(), h_cuda_results.get(), batches * 2);
    REQUIRE(diff == 0);
}

TEMPLATE_TEST_CASE("cuda::math:: reductions, minMax, padded", "[noa][cuda][math]",
                   float, double, int) {
    size_t batches = test::IntRandomizer<size_t>(1, 5).get();
    size3_t shape = test::getRandomShape(3);
    size_t elements = noa::elements(shape);
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);

    cpu::memory::PtrHost<TestType> h_data(elements * batches);
    cpu::memory::PtrHost<TestType> h_results(batches * 2); // all mins, then all maxs.
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
    cuda::memory::PtrDevice<TestType> d_results(batches * 2);
    cpu::memory::PtrHost<TestType> h_cuda_results(batches * 2);

    cuda::Stream stream(cuda::Stream::SERIAL);
    test::Randomizer<TestType> randomizer(-100., 100.);
    test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

    cuda::math::minMax(d_data.get(), d_data.pitch(), d_results.get(), d_results.get() + batches,
                       shape, batches, stream);
    cpu::math::minMax(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
    cuda::Stream::synchronize(stream);
    cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    TestType diff = test::getDifference(h_results.get(), h_cuda_results.get(), batches * 2);
    REQUIRE(diff == 0);
}

TEMPLATE_TEST_CASE("cuda::math:: reductions, sumMean, contiguous", "[noa][cuda][math]",
                   float, double, cfloat_t, cdouble_t, int) {
    test::Randomizer<TestType> randomizer(0., 255.);
    using value_t = noa::traits::value_type_t<TestType>;

    value_t abs_epsilon = 0;
    if constexpr (noa::traits::is_float_v<value_t>)
        abs_epsilon = math::Limits<value_t>::epsilon() * 10;

    AND_THEN("general cases") {
        size_t batches = test::IntRandomizer<size_t>(1, 5).get();
        size_t elements = test::IntRandomizer<size_t>(1, 262144).get();
        cpu::memory::PtrHost<TestType> h_data(elements * batches);
        cpu::memory::PtrHost<TestType> h_results(2 * batches);
        cuda::memory::PtrDevice<TestType> d_data(elements * batches);
        cuda::memory::PtrDevice<TestType> d_results(2 * batches);
        cpu::memory::PtrHost<TestType> h_cuda_results(2 * batches);

        test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);
        cuda::math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, elements, batches, stream);
        cpu::math::sumMean(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
        cuda::Stream::synchronize(stream);

        cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
        TestType diff = test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches * 2);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == abs_epsilon);
        else
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0.), abs_epsilon));
    }

    AND_THEN("large batches") {
        size_t batches = test::IntRandomizer<size_t>(500, 40000).get();
        size_t elements = test::IntRandomizer<size_t>(1, 1024).get();
        cpu::memory::PtrHost<TestType> h_data(elements * batches);
        cpu::memory::PtrHost<TestType> h_results(2 * batches);
        cuda::memory::PtrDevice<TestType> d_data(elements * batches);
        cuda::memory::PtrDevice<TestType> d_results(2 * batches);
        cpu::memory::PtrHost<TestType> h_cuda_results(2 * batches);

        test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);
        cuda::math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, elements, batches, stream);
        cpu::math::sumMean(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
        cuda::Stream::synchronize(stream);

        cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
        TestType diff = test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches * 2);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == abs_epsilon);
        else
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0.), abs_epsilon));
    }

    AND_THEN("multiple of 1024") {
        size_t batches = test::IntRandomizer<size_t>(1, 3).get();
        size_t elements = 1024 * test::IntRandomizer<size_t>(1, 20).get();
        cpu::memory::PtrHost<TestType> h_data(elements * batches);
        cpu::memory::PtrHost<TestType> h_results(2 * batches);
        cuda::memory::PtrDevice<TestType> d_data(elements * batches);
        cuda::memory::PtrDevice<TestType> d_results(2 * batches);
        cpu::memory::PtrHost<TestType> h_cuda_results(2 * batches);

        test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);
        cuda::math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, elements, batches, stream);
        cpu::math::sumMean(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
        cuda::Stream::synchronize(stream);

        cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
        TestType diff = test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches * 2);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == abs_epsilon);
        else
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0.), abs_epsilon));
    }

    AND_THEN("mean only") {
        size_t batches = test::IntRandomizer<size_t>(1, 5).get();
        size_t elements = test::IntRandomizer<size_t>(1, 262144).get();
        cpu::memory::PtrHost<TestType> h_data(elements * batches);
        cpu::memory::PtrHost<TestType> h_results(batches);
        cuda::memory::PtrDevice<TestType> d_data(elements * batches);
        cuda::memory::PtrDevice<TestType> d_results(batches);
        cpu::memory::PtrHost<TestType> h_cuda_results(batches);

        test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        TestType* empty = nullptr;
        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);
        cuda::math::sumMean(d_data.get(), empty, d_results.get(), elements, batches, stream);
        cpu::math::sumMean(h_data.get(), empty, h_results.get(), elements, batches);
        cuda::Stream::synchronize(stream);

        cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
        TestType diff = test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == 0);
        else
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0.), abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("cuda::math:: reduction, sumMean, padded", "[noa][cuda][math]",
                   float, double, cfloat_t, cdouble_t, int) {
    test::Randomizer<TestType> randomizer(0., 255.);
    using value_t = noa::traits::value_type_t<TestType>;

    value_t abs_epsilon = 0;
    if constexpr (noa::traits::is_float_v<value_t>)
        abs_epsilon = math::Limits<value_t>::epsilon() * 10;

    uint ndim = GENERATE(2U, 3U);

    AND_THEN("general cases") {
        size_t batches = test::IntRandomizer<size_t>(1, 3).get();
        size3_t shape = test::getRandomShape(ndim);
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<TestType> h_data(elements * batches);
        cpu::memory::PtrHost<TestType> h_results(2 * batches);
        cuda::memory::PtrDevicePadded<TestType> d_data(size3_t(shape.x, shape.y, shape.z * batches));
        cuda::memory::PtrDevice<TestType> d_results(2 * batches);
        cpu::memory::PtrHost<TestType> h_cuda_results(2 * batches);

        test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), d_data.shape(), stream);
        cuda::math::sumMean(d_data.get(), d_data.pitch(), d_results.get(), d_results.get() + batches,
                            shape, batches, stream);
        cpu::math::sumMean(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
        cuda::Stream::synchronize(stream);

        cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
        TestType diff = test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches * 2);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == abs_epsilon);
        else
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0.), abs_epsilon));
    }

    AND_THEN("row elements multiple of 64") {
        size_t batches = test::IntRandomizer<size_t>(1, 3).get();
        size3_t shape(64 * test::IntRandomizer<size_t>(1, 4).get(), 32, 1);
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<TestType> h_data(elements * batches);
        cpu::memory::PtrHost<TestType> h_results(2 * batches);
        cuda::memory::PtrDevicePadded<TestType> d_data(size3_t(shape.x, shape.y, shape.z * batches));
        cuda::memory::PtrDevice<TestType> d_results(2 * batches);
        cpu::memory::PtrHost<TestType> h_cuda_results(2 * batches);

        test::initDataRandom(h_data.get(), h_data.elements(), randomizer);

        cuda::Stream stream(cuda::Stream::SERIAL);
        cuda::memory::copy(h_data.get(), shape.x,
                           d_data.get(), d_data.pitch(), d_data.shape(), stream);
        cuda::math::sumMean(d_data.get(), d_data.pitch(), d_results.get(), d_results.get() + batches,
                            shape, batches, stream);
        cpu::math::sumMean(h_data.get(), h_results.get(), h_results.get() + batches, elements, batches);
        cuda::Stream::synchronize(stream);

        cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
        TestType diff = test::getAverageNormalizedDifference(h_results.get(), h_cuda_results.get(), batches * 2);
        if constexpr (std::is_integral_v<TestType>)
            REQUIRE(diff == abs_epsilon);
        else
            REQUIRE_THAT(diff, test::isWithinAbs(TestType(0.), abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("cuda::math:: reductions, minMaxSumMean, contiguous", "[noa][cuda][math]",
                   float, double, int) {
    size_t batches = test::IntRandomizer<size_t>(1, 5).get();
    size_t elements = test::IntRandomizer<size_t>(1, 262144).get();

    cpu::memory::PtrHost<TestType> h_data(elements * batches);
    cpu::memory::PtrHost<TestType> h_results(batches * 4); // all mins, all maxs, all sums, all means.
    cuda::memory::PtrDevice<TestType> d_data(elements * batches);
    cuda::memory::PtrDevice<TestType> d_results(batches * 4);
    cpu::memory::PtrHost<TestType> h_cuda_results(batches * 4);

    cuda::Stream stream(cuda::Stream::SERIAL);
    test::Randomizer<TestType> randomizer(-100., 100.);
    test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);

    cuda::math::minMaxSumMean(d_data.get(),
                              d_results.get(),
                              d_results.get() + batches,
                              d_results.get() + batches * 2,
                              d_results.get() + batches * 3,
                              elements, batches, stream);
    cpu::math::minMaxSumMean(h_data.get(),
                             h_results.get(),
                             h_results.get() + batches,
                             h_results.get() + batches * 2,
                             h_results.get() + batches * 3,
                             elements, batches);
    cuda::Stream::synchronize(stream);
    cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    TestType diff = test::getDifference(h_results.get(), h_cuda_results.get(), batches * 2);
    REQUIRE(diff == 0);
    diff = test::getAverageNormalizedDifference(h_results.get() + batches * 2,
                                                h_cuda_results.get() + batches * 2, batches * 2);
    if constexpr (std::is_floating_point_v<TestType>) {
        REQUIRE_THAT(diff, test::isWithinAbs(0., math::Limits<TestType>::epsilon() * 100));
    } else {
        REQUIRE(diff == 0);
    }
}

TEMPLATE_TEST_CASE("cuda::math:: reductions, minMaxSumMean, padded", "[noa][cuda][math]",
                   float, double, int) {
    size_t batches = test::IntRandomizer<size_t>(1, 5).get();
    size3_t shape = test::getRandomShape(3);
    size_t elements = noa::elements(shape);
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);

    cpu::memory::PtrHost<TestType> h_data(elements * batches);
    cpu::memory::PtrHost<TestType> h_results(batches * 4); // all mins, all maxs, all sums, all means.
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
    cuda::memory::PtrDevice<TestType> d_results(batches * 4);
    cpu::memory::PtrHost<TestType> h_cuda_results(batches * 4);

    cuda::Stream stream(cuda::Stream::SERIAL);
    test::Randomizer<TestType> randomizer(-100., 100.);
    test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

    cuda::math::minMaxSumMean(d_data.get(), d_data.pitch(),
                              d_results.get(),
                              d_results.get() + batches,
                              d_results.get() + batches * 2,
                              d_results.get() + batches * 3,
                              shape, batches, stream);
    cpu::math::minMaxSumMean(h_data.get(),
                             h_results.get(),
                             h_results.get() + batches,
                             h_results.get() + batches * 2,
                             h_results.get() + batches * 3,
                             elements, batches);
    cuda::Stream::synchronize(stream);
    cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);
    TestType diff = test::getDifference(h_results.get(), h_cuda_results.get(), batches * 2);
    REQUIRE(diff == 0);
    diff = test::getAverageNormalizedDifference(h_results.get() + batches * 2,
                                                h_cuda_results.get() + batches * 2, batches * 2);
    if constexpr (std::is_floating_point_v<TestType>) {
        REQUIRE_THAT(diff, test::isWithinAbs(0., math::Limits<TestType>::epsilon() * 100));
    } else {
        REQUIRE(diff == 0);
    }
}

TEMPLATE_TEST_CASE("cuda::math:: reductions, statistics, contiguous", "[noa][cuda][math]", float, double) {
    size_t batches = test::IntRandomizer<size_t>(1, 5).get();
    size_t elements = test::IntRandomizer<size_t>(1, 262144).get();

    cpu::memory::PtrHost<TestType> h_data(elements * batches);
    cpu::memory::PtrHost<TestType> h_results(
            batches * 6); // all mins, all maxs, all sums, all means, all variances, all stddevs.
    cuda::memory::PtrDevice<TestType> d_data(elements * batches);
    cuda::memory::PtrDevice<TestType> d_results(batches * 6);
    cpu::memory::PtrHost<TestType> h_cuda_results(batches * 6);

    cuda::Stream stream(cuda::Stream::SERIAL);
    test::Randomizer<TestType> randomizer(-100., 100.);
    test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);

    cuda::math::statistics(d_data.get(),
                           d_results.get(),
                           d_results.get() + batches,
                           d_results.get() + batches * 2,
                           d_results.get() + batches * 3,
                           d_results.get() + batches * 4,
                           d_results.get() + batches * 5,
                           elements, batches, stream);
    cpu::math::statistics(h_data.get(),
                          h_results.get(),
                          h_results.get() + batches,
                          h_results.get() + batches * 2,
                          h_results.get() + batches * 3,
                          h_results.get() + batches * 4,
                          h_results.get() + batches * 5,
                          elements, batches);
    cuda::Stream::synchronize(stream);
    cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);

    TestType diff = test::getDifference(h_results.get(), h_cuda_results.get(), batches * 2);
    REQUIRE(diff == 0);
    diff = test::getAverageNormalizedDifference(h_results.get() + batches * 2,
                                                h_cuda_results.get() + batches * 2, batches * 4);
    if constexpr (std::is_floating_point_v<TestType>) {
        REQUIRE_THAT(diff, test::isWithinAbs(0., math::Limits<TestType>::epsilon() * 100));
    } else {
        REQUIRE(diff == 0);
    }
}

TEMPLATE_TEST_CASE("cuda::math:: reductions, statistics, padded", "[noa][cuda][math]", float, double) {
    size_t batches = test::IntRandomizer<size_t>(1, 5).get();
    size3_t shape = test::getRandomShape(3);
    size_t elements = noa::elements(shape);
    size3_t shape_batched(shape.x, shape.y * shape.z, batches);

    cpu::memory::PtrHost<TestType> h_data(elements * batches);
    cpu::memory::PtrHost<TestType> h_results(
            batches * 6); // all mins, all maxs, all sums, all means, all variances, all stddevs.
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
    cuda::memory::PtrDevice<TestType> d_results(batches * 6);
    cpu::memory::PtrHost<TestType> h_cuda_results(batches * 6);

    cuda::Stream stream(cuda::Stream::SERIAL);
    test::Randomizer<TestType> randomizer(-100., 100.);
    test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

    cuda::math::statistics(d_data.get(), d_data.pitch(),
                           d_results.get(),
                           d_results.get() + batches,
                           d_results.get() + batches * 2,
                           d_results.get() + batches * 3,
                           d_results.get() + batches * 4,
                           d_results.get() + batches * 5,
                           shape, batches, stream);
    cpu::math::statistics(h_data.get(),
                          h_results.get(),
                          h_results.get() + batches,
                          h_results.get() + batches * 2,
                          h_results.get() + batches * 3,
                          h_results.get() + batches * 4,
                          h_results.get() + batches * 5,
                          elements, batches);
    cuda::Stream::synchronize(stream);
    cuda::memory::copy(d_results.get(), h_cuda_results.get(), h_cuda_results.size(), stream);

    TestType diff = test::getDifference(h_results.get(), h_cuda_results.get(), batches * 2);
    REQUIRE(diff == 0);

    // sum & mean
    diff = test::getAverageNormalizedDifference(h_results.get() + batches * 2,
                                                h_cuda_results.get() + batches * 2, batches * 2);
    if constexpr (std::is_floating_point_v<TestType>) {
        REQUIRE_THAT(diff, test::isWithinAbs(0., math::Limits<TestType>::epsilon() * 100));
    } else {
        REQUIRE(diff == 0);
    }

    // variance & stddev: expect slightly lower precision
    diff = test::getAverageNormalizedDifference(h_results.get() + batches * 4,
                                                h_cuda_results.get() + batches * 4, batches * 2);
    if constexpr (std::is_floating_point_v<TestType>) {
        REQUIRE_THAT(diff, test::isWithinAbs(0., math::Limits<TestType>::epsilon() * 1000));
    } else {
        REQUIRE(diff == 0);
    }
}

TEMPLATE_TEST_CASE("cuda::math:: reductions, reduce*, contiguous", "[noa][cuda][math]",
                   int, float, double, cfloat_t, cdouble_t) {
    size_t batches = test::IntRandomizer<size_t>(1, 5).get();
    size_t vectors = test::IntRandomizer<size_t>(1, 5).get();
    size_t elements = test::IntRandomizer<size_t>(1, 100000).get();

    cpu::memory::PtrHost<TestType> h_vectors(elements * vectors * batches);
    cpu::memory::PtrHost<TestType> h_reduced(elements * batches);

    test::Randomizer<TestType> randomizer(-100., 100.);
    test::initDataRandom(h_vectors.get(), h_vectors.elements(), randomizer);

    cuda::Stream stream(cuda::Stream::SERIAL);
    cuda::memory::PtrDevice<TestType> d_vectors(h_vectors.elements());
    cuda::memory::PtrDevice<TestType> d_reduced(h_reduced.elements());
    cpu::memory::PtrHost<TestType> h_cuda_reduced(h_reduced.elements());
    cuda::memory::copy(h_vectors.get(), d_vectors.get(), h_vectors.size(), stream);

    AND_THEN("reduceAdd") {
        cuda::math::reduceAdd(d_vectors.get(), d_reduced.get(), elements, vectors, batches, stream);
        cuda::memory::copy(d_reduced.get(), h_cuda_reduced.get(), d_reduced.size(), stream);
        cpu::math::reduceAdd(h_vectors.get(), h_reduced.get(), elements, vectors, batches);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_reduced.get(), h_cuda_reduced.get(), h_reduced.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0.), 1e-5));
    }

    AND_THEN("reduceMean") {
        cuda::math::reduceMean(d_vectors.get(), d_reduced.get(), elements, vectors, batches, stream);
        cuda::memory::copy(d_reduced.get(), h_cuda_reduced.get(), d_reduced.size(), stream);
        cpu::math::reduceMean(h_vectors.get(), h_reduced.get(), elements, vectors, batches);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_reduced.get(), h_cuda_reduced.get(), h_reduced.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0.), 1e-5));
    }

    AND_THEN("reduceMeanWeighted") {
        using real_t = noa::traits::value_type_t<TestType>;
        cpu::memory::PtrHost<real_t> h_weights(elements * vectors * batches);
        cuda::memory::PtrDevice<real_t> d_weights(h_weights.elements());
        test::Randomizer<real_t> randomizer_real(0., 10.);
        test::initDataRandom(h_weights.get(), h_weights.elements(), randomizer_real);
        cuda::memory::copy(h_weights.get(), d_weights.get(), d_weights.size(), stream);

        cuda::math::reduceMeanWeighted(d_vectors.get(), d_weights.get(), d_reduced.get(),
                                       elements, vectors, batches, stream);
        cuda::memory::copy(d_reduced.get(), h_cuda_reduced.get(), d_reduced.size(), stream);
        cpu::math::reduceMeanWeighted(h_vectors.get(), h_weights.get(), h_reduced.get(), elements, vectors, batches);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_reduced.get(), h_cuda_reduced.get(), h_reduced.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0.), 1e-5));
    }
}

TEMPLATE_TEST_CASE("cuda::math:: reductions, reduce*, padded", "[noa][cuda][math]",
                   int, float, double, cfloat_t, cdouble_t) {
    size_t batches = test::IntRandomizer<size_t>(1, 3).get();
    size_t vectors = test::IntRandomizer<size_t>(1, 3).get();
    size3_t shape = test::getRandomShape(2);
    size_t elements = noa::elements(shape);
    size3_t shape_batched(shape.x, shape.y * shape.z, vectors * batches);
    size3_t shape_reduced(shape.x, shape.y * shape.z, batches);

    cpu::memory::PtrHost<TestType> h_vectors(elements * vectors * batches);
    cpu::memory::PtrHost<TestType> h_reduced(elements * batches);

    test::Randomizer<TestType> randomizer(-100., 100.);
    test::initDataRandom(h_vectors.get(), h_vectors.elements(), randomizer);

    cuda::Stream stream(cuda::Stream::SERIAL);
    cuda::memory::PtrDevicePadded<TestType> d_vectors(shape_batched);
    cuda::memory::PtrDevicePadded<TestType> d_reduced(shape_reduced);
    cpu::memory::PtrHost<TestType> h_cuda_reduced(h_reduced.elements());
    cuda::memory::copy(h_vectors.get(), shape.x, d_vectors.get(), d_vectors.pitch(), shape_batched, stream);

    AND_THEN("reduceAdd") {
        cuda::math::reduceAdd(d_vectors.get(), d_vectors.pitch(), d_reduced.get(), d_reduced.pitch(),
                              shape, vectors, batches, stream);
        cuda::memory::copy(d_reduced.get(), d_reduced.pitch(), h_cuda_reduced.get(), shape.x,
                           shape_reduced, stream);
        cpu::math::reduceAdd(h_vectors.get(), h_reduced.get(), elements, vectors, batches);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_reduced.get(), h_cuda_reduced.get(), h_reduced.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0.), 1e-5));
    }

    AND_THEN("reduceMean") {
        cuda::math::reduceMean(d_vectors.get(), d_vectors.pitch(), d_reduced.get(), d_reduced.pitch(),
                               shape, vectors, batches, stream);
        cuda::memory::copy(d_reduced.get(), d_reduced.pitch(), h_cuda_reduced.get(), shape.x,
                           shape_reduced, stream);
        cpu::math::reduceMean(h_vectors.get(), h_reduced.get(), elements, vectors, batches);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_reduced.get(), h_cuda_reduced.get(), h_reduced.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0.), 1e-5));
    }

    AND_THEN("reduceMeanWeighted") {
        using real_t = noa::traits::value_type_t<TestType>;
        cpu::memory::PtrHost<real_t> h_weights(elements * vectors * batches);
        cuda::memory::PtrDevicePadded<real_t> d_weights(size3_t(shape.x, shape.y * shape.z, vectors));
        test::Randomizer<real_t> randomizer_real(0., 10.);
        test::initDataRandom(h_weights.get(), h_weights.elements(), randomizer_real);
        cuda::memory::copy(h_weights.get(), shape.x,
                           d_weights.get(), d_weights.pitch(),
                           d_weights.shape(), stream);

        cuda::math::reduceMeanWeighted(d_vectors.get(), d_vectors.pitch(),
                                       d_weights.get(), d_weights.pitch(),
                                       d_reduced.get(), d_reduced.pitch(),
                                       shape, vectors, batches, stream);
        cuda::memory::copy(d_reduced.get(), d_reduced.pitch(), h_cuda_reduced.get(), shape.x,
                           shape_reduced, stream);
        cpu::math::reduceMeanWeighted(h_vectors.get(), h_weights.get(), h_reduced.get(), elements, vectors, batches);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getAverageDifference(h_reduced.get(), h_cuda_reduced.get(), h_reduced.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0.), 1e-5));
    }
}
