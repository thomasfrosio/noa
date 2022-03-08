#include <noa/cpu/math/Ewise.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// Just test a few operators to make sure it compiles and indexes are computed correctly.
TEMPLATE_TEST_CASE("cpu::math::ewise() - unary operators", "[noa][cpu][math]",
                   int, uint, float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    cpu::Stream stream;
    cpu::memory::PtrHost<TestType> data(elements);
    cpu::memory::PtrHost<TestType> results(elements);
    cpu::memory::PtrHost<TestType> expected(elements);

    test::Randomizer<TestType> randomizer(-5, 5);
    test::randomize(data.get(), data.elements(), randomizer);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = TestType(1) - data[idx];

    cpu::math::ewise(data.get(), stride, results.get(), stride, shape, math::one_minus_t{}, stream);
    TestType diff = test::getDifference(expected.get(), results.get(), elements);
    REQUIRE(diff == TestType(0));
}

TEMPLATE_TEST_CASE("cpu::math::ewise() - unary operators - return bool", "[noa][cpu][math]", int, uint) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    cpu::Stream stream;
    cpu::memory::PtrHost<TestType> data(elements);
    cpu::memory::PtrHost<bool> results(elements);
    cpu::memory::PtrHost<bool> expected(elements);

    test::Randomizer<TestType> randomizer(-2, 2);
    test::randomize(data.get(), data.elements(), randomizer);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = !data[idx];

    cpu::math::ewise(data.get(), stride, results.get(), stride, shape, math::logical_not_t{}, stream);
    TestType diff = test::getDifference(expected.get(), results.get(), elements);
    REQUIRE(diff == TestType(0));
}

TEMPLATE_TEST_CASE("cpu::math::ewise() - binary operators", "[noa][cpu][math]",
                   int, uint, float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    cpu::Stream stream;
    cpu::memory::PtrHost<TestType> data(elements);
    cpu::memory::PtrHost<TestType> results(elements);
    cpu::memory::PtrHost<TestType> expected(elements);

    using real_t = noa::traits::value_type_t<TestType>;
    cpu::memory::PtrHost<real_t> values(shape[0]);
    cpu::memory::PtrHost<real_t> array(elements);

    test::Randomizer<real_t> randomizer(-5, 5);
    test::randomize(data.get(), data.elements(), randomizer);
    test::randomize(values.get(), values.elements(), randomizer);
    test::randomize(array.get(), array.elements(), randomizer);

    AND_THEN("value") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = data[idx] + values[0];
        cpu::math::ewise(data.get(), stride, values[0], results.get(), stride, shape, math::plus_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("values") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] + values[batch];
        cpu::math::ewise(data.get(), stride, values.get(), results.get(), stride, shape, math::plus_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] + array[idx];
        cpu::math::ewise(data.get(), stride, array.get(), {0, stride[1], stride[2], stride[3]},
                         results.get(), stride, shape, math::plus_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array-batches") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] + array[batch * stride[0] + idx];
        cpu::math::ewise(data.get(), stride, array.get(), stride,
                         results.get(), stride, shape, math::plus_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }
}

TEMPLATE_TEST_CASE("cpu::math::ewise() - binary operators - return bool", "[noa][cpu][math]",
                   int, uint, float, double) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    cpu::Stream stream;
    cpu::memory::PtrHost<TestType> data(elements);
    cpu::memory::PtrHost<bool> results(elements);
    cpu::memory::PtrHost<bool> expected(elements);

    using real_t = noa::traits::value_type_t<TestType>;
    cpu::memory::PtrHost<real_t> values(shape[0]);
    cpu::memory::PtrHost<real_t> array(elements);

    test::Randomizer<real_t> randomizer(-5, 5);
    test::randomize(data.get(), data.elements(), randomizer);
    test::randomize(values.get(), values.elements(), randomizer);
    test::randomize(array.get(), array.elements(), randomizer);

    AND_THEN("value") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = data[idx] < values[0];
        cpu::math::ewise(data.get(), stride, values[0], results.get(), stride, shape, math::less_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("values") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] < values[batch];
        cpu::math::ewise(data.get(), stride, values.get(), results.get(), stride, shape, math::less_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] < array[idx];
        cpu::math::ewise(data.get(), stride, array.get(), {0, stride[1], stride[2], stride[3]},
                         results.get(), stride, shape, math::less_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array-batched") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] < array[batch * stride[0] + idx];
        cpu::math::ewise(data.get(), stride, array.get(), stride, results.get(), stride,
                         shape, math::less_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }
}

TEMPLATE_TEST_CASE("cpu::math::ewise() - trinary operators", "[noa][cpu][math]",
                   int, uint, float, double) {
    test::Randomizer<TestType> randomizer(1., 100.);
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    cpu::Stream stream;
    cpu::memory::PtrHost<TestType> data(elements);
    cpu::memory::PtrHost<TestType> multiplicands(elements);
    cpu::memory::PtrHost<TestType> addends(elements);
    cpu::memory::PtrHost<TestType> results(elements);
    cpu::memory::PtrHost<TestType> expected(elements);

    test::randomize(data.get(), data.elements(), randomizer);
    test::randomize(multiplicands.get(), multiplicands.elements(), randomizer);
    test::randomize(addends.get(), addends.elements(), randomizer);

    AND_THEN("value") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] *
                                                    multiplicands[0] + addends[0];
        cpu::math::ewise(data.get(), stride, multiplicands[0], addends[0],
                         results.get(), stride, shape, math::fma_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("values") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] *
                                                    multiplicands[batch] + addends[batch];
        cpu::math::ewise(data.get(), stride, multiplicands.get(), addends.get(),
                         results.get(), stride, shape, math::fma_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] * multiplicands[idx] + addends[idx];
        cpu::math::ewise(data.get(), stride,
                         multiplicands.get(), {0, stride[1], stride[2], stride[3]},
                         addends.get(), {0, stride[1], stride[2], stride[3]},
                         results.get(), stride,
                         shape, math::fma_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array-batched") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] *
                                                   multiplicands[batch * stride[0] + idx] +
                                                   addends[batch * stride[0] + idx];
        cpu::math::ewise(data.get(), stride, multiplicands.get(), stride, addends.get(), stride,
                         results.get(), stride, shape, math::fma_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }
}

TEMPLATE_TEST_CASE("cpu::math::ewise() - trinary operators - return bool", "[noa][cpu][math]",
                   int, uint, float, double) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    cpu::Stream stream;
    cpu::memory::PtrHost<TestType> data(elements);
    cpu::memory::PtrHost<TestType> low(elements);
    cpu::memory::PtrHost<TestType> high(elements);
    cpu::memory::PtrHost<bool> results(elements);
    cpu::memory::PtrHost<bool> expected(elements);

    test::Randomizer<TestType> randomizer(-5, 5);
    test::Randomizer<TestType> randomizer_high(5, 10);
    test::randomize(data.get(), data.elements(), randomizer);
    test::randomize(low.get(), low.elements(), randomizer);
    test::randomize(high.get(), high.elements(), randomizer_high);

    AND_THEN("value") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] >= low[0] &&
                                                   data[batch * stride[0] + idx] <= high[0];
        cpu::math::ewise(data.get(), stride, low[0], high[0],
                         results.get(), stride, shape, math::within_equal_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("values") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] >= low[batch] &&
                                                   data[batch * stride[0] + idx] <= high[batch];
        cpu::math::ewise(data.get(), stride, low.get(), high.get(),
                         results.get(), stride, shape, math::within_equal_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] >= low[idx] &&
                                                   data[batch * stride[0] + idx] <= high[idx];
        cpu::math::ewise(data.get(), stride,
                         low.get(), {0, stride[1], stride[2], stride[3]},
                         high.get(), {0, stride[1], stride[2], stride[3]},
                         results.get(), stride,
                         shape, math::within_equal_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array - batched") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] >= low[batch * stride[0] + idx] &&
                                                   data[batch * stride[0] + idx] <= high[batch * stride[0] + idx];
        cpu::math::ewise(data.get(), stride, low.get(), stride, high.get(), stride,
                         results.get(), stride, shape, math::within_equal_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }
}
