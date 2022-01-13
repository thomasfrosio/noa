#include <noa/cpu/math/Ewise.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// Just test a few operators to make sure it compiles and indexes are computed correctly.
TEMPLATE_TEST_CASE("cpu::math::ewise() - unary operators", "[noa][cpu][math]",
                   int, uint, float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size3_t shape = test::getRandomShape(ndim);
    const size_t elements = noa::elements(shape);
    const size_t batches = test::Randomizer<size_t>(1, 4).get();

    cpu::Stream stream;
    cpu::memory::PtrHost<TestType> data(elements * batches);
    cpu::memory::PtrHost<TestType> results(elements * batches);
    cpu::memory::PtrHost<TestType> expected(elements * batches);

    test::Randomizer<TestType> randomizer(-5, 5);
    test::randomize(data.get(), data.elements(), randomizer);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = TestType(1) - data[idx];

    cpu::math::ewise(data.get(), shape, results.get(), shape, shape, batches, math::one_minus_t{}, stream);
    TestType diff = test::getDifference(expected.get(), results.get(), elements);
    REQUIRE(diff == TestType(0));
}

TEMPLATE_TEST_CASE("cpu::math::ewise() - unary operators - return bool", "[noa][cpu][math]", int, uint) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size3_t shape = test::getRandomShape(ndim);
    const size_t elements = noa::elements(shape);
    const size_t batches = test::Randomizer<size_t>(1, 4).get();

    cpu::Stream stream;
    cpu::memory::PtrHost<TestType> data(elements * batches);
    cpu::memory::PtrHost<bool> results(elements * batches);
    cpu::memory::PtrHost<bool> expected(elements * batches);

    test::Randomizer<TestType> randomizer(-2, 2);
    test::randomize(data.get(), data.elements(), randomizer);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = !data[idx];

    cpu::math::ewise(data.get(), shape, results.get(), shape, shape, batches, math::not_t{}, stream);
    TestType diff = test::getDifference(expected.get(), results.get(), elements);
    REQUIRE(diff == TestType(0));
}

TEMPLATE_TEST_CASE("cpu::math::ewise() - binary operators", "[noa][cpu][math]",
                   int, uint, float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size3_t shape = test::getRandomShape(ndim);
    const size_t elements = noa::elements(shape);
    const size_t batches = test::Randomizer<size_t>(1, 4).get();

    cpu::Stream stream;
    cpu::memory::PtrHost<TestType> data(elements * batches);
    cpu::memory::PtrHost<TestType> results(elements * batches);
    cpu::memory::PtrHost<TestType> expected(elements * batches);

    using real_t = noa::traits::value_type_t<TestType>;
    cpu::memory::PtrHost<real_t> values(batches);
    cpu::memory::PtrHost<real_t> array(elements * batches);

    test::Randomizer<real_t> randomizer(-5, 5);
    test::randomize(data.get(), data.elements(), randomizer);
    test::randomize(values.get(), values.elements(), randomizer);
    test::randomize(array.get(), array.elements(), randomizer);


    AND_THEN("value") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] + values[0];
        cpu::math::ewise(data.get(), shape, values[0], results.get(), shape, shape, batches, math::plus_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("values") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] + values[batch];
        cpu::math::ewise(data.get(), shape, values.get(), results.get(), shape, shape, batches, math::plus_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] + array[idx];
        cpu::math::ewise(data.get(), shape, array.get(), {shape.x, shape.y, 0},
                         results.get(), shape, shape, batches, math::plus_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array-batches") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] + array[batch * elements + idx];
        cpu::math::ewise(data.get(), shape, array.get(), shape,
                         results.get(), shape, shape, batches, math::plus_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0));
    }
}

TEMPLATE_TEST_CASE("cpu::math::ewise() - binary operators - return bool", "[noa][cpu][math]",
                   int, uint, float, double) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size3_t shape = test::getRandomShape(ndim);
    const size_t elements = noa::elements(shape);
    const size_t batches = test::Randomizer<size_t>(1, 4).get();

    cpu::Stream stream;
    cpu::memory::PtrHost<TestType> data(elements * batches);
    cpu::memory::PtrHost<bool> results(elements * batches);
    cpu::memory::PtrHost<bool> expected(elements * batches);

    using real_t = noa::traits::value_type_t<TestType>;
    cpu::memory::PtrHost<real_t> values(batches);
    cpu::memory::PtrHost<real_t> array(elements * batches);

    test::Randomizer<real_t> randomizer(-5, 5);
    TestType value = randomizer.get();
    test::randomize(data.get(), data.elements(), randomizer);
    test::randomize(values.get(), values.elements(), randomizer);
    test::randomize(array.get(), array.elements(), randomizer);

    AND_THEN("value") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] < value;
        cpu::math::ewise(data.get(), shape, value, results.get(), shape, shape, batches, math::less_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("values") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] < values[batch];
        cpu::math::ewise(data.get(), shape, values.get(), results.get(), shape, shape, batches, math::less_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] < array[idx];
        cpu::math::ewise(data.get(), shape, array.get(), {shape.x, shape.y, 0}, results.get(), shape,
                         shape, batches, math::less_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array-batched") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] < array[batch * elements + idx];
        cpu::math::ewise(data.get(), shape, array.get(), shape, results.get(), shape,
                         shape, batches, math::less_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0));
    }
}

TEMPLATE_TEST_CASE("cpu::math::ewise() - trinary operators", "[noa][cpu][math]",
                   int, uint, float, double) {
    test::Randomizer<TestType> randomizer(1., 100.);
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size3_t shape = test::getRandomShape(ndim);
    const size_t batches = test::Randomizer<size_t>(1, 4).get();
    const size_t elements = noa::elements(shape);

    cpu::Stream stream;
    cpu::memory::PtrHost<TestType> data(elements * batches);
    cpu::memory::PtrHost<TestType> multiplicands(elements * batches);
    cpu::memory::PtrHost<TestType> addends(elements * batches);
    cpu::memory::PtrHost<TestType> results(elements * batches);
    cpu::memory::PtrHost<TestType> expected(elements * batches);

    test::randomize(data.get(), data.elements(), randomizer);
    test::randomize(multiplicands.get(), multiplicands.elements(), randomizer);
    test::randomize(addends.get(), addends.elements(), randomizer);

    AND_THEN("value") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] * multiplicands[0] + addends[0];
        cpu::math::ewise(data.get(), shape, multiplicands[0], addends[0],
                         results.get(), shape, shape, batches, math::fma_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("values") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] * multiplicands[batch] + addends[batch];
        cpu::math::ewise(data.get(), shape, multiplicands.get(), addends.get(),
                         results.get(), shape, shape, batches, math::fma_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] * multiplicands[idx] + addends[idx];
        cpu::math::ewise(data.get(), shape,
                         multiplicands.get(), {shape.x, shape.y, 0},
                         addends.get(), {shape.x, shape.y, 0},
                         results.get(), shape,
                         shape, batches, math::fma_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array-batched") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] *
                                                   multiplicands[batch * elements + idx] +
                                                   addends[batch * elements + idx];
        cpu::math::ewise(data.get(), shape, multiplicands.get(), shape, addends.get(), shape,
                         results.get(), shape, shape, batches, math::fma_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0));
    }
}

TEMPLATE_TEST_CASE("cpu::math::ewise() - trinary operators - return bool", "[noa][cpu][math]",
                   int, uint, float, double) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size3_t shape = test::getRandomShape(ndim);
    const size_t batches = test::Randomizer<size_t>(1, 4).get();
    const size_t elements = noa::elements(shape);

    cpu::Stream stream;
    cpu::memory::PtrHost<TestType> data(elements * batches);
    cpu::memory::PtrHost<TestType> low(elements * batches);
    cpu::memory::PtrHost<TestType> high(elements * batches);
    cpu::memory::PtrHost<bool> results(elements * batches);
    cpu::memory::PtrHost<bool> expected(elements * batches);

    test::Randomizer<TestType> randomizer(-5, 5);
    test::Randomizer<TestType> randomizer_high(5, 10);
    test::randomize(data.get(), data.elements(), randomizer);
    test::randomize(low.get(), low.elements(), randomizer);
    test::randomize(high.get(), high.elements(), randomizer_high);

    AND_THEN("value") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] >= low[0] &&
                                                   data[batch * elements + idx] <= high[0];
        cpu::math::ewise(data.get(), shape, low[0], high[0],
                         results.get(), shape, shape, batches, math::within_equal_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("values") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] >= low[batch] &&
                                                   data[batch * elements + idx] <= high[batch];
        cpu::math::ewise(data.get(), shape, low.get(), high.get(),
                         results.get(), shape, shape, batches, math::within_equal_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] >= low[idx] &&
                                                   data[batch * elements + idx] <= high[idx];
        cpu::math::ewise(data.get(), shape,
                         low.get(), {shape.x, shape.y, 0},
                         high.get(), {shape.x, shape.y, 0},
                         results.get(), shape,
                         shape, batches, math::within_equal_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0));
    }

    AND_THEN("array - batched") {
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = data[batch * elements + idx] >= low[batch * elements + idx] &&
                                                   data[batch * elements + idx] <= high[batch * elements + idx];
        cpu::math::ewise(data.get(), shape, low.get(), shape, high.get(), shape,
                         results.get(), shape, shape, batches, math::within_equal_t{}, stream);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0));
    }
}
