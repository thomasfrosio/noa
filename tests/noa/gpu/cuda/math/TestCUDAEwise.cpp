#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/math/Ewise.h>
#include <noa/gpu/cuda/memory/PtrManaged.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// Just test a few operators to make sure it compiles and indexes are computed correctly.
TEMPLATE_TEST_CASE("cuda::math::ewise() - unary operators", "[noa][cuda][math]",
                   int, uint, float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    cuda::Stream stream;
    cuda::memory::PtrManaged<TestType> data(elements, stream);
    cuda::memory::PtrManaged<TestType> results(elements, stream);
    cpu::memory::PtrHost<TestType> expected(elements);

    test::Randomizer<TestType> randomizer(-5, 5);
    test::randomize(data.get(), data.elements(), randomizer);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = data[idx] * data[idx];

    cuda::math::ewise(data.share(), stride, results.share(), stride, shape, math::square_t{}, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, 1e-6));
}

TEMPLATE_TEST_CASE("cuda::math::ewise() - unary operators - return bool", "[noa][cuda][math]",
                   int32_t, uint32_t, int64_t, uint64_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    cuda::Stream stream;
    cuda::memory::PtrManaged<TestType> data(elements, stream);
    cuda::memory::PtrManaged<bool> results(elements, stream);
    cpu::memory::PtrHost<bool> expected(elements);

    test::Randomizer<TestType> randomizer(-2, 2);
    test::randomize(data.get(), data.elements(), randomizer);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = data[idx] != 0;

    cuda::math::ewise(data.share(), stride, results.share(), stride, shape, math::nonzero_t{}, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, 1e-6));
}

TEMPLATE_TEST_CASE("cuda::math::ewise() - binary operators", "[noa][cuda][math]",
                   int, uint, half_t, float, double, chalf_t, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    cuda::Stream stream;
    cuda::memory::PtrManaged<TestType> data(elements, stream);
    cuda::memory::PtrManaged<TestType> results(elements, stream);
    cpu::memory::PtrHost<TestType> expected(elements);

    using real_t = noa::traits::value_type_t<TestType>;
    cpu::memory::PtrHost<real_t> values(2);
    cuda::memory::PtrManaged<real_t> array(elements, stream);

    test::Randomizer<real_t> randomizer(-5, 5);
    test::randomize(data.get(), data.elements(), randomizer);
    test::randomize(values.get(), values.elements(), randomizer);
    test::randomize(array.get(), array.elements(), randomizer);

    double epsilon = std::is_same_v<TestType, half_t> ? 1e-3 : 1e-6;

    AND_THEN("value") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = data[idx] * values[0];
        cuda::math::ewise(data.share(), stride, values[0], results.share(), stride,
                          shape, math::multiply_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));

        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = values[1] - data[idx];
        cuda::math::ewise(values[1], data.share(), stride, results.share(), stride,
                          shape, math::minus_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));
    }

    AND_THEN("array") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] * array[idx];
        cuda::math::ewise(data.share(), stride, array.share(), {0, stride[1], stride[2], stride[3]},
                          results.share(), stride, shape, math::multiply_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));
    }

    AND_THEN("array-batches") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] * array[batch * stride[0] + idx];
        cuda::math::ewise(data.share(), stride, array.share(), stride,
                          results.share(), stride, shape, math::multiply_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));
    }
}

TEMPLATE_TEST_CASE("cuda::math::ewise() - binary operators - return bool", "[noa][cuda][math]",
                   int, uint, half_t, float, double) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    cuda::Stream stream;
    cuda::memory::PtrManaged<TestType> data(elements, stream);
    cuda::memory::PtrManaged<bool> results(elements, stream);
    cpu::memory::PtrHost<bool> expected(elements);

    using real_t = noa::traits::value_type_t<TestType>;
    cpu::memory::PtrHost<real_t> values(2);
    cuda::memory::PtrManaged<real_t> array(elements, stream);

    test::Randomizer<real_t> randomizer(-5, 5);
    test::randomize(data.get(), data.elements(), randomizer);
    test::randomize(values.get(), values.elements(), randomizer);
    test::randomize(array.get(), array.elements(), randomizer);

    double epsilon = std::is_same_v<TestType, half_t> ? 1e-3 : 1e-6;

    AND_THEN("value") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = data[idx] > values[0];
        cuda::math::ewise(data.share(), stride, values[0], results.share(), stride,
                          shape, math::greater_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));

        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = values[1] < data[idx];
        cuda::math::ewise(values[0], data.share(), stride, results.share(), stride,
                          shape, math::less_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));
    }

    AND_THEN("array") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] >= array[idx];
        cuda::math::ewise(data.share(), stride, array.share(), {0, stride[1], stride[2], stride[3]},
                          results.share(), stride, shape, math::greater_equal_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));
    }

    AND_THEN("array-batched") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] >= array[batch * stride[0] + idx];
        cuda::math::ewise(data.share(), stride, array.share(), stride, results.share(), stride,
                          shape, math::greater_equal_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));
    }
}

TEMPLATE_TEST_CASE("cuda::math::ewise() - trinary operators", "[noa][cuda][math]",
                   int, uint, half_t, float, double) {
    test::Randomizer<TestType> randomizer(1., 100.);
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    cuda::Stream stream;
    cuda::memory::PtrManaged<TestType> lhs(elements, stream);
    cuda::memory::PtrManaged<TestType> mhs(elements, stream);
    cuda::memory::PtrManaged<TestType> rhs(elements, stream);
    cuda::memory::PtrManaged<TestType> results(elements, stream);
    cpu::memory::PtrHost<TestType> expected(elements);

    test::randomize(lhs.get(), lhs.elements(), randomizer);
    test::randomize(mhs.get(), mhs.elements(), randomizer);
    test::randomize(rhs.get(), rhs.elements(), randomizer);

    double epsilon = std::is_same_v<TestType, half_t> ? 1e-3 : 1e-6;

    AND_THEN("value") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = lhs[batch * stride[0] + idx] *
                                                    mhs[0] + rhs[0];
        cuda::math::ewise(lhs.share(), stride, mhs[0], rhs[0],
                          results.share(), stride, shape, math::fma_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));
    }

    AND_THEN("array-value") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = (lhs[batch * stride[0] + idx] + mhs[idx]) / rhs[0];
        cuda::math::ewise(
                lhs.share(), stride,
                mhs.share(), {0, stride[1], stride[2], stride[3]},
                rhs[0],
                results.share(), stride,
                shape, math::plus_divide_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));
    }

    AND_THEN("value-array") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = (lhs[batch * stride[0] + idx] + mhs[0]) / rhs[idx];
        cuda::math::ewise(
                lhs.share(), stride,
                mhs[0],
                rhs.share(), {0, stride[1], stride[2], stride[3]},
                results.share(), stride,
                shape, math::plus_divide_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));
    }

    AND_THEN("array") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = lhs[batch * stride[0] + idx] * mhs[idx] + rhs[idx];
        cuda::math::ewise(
                lhs.share(), stride,
                mhs.share(), {0, stride[1], stride[2], stride[3]},
                rhs.share(), {0, stride[1], stride[2], stride[3]},
                results.share(), stride,
                shape, math::fma_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));
    }

    AND_THEN("array-batched") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = lhs[batch * stride[0] + idx] *
                                                    mhs[batch * stride[0] + idx] +
                                                    rhs[batch * stride[0] + idx];
        cuda::math::ewise(
                lhs.share(), stride,
                mhs.share(), stride,
                rhs.share(), stride,
                results.share(), stride,
                shape, math::fma_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));
    }
}

TEMPLATE_TEST_CASE("cuda::math::ewise() - trinary operators - return bool", "[noa][cuda][math]",
                   int, uint, half_t, float, double) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();

    cuda::Stream stream;
    cuda::memory::PtrManaged<TestType> data(elements, stream);
    cuda::memory::PtrManaged<TestType> low(elements, stream);
    cuda::memory::PtrManaged<TestType> high(elements, stream);
    cuda::memory::PtrManaged<bool> results(elements, stream);
    cpu::memory::PtrHost<bool> expected(elements);

    test::Randomizer<TestType> randomizer(-5, 5);
    test::Randomizer<TestType> randomizer_high(5, 10);
    test::randomize(data.get(), data.elements(), randomizer);
    test::randomize(low.get(), low.elements(), randomizer);
    test::randomize(high.get(), high.elements(), randomizer_high);

    double epsilon = std::is_same_v<TestType, half_t> ? 1e-3 : 1e-6;

    AND_THEN("value") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] >= low[0] &&
                                                    data[batch * stride[0] + idx] <= high[0];
        cuda::math::ewise(data.share(), stride, low[0], high[0],
                          results.share(), stride, shape, math::within_equal_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));
    }

    AND_THEN("array") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] >= low[idx] &&
                                                    data[batch * stride[0] + idx] <= high[idx];
        cuda::math::ewise(
                data.share(), stride,
                low.share(), {0, stride[1], stride[2], stride[3]},
                high.share(), {0, stride[1], stride[2], stride[3]},
                results.share(), stride,
                shape, math::within_equal_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));
    }

    AND_THEN("array - batched") {
        for (size_t batch{0}; batch < shape[0]; ++batch)
            for (size_t idx{0}; idx < stride[0]; ++idx)
                expected[batch * stride[0] + idx] = data[batch * stride[0] + idx] >= low[batch * stride[0] + idx] &&
                                                    data[batch * stride[0] + idx] <= high[batch * stride[0] + idx];
        cuda::math::ewise(
                data.share(), stride, low.share(), stride, high.share(), stride,
                results.share(), stride, shape, math::within_equal_t{}, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), results.get(), elements, epsilon));
    }
}
