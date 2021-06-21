#include <noa/cpu/fourier/Plan.h>
#include <noa/cpu/fourier/Transforms.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("Fourier: transforms for real inputs", "[noa][fourier]", float, double) {
    test::RealRandomizer<TestType> randomizer(-5., 5.);
    using complex_t = noa::Complex<TestType>;

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape_real = test::getRandomShape(ndim); // the entire API is ndim "agnostic".
    size_t elements_real = getElements(shape_real);
    size_t elements_complex = getElementsFFT(shape_real);

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 1e-4;
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-12;

    AND_THEN("one time transform; out-of-place") {
        memory::PtrHost<TestType> input(elements_real);
        memory::PtrHost<TestType> output(elements_real);
        memory::PtrHost<complex_t> transform(elements_complex);

        test::initDataRandom(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements_real * sizeof(TestType));

        fourier::r2c(input.get(), transform.get(), shape_real, 1);
        test::initDataZero(input.get(), input.elements()); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements_real));
        fourier::c2r(transform.get(), input.get(), shape_real, 1);

        TestType diff = test::getAverageDifference(input.get(), output.get(), elements_real);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));
    }

    AND_THEN("one time transform; in-place") {
        // Extra padding to store the complex transform.
        size_t padding_per_row = (shape_real.x % 2) ? 1 : 2;
        size_t elements_row = shape_real.x + padding_per_row;
        memory::PtrHost<TestType> input(elements_real + getRows(shape_real) * padding_per_row);
        memory::PtrHost<TestType> output(elements_real);

        test::initDataRandom(input.get(), input.elements(), randomizer);
        for (size_t row{0}; row < getRows(shape_real); ++row) {
            std::memcpy(output.get() + row * shape_real.x, // output is not padded.
                        input.get() + row * elements_row, // input is padded.
                        shape_real.x * sizeof(TestType));
        }

        auto* transform = reinterpret_cast<complex_t*>(input.get());
        fourier::r2c(input.get(), shape_real, 1);
        test::normalize(transform, elements_complex, 1 / static_cast<TestType>(elements_real));
        fourier::c2r(transform, shape_real, 1);

        TestType diff{0};
        for (size_t row{0}; row < getRows(shape_real); ++row) {
            diff += test::getDifference(input.get() + row * elements_row,
                                        output.get() + row * shape_real.x,
                                        shape_real.x);
        }
        diff /= static_cast<TestType>(elements_real);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));
    }

    AND_THEN("execute and new-arrays functions") {
        uint batches = 2;
        memory::PtrHost<TestType> input(elements_real * batches);
        memory::PtrHost<TestType> output(elements_real * batches);
        memory::PtrHost<complex_t> transform(elements_complex * batches);

        fourier::Plan<TestType> plan_forward(input.get(), transform.get(), shape_real, batches, fourier::ESTIMATE);
        fourier::Plan<TestType> plan_backward(transform.get(), input.get(), shape_real, batches, fourier::ESTIMATE);
        test::initDataRandom(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements_real * sizeof(TestType) * batches);

        fourier::execute(plan_forward);
        test::initDataZero(input.get(), input.elements()); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements_real));
        fourier::execute(plan_backward);

        TestType diff = test::getDifference(input.get(), output.get(), elements_real * batches);
        diff /= static_cast<TestType>(elements_real * batches);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));

        // New arrays.
        memory::PtrHost<TestType> input_new(elements_real * batches);
        memory::PtrHost<complex_t> transform_new(elements_complex * batches);
        test::initDataRandom(input_new.get(), input_new.elements(), randomizer);
        std::memcpy(output.get(), input_new.get(), elements_real * sizeof(TestType) * batches);

        fourier::r2c(input_new.get(), transform_new.get(), plan_forward);
        test::initDataZero(input_new.get(), input_new.elements()); // just make sure new data is written.
        test::normalize(transform_new.get(), transform_new.elements(), 1 / static_cast<TestType>(elements_real));
        fourier::c2r(transform_new.get(), input_new.get(), plan_backward);

        diff = test::getDifference(input_new.get(), output.get(), elements_real * batches);
        diff /= static_cast<TestType>(elements_real);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("Fourier: transforms of complex inputs", "[noa][fourier]", float, double) {
    using complex_t = noa::Complex<TestType>;
    test::RealRandomizer<complex_t> randomizer(-5., 5.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim); // the entire API is ndim "agnostic".
    size_t elements = getElements(shape);

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 1e-4;
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-12;

    AND_THEN("one time transform; out-of-place") {
        memory::PtrHost<complex_t> input(elements);
        memory::PtrHost<complex_t> output(elements);
        memory::PtrHost<complex_t> transform(elements);

        test::initDataRandom(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements * sizeof(complex_t));

        fourier::c2c(input.get(), transform.get(), shape, 1, fourier::FORWARD);
        test::initDataZero(input.get(), input.elements()); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements));
        fourier::c2c(transform.get(), input.get(), shape, 1, fourier::BACKWARD);

        complex_t diff = test::getAverageDifference(input.get(), output.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));
    }

    AND_THEN("one time transform; in-place") {
        memory::PtrHost<complex_t> input(elements);
        memory::PtrHost<complex_t> output(elements);

        test::initDataRandom(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements * sizeof(complex_t));

        fourier::c2c(input.get(), input.get(), shape, 1, fourier::FORWARD);
        test::normalize(input.get(), input.elements(), 1 / static_cast<TestType>(elements));
        fourier::c2c(input.get(), input.get(), shape, 1, fourier::BACKWARD);

        complex_t diff = test::getAverageDifference(input.get(), output.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));
    }

    AND_THEN("execute and new-arrays functions") {
        uint batches = 2;
        memory::PtrHost<complex_t> input(elements * batches);
        memory::PtrHost<complex_t> output(elements * batches);
        memory::PtrHost<complex_t> transform(elements * batches);

        fourier::Plan<TestType> plan_fwd(input.get(), transform.get(), shape, batches,
                                         fourier::FORWARD, fourier::ESTIMATE);
        fourier::Plan<TestType> plan_bwd(transform.get(), input.get(), shape, batches,
                                         fourier::BACKWARD, fourier::ESTIMATE);
        test::initDataRandom(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements * sizeof(complex_t) * batches);

        fourier::execute(plan_fwd);
        test::initDataZero(input.get(), input.elements()); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements));
        fourier::execute(plan_bwd);

        complex_t diff = test::getDifference(input.get(), output.get(), elements * batches);
        diff /= static_cast<TestType>(elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));

        // New arrays.
        memory::PtrHost<complex_t> input_new(elements * batches);
        memory::PtrHost<complex_t> transform_new(elements * batches);
        test::initDataRandom(input_new.get(), input_new.elements(), randomizer);
        std::memcpy(output.get(), input_new.get(), elements * sizeof(complex_t) * batches);

        fourier::c2c(input_new.get(), transform_new.get(), plan_fwd);
        test::initDataZero(input_new.get(), input_new.elements()); // just make sure new data is written.
        test::normalize(transform_new.get(), transform_new.elements(), 1 / static_cast<TestType>(elements));
        fourier::c2c(transform_new.get(), input_new.get(), plan_bwd);

        diff = test::getDifference(input_new.get(), output.get(), elements * batches);
        diff /= static_cast<TestType>(elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));
    }
}
