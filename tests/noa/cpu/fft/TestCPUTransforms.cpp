#include <noa/cpu/fft/Plan.h>
#include <noa/cpu/fft/Transforms.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cpu::fft::r2c(), c2r()", "[noa][cpu][fft]", float, double) {
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
        cpu::memory::PtrHost<TestType> input(elements_real);
        cpu::memory::PtrHost<TestType> output(elements_real);
        cpu::memory::PtrHost<complex_t> transform(elements_complex);

        test::initDataRandom(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements_real * sizeof(TestType));

        cpu::fft::r2c(input.get(), transform.get(), shape_real, 1);
        test::initDataZero(input.get(), input.elements()); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements_real));
        cpu::fft::c2r(transform.get(), input.get(), shape_real, 1);

        TestType diff = test::getAverageDifference(input.get(), output.get(), elements_real);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));
    }

    AND_THEN("one time transform; in-place") {
        // Extra padding to store the complex transform.
        size_t padding_per_row = (shape_real.x % 2) ? 1 : 2;
        size_t elements_row = shape_real.x + padding_per_row;
        cpu::memory::PtrHost<TestType> input(elements_real + getRows(shape_real) * padding_per_row);
        cpu::memory::PtrHost<TestType> output(elements_real);

        test::initDataRandom(input.get(), input.elements(), randomizer);
        for (size_t row{0}; row < getRows(shape_real); ++row) {
            std::memcpy(output.get() + row * shape_real.x, // output is not padded.
                        input.get() + row * elements_row, // input is padded.
                        shape_real.x * sizeof(TestType));
        }

        auto* transform = reinterpret_cast<complex_t*>(input.get());
        cpu::fft::r2c(input.get(), shape_real, 1);
        test::normalize(transform, elements_complex, 1 / static_cast<TestType>(elements_real));
        cpu::fft::c2r(transform, shape_real, 1);

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
        cpu::memory::PtrHost<TestType> input(elements_real * batches);
        cpu::memory::PtrHost<TestType> output(elements_real * batches);
        cpu::memory::PtrHost<complex_t> transform(elements_complex * batches);

        cpu::fft::Plan<TestType> plan_forward(input.get(), transform.get(), shape_real, batches,
                                              cpu::fft::ESTIMATE);
        cpu::fft::Plan<TestType> plan_backward(transform.get(), input.get(), shape_real, batches,
                                               cpu::fft::ESTIMATE);
        test::initDataRandom(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements_real * sizeof(TestType) * batches);

        cpu::fft::execute(plan_forward);
        test::initDataZero(input.get(), input.elements()); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements_real));
        cpu::fft::execute(plan_backward);

        TestType diff = test::getDifference(input.get(), output.get(), elements_real * batches);
        diff /= static_cast<TestType>(elements_real * batches);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));

        // New arrays.
        cpu::memory::PtrHost<TestType> input_new(elements_real * batches);
        cpu::memory::PtrHost<complex_t> transform_new(elements_complex * batches);
        test::initDataRandom(input_new.get(), input_new.elements(), randomizer);
        std::memcpy(output.get(), input_new.get(), elements_real * sizeof(TestType) * batches);

        cpu::fft::r2c(input_new.get(), transform_new.get(), plan_forward);
        test::initDataZero(input_new.get(), input_new.elements()); // just make sure new data is written.
        test::normalize(transform_new.get(), transform_new.elements(), 1 / static_cast<TestType>(elements_real));
        cpu::fft::c2r(transform_new.get(), input_new.get(), plan_backward);

        diff = test::getDifference(input_new.get(), output.get(), elements_real * batches);
        diff /= static_cast<TestType>(elements_real);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::c2c()", "[noa][cpu][fft]", float, double) {
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
        cpu::memory::PtrHost<complex_t> input(elements);
        cpu::memory::PtrHost<complex_t> output(elements);
        cpu::memory::PtrHost<complex_t> transform(elements);

        test::initDataRandom(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements * sizeof(complex_t));

        cpu::fft::c2c(input.get(), transform.get(), shape, 1, cpu::fft::FORWARD);
        test::initDataZero(input.get(), input.elements()); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements));
        cpu::fft::c2c(transform.get(), input.get(), shape, 1, cpu::fft::BACKWARD);

        complex_t diff = test::getAverageDifference(input.get(), output.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));
    }

    AND_THEN("one time transform; in-place") {
        cpu::memory::PtrHost<complex_t> input(elements);
        cpu::memory::PtrHost<complex_t> output(elements);

        test::initDataRandom(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements * sizeof(complex_t));

        cpu::fft::c2c(input.get(), input.get(), shape, 1, cpu::fft::FORWARD);
        test::normalize(input.get(), input.elements(), 1 / static_cast<TestType>(elements));
        cpu::fft::c2c(input.get(), input.get(), shape, 1, cpu::fft::BACKWARD);

        complex_t diff = test::getAverageDifference(input.get(), output.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));
    }

    AND_THEN("execute and new-arrays functions") {
        uint batches = 2;
        cpu::memory::PtrHost<complex_t> input(elements * batches);
        cpu::memory::PtrHost<complex_t> output(elements * batches);
        cpu::memory::PtrHost<complex_t> transform(elements * batches);

        cpu::fft::Plan<TestType> plan_fwd(input.get(), transform.get(), shape, batches,
                                          cpu::fft::FORWARD, cpu::fft::ESTIMATE);
        cpu::fft::Plan<TestType> plan_bwd(transform.get(), input.get(), shape, batches,
                                          cpu::fft::BACKWARD, cpu::fft::ESTIMATE);
        test::initDataRandom(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements * sizeof(complex_t) * batches);

        cpu::fft::execute(plan_fwd);
        test::initDataZero(input.get(), input.elements()); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements));
        cpu::fft::execute(plan_bwd);

        complex_t diff = test::getDifference(input.get(), output.get(), elements * batches);
        diff /= static_cast<TestType>(elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));

        // New arrays.
        cpu::memory::PtrHost<complex_t> input_new(elements * batches);
        cpu::memory::PtrHost<complex_t> transform_new(elements * batches);
        test::initDataRandom(input_new.get(), input_new.elements(), randomizer);
        std::memcpy(output.get(), input_new.get(), elements * sizeof(complex_t) * batches);

        cpu::fft::c2c(input_new.get(), transform_new.get(), plan_fwd);
        test::initDataZero(input_new.get(), input_new.elements()); // just make sure new data is written.
        test::normalize(transform_new.get(), transform_new.elements(), 1 / static_cast<TestType>(elements));
        cpu::fft::c2c(transform_new.get(), input_new.get(), plan_bwd);

        diff = test::getDifference(input_new.get(), output.get(), elements * batches);
        diff /= static_cast<TestType>(elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));
    }
}
