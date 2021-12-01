#include <noa/cpu/fft/Plan.h>
#include <noa/cpu/fft/Transforms.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cpu::fft::r2c(), c2r()", "[noa][cpu][fft]", float, double) {
    test::Randomizer<TestType> randomizer(-5., 5.);
    using complex_t = noa::Complex<TestType>;

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = noa::elements(shape);
    size_t elements_fft = elementsFFT(shape);

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 1e-4;
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-12;

    AND_THEN("one time transform; out-of-place") {
        cpu::memory::PtrHost<TestType> input(elements);
        cpu::memory::PtrHost<TestType> expected(elements);
        cpu::memory::PtrHost<complex_t> transform(elements_fft);

        test::randomize(input.get(), input.elements(), randomizer);
        std::memcpy(expected.get(), input.get(), input.bytes());

        cpu::fft::r2c(input.get(), transform.get(), shape, 1);
        test::memset(input.get(), input.elements(), 0); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements));
        cpu::fft::c2r(transform.get(), input.get(), shape, 1);

        TestType diff = test::getAverageDifference(input.get(), expected.get(), elements);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));
    }

    AND_THEN("one time transform; in-place") {
        // Extra padding to store the complex transform.
        size_t pitch = (shape.x / 2 + 1) * 2;
        cpu::memory::PtrHost<TestType> input(elements_fft * 2);
        cpu::memory::PtrHost<TestType> expected(elements);

        test::randomize(input.get(), input.elements(), randomizer);
        for (size_t row = 0; row < rows(shape); ++row) {
            std::memcpy(expected.get() + row * shape.x, // expected is not padded.
                        input.get() + row * pitch, // input is padded.
                        shape.x * sizeof(TestType));
        }

        cpu::fft::r2c(input.get(), shape, 1);
        test::normalize(input.get(), input.elements(), 1 / static_cast<TestType>(elements));
        cpu::fft::c2r(reinterpret_cast<complex_t*>(input.get()), shape, 1);

        TestType diff = 0;
        for (size_t row = 0; row < rows(shape); ++row) {
            diff += test::getDifference(input.get() + row * pitch,
                                        expected.get() + row * shape.x,
                                        shape.x);
        }
        diff /= static_cast<TestType>(elements);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));
    }

    AND_THEN("execute and new-arrays functions") {
        size_t batches = 2;
        cpu::memory::PtrHost<TestType> input(elements * batches);
        cpu::memory::PtrHost<TestType> output(elements * batches);
        cpu::memory::PtrHost<complex_t> transform(elements_fft * batches);

        cpu::fft::Flag flag = cpu::fft::ESTIMATE;
        cpu::fft::Plan<TestType> plan_forward(input.get(), transform.get(), shape, batches, flag);
        cpu::fft::Plan<TestType> plan_backward(transform.get(), input.get(), shape, batches, flag);
        test::randomize(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements * sizeof(TestType) * batches);

        cpu::fft::execute(plan_forward);
        test::memset(input.get(), input.elements(), 0); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements));
        cpu::fft::execute(plan_backward);

        TestType diff = test::getDifference(input.get(), output.get(), elements * batches);
        diff /= static_cast<TestType>(elements * batches);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));

        // New arrays.
        cpu::memory::PtrHost<TestType> input_new(elements * batches);
        cpu::memory::PtrHost<complex_t> transform_new(elements_fft * batches);
        test::randomize(input_new.get(), input_new.elements(), randomizer);
        std::memcpy(output.get(), input_new.get(), elements * sizeof(TestType) * batches);

        cpu::fft::r2c(input_new.get(), transform_new.get(), plan_forward);
        test::memset(input_new.get(), input_new.elements(), 0); // just make sure new data is written.
        test::normalize(transform_new.get(), transform_new.elements(), 1 / static_cast<TestType>(elements));
        cpu::fft::c2r(transform_new.get(), input_new.get(), plan_backward);

        diff = test::getDifference(input_new.get(), output.get(), elements * batches);
        diff /= static_cast<TestType>(elements);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::c2c()", "[noa][cpu][fft]", float, double) {
    using complex_t = noa::Complex<TestType>;
    test::Randomizer<complex_t> randomizer(-5., 5.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim); // the entire API is ndim "agnostic".
    size_t size = elements(shape);

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 1e-4;
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-12;

    AND_THEN("one time transform; out-of-place") {
        cpu::memory::PtrHost<complex_t> input(size);
        cpu::memory::PtrHost<complex_t> output(size);
        cpu::memory::PtrHost<complex_t> transform(size);

        test::randomize(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), size * sizeof(complex_t));

        cpu::fft::c2c(input.get(), transform.get(), shape, 1, fft::FORWARD);
        test::memset(input.get(), input.elements(), 0); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(size));
        cpu::fft::c2c(transform.get(), input.get(), shape, 1, fft::BACKWARD);

        complex_t diff = test::getAverageDifference(input.get(), output.get(), size);
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));
    }

    AND_THEN("one time transform; in-place") {
        cpu::memory::PtrHost<complex_t> input(size);
        cpu::memory::PtrHost<complex_t> output(size);

        test::randomize(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), size * sizeof(complex_t));

        cpu::fft::c2c(input.get(), input.get(), shape, 1, fft::FORWARD);
        test::normalize(input.get(), input.elements(), 1 / static_cast<TestType>(size));
        cpu::fft::c2c(input.get(), input.get(), shape, 1, fft::BACKWARD);

        complex_t diff = test::getAverageDifference(input.get(), output.get(), size);
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));
    }

    AND_THEN("execute and new-arrays functions") {
        size_t batches = 2;
        cpu::memory::PtrHost<complex_t> input(size * batches);
        cpu::memory::PtrHost<complex_t> output(size * batches);
        cpu::memory::PtrHost<complex_t> transform(size * batches);

        cpu::fft::Flag flag = cpu::fft::ESTIMATE;
        cpu::fft::Plan<TestType> plan_fwd(input.get(), transform.get(), shape, batches, fft::FORWARD, flag);
        cpu::fft::Plan<TestType> plan_bwd(transform.get(), input.get(), shape, batches, fft::BACKWARD, flag);
        test::randomize(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), size * sizeof(complex_t) * batches);

        cpu::fft::execute(plan_fwd);
        test::memset(input.get(), input.elements(), 0); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(size));
        cpu::fft::execute(plan_bwd);

        complex_t diff = test::getDifference(input.get(), output.get(), size * batches);
        diff /= static_cast<TestType>(size * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));

        // New arrays.
        cpu::memory::PtrHost<complex_t> input_new(size * batches);
        cpu::memory::PtrHost<complex_t> transform_new(size * batches);
        test::randomize(input_new.get(), input_new.elements(), randomizer);
        std::memcpy(output.get(), input_new.get(), size * sizeof(complex_t) * batches);

        cpu::fft::c2c(input_new.get(), transform_new.get(), plan_fwd);
        test::memset(input_new.get(), input_new.elements(), 0); // just make sure new data is written.
        test::normalize(transform_new.get(), transform_new.elements(), 1 / static_cast<TestType>(size));
        cpu::fft::c2c(transform_new.get(), input_new.get(), plan_bwd);

        diff = test::getDifference(input_new.get(), output.get(), size * batches);
        diff /= static_cast<TestType>(size * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));
    }
}
