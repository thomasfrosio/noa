#include <noa/cpu/fourier/Plan.h>
#include <noa/cpu/fourier/Transforms.h>
#include <noa/cpu/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("Fourier: transforms for real inputs", "[noa][fourier]", float, double) {
    Test::RealRandomizer<TestType> randomizer(-5, 5);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape_real = Test::getShapeReal(ndim); // the entire API is ndim "agnostic".
    size3_t shape_complex = {shape_real.x / 2 + 1, shape_real.y, shape_real.z};
    size_t elements_real = getElements(shape_real);
    size_t elements_complex = getElements(shape_complex);

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 1e-4;
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-12;

    AND_THEN("one time transform; out-of-place") {
        PtrHost<TestType> input(elements_real);
        PtrHost<TestType> output(elements_real);
        PtrHost<Noa::Complex<TestType>> transform(elements_complex);

        Test::initDataRandom(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements_real * sizeof(TestType));

        Fourier::R2C(input.get(), transform.get(), shape_real);
        Test::initDataZero(input.get(), input.elements()); // just make sure new data is written.
        Test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements_real));
        Fourier::C2R(transform.get(), input.get(), shape_real);

        TestType diff = Test::getDifference(input.get(), output.get(), elements_real);
        diff /= static_cast<TestType>(elements_real);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));
    }

    AND_THEN("one time transform; in-place") {
        // Extra physical padding to store the complex transform.
        size_t padding_per_row = (shape_real.x % 2) ? 1 : 2;
        size_t elements_row = shape_real.x + padding_per_row;
        PtrHost<TestType> input(elements_real + getRows(shape_real) * padding_per_row);
        PtrHost<TestType> output(elements_real);

        Test::initDataRandom(input.get(), input.elements(), randomizer);
        for (size_t row{0}; row < getRows(shape_real); ++row) {
            std::memcpy(output.get() + row * shape_real.x, // output is not padded.
                        input.get() + row * elements_row, // input is padded.
                        shape_real.x * sizeof(TestType));
        }

        auto* transform = reinterpret_cast<Complex <TestType>*>(input.get());
        Fourier::R2C(input.get(), shape_real);
        Test::normalize(transform, elements_complex, 1 / static_cast<TestType>(elements_real));
        Fourier::C2R(transform, shape_real);

        TestType diff{0};
        for (size_t row{0}; row < getRows(shape_real); ++row) {
            diff += Test::getDifference(input.get() + row * elements_row,
                                        output.get() + row * shape_real.x,
                                        shape_real.x);
        }
        diff /= static_cast<TestType>(elements_real);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));
    }

    AND_THEN("execute and new-arrays functions") {
        PtrHost<TestType> input(elements_real);
        PtrHost<TestType> output(elements_real);
        PtrHost<Noa::Complex<TestType>> transform(elements_complex);

        Fourier::Plan<TestType> plan_forward(input.get(), transform.get(), shape_real, Fourier::MEASURE);
        Fourier::Plan<TestType> plan_backward(transform.get(), input.get(), shape_real, Fourier::MEASURE);
        Test::initDataRandom(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements_real * sizeof(TestType));

        for (int i = 0; i < 2; ++i) { // just make sure everything runs as expected
            Fourier::execute(plan_forward);
            Test::initDataZero(input.get(), input.elements()); // just make sure new data is written.
            Test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements_real));
            Fourier::execute(plan_backward);
        }

        TestType diff = Test::getDifference(input.get(), output.get(), elements_real);
        diff /= static_cast<TestType>(elements_real);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));

        // New arrays.
        PtrHost<TestType> input_new(elements_real);
        PtrHost<Noa::Complex<TestType>> transform_new(elements_complex);
        Test::initDataRandom(input_new.get(), input_new.elements(), randomizer);
        std::memcpy(output.get(), input_new.get(), elements_real * sizeof(TestType));

        for (int i = 0; i < 2; ++i) { // just make sure everything runs as expected
            Fourier::R2C(input_new.get(), transform_new.get(), plan_forward);
            Test::initDataZero(input_new.get(), input_new.elements()); // just make sure new data is written.
            Test::normalize(transform_new.get(), transform_new.elements(), 1 / static_cast<TestType>(elements_real));
            Fourier::C2R(transform_new.get(), input_new.get(), plan_backward);
        }

        diff = Test::getDifference(input_new.get(), output.get(), elements_real);
        diff /= static_cast<TestType>(elements_real);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("Fourier: transforms of complex inputs", "[noa][fourier]", float, double) {
    Test::RealRandomizer<TestType> randomizer(-5, 5);
    using complex_t = Noa::Complex<TestType>;

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = Test::getShapeReal(ndim); // the entire API is ndim "agnostic".
    size_t elements = getElements(shape);

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 1e-4;
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-12;

    AND_THEN("one time transform; out-of-place") {
        PtrHost<complex_t> input(elements);
        PtrHost<complex_t> output(elements);
        PtrHost<complex_t> transform(elements);

        Test::initDataRandom(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements * sizeof(complex_t));

        Fourier::C2C(input.get(), transform.get(), shape, Fourier::FORWARD);
        Test::initDataZero(input.get(), input.elements()); // just make sure new data is written.
        Test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements));
        Fourier::C2C(transform.get(), input.get(), shape, Fourier::BACKWARD);

        complex_t diff = Test::getDifference(input.get(), output.get(), elements);
        diff /= static_cast<TestType>(elements);
        REQUIRE_THAT(diff.real(), Catch::WithinAbs(0, abs_epsilon));
        REQUIRE_THAT(diff.imag(), Catch::WithinAbs(0, abs_epsilon));
    }

    AND_THEN("one time transform; in-place") {
        PtrHost<complex_t> input(elements);
        PtrHost<complex_t> output(elements);

        Test::initDataRandom(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements * sizeof(complex_t));

        Fourier::C2C(input.get(), input.get(), shape, Fourier::FORWARD);
        Test::normalize(input.get(), input.elements(), 1 / static_cast<TestType>(elements));
        Fourier::C2C(input.get(), input.get(), shape, Fourier::BACKWARD);

        complex_t diff = Test::getDifference(input.get(), output.get(), elements);
        diff /= static_cast<TestType>(elements);
        REQUIRE_THAT(diff.real(), Catch::WithinAbs(0, abs_epsilon));
        REQUIRE_THAT(diff.imag(), Catch::WithinAbs(0, abs_epsilon));
    }

    AND_THEN("execute and new-arrays functions") {
        PtrHost<complex_t> input(elements);
        PtrHost<complex_t> output(elements);
        PtrHost<complex_t> transform(elements);

        Fourier::Plan<TestType> plan_forward(input.get(), transform.get(), shape, Fourier::FORWARD, Fourier::MEASURE);
        Fourier::Plan<TestType> plan_backward(transform.get(), input.get(), shape, Fourier::BACKWARD, Fourier::MEASURE);
        Test::initDataRandom(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements * sizeof(complex_t));

        for (int i = 0; i < 2; ++i) { // just make sure everything runs as expected
            Fourier::execute(plan_forward);
            Test::initDataZero(input.get(), input.elements()); // just make sure new data is written.
            Test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements));
            Fourier::execute(plan_backward);
        }

        complex_t diff = Test::getDifference(input.get(), output.get(), elements);
        diff /= static_cast<TestType>(elements);
        REQUIRE_THAT(diff.real(), Catch::WithinAbs(0, abs_epsilon));
        REQUIRE_THAT(diff.imag(), Catch::WithinAbs(0, abs_epsilon));

        // New arrays.
        PtrHost<complex_t> input_new(elements);
        PtrHost<complex_t> transform_new(elements);
        Test::initDataRandom(input_new.get(), input_new.elements(), randomizer);
        std::memcpy(output.get(), input_new.get(), elements * sizeof(complex_t));

        for (int i = 0; i < 2; ++i) { // just make sure everything runs as expected
            Fourier::C2C(input_new.get(), transform_new.get(), plan_forward);
            Test::initDataZero(input_new.get(), input_new.elements()); // just make sure new data is written.
            Test::normalize(transform_new.get(), transform_new.elements(), 1 / static_cast<TestType>(elements));
            Fourier::C2C(transform_new.get(), input_new.get(), plan_backward);
        }

        diff = Test::getDifference(input_new.get(), output.get(), elements);
        diff /= static_cast<TestType>(elements);
        REQUIRE_THAT(diff.real(), Catch::WithinAbs(0, abs_epsilon));
        REQUIRE_THAT(diff.imag(), Catch::WithinAbs(0, abs_epsilon));
    }
}
