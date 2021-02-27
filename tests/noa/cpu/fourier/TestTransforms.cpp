#include <noa/cpu/fourier/Plan.h>
#include <noa/cpu/fourier/Transforms.h>
#include <noa/cpu/PtrHost.h>

#include <catch2/catch.hpp>
#include "../../../Helpers.h"

using namespace Noa;

size3_t getShapeReal(uint ndim) {
    if (ndim == 2) {
        Test::IntRandomizer<size_t> randomizer(32, 512);
        return size3_t{randomizer.get(), randomizer.get(), 1};
    } else if (ndim == 3) {
        Test::IntRandomizer<size_t> randomizer(32, 128);
        return size3_t{randomizer.get(), randomizer.get(), randomizer.get()};
    } else {
        Test::IntRandomizer<size_t> randomizer(32, 1024);
        return size3_t{randomizer.get(), 1, 1};
    }
}

TEMPLATE_TEST_CASE("Fourier: transforms", "[noa][fourier]", float, double) {
    Test::RealRandomizer<TestType> randomizer(-5, 5);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape_real = getShapeReal(ndim); // the entire API is ndim "agnostic".
    size3_t shape_complex = {shape_real.x / 2 + 1, shape_real.y, shape_real.z};
    size_t elements_real = elements(shape_real);
    size_t elements_complex = elements(shape_complex);

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

        Fourier::r2c(input.get(), transform.get(), shape_real);
        Test::initDataZero(input.get(), input.elements()); // just make sure new data is written.
        Test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements_real));
        Fourier::c2r(transform.get(), input.get(), shape_real);

        TestType diff = Test::getDifference(input.get(), output.get(), elements_real);
        diff /= static_cast<TestType>(elements_real);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));
    }

    AND_THEN("one time transform; in-place") {
        // Extra physical padding to store the complex transform.
        size_t padding_per_row = (shape_real.x % 2) ? 1 : 2;
        size_t elements_row = shape_real.x + padding_per_row;
        PtrHost<TestType> input(elements_real + rows(shape_real) * padding_per_row);
        PtrHost<TestType> output(elements_real);

        Test::initDataRandom(input.get(), input.elements(), randomizer);
        for (size_t row{0}; row < rows(shape_real); ++row) {
            std::memcpy(output.get() + row * shape_real.x, // output is not padded.
                        input.get() + row * elements_row, // input is padded.
                        shape_real.x * sizeof(TestType));
        }

        auto* transform = reinterpret_cast<Complex <TestType>*>(input.get());
        Fourier::r2c(input.get(), shape_real);
        Test::normalize(transform, elements_complex, 1 / static_cast<TestType>(elements_real));
        Fourier::c2r(transform, shape_real);

        TestType diff{0};
        for (size_t row{0}; row < rows(shape_real); ++row) {
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
            Fourier::r2c(input_new.get(), transform_new.get(), plan_forward);
            Test::initDataZero(input_new.get(), input_new.elements()); // just make sure new data is written.
            Test::normalize(transform_new.get(), transform_new.elements(), 1 / static_cast<TestType>(elements_real));
            Fourier::c2r(transform_new.get(), input_new.get(), plan_backward);
        }

        diff = Test::getDifference(input_new.get(), output.get(), elements_real);
        diff /= static_cast<TestType>(elements_real);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));
    }
}
