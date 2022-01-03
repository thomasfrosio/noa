#include <noa/cpu/fft/Plan.h>
#include <noa/cpu/fft/Transforms.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Copy.h>

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
        abs_epsilon = 1e-5;
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-9;

    cpu::Stream stream;

    AND_THEN("one time transform; out-of-place") {
        cpu::memory::PtrHost<TestType> input(elements);
        cpu::memory::PtrHost<TestType> expected(elements);
        cpu::memory::PtrHost<complex_t> transform(elements_fft);

        test::randomize(input.get(), input.elements(), randomizer);
        std::memcpy(expected.get(), input.get(), input.bytes());

        cpu::fft::r2c(input.get(), transform.get(), shape, 1, stream);
        test::memset(input.get(), input.elements(), 0); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements));
        cpu::fft::c2r(transform.get(), input.get(), shape, 1, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input.get(), expected.get(), elements, abs_epsilon));
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

        cpu::fft::r2c(input.get(), shape, 1, stream);
        test::normalize(input.get(), input.elements(), 1 / static_cast<TestType>(elements));
        cpu::fft::c2r(reinterpret_cast<complex_t*>(input.get()), shape, 1, stream);

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
        cpu::fft::Plan<TestType> plan_forward(input.get(), transform.get(), shape, batches, flag, stream);
        cpu::fft::Plan<TestType> plan_backward(transform.get(), input.get(), shape, batches, flag, stream);
        test::randomize(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), elements * sizeof(TestType) * batches);

        cpu::fft::execute(plan_forward, stream);
        test::memset(input.get(), input.elements(), 0); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(elements));
        cpu::fft::execute(plan_backward, stream);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input.get(), output.get(), elements * batches, abs_epsilon));

        // New arrays.
        cpu::memory::PtrHost<TestType> input_new(elements * batches);
        cpu::memory::PtrHost<complex_t> transform_new(elements_fft * batches);
        test::randomize(input_new.get(), input_new.elements(), randomizer);
        std::memcpy(output.get(), input_new.get(), elements * sizeof(TestType) * batches);

        cpu::fft::r2c(input_new.get(), transform_new.get(), plan_forward, stream);
        test::memset(input_new.get(), input_new.elements(), 0); // just make sure new data is written.
        test::normalize(transform_new.get(), transform_new.elements(), 1 / static_cast<TestType>(elements));
        cpu::fft::c2r(transform_new.get(), input_new.get(), plan_backward, stream);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input_new.get(), output.get(), elements * batches, abs_epsilon));
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
        abs_epsilon = 1e-5;
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-9;

    cpu::Stream stream;
    stream.threads(1);

    AND_THEN("one time transform; out-of-place") {
        cpu::memory::PtrHost<complex_t> input(size);
        cpu::memory::PtrHost<complex_t> output(size);
        cpu::memory::PtrHost<complex_t> transform(size);

        test::randomize(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), size * sizeof(complex_t));

        cpu::fft::c2c(input.get(), transform.get(), shape, 1, fft::FORWARD, stream);
        test::memset(input.get(), input.elements(), 0); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(size));
        cpu::fft::c2c(transform.get(), input.get(), shape, 1, fft::BACKWARD, stream);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input.get(), output.get(), size, abs_epsilon));
    }

    AND_THEN("one time transform; in-place") {
        cpu::memory::PtrHost<complex_t> input(size);
        cpu::memory::PtrHost<complex_t> output(size);

        test::randomize(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), size * sizeof(complex_t));

        cpu::fft::c2c(input.get(), input.get(), shape, 1, fft::FORWARD, stream);
        test::normalize(input.get(), input.elements(), 1 / static_cast<TestType>(size));
        cpu::fft::c2c(input.get(), input.get(), shape, 1, fft::BACKWARD, stream);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input.get(), output.get(), size, abs_epsilon));
    }

    AND_THEN("execute and new-arrays functions") {
        size_t batches = 2;
        cpu::memory::PtrHost<complex_t> input(size * batches);
        cpu::memory::PtrHost<complex_t> output(size * batches);
        cpu::memory::PtrHost<complex_t> transform(size * batches);

        cpu::fft::Flag flag = cpu::fft::ESTIMATE;
        cpu::fft::Plan<TestType> plan_fwd(input.get(), transform.get(), shape, batches, fft::FORWARD, flag, stream);
        cpu::fft::Plan<TestType> plan_bwd(transform.get(), input.get(), shape, batches, fft::BACKWARD, flag, stream);
        test::randomize(input.get(), input.elements(), randomizer);
        std::memcpy(output.get(), input.get(), size * sizeof(complex_t) * batches);

        cpu::fft::execute(plan_fwd, stream);
        test::memset(input.get(), input.elements(), 0); // just make sure new data is written.
        test::normalize(transform.get(), transform.elements(), 1 / static_cast<TestType>(size));
        cpu::fft::execute(plan_bwd, stream);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input.get(), output.get(), size * batches, abs_epsilon));

        // New arrays.
        cpu::memory::PtrHost<complex_t> input_new(size * batches);
        cpu::memory::PtrHost<complex_t> transform_new(size * batches);
        test::randomize(input_new.get(), input_new.elements(), randomizer);
        std::memcpy(output.get(), input_new.get(), size * sizeof(complex_t) * batches);

        cpu::fft::c2c(input_new.get(), transform_new.get(), plan_fwd, stream);
        test::memset(input_new.get(), input_new.elements(), 0); // just make sure new data is written.
        test::normalize(transform_new.get(), transform_new.elements(), 1 / static_cast<TestType>(size));
        cpu::fft::c2c(transform_new.get(), input_new.get(), plan_bwd, stream);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input_new.get(), output.get(), size * batches, abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::c2c(), padded", "[noa][cpu][fft]", float, double) {
    using complex_t = noa::Complex<TestType>;
    test::Randomizer<complex_t> randomizer(-5., 5.);

    size_t batches = 2;
    size3_t shape = test::getRandomShape(3);
    size3_t pitch = shape + 10;

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 1e-5;
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-9;

    cpu::Stream stream;
    fft::Sign sign = GENERATE(fft::FORWARD, fft::BACKWARD);

    AND_THEN("in-place") {
        cpu::memory::PtrHost<complex_t> input(elements(pitch) * batches);
        cpu::memory::PtrHost<complex_t> subregion(elements(shape) * batches);
        test::randomize(input.get(), input.elements(), randomizer);
        cpu::memory::copy(input.get(), pitch, subregion.get(), shape, shape, batches);

        cpu::fft::c2c(subregion.get(), shape, batches, sign, stream);
        cpu::fft::c2c(input.get(), pitch, shape, batches, sign, stream);
        cpu::memory::PtrHost<complex_t> output(subregion.size());
        cpu::memory::copy(input.get(), pitch, output.get(), shape, shape, batches);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, subregion.get(), output.get(), subregion.size(), abs_epsilon));
    }

    AND_THEN("out of place") {
        cpu::memory::PtrHost<complex_t> input(elements(pitch) * batches);
        cpu::memory::PtrHost<complex_t> subregion(elements(shape) * batches);
        test::randomize(input.get(), input.elements(), randomizer);
        cpu::memory::copy(input.get(), pitch, subregion.get(), shape, shape, batches);

        cpu::memory::PtrHost<complex_t> output1(subregion.size());
        cpu::memory::PtrHost<complex_t> output2(subregion.size());

        cpu::fft::c2c(subregion.get(), output1.get(), shape, batches, sign, stream);
        cpu::fft::c2c(input.get(), pitch, output2.get(), shape, shape, batches, sign, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output1.get(), output2.get(), subregion.size(), abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::c2r(), padded", "[noa][cpu][fft]", float, double) {
    using complex_t = noa::Complex<TestType>;
    test::Randomizer<complex_t> randomizer(-5., 5.);

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 5e-4; // mostly are below 1e-5 but there's some bad ones
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-9;

    cpu::Stream stream;
    size_t batches = 2;
    size3_t shape = test::getRandomShape(3);
    size3_t shape_fft = shapeFFT(shape);

    AND_THEN("in-place") {
        size3_t input_pitch = shape_fft + 13;
        size3_t output_pitch = {input_pitch.x * 2, input_pitch.y, input_pitch.z};
        cpu::memory::PtrHost<complex_t> input_padded(elements(input_pitch) * batches);
        auto* output_padded = reinterpret_cast<TestType*>(input_padded.get());
        cpu::memory::PtrHost<complex_t> input(elements(shape_fft) * batches);
        cpu::memory::PtrHost<TestType> output(elements(shape) * batches);
        cpu::memory::PtrHost<TestType> output_contiguous(output.size());

        test::randomize(input_padded.get(), input_padded.elements(), randomizer);
        cpu::memory::copy(input_padded.get(), input_pitch, input.get(), shape_fft, shape_fft, batches);

        cpu::fft::c2r(input.get(), output.get(), shape, batches, stream);
        cpu::fft::c2r(input_padded.get(), input_pitch, shape, batches, stream);
        cpu::memory::copy(output_padded, output_pitch, output_contiguous.get(), shape, shape, batches);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output.get(), output_contiguous.get(), output.size(), abs_epsilon));
    }

    AND_THEN("out of place") {
        size3_t input_pitch = shape_fft + 13;
        size3_t output_pitch = shape + 10;
        cpu::memory::PtrHost<complex_t> input_padded(elements(input_pitch) * batches);
        cpu::memory::PtrHost<TestType> output_padded(elements(output_pitch) * batches);
        cpu::memory::PtrHost<complex_t> input(elements(shape_fft) * batches);
        cpu::memory::PtrHost<TestType> output(elements(shape) * batches);
        cpu::memory::PtrHost<TestType> output_contiguous(output.size());

        test::randomize(input_padded.get(), input_padded.elements(), randomizer);
        cpu::memory::copy(input_padded.get(), input_pitch, input.get(), shape_fft, shape_fft, batches);

        cpu::fft::c2r(input.get(), output.get(), shape, batches, stream);
        cpu::fft::c2r(input_padded.get(), input_pitch, output_padded.get(), output_pitch, shape, batches, stream);
        cpu::memory::copy(output_padded.get(), output_pitch, output_contiguous.get(), shape, shape, batches);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output.get(), output_contiguous.get(), output.size(), abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::r2c(), padded", "[noa][cpu][fft]", float, double) {
    using complex_t = noa::Complex<TestType>;
    test::Randomizer<TestType> randomizer(-5., 5.);

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 5e-4; // mostly are below 1e-5 but there's some bad ones
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-9;

    cpu::Stream stream;
    size_t batches = 2;

    AND_THEN("in-place") {
        size3_t shape = test::getRandomShape(3, true);
        size3_t shape_fft = shapeFFT(shape);
        size3_t input_pitch = shape + 12;
        size3_t output_pitch = {input_pitch.x / 2, input_pitch.y, input_pitch.z};
        cpu::memory::PtrHost<TestType> input_padded(elements(input_pitch) * batches);
        auto* output_padded = reinterpret_cast<complex_t*>(input_padded.get());
        cpu::memory::PtrHost<TestType> input(elements(shape) * batches);
        cpu::memory::PtrHost<complex_t> output(elements(shape_fft) * batches);
        cpu::memory::PtrHost<complex_t> output_contiguous(output.size());

        test::randomize(input_padded.get(), input_padded.elements(), randomizer);
        cpu::memory::copy(input_padded.get(), input_pitch, input.get(), shape, shape, batches);

        cpu::fft::r2c(input.get(), output.get(), shape, batches, stream);
        cpu::fft::r2c(input_padded.get(), input_pitch, shape, batches, stream);
        cpu::memory::copy(output_padded, output_pitch, output_contiguous.get(), shape_fft, shape_fft, batches);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output.get(), output_contiguous.get(), output.size(), abs_epsilon));
    }

    AND_THEN("out of place") {
        size3_t shape = test::getRandomShape(3, true);
        size3_t shape_fft = shapeFFT(shape);
        size3_t input_pitch = shape + 12;
        size3_t output_pitch = {input_pitch.x / 2, input_pitch.y, input_pitch.z};
        cpu::memory::PtrHost<TestType> input_padded(elements(input_pitch) * batches);
        cpu::memory::PtrHost<complex_t> output_padded(elements(output_pitch) * batches);
        cpu::memory::PtrHost<TestType> input(elements(shape) * batches);
        cpu::memory::PtrHost<complex_t> output(elements(shape_fft) * batches);
        cpu::memory::PtrHost<complex_t> output_contiguous(output.size());

        test::randomize(input_padded.get(), input_padded.elements(), randomizer);
        cpu::memory::copy(input_padded.get(), input_pitch, input.get(), shape, shape, batches);

        cpu::fft::r2c(input.get(), output.get(), shape, batches, stream);
        cpu::fft::r2c(input_padded.get(), input_pitch, output_padded.get(), output_pitch, shape, batches, stream);
        cpu::memory::copy(output_padded.get(), output_pitch, output_contiguous.get(), shape_fft, shape_fft, batches);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output.get(), output_contiguous.get(), output.size(), abs_epsilon));
    }
}
