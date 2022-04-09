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

    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size_t elements = shape.elements();
    const size_t elements_fft = shape.fft().elements();
    const TestType scaling = 1 / static_cast<TestType>(size3_t{shape.get() + 1}.elements()); // unbatched

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 1e-5;
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-9;

    cpu::Stream stream(cpu::Stream::DEFAULT);

    AND_THEN("one time transform; out-of-place") {
        cpu::memory::PtrHost<TestType> input(elements);
        cpu::memory::PtrHost<TestType> expected(elements);
        cpu::memory::PtrHost<complex_t> transform(elements_fft);

        test::randomize(input.get(), input.elements(), randomizer);
        test::copy(input.get(), expected.get(), input.elements());

        cpu::fft::r2c(input.share(), transform.share(), shape, cpu::fft::ESTIMATE, stream);
        test::memset(input.get(), input.elements(), 0); // just make sure new data is written.
        test::scale(transform.get(), transform.elements(), scaling);
        cpu::fft::c2r(transform.share(), input.share(), shape, cpu::fft::ESTIMATE, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input.get(), expected.get(), elements, abs_epsilon));
    }

    AND_THEN("one time transform; in-place") {
        // Extra padding to store the complex transform.
        const size_t pitch = (shape[3] / 2 + 1) * 2;
        const size_t rows = shape[0] * shape[1] * shape[2];
        cpu::memory::PtrHost<TestType> input(elements_fft * 2);
        cpu::memory::PtrHost<TestType> expected(elements);

        test::randomize(input.get(), input.elements(), randomizer);
        for (size_t row = 0; row < rows; ++row)
            test::copy(input.get() + row * pitch, expected.get() + row * shape[3], shape[3]);

        cpu::fft::r2c(input.share(), shape, cpu::fft::ESTIMATE, stream);
        test::scale(input.get(), input.elements(), scaling);
        cpu::fft::c2r(std::reinterpret_pointer_cast<complex_t[]>(input.share()), shape, cpu::fft::ESTIMATE, stream);

        TestType diff = 0;
        for (size_t row = 0; row < rows; ++row) {
            diff += test::getDifference(input.get() + row * pitch,
                                        expected.get() + row * shape[3],
                                        shape[3]);
        }
        diff *= scaling;
        REQUIRE_THAT(diff, Catch::WithinAbs(0, abs_epsilon));
    }

    AND_THEN("execute and new-arrays functions") {
        cpu::memory::PtrHost<TestType> input(elements);
        cpu::memory::PtrHost<TestType> output(elements);
        cpu::memory::PtrHost<complex_t> transform(elements_fft);

        cpu::fft::Flag flag = cpu::fft::ESTIMATE;
        cpu::fft::Plan<TestType> plan_forward(input.share(), transform.share(), shape, flag, stream);
        cpu::fft::Plan<TestType> plan_backward(transform.share(), input.share(), shape, flag, stream);
        test::randomize(input.get(), input.elements(), randomizer);
        test::copy(input.get(), output.get(), input.elements());

        cpu::fft::execute(plan_forward, stream);
        test::memset(input.get(), input.elements(), 0); // just make sure new data is written.
        test::scale(transform.get(), transform.elements(), scaling);
        cpu::fft::execute(plan_backward, stream);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input.get(), output.get(), elements, abs_epsilon));

        // New arrays.
        cpu::memory::PtrHost<TestType> input_new(elements);
        cpu::memory::PtrHost<complex_t> transform_new(elements_fft);
        test::randomize(input_new.get(), input_new.elements(), randomizer);
        test::copy(input_new.get(), output.get(), input.elements());

        cpu::fft::r2c(input_new.share(), transform_new.share(), plan_forward, stream);
        test::memset(input_new.get(), input_new.elements(), 0); // just make sure new data is written.
        test::scale(transform_new.get(), transform_new.elements(), scaling);
        cpu::fft::c2r(transform_new.share(), input_new.share(), plan_backward, stream);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input_new.get(), output.get(), elements, abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::c2c()", "[noa][cpu][fft]", float, double) {
    using complex_t = noa::Complex<TestType>;
    test::Randomizer<complex_t> randomizer(-5., 5.);

    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size_t elements = shape.elements();
    const TestType scaling = 1 / static_cast<TestType>(size3_t{shape.get() + 1}.elements()); // unbatched

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 1e-5;
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-9;

    cpu::Stream stream(cpu::Stream::DEFAULT);

    AND_THEN("one time transform; out-of-place") {
        cpu::memory::PtrHost<complex_t> input(elements);
        cpu::memory::PtrHost<complex_t> output(elements);
        cpu::memory::PtrHost<complex_t> transform(elements);

        test::randomize(input.get(), input.elements(), randomizer);
        test::copy(input.get(), output.get(), input.elements());

        cpu::fft::c2c(input.share(), transform.share(), shape, fft::FORWARD, cpu::fft::ESTIMATE, stream);
        test::memset(input.get(), input.elements(), 0); // just make sure new data is written.
        test::scale(transform.get(), transform.elements(), scaling);
        cpu::fft::c2c(transform.share(), input.share(), shape, fft::BACKWARD, cpu::fft::ESTIMATE, stream);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input.get(), output.get(), input.size(), abs_epsilon));
    }

    AND_THEN("one time transform; in-place") {
        cpu::memory::PtrHost<complex_t> input(elements);
        cpu::memory::PtrHost<complex_t> output(elements);

        test::randomize(input.get(), input.elements(), randomizer);
        test::copy(input.get(), output.get(), input.elements());

        cpu::fft::c2c(input.share(), input.share(), shape, fft::FORWARD, cpu::fft::ESTIMATE, stream);
        test::scale(input.get(), input.elements(), scaling);
        cpu::fft::c2c(input.share(), input.share(), shape, fft::BACKWARD, cpu::fft::ESTIMATE, stream);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input.get(), output.get(), input.size(), abs_epsilon));
    }

    AND_THEN("execute and new-arrays functions") {
        cpu::memory::PtrHost<complex_t> input(elements);
        cpu::memory::PtrHost<complex_t> output(elements);
        cpu::memory::PtrHost<complex_t> transform(elements);

        cpu::fft::Flag flag = cpu::fft::ESTIMATE;
        cpu::fft::Plan<TestType> plan_fwd(input.share(), transform.share(), shape, fft::FORWARD, flag, stream);
        cpu::fft::Plan<TestType> plan_bwd(transform.share(), input.share(), shape, fft::BACKWARD, flag, stream);
        test::randomize(input.get(), input.elements(), randomizer);
        test::copy(input.get(), output.get(), input.elements());

        cpu::fft::execute(plan_fwd, stream);
        test::memset(input.get(), input.elements(), 0); // just make sure new data is written.
        test::scale(transform.get(), transform.elements(), scaling);
        cpu::fft::execute(plan_bwd, stream);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input.get(), output.get(), input.size(), abs_epsilon));

        // New arrays.
        cpu::memory::PtrHost<complex_t> input_new(elements);
        cpu::memory::PtrHost<complex_t> transform_new(elements);
        test::randomize(input_new.get(), input_new.elements(), randomizer);
        test::copy(input_new.get(), output.get(), input.elements());

        cpu::fft::c2c(input_new.share(), transform_new.share(), plan_fwd, stream);
        test::memset(input_new.get(), input_new.elements(), 0); // just make sure new data is written.
        test::scale(transform_new.get(), transform_new.elements(), scaling);
        cpu::fft::c2c(transform_new.share(), input_new.share(), plan_bwd, stream);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input_new.get(), output.get(), elements, abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::c2c(), padded", "[noa][cpu][fft]", float, double) {
    using complex_t = noa::Complex<TestType>;
    test::Randomizer<complex_t> randomizer(-5., 5.);

    const size4_t shape = test::getRandomShapeBatched(3);
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();

    const size4_t shape_padded = shape + 10 * size4_t{shape != 1};
    const size4_t stride_padded = shape_padded.stride();
    const size_t elements_padded = shape_padded.elements();

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 1e-5;
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-9;

    cpu::Stream stream(cpu::Stream::DEFAULT);
    fft::Sign sign = GENERATE(fft::FORWARD, fft::BACKWARD);

    AND_THEN("in-place") {
        cpu::memory::PtrHost<complex_t> input(elements_padded);
        cpu::memory::PtrHost<complex_t> subregion(elements);
        test::randomize(input.get(), input.elements(), randomizer);
        cpu::memory::copy(input.get(), stride_padded, subregion.get(), stride, shape);

        cpu::fft::c2c(subregion.share(), shape, sign, cpu::fft::ESTIMATE, stream);
        cpu::fft::c2c(input.share(), stride_padded, shape, sign, cpu::fft::ESTIMATE, stream);
        cpu::memory::PtrHost<complex_t> output(subregion.size());
        cpu::memory::copy(input.get(), stride_padded, output.get(), stride, shape);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, subregion.get(), output.get(), subregion.size(), abs_epsilon));
    }

    AND_THEN("out of place") {
        cpu::memory::PtrHost<complex_t> input(elements_padded);
        cpu::memory::PtrHost<complex_t> subregion(elements);
        test::randomize(input.get(), input.elements(), randomizer);
        cpu::memory::copy(input.get(), stride_padded, subregion.get(), stride, shape);

        cpu::memory::PtrHost<complex_t> output1(subregion.size());
        cpu::memory::PtrHost<complex_t> output2(subregion.size());

        cpu::fft::c2c(subregion.share(), output1.share(), shape, sign, cpu::fft::ESTIMATE, stream);
        cpu::fft::c2c(input.share(), stride_padded, output2.share(), stride, shape, sign, cpu::fft::ESTIMATE, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output1.get(), output2.get(), subregion.size(), abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::c2r(), padded", "[noa][cpu][fft]", float, double) {
    using complex_t = noa::Complex<TestType>;
    test::Randomizer<complex_t> randomizer(-5., 5.);

    double epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        epsilon = 5e-4; // mostly are below 1e-5 but there's some bad ones
    else if constexpr (std::is_same_v<TestType, double>)
        epsilon = 1e-9;

    cpu::Stream stream(cpu::Stream::DEFAULT);
    const size4_t shape = test::getRandomShapeBatched(3);
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();

    const size4_t shape_padded = shape + 10 * size4_t{shape != 1};
    const size4_t stride_padded = shape_padded.stride();
    const size_t elements_padded = shape_padded.elements();

    const size4_t shape_fft = shape.fft();
    const size4_t stride_fft = shape_fft.stride();
    const size_t elements_fft = shape_fft.elements();

    const size4_t shape_fft_padded = shape_padded.fft();
    const size4_t stride_fft_padded = shape_fft_padded.stride();
    const size_t elements_fft_padded = shape_fft_padded.elements();

    AND_THEN("in-place") {
        cpu::memory::PtrHost<complex_t> input_padded(elements_fft_padded);
        auto* output_padded = reinterpret_cast<TestType*>(input_padded.get());
        cpu::memory::PtrHost<complex_t> input(elements_fft_padded);
        cpu::memory::PtrHost<TestType> output(elements);
        cpu::memory::PtrHost<TestType> output_contiguous(output.size());

        test::randomize(input_padded.get(), input_padded.elements(), randomizer);
        cpu::memory::copy(input_padded.get(), stride_fft_padded, input.get(), stride_fft, shape_fft);

        const size4_t real_stride{stride_fft_padded[0] * 2, stride_fft_padded[1] * 2,
                                  stride_fft_padded[2] * 2, stride_fft_padded[3]};
        cpu::fft::c2r(input.share(), output.share(), shape, cpu::fft::ESTIMATE, stream);
        cpu::fft::c2r(input_padded.share(), stride_fft_padded, shape, cpu::fft::ESTIMATE, stream);
        cpu::memory::copy(output_padded, real_stride, output_contiguous.get(), stride, shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output.get(), output_contiguous.get(), output.size(), epsilon));
    }

    AND_THEN("out of place") {
        cpu::memory::PtrHost<complex_t> input_padded(elements_fft_padded);
        cpu::memory::PtrHost<TestType> output_padded(elements_padded);
        cpu::memory::PtrHost<complex_t> input(elements_fft);
        cpu::memory::PtrHost<TestType> output(elements);
        cpu::memory::PtrHost<TestType> output_contiguous(output.size());

        test::randomize(input_padded.get(), input_padded.elements(), randomizer);
        cpu::memory::copy(input_padded.get(), stride_fft_padded, input.get(), stride_fft, shape_fft);

        cpu::fft::c2r(input.share(), output.share(), shape, cpu::fft::ESTIMATE, stream);
        cpu::fft::c2r(input_padded.share(), stride_fft_padded,
                      output_padded.share(), stride_padded,
                      shape, cpu::fft::ESTIMATE, stream);
        cpu::memory::copy(output_padded.get(), stride_padded, output_contiguous.get(), stride, shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output.get(), output_contiguous.get(), output.size(), epsilon));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::r2c(), padded", "[noa][cpu][fft]", float, double) {
    using complex_t = noa::Complex<TestType>;
    test::Randomizer<TestType> randomizer(-5., 5.);

    double epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        epsilon = 5e-4; // mostly are below 1e-5 but there's some bad ones
    else if constexpr (std::is_same_v<TestType, double>)
        epsilon = 1e-9;

    cpu::Stream stream(cpu::Stream::DEFAULT);
    const size4_t shape = test::getRandomShapeBatched(3, true);
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();
    INFO(shape);

    const size4_t shape_padded = shape + 10 * size4_t{shape != 1};
    const size4_t stride_padded = shape_padded.stride();
    const size_t elements_padded = shape_padded.elements();

    const size4_t shape_fft = shape.fft();
    const size4_t stride_fft = shape_fft.stride();
    const size_t elements_fft = shape_fft.elements();

    const size4_t shape_fft_padded = shape_padded.fft();
    const size4_t stride_fft_padded = shape_fft_padded.stride();
    const size_t elements_fft_padded = shape_fft_padded.elements();

    AND_THEN("in-place") {
        cpu::memory::PtrHost<TestType> input_padded(elements_padded);
        auto* output_padded = reinterpret_cast<complex_t*>(input_padded.get());
        cpu::memory::PtrHost<TestType> input(elements);
        cpu::memory::PtrHost<complex_t> output(elements_fft);
        cpu::memory::PtrHost<complex_t> output_contiguous(output.size());

        test::randomize(input.get(), input.elements(), randomizer);
        test::memset(input_padded.get(), input_padded.elements(), 1);
        cpu::memory::copy(input.get(), stride, input_padded.get(), stride_padded, shape);

        cpu::fft::r2c(input.share(), output.share(), shape, cpu::fft::ESTIMATE, stream);
        cpu::fft::r2c(input_padded.share(), stride_padded, shape, cpu::fft::ESTIMATE, stream);

        const size4_t complex_stride{stride_padded[0] / 2, stride_padded[1] / 2,
                                     stride_padded[2] / 2, stride_padded[3]};
        cpu::memory::copy(output_padded, complex_stride, output_contiguous.get(), stride_fft, shape_fft);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output.get(), output_contiguous.get(), output.size(), epsilon));
    }

    AND_THEN("out of place") {
        cpu::memory::PtrHost<TestType> input_padded(elements_padded);
        cpu::memory::PtrHost<complex_t> output_padded(elements_fft_padded);
        cpu::memory::PtrHost<TestType> input(elements);
        cpu::memory::PtrHost<complex_t> output(elements_fft);
        cpu::memory::PtrHost<complex_t> output_contiguous(output.size());

        test::randomize(input_padded.get(), input_padded.elements(), randomizer);
        cpu::memory::copy(input_padded.get(), stride_padded, input.get(), stride, shape);

        cpu::fft::r2c(input.share(), output.share(), shape, cpu::fft::ESTIMATE, stream);
        cpu::fft::r2c(input_padded.share(), stride_padded,
                      output_padded.share(), stride_fft_padded,
                      shape, cpu::fft::ESTIMATE, stream);
        cpu::memory::copy(output_padded.get(), stride_fft_padded, output_contiguous.get(), stride_fft, shape_fft);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output.get(), output_contiguous.get(), output.size(), epsilon));
    }
}
