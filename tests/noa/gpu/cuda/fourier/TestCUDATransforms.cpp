#include "noa/gpu/cuda/fourier/Transforms.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrDevicePadded.h"
#include "noa/gpu/cuda/memory/Copy.h"

#include <noa/cpu/fourier/Transforms.h>
#include <noa/cpu/memory/PtrHost.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace noa;

TEMPLATE_TEST_CASE("cuda::fourier::r2c(), c2r()", "[noa][cuda][fourier]", float, double) {
    using complex_t = Complex<TestType>;
    test::RealRandomizer<TestType> randomizer(-1, 1);
    test::RealRandomizer<complex_t> randomizer_complex(-1., 1.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape_real = test::getRandomShape(ndim); // the entire API is ndim "agnostic".
    size3_t shape_complex = getShapeFFT(shape_real);
    size_t elements_real = getElements(shape_real);
    size_t elements_complex = getElements(shape_complex);
    INFO(shape_real);

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 5e-3; // I should really use WithinRel, as opposed to WithinAbs
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 5e-11;

    cuda::Stream stream(cuda::Stream::CONCURRENT);

    AND_THEN("one time transform; out-of-place; R2C/C2R") {
        cpu::memory::PtrHost<TestType> h_real(elements_real);
        cpu::memory::PtrHost<complex_t> h_transform(elements_complex);
        cpu::memory::PtrHost<complex_t> h_transform_cuda(elements_complex);
        cuda::memory::PtrDevice<TestType> d_real(elements_real);
        cuda::memory::PtrDevice<complex_t> d_transform(elements_complex);

        test::initDataRandom(h_real.get(), h_real.elements(), randomizer);
        cuda::memory::copy(h_real.get(), d_real.get(), h_real.size());

        // R2C
        cuda::fourier::r2c(d_real.get(), d_transform.get(), shape_real, 1, stream);
        cuda::memory::copy(d_transform.get(), h_transform_cuda.get(), h_transform.size(), stream);
        cpu::fourier::r2c(h_real.get(), h_transform.get(), shape_real, 1);
        cuda::Stream::synchronize(stream);

        complex_t diff = test::getAverageDifference(h_transform.get(), h_transform_cuda.get(), h_transform.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));

        // Reset data
        test::initDataRandom(h_transform.get(), h_transform.elements(), randomizer_complex);
        cuda::memory::copy(h_transform.get(), d_transform.get(), h_transform.size(), stream);

        cpu::memory::PtrHost<TestType> h_real_cuda(h_real.elements());
        test::initDataZero(h_real.get(), h_real.elements());
        test::initDataZero(h_real_cuda.get(), h_real.elements());

        // C2R
        cuda::fourier::c2r(d_transform.get(), d_real.get(), shape_real, 1, stream);
        cuda::memory::copy(d_real.get(), h_real_cuda.get(), d_real.size(), stream);
        cpu::fourier::c2r(h_transform.get(), h_real.get(), shape_real, 1);
        cuda::Stream::synchronize(stream);

        diff = test::getAverageDifference(h_real.get(), h_real_cuda.get(), h_real_cuda.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));
    }

    AND_THEN("one time transform; out-of-place; R2C/C2R; padded memory") {
        cpu::memory::PtrHost<TestType> h_real(elements_real);
        cpu::memory::PtrHost<complex_t> h_transform(elements_complex);
        cuda::memory::PtrDevicePadded<TestType> d_real(shape_real);
        cuda::memory::PtrDevicePadded<complex_t> d_transform(shape_complex);
        cpu::memory::PtrHost<complex_t> h_transform_cuda(elements_complex);

        test::initDataRandom(h_real.get(), h_real.elements(), randomizer);
        cuda::memory::copy(h_real.get(), shape_real.x, d_real.get(), d_real.pitch(), shape_real);

        // R2C
        cuda::fourier::Plan<TestType> plan_r2c(shape_real, 1, d_real.pitch(), d_transform.pitch(),
                                               cuda::fourier::R2C, stream);
        cuda::fourier::r2c(d_real.get(), d_transform.get(), plan_r2c);
        cuda::memory::copy(d_transform.get(), d_transform.pitch(),
                           h_transform_cuda.get(), shape_complex.x, shape_complex, stream);
        cpu::fourier::r2c(h_real.get(), h_transform.get(), shape_real, 1);
        cuda::Stream::synchronize(stream);

        complex_t diff = test::getAverageDifference(h_transform.get(), h_transform_cuda.get(), h_transform.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));

        // Reset data
        test::initDataRandom(h_transform.get(), h_transform.elements(), randomizer_complex);
        cuda::memory::copy(h_transform.get(), shape_complex.x,
                           d_transform.get(), d_transform.pitch(), shape_complex);

        cpu::memory::PtrHost<TestType> h_real_cuda(h_real.elements());
        test::initDataZero(h_real.get(), h_real.elements());
        test::initDataZero(h_real_cuda.get(), h_real.elements());

        // C2R
        cuda::fourier::Plan<TestType> plan_c2r(shape_real, 1, d_transform.pitch(), d_real.pitch(),
                                               cuda::fourier::C2R, stream);
        cuda::fourier::c2r(d_transform.get(), d_real.get(), plan_c2r);
        cuda::memory::copy(d_real.get(), d_real.pitch(), h_real_cuda.get(), shape_real.x, shape_real, stream);
        cpu::fourier::c2r(h_transform.get(), h_real.get(), shape_real, 1);
        cuda::Stream::synchronize(stream);

        diff = test::getAverageDifference(h_real.get(), h_real_cuda.get(), h_real_cuda.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));
    }

    AND_THEN("one time transform; in-place; R2C/C2R") {
        // We tend to not pad the rows, we tend to ignore in-place transforms with FFTW (they are supported and
        // tested but one need to take care of the padding) since it makes everything more complicated...
        // In this case, just ignore the padding and compare to FFTW.
        cpu::memory::PtrHost<TestType> h_real(elements_real);
        cpu::memory::PtrHost<complex_t> h_transform(elements_complex);

        cuda::memory::PtrDevice<complex_t> d_input(
                elements_complex); // enough to contain the real data and the transform.
        auto* d_real = reinterpret_cast<TestType*>(d_input.get());
        auto* d_transform = d_input.get();
        size_t pitch_real = shape_real.x + ((shape_real.x % 2) ? 1 : 2);

        test::initDataRandom(h_real.get(), h_real.elements(), randomizer);
        cuda::memory::copy(h_real.get(), shape_real.x, d_real, pitch_real, shape_real);

        // R2C
        cpu::fourier::r2c(h_real.get(), h_transform.get(), shape_real, 1);
        cuda::fourier::r2c(d_real, d_transform, shape_real, 1, stream);
        cuda::Stream::synchronize(stream);

        cpu::memory::PtrHost<complex_t> h_transform_cuda(elements_complex);
        cuda::memory::copy(d_transform, h_transform_cuda.get(), h_transform_cuda.size());

        complex_t diff = test::getAverageDifference(h_transform.get(), h_transform_cuda.get(), h_transform.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));

        // Reset data
        test::initDataRandom(h_transform.get(), elements_complex, randomizer_complex);
        cuda::memory::copy(h_transform.get(), d_transform, h_transform.size());

        cpu::memory::PtrHost<TestType> h_real_cuda(elements_real);
        test::initDataZero(h_real.get(), elements_real);
        test::initDataZero(h_real_cuda.get(), h_real_cuda.elements());

        // C2R
        cpu::fourier::c2r(h_transform.get(), h_real.get(), shape_real, 1);
        cuda::fourier::c2r(d_transform, d_real, shape_real, 1, stream);
        cuda::memory::copy(d_real, pitch_real, h_real_cuda.get(), shape_real.x, shape_real, stream);
        cuda::Stream::synchronize(stream);

        diff = test::getAverageDifference(h_real.get(), h_real_cuda.get(), h_real_cuda.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(complex_t(0), abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("cuda::fourier::c2c()", "[noa][cuda][fourier]", cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;
    test::RealRandomizer<TestType> randomizer(-1., 1.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim); // the entire API is ndim "agnostic".
    size_t elements = getElements(shape);
    INFO(shape);

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, cfloat_t>)
        abs_epsilon = 5e-3; // I should really use WithinRel, as opposed to WithinAbs
    else if constexpr (std::is_same_v<TestType, cdouble_t>)
        abs_epsilon = 5e-11;

    cuda::Stream stream(cuda::Stream::CONCURRENT);

    AND_THEN("one time transform; out-of-place; C2C") {
        cpu::memory::PtrHost<TestType> h_input(elements);
        cpu::memory::PtrHost<TestType> h_output(elements);
        cuda::memory::PtrDevice<TestType> d_input(elements);
        cuda::memory::PtrDevice<TestType> d_output(elements);

        test::initDataRandom(h_input.get(), elements, randomizer);
        cuda::memory::copy(h_input.get(), d_input.get(), h_input.size());

        // Forward
        cpu::fourier::c2c(h_input.get(), h_output.get(), shape, 1, cpu::fourier::FORWARD);
        cuda::fourier::c2c(d_input.get(), d_output.get(), shape, 1, cuda::fourier::FORWARD, stream);

        cpu::memory::PtrHost<TestType> h_output_cuda(elements);
        cuda::memory::copy(d_output.get(), h_output_cuda.get(), d_output.size(), stream);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_output.get(), h_output_cuda.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), abs_epsilon));

        // Reset data
        test::initDataRandom(h_input.get(), elements, randomizer);
        cuda::memory::copy(h_input.get(), d_input.get(), h_input.size());

        // Backward
        cpu::fourier::c2c(h_input.get(), h_output.get(), shape, 1, cpu::fourier::BACKWARD);
        cuda::fourier::c2c(d_input.get(), d_output.get(), shape, 1, cuda::fourier::BACKWARD, stream);
        cuda::memory::copy(d_output.get(), h_output_cuda.get(), d_output.size(), stream);
        cuda::Stream::synchronize(stream);

        diff = test::getAverageDifference(h_output.get(), h_output_cuda.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), abs_epsilon));
    }

    AND_THEN("one time transform; out-of-place; C2C; padded memory") {
        cpu::memory::PtrHost<TestType> h_input(elements);
        cpu::memory::PtrHost<TestType> h_output(elements);
        cpu::memory::PtrHost<TestType> h_output_cuda(elements);
        cuda::memory::PtrDevicePadded<TestType> d_input(shape);
        cuda::memory::PtrDevicePadded<TestType> d_output(shape);

        test::initDataRandom(h_input.get(), elements, randomizer);
        cuda::memory::copy(h_input.get(), shape.x, d_input.get(), d_input.pitch(), shape);

        // Forward
        cuda::fourier::Plan<real_t> plan_c2c(shape, 1, d_input.pitch(), d_output.pitch(),
                                             cuda::fourier::C2C, stream);
        cuda::fourier::c2c(d_input.get(), d_output.get(), plan_c2c, cuda::fourier::FORWARD);
        cuda::memory::copy(d_output.get(), d_output.pitch(), h_output_cuda.get(), shape.x, shape, stream);
        cpu::fourier::c2c(h_input.get(), h_output.get(), shape, 1, cpu::fourier::FORWARD);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getAverageDifference(h_output.get(), h_output_cuda.get(), elements);
        REQUIRE_THAT(diff.real, Catch::WithinAbs(0, abs_epsilon));
        REQUIRE_THAT(diff.imag, Catch::WithinAbs(0, abs_epsilon));

        // Reset data
        test::initDataRandom(h_input.get(), elements, randomizer);
        cuda::memory::copy(h_input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);

        // Backward
        // We can reuse the plan since both arrays have the same shape and therefore same pitch.
        cuda::fourier::c2c(d_input.get(), d_output.get(), plan_c2c, cuda::fourier::BACKWARD);
        cuda::memory::copy(d_output.get(), d_output.pitch(), h_output_cuda.get(), shape.x, shape, stream);
        cpu::fourier::c2c(h_input.get(), h_output.get(), shape, 1, cpu::fourier::BACKWARD);
        cuda::Stream::synchronize(stream);

        diff = test::getAverageDifference(h_output.get(), h_output_cuda.get(), elements);
        REQUIRE_THAT(diff.real, Catch::WithinAbs(0, abs_epsilon));
        REQUIRE_THAT(diff.imag, Catch::WithinAbs(0, abs_epsilon));
    }
}
