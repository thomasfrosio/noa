#include "noa/gpu/cuda/fourier/Transforms.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrDevicePadded.h"
#include "noa/gpu/cuda/memory/Copy.h"

#include <noa/cpu/fourier/Transforms.h>
#include <noa/cpu/memory/PtrHost.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA::Fourier: transforms for real inputs", "[noa][cuda][fourier]", float, double) {
    using complex_t = Complex<TestType>;
    Test::RealRandomizer<TestType> randomizer(-1, 1);
    Test::RealRandomizer<complex_t> randomizer_complex(-1., 1.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape_real = Test::getRandomShape(ndim); // the entire API is ndim "agnostic".
    size3_t shape_complex = getShapeFFT(shape_real);
    size_t elements_real = getElements(shape_real);
    size_t elements_complex = getElements(shape_complex);
    INFO(shape_real);

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 5e-3; // I should really use WithinRel, as opposed to WithinAbs
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 5e-11;

    CUDA::Stream stream(CUDA::STREAM_CONCURRENT);

    AND_THEN("one time transform; out-of-place; R2C/C2R") {
        Memory::PtrHost<TestType> h_real(elements_real);
        Memory::PtrHost<complex_t> h_transform(elements_complex);
        Memory::PtrHost<complex_t> h_transform_cuda(elements_complex);
        CUDA::Memory::PtrDevice<TestType> d_real(elements_real);
        CUDA::Memory::PtrDevice<complex_t> d_transform(elements_complex);

        Test::initDataRandom(h_real.get(), h_real.elements(), randomizer);
        CUDA::Memory::copy(h_real.get(), d_real.get(), h_real.size());

        // R2C
        CUDA::Fourier::R2C(d_real.get(), d_transform.get(), shape_real, 1, stream);
        CUDA::Memory::copy(d_transform.get(), h_transform_cuda.get(), h_transform.size(), stream);
        Fourier::R2C(h_real.get(), h_transform.get(), shape_real, 1);
        CUDA::Stream::synchronize(stream);

        complex_t diff = Test::getAverageDifference(h_transform.get(), h_transform_cuda.get(), h_transform.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(complex_t(0), abs_epsilon));

        // Reset data
        Test::initDataRandom(h_transform.get(), h_transform.elements(), randomizer_complex);
        CUDA::Memory::copy(h_transform.get(), d_transform.get(), h_transform.size(), stream);

        Memory::PtrHost<TestType> h_real_cuda(h_real.elements());
        Test::initDataZero(h_real.get(), h_real.elements());
        Test::initDataZero(h_real_cuda.get(), h_real.elements());

        // C2R
        CUDA::Fourier::C2R(d_transform.get(), d_real.get(), shape_real, 1, stream);
        CUDA::Memory::copy(d_real.get(), h_real_cuda.get(), d_real.size(), stream);
        Fourier::C2R(h_transform.get(), h_real.get(), shape_real, 1);
        CUDA::Stream::synchronize(stream);

        diff = Test::getAverageDifference(h_real.get(), h_real_cuda.get(), h_real_cuda.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(complex_t(0), abs_epsilon));
    }

    AND_THEN("one time transform; out-of-place; R2C/C2R; padded memory") {
        Memory::PtrHost<TestType> h_real(elements_real);
        Memory::PtrHost<complex_t> h_transform(elements_complex);
        CUDA::Memory::PtrDevicePadded<TestType> d_real(shape_real);
        CUDA::Memory::PtrDevicePadded<complex_t> d_transform(shape_complex);
        Memory::PtrHost<complex_t> h_transform_cuda(elements_complex);

        Test::initDataRandom(h_real.get(), h_real.elements(), randomizer);
        CUDA::Memory::copy(h_real.get(), shape_real.x, d_real.get(), d_real.pitch(), shape_real);

        // R2C
        CUDA::Fourier::Plan<TestType> plan_r2c(shape_real, 1, d_real.pitch(), d_transform.pitch(),
                                               CUDA::Fourier::PLAN_R2C, stream);
        CUDA::Fourier::R2C(d_real.get(), d_transform.get(), plan_r2c);
        CUDA::Memory::copy(d_transform.get(), d_transform.pitch(),
                           h_transform_cuda.get(), shape_complex.x, shape_complex, stream);
        Fourier::R2C(h_real.get(), h_transform.get(), shape_real, 1);
        CUDA::Stream::synchronize(stream);

        complex_t diff = Test::getAverageDifference(h_transform.get(), h_transform_cuda.get(), h_transform.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(complex_t(0), abs_epsilon));

        // Reset data
        Test::initDataRandom(h_transform.get(), h_transform.elements(), randomizer_complex);
        CUDA::Memory::copy(h_transform.get(), shape_complex.x,
                           d_transform.get(), d_transform.pitch(), shape_complex);

        Memory::PtrHost<TestType> h_real_cuda(h_real.elements());
        Test::initDataZero(h_real.get(), h_real.elements());
        Test::initDataZero(h_real_cuda.get(), h_real.elements());

        // C2R
        CUDA::Fourier::Plan<TestType> plan_c2r(shape_real, 1, d_transform.pitch(), d_real.pitch(),
                                               CUDA::Fourier::PLAN_C2R, stream);
        CUDA::Fourier::C2R(d_transform.get(), d_real.get(), plan_c2r);
        CUDA::Memory::copy(d_real.get(), d_real.pitch(), h_real_cuda.get(), shape_real.x, shape_real, stream);
        Fourier::C2R(h_transform.get(), h_real.get(), shape_real, 1);
        CUDA::Stream::synchronize(stream);

        diff = Test::getAverageDifference(h_real.get(), h_real_cuda.get(), h_real_cuda.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(complex_t(0), abs_epsilon));
    }

    AND_THEN("one time transform; in-place; R2C/C2R") {
        // We tend to not pad the rows, we tend to ignore in-place transforms with FFTW (they are supported and
        // tested but one need to take care of the padding) since it makes everything more complicated...
        // In this case, just ignore the padding and compare to FFTW.
        Memory::PtrHost<TestType> h_real(elements_real);
        Memory::PtrHost<complex_t> h_transform(elements_complex);

        CUDA::Memory::PtrDevice<complex_t> d_input(elements_complex); // enough to contain the real data and the transform.
        auto* d_real = reinterpret_cast<TestType*>(d_input.get());
        auto* d_transform = d_input.get();
        size_t pitch_real = shape_real.x + ((shape_real.x % 2) ? 1 : 2);

        Test::initDataRandom(h_real.get(), h_real.elements(), randomizer);
        CUDA::Memory::copy(h_real.get(), shape_real.x, d_real, pitch_real, shape_real);

        // R2C
        Fourier::R2C(h_real.get(), h_transform.get(), shape_real, 1);
        CUDA::Fourier::R2C(d_real, d_transform, shape_real, 1, stream);
        CUDA::Stream::synchronize(stream);

        Memory::PtrHost<complex_t> h_transform_cuda(elements_complex);
        CUDA::Memory::copy(d_transform, h_transform_cuda.get(), h_transform_cuda.size());

        complex_t diff = Test::getAverageDifference(h_transform.get(), h_transform_cuda.get(), h_transform.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(complex_t(0), abs_epsilon));

        // Reset data
        Test::initDataRandom(h_transform.get(), elements_complex, randomizer_complex);
        CUDA::Memory::copy(h_transform.get(), d_transform, h_transform.size());

        Memory::PtrHost<TestType> h_real_cuda(elements_real);
        Test::initDataZero(h_real.get(), elements_real);
        Test::initDataZero(h_real_cuda.get(), h_real_cuda.elements());

        // C2R
        Fourier::C2R(h_transform.get(), h_real.get(), shape_real, 1);
        CUDA::Fourier::C2R(d_transform, d_real, shape_real, 1, stream);
        CUDA::Memory::copy(d_real, pitch_real, h_real_cuda.get(), shape_real.x, shape_real, stream);
        CUDA::Stream::synchronize(stream);

        diff = Test::getAverageDifference(h_real.get(), h_real_cuda.get(), h_real_cuda.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(complex_t(0), abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("CUDA::Fourier: transforms for complex inputs", "[noa][cuda][fourier]", cfloat_t, cdouble_t) {
    using real_t = Noa::Traits::value_type_t<TestType>;
    Test::RealRandomizer<TestType> randomizer(-1., 1.);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = Test::getRandomShape(ndim); // the entire API is ndim "agnostic".
    size_t elements = getElements(shape);
    INFO(shape);

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, cfloat_t>)
        abs_epsilon = 5e-3; // I should really use WithinRel, as opposed to WithinAbs
    else if constexpr (std::is_same_v<TestType, cdouble_t>)
        abs_epsilon = 5e-11;

    CUDA::Stream stream(CUDA::STREAM_CONCURRENT);

    AND_THEN("one time transform; out-of-place; C2C") {
        Memory::PtrHost<TestType> h_input(elements);
        Memory::PtrHost<TestType> h_output(elements);
        CUDA::Memory::PtrDevice<TestType> d_input(elements);
        CUDA::Memory::PtrDevice<TestType> d_output(elements);

        Test::initDataRandom(h_input.get(), elements, randomizer);
        CUDA::Memory::copy(h_input.get(), d_input.get(), h_input.size());

        // Forward
        Fourier::C2C(h_input.get(), h_output.get(), shape, 1, Fourier::FORWARD);
        CUDA::Fourier::C2C(d_input.get(), d_output.get(), shape, 1, CUDA::Fourier::FORWARD, stream);

        Memory::PtrHost<TestType> h_output_cuda(elements);
        CUDA::Memory::copy(d_output.get(), h_output_cuda.get(), d_output.size(), stream);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(h_output.get(), h_output_cuda.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), abs_epsilon));

        // Reset data
        Test::initDataRandom(h_input.get(), elements, randomizer);
        CUDA::Memory::copy(h_input.get(), d_input.get(), h_input.size());

        // Backward
        Fourier::C2C(h_input.get(), h_output.get(), shape, 1, Fourier::BACKWARD);
        CUDA::Fourier::C2C(d_input.get(), d_output.get(), shape, 1, CUDA::Fourier::BACKWARD, stream);
        CUDA::Memory::copy(d_output.get(), h_output_cuda.get(), d_output.size(), stream);
        CUDA::Stream::synchronize(stream);

        diff = Test::getAverageDifference(h_output.get(), h_output_cuda.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(TestType(0), abs_epsilon));
    }

    AND_THEN("one time transform; out-of-place; C2C; padded memory") {
        Memory::PtrHost<TestType> h_input(elements);
        Memory::PtrHost<TestType> h_output(elements);
        Memory::PtrHost<TestType> h_output_cuda(elements);
        CUDA::Memory::PtrDevicePadded<TestType> d_input(shape);
        CUDA::Memory::PtrDevicePadded<TestType> d_output(shape);

        Test::initDataRandom(h_input.get(), elements, randomizer);
        CUDA::Memory::copy(h_input.get(), shape.x, d_input.get(), d_input.pitch(), shape);

        // Forward
        CUDA::Fourier::Plan<real_t> plan_c2c(shape, 1, d_input.pitch(), d_output.pitch(),
                                             CUDA::Fourier::PLAN_C2C, stream);
        CUDA::Fourier::C2C(d_input.get(), d_output.get(), plan_c2c, CUDA::Fourier::FORWARD);
        CUDA::Memory::copy(d_output.get(), d_output.pitch(), h_output_cuda.get(), shape.x, shape, stream);
        Fourier::C2C(h_input.get(), h_output.get(), shape, 1, Fourier::FORWARD);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getAverageDifference(h_output.get(), h_output_cuda.get(), elements);
        REQUIRE_THAT(diff.real(), Catch::WithinAbs(0, abs_epsilon));
        REQUIRE_THAT(diff.imag(), Catch::WithinAbs(0, abs_epsilon));

        // Reset data
        Test::initDataRandom(h_input.get(), elements, randomizer);
        CUDA::Memory::copy(h_input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);

        // Backward
        // We can reuse the plan since both arrays have the same shape and therefore same pitch.
        CUDA::Fourier::C2C(d_input.get(), d_output.get(), plan_c2c, CUDA::Fourier::BACKWARD);
        CUDA::Memory::copy(d_output.get(), d_output.pitch(), h_output_cuda.get(), shape.x, shape, stream);
        Fourier::C2C(h_input.get(), h_output.get(), shape, 1, Fourier::BACKWARD);
        CUDA::Stream::synchronize(stream);

        diff = Test::getAverageDifference(h_output.get(), h_output_cuda.get(), elements);
        REQUIRE_THAT(diff.real(), Catch::WithinAbs(0, abs_epsilon));
        REQUIRE_THAT(diff.imag(), Catch::WithinAbs(0, abs_epsilon));
    }
}
