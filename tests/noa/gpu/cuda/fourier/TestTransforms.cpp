#include "noa/gpu/cuda/fourier/Transforms.h"
#include "noa/gpu/cuda/PtrDevice.h"
#include "noa/gpu/cuda/PtrDevicePadded.h"
#include "noa/gpu/cuda/Memory.h"

#include <noa/cpu/fourier/Transforms.h>
#include <noa/cpu/PtrHost.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA::Fourier: transforms for real inputs", "[noa][cuda][fourier]", float, double) {
    Test::RealRandomizer<TestType> randomizer(-1, 1);

    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape_real = Test::getShapeReal(ndim); // the entire API is ndim "agnostic".
    size3_t shape_complex = {shape_real.x / 2 + 1, shape_real.y, shape_real.z};
    size_t elements_real = getElements(shape_real);
    size_t elements_complex = getElements(shape_complex);

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 5e-3; // I should really use WithinRel, as opposed to WithinAbs
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 5e-11;

    using complex_t = Complex<TestType>;

    AND_THEN("one time transform; out-of-place; R2C/C2R") {
        PtrHost<TestType> h_real(elements_real);
        PtrHost<complex_t> h_transform(elements_complex);
        CUDA::PtrDevice<TestType> d_real(elements_real);
        CUDA::PtrDevice<complex_t> d_transform(elements_complex);

        Test::initDataRandom(h_real.get(), h_real.elements(), randomizer);
        CUDA::Memory::copy(d_real.get(), h_real.get(), h_real.bytes());

        // R2C
        Fourier::R2C(h_real.get(), h_transform.get(), shape_real, 1);
        CUDA::Fourier::R2C(d_real.get(), d_transform.get(), shape_real, 1);

        PtrHost<complex_t> h_transform_cuda(elements_complex);
        CUDA::Memory::copy(h_transform_cuda.get(), d_transform.get(), h_transform.bytes());

        complex_t diff = Test::getDifference(h_transform.get(), h_transform_cuda.get(), h_transform.elements());
        diff /= static_cast<TestType>(h_transform.elements());
        REQUIRE_THAT(diff.real(), Catch::WithinAbs(0, abs_epsilon));
        REQUIRE_THAT(diff.imag(), Catch::WithinAbs(0, abs_epsilon));

        // Reset data
        Test::initDataRandom(h_transform.get(), h_transform.elements(), randomizer);
        CUDA::Memory::copy(d_transform.get(), h_transform.get(), h_transform.bytes());
        Test::initDataZero(h_real.get(), h_real.elements());
        PtrHost<TestType> h_real_cuda(h_real.elements());
        CUDA::Memory::copy(h_real_cuda.get(), h_real.get(), h_real.bytes());

        // C2R
        Fourier::C2R(h_transform.get(), h_real.get(), shape_real, 1);
        CUDA::Fourier::C2R(d_transform.get(), d_real.get(), shape_real, 1);
        CUDA::Memory::copy(h_real_cuda.get(), d_real.get(), d_real.bytes());

        diff = Test::getDifference(h_real.get(), h_real_cuda.get(), h_real_cuda.elements());
        diff /= static_cast<TestType>(h_real.elements());
        REQUIRE_THAT(diff.real(), Catch::WithinAbs(0, abs_epsilon));
        REQUIRE_THAT(diff.imag(), Catch::WithinAbs(0, abs_epsilon));
    }

    AND_THEN("one time transform; out-of-place; C2C") {
        size_t elements = elements_real;
        PtrHost<complex_t> h_input(elements);
        PtrHost<complex_t> h_output(elements);
        CUDA::PtrDevice<complex_t> d_input(elements);
        CUDA::PtrDevice<complex_t> d_output(elements);

        Test::initDataRandom(h_input.get(), elements, randomizer);
        CUDA::Memory::copy(d_input.get(), h_input.get(), h_input.bytes());

        // Forward
        Fourier::C2C(h_input.get(), h_output.get(), shape_real, 1, Fourier::FORWARD);
        CUDA::Fourier::C2C(d_input.get(), d_output.get(), shape_real, 1, CUDA::Fourier::FORWARD);

        PtrHost<complex_t> h_output_cuda(elements);
        CUDA::Memory::copy(h_output_cuda.get(), d_output.get(), d_output.bytes());

        complex_t diff = Test::getDifference(h_output.get(), h_output_cuda.get(), elements);
        diff /= static_cast<TestType>(elements);
        REQUIRE_THAT(diff.real(), Catch::WithinAbs(0, abs_epsilon));
        REQUIRE_THAT(diff.imag(), Catch::WithinAbs(0, abs_epsilon));

        // Reset data
        Test::initDataRandom(h_input.get(), elements, randomizer);
        CUDA::Memory::copy(d_input.get(), h_input.get(), h_input.bytes());

        // Backward
        Fourier::C2C(h_input.get(), h_output.get(), shape_real, 1, Fourier::BACKWARD);
        CUDA::Fourier::C2C(d_input.get(), d_output.get(), shape_real, 1, CUDA::Fourier::BACKWARD);
        CUDA::Memory::copy(h_output_cuda.get(), d_output.get(), d_output.bytes());

        diff = Test::getDifference(h_output.get(), h_output_cuda.get(), elements);
        diff /= static_cast<TestType>(elements);
        REQUIRE_THAT(diff.real(), Catch::WithinAbs(0, abs_epsilon));
        REQUIRE_THAT(diff.imag(), Catch::WithinAbs(0, abs_epsilon));
    }

    AND_THEN("one time transform; out-of-place; C2C; padded memory") {
        size3_t shape = shape_real;
        size_t elements = elements_real;
        PtrHost<complex_t> h_input(elements);
        PtrHost<complex_t> h_output(elements);
        CUDA::PtrDevicePadded<complex_t> d_input(shape);
        CUDA::PtrDevicePadded<complex_t> d_output(shape);


    }
}
