#include "noa/gpu/cuda/fft/Transforms.h"
#include "noa/gpu/cuda/memory/PtrManaged.h"
#include "noa/gpu/cuda/memory/PtrDevicePadded.h"
#include "noa/gpu/cuda/memory/Copy.h"

#include <noa/cpu/fft/Transforms.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace noa;

TEMPLATE_TEST_CASE("cuda::fft::r2c(), c2r() - out-of-place", "[noa][cuda][fft]", float, double) {
    using complex_t = Complex<TestType>;
    test::Randomizer<TestType> randomizer(-1, 1);
    test::Randomizer<complex_t> randomizer_complex(-1., 1.);

    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t shape_fft = shape.fft();
    const size4_t stride = shape.stride();
    const size4_t stride_fft = shape_fft.stride();
    const size_t elements = shape.elements();
    const size_t elements_fft = shape_fft.elements();
    INFO(shape);

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 1e-3; // most are okay at 1e-5 but there are some outliers...
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-9;

    cuda::Stream gpu_stream;
    cpu::Stream cpu_stream;

    AND_THEN("one time transform; R2C/C2R") {
        cpu::memory::PtrHost<TestType> h_real(elements);
        cpu::memory::PtrHost<complex_t> h_transform(elements_fft);
        cuda::memory::PtrManaged<TestType> d_real(elements, gpu_stream);
        cuda::memory::PtrManaged<complex_t> d_transform(elements_fft, gpu_stream);
        test::randomize(h_real.get(), h_real.elements(), randomizer);
        cuda::memory::copy(h_real.get(), d_real.get(), elements, gpu_stream);

        // R2C
        cuda::fft::r2c(d_real.get(), d_transform.get(), shape, gpu_stream);
        cpu::fft::r2c(h_real.get(), h_transform.get(), shape, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_transform.get(), d_transform.get(), elements_fft, abs_epsilon));

        // Reset data
        test::randomize(h_transform.get(), h_transform.elements(), randomizer_complex);
        cuda::memory::copy(h_transform.get(), d_transform.get(), h_transform.size(), gpu_stream);
        test::randomize(h_real.get(), h_real.elements(), randomizer);
        test::randomize(d_real.get(), d_real.elements(), randomizer);

        // C2R
        cuda::fft::c2r(d_transform.get(), d_real.get(), shape, gpu_stream);
        cpu::fft::c2r(h_transform.get(), h_real.get(), shape, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_real.get(), d_real.get(), h_real.elements(), abs_epsilon));
    }

    AND_THEN("one time transform; R2C/C2R; padded") {
        cpu::memory::PtrHost<TestType> h_real(elements);
        cpu::memory::PtrHost<complex_t> h_transform(elements_fft);
        cuda::memory::PtrDevicePadded<TestType> d_real(shape);
        cuda::memory::PtrDevicePadded<complex_t> d_transform(shape_fft);
        cpu::memory::PtrHost<complex_t> h_transform_cuda(elements_fft);

        test::randomize(h_real.get(), h_real.elements(), randomizer);
        cuda::memory::copy(h_real.get(), stride, d_real.get(), d_real.stride(), shape, gpu_stream);

        // R2C
        cuda::fft::Plan<TestType> plan_r2c(cuda::fft::R2C, d_real.stride(), d_transform.stride(), shape, gpu_stream);
        cuda::fft::r2c(d_real.get(), d_transform.get(), plan_r2c);
        cuda::memory::copy(d_transform.get(), d_transform.stride(),
                           h_transform_cuda.get(), stride_fft, shape_fft, gpu_stream);
        cpu::fft::r2c(h_real.get(), h_transform.get(), shape, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE,
                              h_transform.get(), h_transform_cuda.get(), h_transform.elements(), abs_epsilon));

        // Reset data
        test::randomize(h_transform.get(), h_transform.elements(), randomizer_complex);
        cuda::memory::copy(h_transform.get(), stride_fft,
                           d_transform.get(), d_transform.stride(), shape_fft, gpu_stream);

        cpu::memory::PtrHost<TestType> h_real_cuda(h_real.elements());
        test::memset(h_real.get(), h_real.elements(), 0);
        test::memset(h_real_cuda.get(), h_real.elements(), 0);

        // C2R
        cuda::fft::Plan<TestType> plan_c2r(cuda::fft::C2R, d_transform.stride(), d_real.stride(), shape, gpu_stream);
        cuda::fft::c2r(d_transform.get(), d_real.get(), plan_c2r);
        cuda::memory::copy(d_real.get(), d_real.stride(), h_real_cuda.get(), stride, shape, gpu_stream);
        cpu::fft::c2r(h_transform.get(), h_real.get(), shape, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE,
                              h_real.get(), h_real_cuda.get(), h_real_cuda.elements(), abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("cuda::fft::r2c(), c2r() - in-place", "[noa][cuda][fft]", float, double) {
    using complex_t = Complex<TestType>;
    test::Randomizer<TestType> randomizer(-1, 1);
    test::Randomizer<complex_t> randomizer_complex(-1., 1.);

    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size_t padding = (shape[3] % 2) ? 1 : 2;
    const size4_t pitch{shape[0], shape[1], shape[2], shape[3] + padding};
    const size_t elements = pitch.elements();

    const size4_t shape_fft = shape.fft();
    const size_t elements_fft = shape_fft.elements();

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, float>)
        abs_epsilon = 1e-3; // most are okay at 1e-5 but there are some outliers...
    else if constexpr (std::is_same_v<TestType, double>)
        abs_epsilon = 1e-9;

    cuda::Stream gpu_stream;
    cpu::Stream cpu_stream;

    cpu::memory::PtrHost<complex_t> h_input(elements_fft); // enough to contain the real and transform.
    auto* h_real = reinterpret_cast<TestType*>(h_input.get());
    auto* h_transform = h_input.get();

    cuda::memory::PtrManaged<complex_t> d_input(elements_fft, gpu_stream);
    auto* d_real = reinterpret_cast<TestType*>(d_input.get());
    auto* d_transform = d_input.get();

    test::randomize(h_real, elements, randomizer);
    cuda::memory::copy(h_real, d_real, elements, gpu_stream);

    // R2C
    cuda::fft::r2c(d_real, d_transform, shape, gpu_stream);
    cpu::fft::r2c(h_real, h_transform, shape, cpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_transform, d_transform, elements_fft, abs_epsilon));

    // Reset data
    test::randomize(h_transform, elements_fft, randomizer_complex);
    cuda::memory::copy(h_transform, d_transform, elements_fft, gpu_stream);

    // C2R
    cuda::fft::c2r(d_transform, d_real, shape, gpu_stream);
    cpu::fft::c2r(h_transform, h_real, shape, cpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();

    // Ignore the extra padding for the comparison
    cpu::memory::set(h_real + shape[3], pitch.stride(), size4_t{shape[0], shape[1], shape[2], padding}, TestType(0));
    cpu::memory::set(d_real + shape[3], pitch.stride(), size4_t{shape[0], shape[1], shape[2], padding}, TestType(0));
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_real, d_real, elements, abs_epsilon));
}

TEMPLATE_TEST_CASE("cuda::fft::c2c()", "[noa][cuda][fft]", cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;
    test::Randomizer<TestType> randomizer(-1., 1.);

    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();

    double abs_epsilon;
    if constexpr (std::is_same_v<TestType, cfloat_t>)
        abs_epsilon = 1e-3;
    else if constexpr (std::is_same_v<TestType, cdouble_t>)
        abs_epsilon = 1e-9;

    cuda::Stream stream;
    cpu::Stream cpu_stream;

    AND_THEN("one time transform; out-of-place; C2C") {
        cpu::memory::PtrHost<TestType> h_input(elements);
        cpu::memory::PtrHost<TestType> h_output(elements);
        cuda::memory::PtrManaged<TestType> d_input(elements);
        cuda::memory::PtrManaged<TestType> d_output(elements);

        test::randomize(h_input.get(), elements, randomizer);
        cuda::memory::copy(h_input.get(), d_input.get(), h_input.size());

        // Forward
        cuda::fft::c2c(d_input.get(), d_output.get(), shape, cuda::fft::FORWARD, stream);
        cpu::fft::c2c(h_input.get(), h_output.get(), shape, cpu::fft::FORWARD, cpu_stream);
        stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), elements, abs_epsilon));

        // Reset data
        test::randomize(h_input.get(), elements, randomizer);
        cuda::memory::copy(h_input.get(), d_input.get(), h_input.size());

        // Backward
        cuda::fft::c2c(d_input.get(), d_output.get(), shape, cuda::fft::BACKWARD, stream);
        cpu::fft::c2c(h_input.get(), h_output.get(), shape, cpu::fft::BACKWARD, cpu_stream);
        stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), elements, abs_epsilon));
    }

    AND_THEN("one time transform; out-of-place; C2C; padded memory") {
        cpu::memory::PtrHost<TestType> h_input(elements);
        cpu::memory::PtrHost<TestType> h_output(elements);
        cpu::memory::PtrHost<TestType> h_output_cuda(elements);
        cuda::memory::PtrDevicePadded<TestType> d_input(shape);
        cuda::memory::PtrDevicePadded<TestType> d_output(shape);

        test::randomize(h_input.get(), elements, randomizer);
        cuda::memory::copy(h_input.get(), stride, d_input.get(), d_input.stride(), shape, stream);

        // Forward
        cuda::fft::Plan<real_t> plan_c2c(cuda::fft::C2C, d_input.stride(), d_output.stride(), shape, stream);
        cuda::fft::c2c(d_input.get(), d_output.get(), plan_c2c, cuda::fft::FORWARD);
        cuda::memory::copy(d_output.get(), d_output.stride(), h_output_cuda.get(), stride, shape, stream);
        cpu::fft::c2c(h_input.get(), h_output.get(), shape, cpu::fft::FORWARD, cpu_stream);
        stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), h_output_cuda.get(), elements, abs_epsilon));

        // Reset data
        test::randomize(h_input.get(), elements, randomizer);
        cuda::memory::copy(h_input.get(), stride, d_input.get(), d_input.stride(), shape, stream);

        // Backward
        // We can reuse the plan since both arrays have the same shape and strides.
        cuda::fft::c2c(d_input.get(), d_output.get(), plan_c2c, cuda::fft::BACKWARD);
        cuda::memory::copy(d_output.get(), d_output.stride(), h_output_cuda.get(), stride, shape, stream);
        cpu::fft::c2c(h_input.get(), h_output.get(), shape, cpu::fft::BACKWARD, cpu_stream);
        stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), h_output_cuda.get(), elements, abs_epsilon));
    }
}
