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
    const size4_t stride = shape.strides();
    const size4_t stride_fft = shape_fft.strides();
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
        cuda::fft::r2c(d_real.share(), d_transform.share(), shape, fft::NORM_BACKWARD, gpu_stream);
        cpu::fft::r2c(h_real.share(), h_transform.share(), shape, cpu::fft::ESTIMATE, fft::NORM_BACKWARD, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_transform.get(), d_transform.get(), elements_fft, abs_epsilon));

        // Reset data
        // FIX: Passing a non-hermitian input to c2r is undefined. For some shapes, CUDA doesn't produce the
        //      same output as FFTW3 or Numpy. This is only for single precision and 3D. Shape {44,99,128}
        //      fails, as well as {128,128,128}. However, {44,99,127} is fine for example. I'm pretty sure this
        //      used to give the same results as FFTW before, so they might have change the algorithm in 11.7,
        //      or maybe it's because I used to have sm6.0 and now I have sm8.6 which might enable another algo...
        //      In any case, this seems completely OK, since doing a C2R transform on a non-hermitian input
        //      doesn't make sense. When randomizing, the redundant planes in the 3D complex input is not
        //      hermitian. When the width is even, there's the extra plane at the real Nyquist as well.
        // test::randomize(h_transform.get(), h_transform.elements(), randomizer_complex);
        cuda::memory::copy(h_transform.get(), d_transform.get(), h_transform.size(), gpu_stream);

        test::randomize(h_real.get(), h_real.elements(), randomizer);
        test::randomize(d_real.get(), d_real.elements(), randomizer);

        // C2R
        cuda::fft::c2r(d_transform.share(), d_real.share(), shape, fft::NORM_BACKWARD, gpu_stream);
        cpu::fft::c2r(h_transform.share(), h_real.share(), shape, cpu::fft::ESTIMATE, fft::NORM_BACKWARD, cpu_stream);
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
        cuda::memory::copy(h_real.share(), stride, d_real.share(), d_real.strides(), shape, gpu_stream);

        // R2C
        cuda::fft::r2c(d_real.share(), d_real.strides(), d_transform.share(), d_transform.strides(),
                       shape, fft::NORM_BACKWARD, gpu_stream);
        cuda::memory::copy(d_transform.share(), d_transform.strides(),
                           h_transform_cuda.share(), stride_fft, shape_fft, gpu_stream);
        cpu::fft::r2c(h_real.share(), h_transform.share(), shape, cpu::fft::ESTIMATE, fft::NORM_BACKWARD, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE,
                              h_transform.get(), h_transform_cuda.get(), h_transform.elements(), abs_epsilon));

        // Reset data
        // test::randomize(h_transform.get(), h_transform.elements(), randomizer_complex);
        cuda::memory::copy(h_transform.share(), stride_fft,
                           d_transform.share(), d_transform.strides(), shape_fft, gpu_stream);

        cpu::memory::PtrHost<TestType> h_real_cuda(h_real.elements());
        test::memset(h_real.get(), h_real.elements(), 0);
        test::memset(h_real_cuda.get(), h_real.elements(), 0);

        // C2R
        cuda::fft::c2r(d_transform.share(), d_transform.strides(), d_real.share(), d_real.strides(),
                       shape, fft::NORM_BACKWARD, gpu_stream);
        cuda::memory::copy(d_real.share(), d_real.strides(), h_real_cuda.share(), stride, shape, gpu_stream);
        cpu::fft::c2r(h_transform.share(), h_real.share(), shape, cpu::fft::ESTIMATE, fft::NORM_BACKWARD, cpu_stream);
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
    auto h_real = std::reinterpret_pointer_cast<TestType[]>(h_input.share());
    const auto& h_transform = h_input.share();

    cuda::memory::PtrManaged<complex_t> d_input(elements_fft, gpu_stream);
    auto d_real = std::reinterpret_pointer_cast<TestType[]>(d_input.share());
    const auto& d_transform = d_input.share();

    test::randomize(h_real.get(), elements, randomizer);
    cuda::memory::copy(h_real, d_real, elements, gpu_stream);

    // R2C
    cuda::fft::r2c(d_real, d_transform, shape, fft::NORM_ORTHO, gpu_stream);
    cpu::fft::r2c(h_real, h_transform, shape, cpu::fft::ESTIMATE, fft::NORM_ORTHO, cpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_transform.get(), d_transform.get(), elements_fft, abs_epsilon));

    // Reset data
    // test::randomize(h_transform.get(), elements_fft, randomizer_complex);
    cuda::memory::copy(h_transform, d_transform, elements_fft, gpu_stream);

    // C2R
    cuda::fft::c2r(d_transform, d_real, shape, fft::NORM_ORTHO, gpu_stream);
    cpu::fft::c2r(h_transform, h_real, shape, cpu::fft::ESTIMATE, fft::NORM_ORTHO, cpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();

    // Ignore the extra padding for the comparison
    cpu::memory::set(h_real.get() + shape[3], pitch.strides(), size4_t{shape[0], shape[1], shape[2], padding}, TestType(0));
    cpu::memory::set(d_real.get() + shape[3], pitch.strides(), size4_t{shape[0], shape[1], shape[2], padding}, TestType(0));
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_real.get(), d_real.get(), elements, abs_epsilon));
}

TEMPLATE_TEST_CASE("cuda::fft::c2c()", "[noa][cuda][fft]", cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;
    test::Randomizer<TestType> randomizer(-1., 1.);

    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
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
        cuda::fft::c2c(d_input.share(), d_output.share(), shape, fft::Sign::FORWARD, fft::NORM_ORTHO, stream);
        cpu::fft::c2c(h_input.share(), h_output.share(), shape, fft::Sign::FORWARD, cpu::fft::ESTIMATE, fft::NORM_ORTHO, cpu_stream);
        stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), elements, abs_epsilon));

        // Reset data
        test::randomize(h_input.get(), elements, randomizer);
        cuda::memory::copy(h_input.get(), d_input.get(), h_input.size());

        // Backward
        cuda::fft::c2c(d_input.share(), d_output.share(), shape, fft::Sign::BACKWARD, fft::NORM_ORTHO, stream);
        cpu::fft::c2c(h_input.share(), h_output.share(), shape, fft::Sign::BACKWARD, cpu::fft::ESTIMATE, fft::NORM_ORTHO, cpu_stream);
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
        cuda::memory::copy(h_input.share(), stride, d_input.share(), d_input.strides(), shape, stream);

        // Forward
        cuda::fft::Plan<real_t> plan_c2c(cuda::fft::C2C, d_input.strides(), d_output.strides(), shape);
        cuda::fft::c2c(d_input.share(), d_output.share(), fft::Sign::FORWARD, plan_c2c, stream);
        cuda::memory::copy(d_output.share(), d_output.strides(), h_output_cuda.share(), stride, shape, stream);
        cpu::fft::c2c(h_input.share(), h_output.share(), shape, fft::Sign::FORWARD, cpu::fft::ESTIMATE, fft::NORM_NONE, cpu_stream);
        stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), h_output_cuda.get(), elements, abs_epsilon));

        // Reset data
        test::randomize(h_input.get(), elements, randomizer);
        cuda::memory::copy(h_input.share(), stride, d_input.share(), d_input.strides(), shape, stream);

        // Backward
        // We can reuse the plan since both arrays have the same shape and strides.
        cuda::fft::c2c(d_input.share(), d_output.share(), fft::Sign::BACKWARD, plan_c2c, stream);
        cuda::memory::copy(d_output.share(), d_output.strides(), h_output_cuda.share(), stride, shape, stream);
        cpu::fft::c2c(h_input.share(), h_output.share(), shape, fft::Sign::BACKWARD, cpu::fft::ESTIMATE, fft::NORM_NONE, cpu_stream);
        stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), h_output_cuda.get(), elements, abs_epsilon));
    }
}
