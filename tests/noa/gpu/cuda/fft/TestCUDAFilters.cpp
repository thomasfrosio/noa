#include <noa/gpu/cuda/fft/Filters.h>
#include <noa/cpu/fft/Filters.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// The implementation for contiguous layouts is using the version with a pitch, so test only padded layouts...
TEMPLATE_TEST_CASE("cuda::fft::lowpass()", "[noa][cuda][fft]", half_t, float, double, chalf_t, cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;

    double epsilon;
    if constexpr (std::is_same_v<real_t, half_t>)
        epsilon = 1e-2;
    else
        epsilon = 5e-4; // the filter is single-precision, so expect the same with T=double

    const uint ndim = GENERATE(2U, 3U);
    const size_t batches = test::Randomizer<size_t>(1, 3).get();
    const size3_t shape = test::getRandomShape(ndim);
    const size3_t shape_fft = shapeFFT(shape);
    const size_t elements_fft = noa::elementsFFT(shape);
    const size3_t shape_fft_batched(shape_fft.x, shape_fft.y * shape_fft.z, batches);

    cpu::memory::PtrHost<real_t> h_filter(elements_fft);
    cpu::memory::PtrHost<TestType> h_data(elements_fft * batches);
    cuda::memory::PtrDevicePadded<real_t> d_filter(shape);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_fft_batched);
    cpu::memory::PtrHost<real_t> h_cuda_filter(elements_fft);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements_fft * batches);
    const size3_t d_filter_pitch{d_filter.pitch(), shape.y, shape.z};
    const size3_t d_data_pitch{d_data.pitch(), shape.y, shape.z};

    cuda::Stream stream(cuda::Stream::SERIAL);
    cpu::Stream cpu_stream;

    // Filter parameters:
    test::Randomizer<float> randomizer_float(0.f, 0.5f);
    const float cutoff = randomizer_float.get();
    const float width = randomizer_float.get();

    test::Randomizer<TestType> randomizer(-5., 5.);
    test::randomize(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), shape_fft.x,
                       d_data.get(), d_data.pitch(),
                       shape_fft_batched, stream);

    // Test saving the mask.
    cuda::fft::lowpass<fft::H2H, real_t>(nullptr, {}, d_filter.get(), d_filter_pitch, shape, 1, cutoff, width, stream);
    cuda::memory::copy(d_filter.get(), d_filter.pitch(),
                       h_cuda_filter.get(), shape_fft.x,
                       shape_fft, stream);
    cpu::fft::lowpass<fft::H2H, real_t>(nullptr, {}, h_filter.get(), shape_fft, shape, 1, cutoff, width, cpu_stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_filter.get(), h_cuda_filter.get(), h_filter.elements(), epsilon));

    // Test on-the-fly, in-place.
    cuda::fft::lowpass<fft::H2H>(d_data.get(), d_data_pitch, d_data.get(), d_data_pitch,
                                 shape, batches, cutoff, width, stream);
    cuda::memory::copy(d_data.get(), d_data.pitch(),
                       h_cuda_data.get(), shape_fft.x,
                       shape_fft_batched, stream);
    cpu::fft::lowpass<fft::H2H>(h_data.get(), shape_fft, h_data.get(), shape_fft,
                                shape, batches, cutoff, width, cpu_stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_data.get(), h_cuda_data.get(), h_data.elements(), epsilon));
}

TEMPLATE_TEST_CASE("cuda::fft::highpass()", "[noa][cuda][fft]", half_t, float, double, chalf_t, cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;

    double epsilon;
    if constexpr (std::is_same_v<real_t, half_t>)
        epsilon = 1e-2;
    else
        epsilon = 5e-4; // the filter is single-precision, so expect the same with T=double

    const uint ndim = GENERATE(2U, 3U);
    const size_t batches = test::Randomizer<size_t>(1, 3).get();
    const size3_t shape = test::getRandomShape(ndim);
    const size3_t shape_fft = shapeFFT(shape);
    const size_t elements_fft = noa::elementsFFT(shape);
    const size3_t shape_fft_batched(shape_fft.x, shape_fft.y * shape_fft.z, batches);

    cpu::memory::PtrHost<real_t> h_filter(elements_fft);
    cpu::memory::PtrHost<TestType> h_data(elements_fft * batches);
    cuda::memory::PtrDevicePadded<real_t> d_filter(shape);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_fft_batched);
    cpu::memory::PtrHost<real_t> h_cuda_filter(elements_fft);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements_fft * batches);
    const size3_t d_filter_pitch{d_filter.pitch(), shape.y, shape.z};
    const size3_t d_data_pitch{d_data.pitch(), shape.y, shape.z};

    cuda::Stream stream(cuda::Stream::SERIAL);
    cpu::Stream cpu_stream;

    // Filter parameters:
    test::Randomizer<float> randomizer_float(0.f, 0.5f);
    const float cutoff = randomizer_float.get();
    const float width = randomizer_float.get();

    test::Randomizer<TestType> randomizer(-5., 5.);
    test::randomize(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), shape_fft.x,
                       d_data.get(), d_data.pitch(),
                       shape_fft_batched, stream);

    // Test saving the mask.
    cuda::fft::highpass<fft::H2H, real_t>(nullptr, {}, d_filter.get(), d_filter_pitch,
                                          shape, 1, cutoff, width, stream);
    cuda::memory::copy(d_filter.get(), d_filter.pitch(),
                       h_cuda_filter.get(), shape_fft.x,
                       shape_fft, stream);
    cpu::fft::highpass<fft::H2H, real_t>(nullptr, {}, h_filter.get(), shape_fft, shape, 1, cutoff, width, cpu_stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_filter.get(), h_cuda_filter.get(), h_filter.elements(), epsilon));

    // Test on-the-fly, in-place.
    cuda::fft::highpass<fft::H2H>(d_data.get(), d_data_pitch, d_data.get(), d_data_pitch,
                                  shape, batches, cutoff, width, stream);
    cuda::memory::copy(d_data.get(), d_data.pitch(),
                       h_cuda_data.get(), shape_fft.x,
                       shape_fft_batched, stream);
    cpu::fft::highpass<fft::H2H>(h_data.get(), shape_fft, h_data.get(), shape_fft,
                                 shape, batches, cutoff, width, cpu_stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_data.get(), h_cuda_data.get(), h_data.elements(), epsilon));
}

TEMPLATE_TEST_CASE("cuda::fft::bandpass()", "[noa][cuda][fft]", half_t, float, double, chalf_t, cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;

    double epsilon;
    if constexpr (std::is_same_v<real_t, half_t>)
        epsilon = 1e-2;
    else
        epsilon = 5e-4; // the filter is single-precision, so expect the same with T=double

    const uint ndim = GENERATE(2U, 3U);
    const size_t batches = test::Randomizer<size_t>(1, 3).get();
    const size3_t shape = test::getRandomShape(ndim);
    const size3_t shape_fft = shapeFFT(shape);
    const size_t elements_fft = noa::elementsFFT(shape);
    const size3_t shape_fft_batched(shape_fft.x, shape_fft.y * shape_fft.z, batches);

    cpu::memory::PtrHost<real_t> h_filter(elements_fft);
    cpu::memory::PtrHost<TestType> h_data(elements_fft * batches);
    cuda::memory::PtrDevicePadded<real_t> d_filter(shape);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_fft_batched);
    cpu::memory::PtrHost<real_t> h_cuda_filter(elements_fft);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements_fft * batches);
    const size3_t d_filter_pitch{d_filter.pitch(), shape.y, shape.z};
    const size3_t d_data_pitch{d_data.pitch(), shape.y, shape.z};

    cuda::Stream stream(cuda::Stream::SERIAL);
    cpu::Stream cpu_stream;

    // Filter parameters:
    test::Randomizer<float> randomizer_float(0.f, 0.5f);
    float cutoff1 = randomizer_float.get(), cutoff2 = cutoff1 + 0.1f;
    float width1 = randomizer_float.get(), width2 = randomizer_float.get();

    test::Randomizer<TestType> randomizer(-5., 5.);
    test::randomize(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), shape_fft.x,
                       d_data.get(), d_data.pitch(),
                       shape_fft_batched, stream);

    // Test saving the mask.
    size3_t pitch = shapeFFT(shape);
    cuda::fft::bandpass<fft::H2H, real_t>(nullptr, {}, d_filter.get(), d_filter_pitch, shape, 1,
                                          cutoff1, cutoff2, width1, width2, stream);
    cuda::memory::copy(d_filter.get(), d_filter.pitch(),
                       h_cuda_filter.get(), shape_fft.x,
                       shape_fft, stream);
    cpu::fft::bandpass<fft::H2H, real_t>(nullptr, pitch, h_filter.get(), pitch, shape, 1,
                                         cutoff1, cutoff2, width1, width2, cpu_stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_filter.get(), h_cuda_filter.get(), h_filter.elements(), epsilon));

    // Test on-the-fly, in-place.
    cuda::fft::bandpass<fft::H2H>(d_data.get(), d_data_pitch, d_data.get(), d_data_pitch,
                                  shape, batches, cutoff1, cutoff2, width1, width2, stream);
    cuda::memory::copy(d_data.get(), d_data.pitch(),
                       h_cuda_data.get(), shape_fft.x,
                       shape_fft_batched, stream);
    cpu::fft::bandpass<fft::H2H>(h_data.get(), pitch, h_data.get(), pitch, shape, batches,
                                 cutoff1, cutoff2, width1, width2, cpu_stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_data.get(), h_cuda_data.get(), h_data.elements(), epsilon));
}
