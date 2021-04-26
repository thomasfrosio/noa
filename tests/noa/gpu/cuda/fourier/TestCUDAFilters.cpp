#include <noa/gpu/cuda/fourier/Filters.h>
#include <noa/cpu/fourier/Filters.h>

#include <noa/cpu/PtrHost.h>
#include <noa/gpu/cuda/PtrDevicePadded.h>
#include <noa/gpu/cuda/Memory.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

// The implementation for contiguous layouts is using the version with a pitch, so test only padded layouts...
TEMPLATE_TEST_CASE("Fourier: lowpass filters", "[noa][cpu][fourier]", float, double, cfloat_t, cdouble_t) {
    using real_t = Noa::Traits::value_type_t<TestType>;

    uint batches = Test::IntRandomizer<uint>(1, 3).get();
    uint ndim = GENERATE(2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size3_t shape_fft = getShapeFFT(shape);
    size_t elements_fft = getElementsFFT(shape);
    size3_t shape_fft_batched(shape_fft.x, shape_fft.y * shape_fft.z, batches);

    PtrHost<real_t> h_filter(elements_fft);
    PtrHost<TestType> h_data(elements_fft * batches);

    CUDA::PtrDevicePadded<real_t> d_filter(shape);
    CUDA::PtrDevicePadded<TestType> d_data(shape_fft_batched);
    PtrHost<real_t> h_cuda_filter(elements_fft);
    PtrHost<TestType> h_cuda_data(elements_fft * batches);

    CUDA::Stream stream(CUDA::Stream::SERIAL);

    // Filter parameters:
    Test::RealRandomizer<float> randomizer_float(0.f, 0.5f);
    float cutoff = randomizer_float.get();
    float width = randomizer_float.get();

    Test::Randomizer<TestType> randomizer(-5., 5.);
    Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    CUDA::Memory::copy(h_data.get(), shape_fft.x * sizeof(TestType),
                       d_data.get(), d_data.pitch(),
                       shape_fft_batched, stream);

    // Test saving the mask.
    CUDA::Fourier::lowpass(d_filter.get(), d_filter.pitchElements(), shape, cutoff, width, stream);
    CUDA::Memory::copy(d_filter.get(), d_filter.pitch(),
                       h_cuda_filter.get(), shape_fft.x * sizeof(real_t),
                       shape_fft, stream);
    Fourier::lowpass(h_filter.get(), shape, cutoff, width);
    CUDA::Stream::synchronize(stream);
    real_t diff_filter = Test::getAverageDifference(h_filter.get(), h_cuda_filter.get(), h_filter.elements());
    REQUIRE_THAT(diff_filter, Test::isWithinAbs(real_t(0.), 1e-6));

    // Test on-the-fly, in-place.
    CUDA::Fourier::lowpass(d_data.get(), d_data.pitchElements(), d_data.get(), d_data.pitchElements(),
                           shape, cutoff, width, batches, stream);
    CUDA::Memory::copy(d_data.get(), d_data.pitch(),
                       h_cuda_data.get(), shape_fft.x * sizeof(TestType),
                       shape_fft_batched, stream);
    Fourier::lowpass(h_data.get(), h_data.get(), shape, cutoff, width, batches);
    CUDA::Stream::synchronize(stream);
    TestType diff_data = Test::getAverageDifference(h_data.get(), h_cuda_data.get(), h_data.elements());
    REQUIRE_THAT(diff_data, Test::isWithinAbs(TestType(0.), 1e-6));
}

TEMPLATE_TEST_CASE("Fourier: highpass filters", "[noa][cpu][fourier]", float, double, cfloat_t, cdouble_t) {
    using real_t = Noa::Traits::value_type_t<TestType>;

    uint batches = Test::IntRandomizer<uint>(1, 3).get();
    uint ndim = GENERATE(2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size3_t shape_fft = getShapeFFT(shape);
    size_t elements_fft = getElementsFFT(shape);
    size3_t shape_fft_batched(shape_fft.x, shape_fft.y * shape_fft.z, batches);

    PtrHost<real_t> h_filter(elements_fft);
    PtrHost<TestType> h_data(elements_fft * batches);

    CUDA::PtrDevicePadded<real_t> d_filter(shape);
    CUDA::PtrDevicePadded<TestType> d_data(shape_fft_batched);
    PtrHost<real_t> h_cuda_filter(elements_fft);
    PtrHost<TestType> h_cuda_data(elements_fft * batches);

    CUDA::Stream stream(CUDA::Stream::SERIAL);

    // Filter parameters:
    Test::RealRandomizer<float> randomizer_float(0.f, 0.5f);
    float cutoff = randomizer_float.get();
    float width = randomizer_float.get();

    Test::Randomizer<TestType> randomizer(-5., 5.);
    Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    CUDA::Memory::copy(h_data.get(), shape_fft.x * sizeof(TestType),
                       d_data.get(), d_data.pitch(),
                       shape_fft_batched, stream);

    // Test saving the mask.
    CUDA::Fourier::highpass(d_filter.get(), d_filter.pitchElements(), shape, cutoff, width, stream);
    CUDA::Memory::copy(d_filter.get(), d_filter.pitch(),
                       h_cuda_filter.get(), shape_fft.x * sizeof(real_t),
                       shape_fft, stream);
    Fourier::highpass(h_filter.get(), shape, cutoff, width);
    CUDA::Stream::synchronize(stream);
    real_t diff_filter = Test::getAverageDifference(h_filter.get(), h_cuda_filter.get(), h_filter.elements());
    REQUIRE_THAT(diff_filter, Test::isWithinAbs(real_t(0.), 1e-6));

    // Test on-the-fly, in-place.
    CUDA::Fourier::highpass(d_data.get(), d_data.pitchElements(), d_data.get(), d_data.pitchElements(),
                           shape, cutoff, width, batches, stream);
    CUDA::Memory::copy(d_data.get(), d_data.pitch(),
                       h_cuda_data.get(), shape_fft.x * sizeof(TestType),
                       shape_fft_batched, stream);
    Fourier::highpass(h_data.get(), h_data.get(), shape, cutoff, width, batches);
    CUDA::Stream::synchronize(stream);
    TestType diff_data = Test::getAverageDifference(h_data.get(), h_cuda_data.get(), h_data.elements());
    REQUIRE_THAT(diff_data, Test::isWithinAbs(TestType(0.), 1e-6));
}

TEMPLATE_TEST_CASE("Fourier: bandpass filters", "[noa][cpu][fourier]", float, double, cfloat_t, cdouble_t) {
    using real_t = Noa::Traits::value_type_t<TestType>;

    uint batches = Test::IntRandomizer<uint>(1, 3).get();
    uint ndim = GENERATE(2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size3_t shape_fft = getShapeFFT(shape);
    size_t elements_fft = getElementsFFT(shape);
    size3_t shape_fft_batched(shape_fft.x, shape_fft.y * shape_fft.z, batches);

    PtrHost<real_t> h_filter(elements_fft);
    PtrHost<TestType> h_data(elements_fft * batches);

    CUDA::PtrDevicePadded<real_t> d_filter(shape);
    CUDA::PtrDevicePadded<TestType> d_data(shape_fft_batched);
    PtrHost<real_t> h_cuda_filter(elements_fft);
    PtrHost<TestType> h_cuda_data(elements_fft * batches);

    CUDA::Stream stream(CUDA::Stream::SERIAL);

    // Filter parameters:
    Test::RealRandomizer<float> randomizer_float(0.f, 0.5f);
    float cutoff1 = randomizer_float.get(), cutoff2 = cutoff1 + 0.1f;
    float width1 = randomizer_float.get(), width2 = randomizer_float.get();

    Test::Randomizer<TestType> randomizer(-5., 5.);
    Test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    CUDA::Memory::copy(h_data.get(), shape_fft.x * sizeof(TestType),
                       d_data.get(), d_data.pitch(),
                       shape_fft_batched, stream);

    // Test saving the mask.
    CUDA::Fourier::bandpass(d_filter.get(), d_filter.pitchElements(), shape, cutoff1, cutoff2, width1, width2, stream);
    CUDA::Memory::copy(d_filter.get(), d_filter.pitch(),
                       h_cuda_filter.get(), shape_fft.x * sizeof(real_t),
                       shape_fft, stream);
    Fourier::bandpass(h_filter.get(), shape, cutoff1, cutoff2, width1, width2);
    CUDA::Stream::synchronize(stream);
    real_t diff_filter = Test::getAverageDifference(h_filter.get(), h_cuda_filter.get(), h_filter.elements());
    REQUIRE_THAT(diff_filter, Test::isWithinAbs(real_t(0.), 1e-6));

    // Test on-the-fly, in-place.
    CUDA::Fourier::bandpass(d_data.get(), d_data.pitchElements(), d_data.get(), d_data.pitchElements(),
                           shape, cutoff1, cutoff2, width1, width2, batches, stream);
    CUDA::Memory::copy(d_data.get(), d_data.pitch(),
                       h_cuda_data.get(), shape_fft.x * sizeof(TestType),
                       shape_fft_batched, stream);
    Fourier::bandpass(h_data.get(), h_data.get(), shape, cutoff1, cutoff2, width1, width2, batches);
    CUDA::Stream::synchronize(stream);
    TestType diff_data = Test::getAverageDifference(h_data.get(), h_cuda_data.get(), h_data.elements());
    REQUIRE_THAT(diff_data, Test::isWithinAbs(TestType(0.), 1e-6));
}
