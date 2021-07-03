#include <noa/gpu/cuda/fourier/Filters.h>
#include <noa/cpu/fourier/Filters.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// The implementation for contiguous layouts is using the version with a pitch, so test only padded layouts...
TEMPLATE_TEST_CASE("cuda::fourier::lowpass()", "[noa][cuda][fourier]", float, double, cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;

    uint batches = test::IntRandomizer<uint>(1, 3).get();
    uint ndim = GENERATE(2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_fft = getShapeFFT(shape);
    size_t elements_fft = getElementsFFT(shape);
    size3_t shape_fft_batched(shape_fft.x, shape_fft.y * shape_fft.z, batches);

    memory::PtrHost<real_t> h_filter(elements_fft);
    memory::PtrHost<TestType> h_data(elements_fft * batches);

    cuda::memory::PtrDevicePadded<real_t> d_filter(shape);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_fft_batched);
    memory::PtrHost<real_t> h_cuda_filter(elements_fft);
    memory::PtrHost<TestType> h_cuda_data(elements_fft * batches);

    cuda::Stream stream(cuda::Stream::SERIAL);

    // Filter parameters:
    test::RealRandomizer<float> randomizer_float(0.f, 0.5f);
    float cutoff = randomizer_float.get();
    float width = randomizer_float.get();

    test::Randomizer<TestType> randomizer(-5., 5.);
    test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), shape_fft.x,
                       d_data.get(), d_data.pitch(),
                       shape_fft_batched, stream);

    // Test saving the mask.
    cuda::fourier::lowpass(d_filter.get(), d_filter.pitch(), shape, cutoff, width, stream);
    cuda::memory::copy(d_filter.get(), d_filter.pitch(),
                       h_cuda_filter.get(), shape_fft.x,
                       shape_fft, stream);
    fourier::lowpass(h_filter.get(), shape, cutoff, width);
    cuda::Stream::synchronize(stream);
    real_t diff_filter = test::getAverageDifference(h_filter.get(), h_cuda_filter.get(), h_filter.elements());
    REQUIRE_THAT(diff_filter, test::isWithinAbs(real_t(0.), 1e-6));

    // Test on-the-fly, in-place.
    cuda::fourier::lowpass(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(),
                           shape, cutoff, width, batches, stream);
    cuda::memory::copy(d_data.get(), d_data.pitch(),
                       h_cuda_data.get(), shape_fft.x,
                       shape_fft_batched, stream);
    fourier::lowpass(h_data.get(), h_data.get(), shape, cutoff, width, batches);
    cuda::Stream::synchronize(stream);
    TestType diff_data = test::getAverageDifference(h_data.get(), h_cuda_data.get(), h_data.elements());
    REQUIRE_THAT(diff_data, test::isWithinAbs(TestType(0.), 1e-6));
}

TEMPLATE_TEST_CASE("cuda::fourier::highpass()", "[noa][cuda][fourier]", float, double, cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;

    uint batches = test::IntRandomizer<uint>(1, 3).get();
    uint ndim = GENERATE(2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_fft = getShapeFFT(shape);
    size_t elements_fft = getElementsFFT(shape);
    size3_t shape_fft_batched(shape_fft.x, shape_fft.y * shape_fft.z, batches);

    memory::PtrHost<real_t> h_filter(elements_fft);
    memory::PtrHost<TestType> h_data(elements_fft * batches);

    cuda::memory::PtrDevicePadded<real_t> d_filter(shape);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_fft_batched);
    memory::PtrHost<real_t> h_cuda_filter(elements_fft);
    memory::PtrHost<TestType> h_cuda_data(elements_fft * batches);

    cuda::Stream stream(cuda::Stream::SERIAL);

    // Filter parameters:
    test::RealRandomizer<float> randomizer_float(0.f, 0.5f);
    float cutoff = randomizer_float.get();
    float width = randomizer_float.get();

    test::Randomizer<TestType> randomizer(-5., 5.);
    test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), shape_fft.x,
                       d_data.get(), d_data.pitch(),
                       shape_fft_batched, stream);

    // Test saving the mask.
    cuda::fourier::highpass(d_filter.get(), d_filter.pitch(), shape, cutoff, width, stream);
    cuda::memory::copy(d_filter.get(), d_filter.pitch(),
                       h_cuda_filter.get(), shape_fft.x,
                       shape_fft, stream);
    fourier::highpass(h_filter.get(), shape, cutoff, width);
    cuda::Stream::synchronize(stream);
    real_t diff_filter = test::getAverageDifference(h_filter.get(), h_cuda_filter.get(), h_filter.elements());
    REQUIRE_THAT(diff_filter, test::isWithinAbs(real_t(0.), 1e-6));

    // Test on-the-fly, in-place.
    cuda::fourier::highpass(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(),
                           shape, cutoff, width, batches, stream);
    cuda::memory::copy(d_data.get(), d_data.pitch(),
                       h_cuda_data.get(), shape_fft.x,
                       shape_fft_batched, stream);
    fourier::highpass(h_data.get(), h_data.get(), shape, cutoff, width, batches);
    cuda::Stream::synchronize(stream);
    TestType diff_data = test::getAverageDifference(h_data.get(), h_cuda_data.get(), h_data.elements());
    REQUIRE_THAT(diff_data, test::isWithinAbs(TestType(0.), 1e-6));
}

TEMPLATE_TEST_CASE("cuda::fourier::bandpass()", "[noa][cuda][fourier]", float, double, cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;

    uint batches = test::IntRandomizer<uint>(1, 3).get();
    uint ndim = GENERATE(2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_fft = getShapeFFT(shape);
    size_t elements_fft = getElementsFFT(shape);
    size3_t shape_fft_batched(shape_fft.x, shape_fft.y * shape_fft.z, batches);

    memory::PtrHost<real_t> h_filter(elements_fft);
    memory::PtrHost<TestType> h_data(elements_fft * batches);

    cuda::memory::PtrDevicePadded<real_t> d_filter(shape);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_fft_batched);
    memory::PtrHost<real_t> h_cuda_filter(elements_fft);
    memory::PtrHost<TestType> h_cuda_data(elements_fft * batches);

    cuda::Stream stream(cuda::Stream::SERIAL);

    // Filter parameters:
    test::RealRandomizer<float> randomizer_float(0.f, 0.5f);
    float cutoff1 = randomizer_float.get(), cutoff2 = cutoff1 + 0.1f;
    float width1 = randomizer_float.get(), width2 = randomizer_float.get();

    test::Randomizer<TestType> randomizer(-5., 5.);
    test::initDataRandom(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), shape_fft.x,
                       d_data.get(), d_data.pitch(),
                       shape_fft_batched, stream);

    // Test saving the mask.
    cuda::fourier::bandpass(d_filter.get(), d_filter.pitch(), shape, cutoff1, cutoff2, width1, width2, stream);
    cuda::memory::copy(d_filter.get(), d_filter.pitch(),
                       h_cuda_filter.get(), shape_fft.x,
                       shape_fft, stream);
    fourier::bandpass(h_filter.get(), shape, cutoff1, cutoff2, width1, width2);
    cuda::Stream::synchronize(stream);
    real_t diff_filter = test::getAverageDifference(h_filter.get(), h_cuda_filter.get(), h_filter.elements());
    REQUIRE_THAT(diff_filter, test::isWithinAbs(real_t(0.), 1e-6));

    // Test on-the-fly, in-place.
    cuda::fourier::bandpass(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(),
                           shape, cutoff1, cutoff2, width1, width2, batches, stream);
    cuda::memory::copy(d_data.get(), d_data.pitch(),
                       h_cuda_data.get(), shape_fft.x,
                       shape_fft_batched, stream);
    fourier::bandpass(h_data.get(), h_data.get(), shape, cutoff1, cutoff2, width1, width2, batches);
    cuda::Stream::synchronize(stream);
    TestType diff_data = test::getAverageDifference(h_data.get(), h_cuda_data.get(), h_data.elements());
    REQUIRE_THAT(diff_data, test::isWithinAbs(TestType(0.), 1e-6));
}
