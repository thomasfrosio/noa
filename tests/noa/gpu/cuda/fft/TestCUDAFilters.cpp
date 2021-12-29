#include <noa/gpu/cuda/fft/Filters.h>
#include <noa/cpu/fft/Filters.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// The implementation for contiguous layouts is using the version with a pitch, so test only padded layouts...
TEMPLATE_TEST_CASE("cuda::fft::lowpass()", "[noa][cuda][fft]", float, double, cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;

    size_t batches = test::Randomizer<size_t>(1, 3).get();
    uint ndim = GENERATE(2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_fft = shapeFFT(shape);
    size_t elements_fft = noa::elementsFFT(shape);
    size3_t shape_fft_batched(shape_fft.x, shape_fft.y * shape_fft.z, batches);

    cpu::memory::PtrHost<real_t> h_filter(elements_fft);
    cpu::memory::PtrHost<TestType> h_data(elements_fft * batches);

    cuda::memory::PtrDevicePadded<real_t> d_filter(shape);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_fft_batched);
    cpu::memory::PtrHost<real_t> h_cuda_filter(elements_fft);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements_fft * batches);

    cuda::Stream stream(cuda::Stream::SERIAL);

    // Filter parameters:
    test::Randomizer<float> randomizer_float(0.f, 0.5f);
    float cutoff = randomizer_float.get();
    float width = randomizer_float.get();

    test::Randomizer<TestType> randomizer(-5., 5.);
    test::randomize(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), shape_fft.x,
                       d_data.get(), d_data.pitch(),
                       shape_fft_batched, stream);

    // Test saving the mask.
    cpu::Stream cpu_stream;
    size3_t pitch = shapeFFT(shape);
    cuda::fft::lowpass<real_t>(nullptr, 0, d_filter.get(), d_filter.pitch(), shape, 1, cutoff, width, stream);
    cuda::memory::copy(d_filter.get(), d_filter.pitch(),
                       h_cuda_filter.get(), shape_fft.x,
                       shape_fft, stream);
    cpu::fft::lowpass<fft::H2H, real_t>(nullptr, pitch, h_filter.get(), pitch, shape, 1, cutoff, width, cpu_stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_filter.get(), h_cuda_filter.get(), h_filter.elements(), 1e-4));

    // Test on-the-fly, in-place.
    cuda::fft::lowpass(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(),
                       shape, batches, cutoff, width, stream);
    cuda::memory::copy(d_data.get(), d_data.pitch(),
                       h_cuda_data.get(), shape_fft.x,
                       shape_fft_batched, stream);
    cpu::fft::lowpass<fft::H2H>(h_data.get(), pitch, h_data.get(), pitch, shape, batches, cutoff, width, cpu_stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_data.get(), h_cuda_data.get(), h_data.elements(), 1e-4));
}

TEMPLATE_TEST_CASE("cuda::fft::highpass()", "[noa][cuda][fft]", float, double, cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;

    size_t batches = test::Randomizer<size_t>(1, 3).get();
    uint ndim = GENERATE(2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_fft = shapeFFT(shape);
    size_t elements_fft = noa::elementsFFT(shape);
    size3_t shape_fft_batched(shape_fft.x, shape_fft.y * shape_fft.z, batches);

    cpu::memory::PtrHost<real_t> h_filter(elements_fft);
    cpu::memory::PtrHost<TestType> h_data(elements_fft * batches);

    cuda::memory::PtrDevicePadded<real_t> d_filter(shape);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_fft_batched);
    cpu::memory::PtrHost<real_t> h_cuda_filter(elements_fft);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements_fft * batches);

    cuda::Stream stream(cuda::Stream::SERIAL);

    // Filter parameters:
    test::Randomizer<float> randomizer_float(0.f, 0.5f);
    float cutoff = randomizer_float.get();
    float width = randomizer_float.get();

    test::Randomizer<TestType> randomizer(-5., 5.);
    test::randomize(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), shape_fft.x,
                       d_data.get(), d_data.pitch(),
                       shape_fft_batched, stream);

    // Test saving the mask.
    cpu::Stream cpu_stream;
    size3_t pitch = shapeFFT(shape);
    cuda::fft::highpass<real_t>(nullptr, 0, d_filter.get(), d_filter.pitch(), shape, 1, cutoff, width, stream);
    cuda::memory::copy(d_filter.get(), d_filter.pitch(),
                       h_cuda_filter.get(), shape_fft.x,
                       shape_fft, stream);
    cpu::fft::highpass<fft::H2H, real_t>(nullptr, pitch, h_filter.get(), pitch, shape, 1, cutoff, width, cpu_stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_filter.get(), h_cuda_filter.get(), h_filter.elements(), 1e-4));

    // Test on-the-fly, in-place.
    cuda::fft::highpass(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(),
                        shape, batches, cutoff, width, stream);
    cuda::memory::copy(d_data.get(), d_data.pitch(),
                       h_cuda_data.get(), shape_fft.x,
                       shape_fft_batched, stream);
    cpu::fft::highpass<fft::H2H>(h_data.get(), pitch, h_data.get(), pitch, shape, batches, cutoff, width, cpu_stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_data.get(), h_cuda_data.get(), h_data.elements(), 1e-4));
}

TEMPLATE_TEST_CASE("cuda::fft::bandpass()", "[noa][cuda][fft]", float, double, cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;

    size_t batches = test::Randomizer<size_t>(1, 3).get();
    uint ndim = GENERATE(2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_fft = shapeFFT(shape);
    size_t elements_fft = noa::elementsFFT(shape);
    size3_t shape_fft_batched(shape_fft.x, shape_fft.y * shape_fft.z, batches);

    cpu::memory::PtrHost<real_t> h_filter(elements_fft);
    cpu::memory::PtrHost<TestType> h_data(elements_fft * batches);

    cuda::memory::PtrDevicePadded<real_t> d_filter(shape);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape_fft_batched);
    cpu::memory::PtrHost<real_t> h_cuda_filter(elements_fft);
    cpu::memory::PtrHost<TestType> h_cuda_data(elements_fft * batches);

    cuda::Stream stream(cuda::Stream::SERIAL);

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
    cpu::Stream cpu_stream;
    size3_t pitch = shapeFFT(shape);
    cuda::fft::bandpass<real_t>(nullptr, 0, d_filter.get(), d_filter.pitch(), shape, 1,
                                cutoff1, cutoff2, width1, width2, stream);
    cuda::memory::copy(d_filter.get(), d_filter.pitch(),
                       h_cuda_filter.get(), shape_fft.x,
                       shape_fft, stream);
    cpu::fft::bandpass<fft::H2H, real_t>(nullptr, pitch, h_filter.get(), pitch, shape, 1,
                                         cutoff1, cutoff2, width1, width2, cpu_stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_filter.get(), h_cuda_filter.get(), h_filter.elements(), 1e-4));

    // Test on-the-fly, in-place.
    cuda::fft::bandpass(d_data.get(), d_data.pitch(), d_data.get(), d_data.pitch(),
                        shape, batches, cutoff1, cutoff2, width1, width2, stream);
    cuda::memory::copy(d_data.get(), d_data.pitch(),
                       h_cuda_data.get(), shape_fft.x,
                       shape_fft_batched, stream);
    cpu::fft::bandpass<fft::H2H>(h_data.get(), pitch, h_data.get(), pitch, shape, batches,
                                 cutoff1, cutoff2, width1, width2, cpu_stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_data.get(), h_cuda_data.get(), h_data.elements(), 1e-4));
}
