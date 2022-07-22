#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/signal/fft/Bandpass.h>
#include <noa/gpu/cuda/fft/Remap.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/memory/PtrManaged.h>
#include <noa/gpu/cuda/signal/fft/Bandpass.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::signal::fft::lowpass()", "[noa][cuda][signal][fft]",
                   half_t, float, double, chalf_t, cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;

    double epsilon;
    if constexpr (std::is_same_v<real_t, half_t>)
        epsilon = 1e-2;
    else
        epsilon = 5e-4; // the filter is single-precision, so expect the same with T=double

    const uint ndim = GENERATE(2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t shape_fft = shape.fft();
    const size4_t stride_fft = shape_fft.strides();
    const size_t elements_fft = shape_fft.elements();

    cuda::Stream gpu_stream;
    cpu::Stream cpu_stream;
    cpu::memory::PtrHost<real_t> h_filter(elements_fft);
    cpu::memory::PtrHost<TestType> h_data(elements_fft);
    cuda::memory::PtrManaged<real_t> d_filter(elements_fft, gpu_stream);
    cuda::memory::PtrManaged<TestType> d_data(elements_fft, gpu_stream);

    // Filter parameters:
    test::Randomizer<float> randomizer_float(0.f, 0.5f);
    const float cutoff = randomizer_float.get();
    const float width = randomizer_float.get();

    test::Randomizer<TestType> randomizer(-5., 5.);
    test::randomize(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.share(), stride_fft, d_data.share(), stride_fft, shape_fft, gpu_stream);

    // Test saving the mask.
    cuda::signal::fft::lowpass<fft::H2H, real_t>(nullptr, {}, d_filter.share(), stride_fft, shape, cutoff, width, gpu_stream);
    cpu::signal::fft::lowpass<fft::H2H, real_t>(nullptr, {}, h_filter.share(), stride_fft, shape, cutoff, width, cpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_filter.get(), d_filter.get(), h_filter.elements(), epsilon));

    // Test on-the-fly, in-place.
    cuda::signal::fft::lowpass<fft::H2H>(d_data.share(), stride_fft, d_data.share(), stride_fft, shape, cutoff, width, gpu_stream);
    cpu::signal::fft::lowpass<fft::H2H>(h_data.share(), stride_fft, h_data.share(), stride_fft, shape, cutoff, width, cpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_data.get(), d_data.get(), h_data.elements(), epsilon));
}

TEMPLATE_TEST_CASE("cuda::signal::fft::lowpass(), remap", "[noa][cuda][signal][fft]", half_t, float) {
    const size4_t shape = test::getRandomShapeBatched(3);
    const float cutoff = 0.4f;
    const float width = 0.1f;

    const size_t elements = shape.fft().elements();
    const size4_t stride = shape.fft().strides();

    cuda::Stream stream;
    cuda::memory::PtrManaged<float> filter_expected(elements, stream);
    cuda::memory::PtrManaged<float> filter_result(elements, stream);
    cuda::memory::PtrManaged<float> filter_remapped(elements, stream);
    test::memset(filter_expected.get(), elements, 1);
    test::memset(filter_result.get(), elements, 1);

    // H2HC
    cuda::signal::fft::lowpass<fft::H2H, float>(nullptr, {}, filter_expected.share(), stride, shape, cutoff, width, stream);
    cuda::signal::fft::lowpass<fft::H2HC, float>(nullptr, {}, filter_result.share(), stride, shape, cutoff, width, stream);
    cuda::fft::remap(fft::HC2H, filter_result.share(), stride, filter_remapped.share(), stride, shape, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2HC
    cuda::signal::fft::lowpass<fft::H2H, float>(nullptr, {}, filter_expected.share(), stride, shape, cutoff, width, stream);
    cuda::signal::fft::lowpass<fft::HC2HC, float>(nullptr, {}, filter_result.share(), stride, shape, cutoff, width, stream);
    cuda::fft::remap(fft::HC2H, filter_result.share(), stride, filter_remapped.share(), stride, shape, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2H
    cuda::signal::fft::lowpass<fft::H2H, float>(nullptr, {}, filter_expected.share(), stride, shape, cutoff, width, stream);
    cuda::signal::fft::lowpass<fft::HC2H, float>(nullptr, {}, filter_result.share(), stride, shape, cutoff, width, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));
}

TEMPLATE_TEST_CASE("cuda::signal::fft::highpass()", "[noa][cuda][signal][fft]",
                   half_t, float, double, chalf_t, cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;

    double epsilon;
    if constexpr (std::is_same_v<real_t, half_t>)
        epsilon = 1e-2;
    else
        epsilon = 5e-4; // the filter is single-precision, so expect the same with T=double

    const uint ndim = GENERATE(2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t shape_fft = shape.fft();
    const size4_t stride_fft = shape_fft.strides();
    const size_t elements_fft = shape_fft.elements();

    cuda::Stream gpu_stream;
    cpu::Stream cpu_stream;
    cpu::memory::PtrHost<real_t> h_filter(elements_fft);
    cpu::memory::PtrHost<TestType> h_data(elements_fft);
    cuda::memory::PtrManaged<real_t> d_filter(elements_fft, gpu_stream);
    cuda::memory::PtrManaged<TestType> d_data(elements_fft, gpu_stream);

    // Filter parameters:
    test::Randomizer<float> randomizer_float(0.f, 0.5f);
    const float cutoff = randomizer_float.get();
    const float width = randomizer_float.get();

    test::Randomizer<TestType> randomizer(-5., 5.);
    test::randomize(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.share(), stride_fft, d_data.share(), stride_fft, shape_fft, gpu_stream);

    // Test saving the mask.
    cuda::signal::fft::highpass<fft::H2H, real_t>(nullptr, {}, d_filter.share(), stride_fft, shape, cutoff, width, gpu_stream);
    cpu::signal::fft::highpass<fft::H2H, real_t>(nullptr, {}, h_filter.share(), stride_fft, shape, cutoff, width, cpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_filter.get(), d_filter.get(), h_filter.elements(), epsilon));

    // Test on-the-fly, in-place.
    cuda::signal::fft::highpass<fft::H2H>(d_data.share(), stride_fft, d_data.share(), stride_fft, shape, cutoff, width, gpu_stream);
    cpu::signal::fft::highpass<fft::H2H>(h_data.share(), stride_fft, h_data.share(), stride_fft, shape, cutoff, width, cpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_data.get(), d_data.get(), h_data.elements(), epsilon));
}

TEMPLATE_TEST_CASE("cuda::signal::fft::highpass(), remap", "[noa][cuda][signal][fft]", half_t, float) {
    const size4_t shape = test::getRandomShapeBatched(3);
    const float cutoff = 0.4f;
    const float width = 0.1f;

    const size_t elements = shape.fft().elements();
    const size4_t stride = shape.fft().strides();

    cuda::Stream stream;
    cuda::memory::PtrManaged<float> filter_expected(elements, stream);
    cuda::memory::PtrManaged<float> filter_result(elements, stream);
    cuda::memory::PtrManaged<float> filter_remapped(elements, stream);
    test::memset(filter_expected.get(), elements, 1);
    test::memset(filter_result.get(), elements, 1);

    // H2HC
    cuda::signal::fft::highpass<fft::H2H, float>(nullptr, {}, filter_expected.share(), stride, shape, cutoff, width, stream);
    cuda::signal::fft::highpass<fft::H2HC, float>(nullptr, {}, filter_result.share(), stride, shape, cutoff, width, stream);
    cuda::fft::remap(fft::HC2H, filter_result.share(), stride, filter_remapped.share(), stride, shape, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2HC
    cuda::signal::fft::highpass<fft::H2H, float>(nullptr, {}, filter_expected.share(), stride, shape, cutoff, width, stream);
    cuda::signal::fft::highpass<fft::HC2HC, float>(nullptr, {}, filter_result.share(), stride, shape, cutoff, width, stream);
    cuda::fft::remap(fft::HC2H, filter_result.share(), stride, filter_remapped.share(), stride, shape, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2H
    cuda::signal::fft::highpass<fft::H2H, float>(nullptr, {}, filter_expected.share(), stride, shape, cutoff, width, stream);
    cuda::signal::fft::highpass<fft::HC2H, float>(nullptr, {}, filter_result.share(), stride, shape, cutoff, width, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));
}

TEMPLATE_TEST_CASE("cuda::signal::fft::bandpass()", "[noa][cuda][signal][fft]", half_t, float, double, chalf_t, cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;

    double epsilon;
    if constexpr (std::is_same_v<real_t, half_t>)
        epsilon = 1e-2;
    else
        epsilon = 5e-4; // the filter is single-precision, so expect the same with T=double

    const uint ndim = GENERATE(2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t shape_fft = shape.fft();
    const size4_t stride_fft = shape_fft.strides();
    const size_t elements_fft = shape_fft.elements();

    cuda::Stream gpu_stream;
    cpu::Stream cpu_stream;
    cpu::memory::PtrHost<real_t> h_filter(elements_fft);
    cpu::memory::PtrHost<TestType> h_data(elements_fft);
    cuda::memory::PtrManaged<real_t> d_filter(elements_fft, gpu_stream);
    cuda::memory::PtrManaged<TestType> d_data(elements_fft, gpu_stream);

    // Filter parameters:
    test::Randomizer<float> randomizer_float(0.f, 0.5f);
    float cutoff1 = randomizer_float.get(), cutoff2 = cutoff1 + 0.1f;
    float width1 = randomizer_float.get(), width2 = randomizer_float.get();

    test::Randomizer<TestType> randomizer(-5., 5.);
    test::randomize(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.share(), stride_fft, d_data.share(), stride_fft, shape_fft, gpu_stream);

    // Test saving the mask.
    cuda::signal::fft::bandpass<fft::H2H, real_t>(nullptr, {}, d_filter.share(), stride_fft, shape,
                                                  cutoff1, cutoff2, width1, width2, gpu_stream);
    cpu::signal::fft::bandpass<fft::H2H, real_t>(nullptr, {}, h_filter.share(), stride_fft, shape,
                                                 cutoff1, cutoff2, width1, width2, cpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_filter.get(), d_filter.get(), h_filter.elements(), epsilon));

    // Test on-the-fly, in-place.
    cuda::signal::fft::bandpass<fft::H2H>(d_data.share(), stride_fft, d_data.share(), stride_fft, shape,
                                          cutoff1, cutoff2, width1, width2, gpu_stream);
    cpu::signal::fft::bandpass<fft::H2H>(h_data.share(), stride_fft, h_data.share(), stride_fft, shape,
                                         cutoff1, cutoff2, width1, width2, cpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_data.get(), d_data.get(), h_data.elements(), epsilon));
}

TEMPLATE_TEST_CASE("cuda::signal::fft::bandpass(), remap", "[noa][cuda][signal][fft]", half_t, float) {
    const size4_t shape = test::getRandomShapeBatched(3);
    const float cutoff1 = 0.3f, cutoff2 = 0.4f;
    const float width = 0.1f;

    const size_t elements = shape.fft().elements();
    const size4_t stride = shape.fft().strides();

    cuda::Stream stream;
    cuda::memory::PtrManaged<float> filter_expected(elements, stream);
    cuda::memory::PtrManaged<float> filter_result(elements, stream);
    cuda::memory::PtrManaged<float> filter_remapped(elements, stream);
    test::memset(filter_expected.get(), elements, 1);
    test::memset(filter_result.get(), elements, 1);

    // H2HC
    cuda::signal::fft::bandpass<fft::H2H, float>(
            nullptr, {}, filter_expected.share(), stride, shape, cutoff1, cutoff2, width, width, stream);
    cuda::signal::fft::bandpass<fft::H2HC, float>(
            nullptr, {}, filter_result.share(), stride, shape, cutoff1, cutoff2, width, width, stream);
    cuda::fft::remap(fft::HC2H, filter_result.share(), stride, filter_remapped.share(), stride, shape, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2HC
    cuda::signal::fft::bandpass<fft::H2H, float>(
            nullptr, {}, filter_expected.share(), stride, shape, cutoff1, cutoff2, width, width, stream);
    cuda::signal::fft::bandpass<fft::HC2HC, float>(
            nullptr, {}, filter_result.share(), stride, shape, cutoff1, cutoff2, width, width, stream);
    cuda::fft::remap(fft::HC2H, filter_result.share(), stride, filter_remapped.share(), stride, shape, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2H
    cuda::signal::fft::bandpass<fft::H2H, float>(
            nullptr, {}, filter_expected.share(), stride, shape, cutoff1, cutoff2, width, width, stream);
    cuda::signal::fft::bandpass<fft::HC2H, float>(
            nullptr, {}, filter_result.share(), stride, shape, cutoff1, cutoff2, width, width, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));
}
