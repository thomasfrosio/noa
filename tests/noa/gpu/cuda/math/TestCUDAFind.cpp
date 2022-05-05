#include <noa/cpu/math/Find.h>
#include <noa/gpu/cuda/math/Find.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/PtrManaged.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::math::find()", "[noa][cuda][math]", int32_t, int64_t, float, double) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();
    cpu::Stream cpu_stream{cpu::Stream::DEFAULT};
    cuda::Stream gpu_stream;

    const bool batch = GENERATE(true, false);
    size_t output_size = batch ? shape[0] : 1;

    cuda::memory::PtrManaged<TestType> data(elements);
    cuda::memory::PtrManaged<size_t> offset_expected(output_size);
    cuda::memory::PtrManaged<size_t> offset_results(output_size);

    test::Randomizer<TestType> randomizer(-100., 100.);
    test::randomize(data.get(), data.elements(), randomizer);

    cpu::math::find(math::min_t{}, data.share(), stride, shape, offset_expected.share(), batch, cpu_stream);
    cuda::math::find(math::min_t{}, data.share(), stride, shape, offset_results.share(), batch, gpu_stream);
    gpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, offset_expected.get(), offset_results.get(), shape[0], 1e-7));

    cpu::math::find(math::max_t{}, data.share(), stride, shape, offset_expected.share(), batch, cpu_stream);
    cuda::math::find(math::max_t{}, data.share(), stride, shape, offset_results.share(), batch, gpu_stream);
    gpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, offset_expected.get(), offset_results.get(), output_size, 1e-7));
}

TEMPLATE_TEST_CASE("cuda::math::find(), padded", "[noa][cuda][math]", int32_t, int64_t, float, double) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();
    cpu::Stream cpu_stream{cpu::Stream::DEFAULT};
    cuda::Stream gpu_stream;

    const bool batch = GENERATE(true, false);
    size_t output_size = batch ? shape[0] : 1;

    cuda::memory::PtrManaged<TestType> data(elements);
    cuda::memory::PtrManaged<size_t> offset_expected(output_size);
    cuda::memory::PtrManaged<size_t> offset_results(output_size);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape);

    test::Randomizer<TestType> randomizer(-100., 100.);
    test::randomize(data.get(), data.elements(), randomizer);
    cuda::memory::copy(data.share(), stride, d_data.share(), d_data.stride(), shape, gpu_stream);

    cpu::math::find(math::min_t{}, data.share(), stride, shape, offset_expected.share(), batch, cpu_stream);
    cuda::math::find(math::min_t{}, d_data.share(), d_data.stride(), shape, offset_results.share(), batch, gpu_stream);
    gpu_stream.synchronize();
    int4_t diff;
    for (size_t i = 0; i < output_size; ++i) {
        const auto idx0 = noa::indexing::indexes(offset_expected[0], shape[1], shape[2], shape[3]);
        const auto idx1 = noa::indexing::indexes(offset_results[0], shape[1], shape[2], d_data.pitch()[2]);
        diff += int4_t{idx0} - int4_t{idx1};
    }
    REQUIRE(all(diff == 0));

    cpu::math::find(math::max_t{}, data.share(), stride, shape, offset_expected.share(), batch, cpu_stream);
    cuda::math::find(math::max_t{}, d_data.share(), d_data.stride(), shape, offset_results.share(), batch, gpu_stream);
    gpu_stream.synchronize();
    for (size_t i = 0; i < output_size; ++i) {
        const auto idx0 = noa::indexing::indexes(offset_expected[0], shape[1], shape[2], shape[3]);
        const auto idx1 = noa::indexing::indexes(offset_results[0], shape[1], shape[2], d_data.pitch()[2]);
        diff += int4_t{idx0} - int4_t{idx1};
    }
    REQUIRE(all(diff == 0));
}
