#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include <noa/io/files/MRCFile.h>
#include <noa/cpu/filter/Median.h>
#include <noa/gpu/cuda/filter/Median.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

TEST_CASE("CUDA::Filter::median", "[noa][cpu][filter]") {
    using namespace Noa;

    int test_number = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    INFO("test_number = " << test_number);
    path_t filename_data;
    path_t filename_expected;
    uint window;
    BorderMode mode;
    size3_t shape;

    Test::Assets::Filter::getMedianData(test_number, &filename_data);
    Test::Assets::Filter::getMedianParams(test_number, &filename_expected, &shape, &mode, &window);

    size_t elements = getElements(shape);
    Memory::PtrHost<float> data(elements);
    Memory::PtrHost<float> expected(elements);

    MRCFile file(filename_data, IO::READ);
    file.readAll(data.get());
    file.open(filename_expected, IO::READ);
    file.readAll(expected.get());

    CUDA::Memory::PtrDevicePadded<float> d_data(shape);
    CUDA::Memory::PtrDevicePadded<float> d_result(shape);
    Memory::PtrHost<float> result(elements);
    CUDA::Stream stream;

    CUDA::Memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);

    if (test_number < 5)
        CUDA::Filter::median1(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                              shape, 1, mode, window, stream);
    else if (test_number < 9)
        CUDA::Filter::median2(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                              shape, 1, mode, window, stream);
    else
        CUDA::Filter::median3(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                              shape, 1, mode, window, stream);

    CUDA::Memory::copy(d_result.get(), d_result.pitch(), result.get(), shape.x, shape, stream);
    CUDA::Stream::synchronize(stream);

    float diff = Test::getAverageDifference(expected.get(), result.get(), elements);
    REQUIRE_THAT(diff, Test::isWithinRel(0.f, 1e-6));
}

TEMPLATE_TEST_CASE("CUDA::Filter::median, random", "[noa][cpu][filter]", int, float, double) {
    using namespace Noa;

    int ndim = GENERATE(1, 2, 3);
    BorderMode mode = GENERATE(BORDER_ZERO, BORDER_MIRROR);
    uint window = Test::IntRandomizer<uint>(2, 11).get();
    if (!(window % 2))
        window -= 1;
    if (ndim == 3 && window > 5)
        window = 3;

    Test::IntRandomizer<size_t> random_size(16, 100);
    size3_t shape{random_size.get(), random_size.get(), random_size.get()};
    if (ndim != 3 && random_size.get() % 2)
        shape.z = 1; // randomly switch to 2D
    size_t elements = getElements(shape);

    uint batches = Test::IntRandomizer<uint>(1, 3).get();
    size_t elements_batched = elements * batches;
    size3_t shape_batched = {shape.x, getRows(shape), batches};

    INFO(String::format("ndim:{}, mode:{}, window:{}, shape:{}, batches:{}",
                        ndim, mode, window, shape, batches));

    Test::Randomizer<TestType> randomizer(-128, 128);
    Memory::PtrHost<TestType> data(elements_batched);
    Test::initDataRandom(data.get(), data.elements(), randomizer);

    CUDA::Memory::PtrDevicePadded<TestType> d_data(shape_batched);
    CUDA::Memory::PtrDevicePadded<TestType> d_result(shape_batched);
    Memory::PtrHost<TestType> cuda_result(elements_batched);
    Memory::PtrHost<TestType> h_result(elements_batched);
    CUDA::Stream stream;

    CUDA::Memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

    if (ndim == 1) {
        CUDA::Filter::median1(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                              shape, batches, mode, window, stream);
        Filter::median1(data.get(), h_result.get(), shape, batches, mode, window);
    } else if (ndim == 2) {
        CUDA::Filter::median2(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                              shape, batches, mode, window, stream);
        Filter::median2(data.get(), h_result.get(), shape, batches, mode, window);
    } else {
        CUDA::Filter::median3(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                              shape, batches, mode, window, stream);
        Filter::median3(data.get(), h_result.get(), shape, batches, mode, window);
    }
    CUDA::Memory::copy(d_result.get(), d_result.pitch(), cuda_result.get(), shape.x, shape_batched, stream);
    CUDA::Stream::synchronize(stream);

    TestType diff = Test::getAverageDifference(h_result.get(), cuda_result.get(), elements_batched);
    REQUIRE_THAT(diff, Test::isWithinRel(0.f, 1e-6));
}
