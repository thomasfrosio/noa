#include <noa/common/files/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include <noa/cpu/filter/Median.h>
#include <noa/gpu/cuda/filter/Median.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

TEST_CASE("cuda::filter::median()", "[noa][cuda][filter]") {
    using namespace noa;

    int test_number = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    INFO("test_number = " << test_number);
    path_t filename_data;
    path_t filename_expected;
    uint window;
    BorderMode mode;
    size3_t shape;

    test::assets::filter::getMedianData(test_number, &filename_data);
    test::assets::filter::getMedianParams(test_number, &filename_expected, &shape, &mode, &window);

    size_t elements = getElements(shape);
    memory::PtrHost<float> data(elements);
    memory::PtrHost<float> expected(elements);

    MRCFile file(filename_data, io::READ);
    file.readAll(data.get());
    file.open(filename_expected, io::READ);
    file.readAll(expected.get());

    cuda::memory::PtrDevicePadded<float> d_data(shape);
    cuda::memory::PtrDevicePadded<float> d_result(shape);
    memory::PtrHost<float> result(elements);
    cuda::Stream stream;

    cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);

    if (test_number < 5)
        cuda::filter::median1(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                              shape, 1, mode, window, stream);
    else if (test_number < 9)
        cuda::filter::median2(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                              shape, 1, mode, window, stream);
    else
        cuda::filter::median3(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                              shape, 1, mode, window, stream);

    cuda::memory::copy(d_result.get(), d_result.pitch(), result.get(), shape.x, shape, stream);
    cuda::Stream::synchronize(stream);

    float diff = test::getAverageDifference(expected.get(), result.get(), elements);
    REQUIRE_THAT(diff, test::isWithinRel(0.f, 1e-6));
}

TEMPLATE_TEST_CASE("cuda::filter::median(), random", "[noa][cuda][filter]", int, float, double) {
    using namespace noa;

    int ndim = GENERATE(1, 2, 3);
    BorderMode mode = GENERATE(BORDER_ZERO, BORDER_REFLECT);
    uint window = test::IntRandomizer<uint>(2, 11).get();
    if (!(window % 2))
        window -= 1;
    if (ndim == 3 && window > 5)
        window = 3;

    test::IntRandomizer<size_t> random_size(16, 100);
    size3_t shape{random_size.get(), random_size.get(), random_size.get()};
    if (ndim != 3 && random_size.get() % 2)
        shape.z = 1; // randomly switch to 2D
    size_t elements = getElements(shape);

    uint batches = test::IntRandomizer<uint>(1, 3).get();
    size_t elements_batched = elements * batches;
    size3_t shape_batched = {shape.x, getRows(shape), batches};

    INFO(string::format("ndim:{}, mode:{}, window:{}, shape:{}, batches:{}",
                        ndim, mode, window, shape, batches));

    test::Randomizer<TestType> randomizer(-128, 128);
    memory::PtrHost<TestType> data(elements_batched);
    test::initDataRandom(data.get(), data.elements(), randomizer);

    cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
    cuda::memory::PtrDevicePadded<TestType> d_result(shape_batched);
    memory::PtrHost<TestType> cuda_result(elements_batched);
    memory::PtrHost<TestType> h_result(elements_batched);
    cuda::Stream stream;

    cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

    if (ndim == 1) {
        cuda::filter::median1(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                              shape, batches, mode, window, stream);
        filter::median1(data.get(), h_result.get(), shape, batches, mode, window);
    } else if (ndim == 2) {
        cuda::filter::median2(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                              shape, batches, mode, window, stream);
        filter::median2(data.get(), h_result.get(), shape, batches, mode, window);
    } else {
        cuda::filter::median3(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                              shape, batches, mode, window, stream);
        filter::median3(data.get(), h_result.get(), shape, batches, mode, window);
    }
    cuda::memory::copy(d_result.get(), d_result.pitch(), cuda_result.get(), shape.x, shape_batched, stream);
    cuda::Stream::synchronize(stream);

    TestType diff = test::getAverageDifference(h_result.get(), cuda_result.get(), elements_batched);
    REQUIRE_THAT(diff, test::isWithinRel(0.f, 1e-6));
}
