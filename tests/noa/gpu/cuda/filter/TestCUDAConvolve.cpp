#include <noa/common/files/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/filter/Convolve.h>

#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/filter/Convolve.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

TEST_CASE("cuda::filter::convolve()", "[noa][cuda][filter]") {
    using namespace noa;

    int test_number = GENERATE(1, 2, 3, 4, 5, 6, 7);
    INFO("test_number = " << test_number);
    path_t filename_data;
    path_t filename_filter;
    path_t filename_expected;
    size3_t shape;
    uint3_t filter_shape;

    test::assets::filter::getConvData(test_number, &filename_data);
    test::assets::filter::getConvFilter(test_number, &filename_filter);
    test::assets::filter::getConvParams(test_number, &filename_expected, &shape, &filter_shape);

    size_t elements = getElements(shape);
    memory::PtrHost<float> data(elements);
    memory::PtrHost<float> filter(160); // for 1D case, the MRC file as an extra row to make it 2D.
    memory::PtrHost<float> expected(elements);

    MRCFile file(filename_data, io::READ);
    file.readAll(data.get());
    file.open(filename_filter, io::READ);
    file.readAll(filter.get());
    file.open(filename_expected, io::READ);
    file.readAll(expected.get());

    cuda::Stream stream;
    cuda::memory::PtrDevicePadded<float> d_data(shape);
    cuda::memory::PtrDevicePadded<float> d_result(shape);
    cuda::memory::PtrDevice<float> d_filter(filter.elements()); // FYI, convolve doesn't need the filter to be on device.
    memory::PtrHost<float> result(elements);

    cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
    cuda::memory::copy(filter.get(), d_filter.get(), filter.elements());
    cuda::filter::convolve(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(), shape, 1,
                           filter.get(), filter_shape, stream);
    cuda::memory::copy(d_result.get(), d_result.pitch(), result.get(), shape.x, shape, stream);
    filter::convolve(data.get(), result.get(), shape, 1, filter.get(), filter_shape);
    cuda::Stream::synchronize(stream);

    float diff = test::getAverageNormalizedDifference(expected.get(), result.get(), elements);
    REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
}

TEST_CASE("cuda::filter::convolve() - separable", "[noa][cuda][filter]") {
    using namespace noa;

    int test_number = GENERATE(8, 9, 10, 11, 12, 13, 14, 15, 16, 17);
    INFO("test_number = " << test_number);
    path_t filename_data;
    path_t filename_filter;
    path_t filename_expected;
    size3_t shape;
    uint3_t filter_shape;

    test::assets::filter::getConvData(test_number, &filename_data);
    test::assets::filter::getConvFilter(test_number, &filename_filter);
    test::assets::filter::getConvParams(test_number, &filename_expected, &shape, &filter_shape);

    size_t elements = getElements(shape);
    memory::PtrHost<float> data(elements);
    memory::PtrHost<float> filter(math::max(filter_shape) * 2); // the MRC file as an extra row to make it 2D.
    memory::PtrHost<float> expected(elements);

    MRCFile file(filename_data, io::READ);
    file.readAll(data.get());
    file.open(filename_filter, io::READ);
    file.readAll(filter.get());
    file.open(filename_expected, io::READ);
    file.readAll(expected.get());

    float* filter0 = nullptr;
    float* filter1 = nullptr;
    float* filter2 = nullptr;
    if (filter_shape[0] > 1)
        filter0 = filter.get();
    if (filter_shape[1] > 1)
        filter1 = filter.get();
    if (filter_shape[2] > 1)
        filter2 = filter.get();

    cuda::Stream stream;
    cuda::memory::PtrDevicePadded<float> d_data(shape);
    cuda::memory::PtrDevicePadded<float> d_result(shape);
    memory::PtrHost<float> result(elements);

    cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
    cuda::filter::convolve(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(), shape, 1,
                           filter0, filter_shape[0], filter1, filter_shape[1], filter2, filter_shape[2], stream);
    cuda::memory::copy(d_result.get(), d_result.pitch(), result.get(), shape.x, shape, stream);
    filter::convolve(data.get(), result.get(), shape, 1,
                     filter0, filter_shape[0], filter1, filter_shape[1], filter2, filter_shape[2]);
    cuda::Stream::synchronize(stream);

    float diff = test::getAverageNormalizedDifference(expected.get(), result.get(), elements);
    REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
}
