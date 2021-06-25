//#include <noa/cpu/memory/PtrHost.h>
//#include <noa/io/files/MRCFile.h>
//#include <noa/cpu/filter/Convolve.h>
//
//#include "Helpers.h"
//#include "Assets.h"
//#include <catch2/catch.hpp>
//
//TEST_CASE("filter::convolve()", "[noa][cpu][filter]") {
//    using namespace noa;
//
//    int test_number = GENERATE(1, 2, 3, 4, 5, 6, 7);
//    INFO("test_number = " << test_number);
//    path_t filename_data;
//    path_t filename_filter;
//    path_t filename_expected;
//    size3_t shape;
//    uint3_t filter_shape;
//
//    test::assets::filter::getConvData(test_number, &filename_data);
//    test::assets::filter::getConvFilter(test_number, &filename_filter);
//    test::assets::filter::getConvParams(test_number, &filename_expected, &shape, &filter_shape);
//
//    size_t elements = getElements(shape);
//    memory::PtrHost<float> data(elements);
//    memory::PtrHost<float> filter(130); // for 1D case, the MRC file as an extra row to make it 2D.
//    memory::PtrHost<float> expected(elements);
//    memory::PtrHost<float> result(elements);
//
//    MRCFile file(filename_data, io::READ);
//    file.readAll(data.get());
//    file.open(filename_filter, io::READ);
//    file.readAll(filter.get());
//    file.open(filename_expected, io::READ);
//    file.readAll(expected.get());
//
//    filter::convolve(data.get(), result.get(), shape, 1, filter.get(), filter_shape);
//    float diff = test::getAverageDifference(expected.get(), result.get(), elements);
//    REQUIRE_THAT(diff, test::isWithinRel(0.f, 1e-6));
//}
//
//TEST_CASE("filter::convolve() - separable", "[noa][cpu][filter]") {
//    using namespace noa;
//
//    int test_number = GENERATE(8, 9, 10, 11, 12, 13, 14, 15, 16, 17);
//    INFO("test_number = " << test_number);
//    path_t filename_data;
//    path_t filename_filter;
//    path_t filename_expected;
//    size3_t shape;
//    uint3_t filter_shape;
//
//    test::assets::filter::getConvData(test_number, &filename_data);
//    test::assets::filter::getConvFilter(test_number, &filename_filter);
//    test::assets::filter::getConvParams(test_number, &filename_expected, &shape, &filter_shape);
//
//    size_t elements = getElements(shape);
//    memory::PtrHost<float> data(elements);
//    memory::PtrHost<float> filter(130);
//    memory::PtrHost<float> expected(elements);
//    memory::PtrHost<float> result(elements);
//
//    MRCFile file(filename_data, io::READ);
//    file.readAll(data.get());
//    file.open(filename_filter, io::READ);
//    file.readAll(filter.get());
//    file.open(filename_expected, io::READ);
//    file.readAll(expected.get());
//
//    float* filter0;
//    float* filter1;
//    float* filter2;
//    if (filter_shape[0] > 1)
//        filter0 = filter.get();
//    if (filter_shape[1] > 1)
//        filter1 = filter.get();
//    if (filter_shape[2] > 1)
//        filter2 = filter.get();
//
//    filter::convolve(data.get(), result.get(), shape, 1,
//                     filter0, filter_shape[0], filter1, filter_shape[1], filter2, filter_shape[2]);
//    float diff = test::getAverageDifference(expected.get(), result.get(), elements);
//    REQUIRE_THAT(diff, test::isWithinRel(0.f, 1e-6));
//}
