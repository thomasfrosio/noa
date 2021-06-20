#include <noa/cpu/fourier/Filters.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/io/files/MRCFile.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("Fourier: lowpass filters", "[noa][cpu][fourier]") {
    test::Randomizer<float> randomizer(-5, 5);
    path_t filename;
    MRCFile file;

    size3_t shape;
    float cutoff{}, width{};

    int test_number = GENERATE(1, 2, 3, 4, 5, 6, 7);
    INFO("test number: " << test_number);
    test::assets::fourier::getLowpassParams(test_number, &filename, &shape, &cutoff, &width);

    size_t elements = getElementsFFT(shape);
    memory::PtrHost<float> filter_expected(elements);
    file.open(filename, io::READ);
    file.readAll(filter_expected.get());

    uint batches = test::IntRandomizer<uint>(1, 3).get();
    memory::PtrHost<float> filter_result(elements * batches);
    memory::PtrHost<float> input_result(elements * batches);
    memory::PtrHost<float> input_expected(elements * batches);

    test::initDataRandom(input_expected.get(), elements * batches, randomizer);
    std::memcpy(input_result.get(), input_expected.get(), elements * batches * sizeof(float));

    // Test saving the mask.
    fourier::lowpass(filter_result.get(), shape, cutoff, width);
    float diff = test::getAverageDifference(filter_expected.get(), filter_result.get(), elements);
    REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));

    // Test on-the-fly, in-place.
    fourier::lowpass(input_result.get(), input_result.get(), shape, cutoff, width, batches);
    for (uint batch = 0; batch < batches; ++batch)
        for (size_t idx = 0; idx < elements; ++idx)
            input_expected[elements * batch + idx] *= filter_expected[idx];
    diff = test::getAverageDifference(input_result.get(), input_expected.get(), elements);
    REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));
}

TEST_CASE("Fourier: highpass filters", "[noa][cpu][fourier]") {
    test::Randomizer<float> randomizer(-5, 5);
    path_t filename;
    MRCFile file;

    size3_t shape;
    float cutoff{}, width{};

    int test_number = GENERATE(1, 2, 3, 4, 5, 6, 7);
    INFO("test number: " << test_number);
    test::assets::fourier::getHighpassParams(test_number, &filename, &shape, &cutoff, &width);

    size_t elements = getElementsFFT(shape);
    memory::PtrHost<float> filter_expected(elements);
    file.open(filename, io::READ);
    file.readAll(filter_expected.get());

    uint batches = test::IntRandomizer<uint>(1, 3).get();
    memory::PtrHost<float> filter_result(elements * batches);
    memory::PtrHost<float> input_result(elements * batches);
    memory::PtrHost<float> input_expected(elements * batches);

    test::initDataRandom(input_expected.get(), elements * batches, randomizer);
    std::memcpy(input_result.get(), input_expected.get(), elements * batches * sizeof(float));

    // Test saving the mask.
    fourier::highpass(filter_result.get(), shape, cutoff, width);
    float diff = test::getAverageDifference(filter_expected.get(), filter_result.get(), elements);
    REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));

    // Test on-the-fly, in-place.
    fourier::highpass(input_result.get(), input_result.get(), shape, cutoff, width, batches);
    for (uint batch = 0; batch < batches; ++batch)
        for (size_t idx = 0; idx < elements; ++idx)
            input_expected[elements * batch + idx] *= filter_expected[idx];
    diff = test::getAverageDifference(input_result.get(), input_expected.get(), elements);
    REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));
}

TEST_CASE("Fourier: bandpass filters", "[noa][cpu][fourier]") {
    test::Randomizer<float> randomizer(-5, 5);
    path_t filename;
    MRCFile file;

    size3_t shape;
    float cutoff1{}, width1{};
    float cutoff2{}, width2{};

    int test_number = GENERATE(1, 2, 3, 4, 5);
    INFO("test number: " << test_number);
    test::assets::fourier::getBandpassParams(test_number, &filename, &shape, &cutoff1, &cutoff2, &width1, &width2);

    size_t elements = getElementsFFT(shape);
    memory::PtrHost<float> filter_expected(elements);
    file.open(filename, io::READ);
    file.readAll(filter_expected.get());

    uint batches = test::IntRandomizer<uint>(1, 3).get();
    memory::PtrHost<float> filter_result(elements * batches);
    memory::PtrHost<float> input_result(elements * batches);
    memory::PtrHost<float> input_expected(elements * batches);

    test::initDataRandom(input_expected.get(), elements * batches, randomizer);
    std::memcpy(input_result.get(), input_expected.get(), elements * batches * sizeof(float));

    // Test saving the mask.
    fourier::bandpass(filter_result.get(), shape, cutoff1, cutoff2, width1, width2);
    float diff = test::getAverageDifference(filter_expected.get(), filter_result.get(), elements);
    REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));

    // Test on-the-fly, in-place.
    fourier::bandpass(input_result.get(), input_result.get(), shape, cutoff1, cutoff2, width1, width2, batches);
    for (uint batch = 0; batch < batches; ++batch)
        for (size_t idx = 0; idx < elements; ++idx)
            input_expected[elements * batch + idx] *= filter_expected[idx];
    diff = test::getAverageDifference(input_result.get(), input_expected.get(), elements);
    REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));
}
