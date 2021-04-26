#include <noa/cpu/fourier/Filters.h>

#include <noa/cpu/PtrHost.h>
#include <noa/io/files/MRCFile.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEST_CASE("Fourier: lowpass filters", "[noa][cpu][fourier]") {
    Test::Randomizer<float> randomizer(-5, 5);
    path_t path_data = Test::PATH_TEST_DATA / "fourier";
    MRCFile file;

    size3_t shape;
    float cutoff{}, width{};

    int test_number = GENERATE(1, 2, 3, 4, 5, 6, 7);
    if (test_number == 1) {
        shape = {256, 256, 1};
        cutoff = 0;
        width = 0;
        path_data /= "lowpass_01.mrc";
    } else if (test_number == 2) {
        shape = {256, 256, 1};
        cutoff = 0.5f;
        width = 0;
        path_data /= "lowpass_02.mrc";
    } else if (test_number == 3) {
        shape = {256, 256, 1};
        cutoff = 0.35f;
        width = 0.1f;
        path_data /= "lowpass_03.mrc";
    } else if (test_number == 4) {
        shape = {512, 256, 1};
        cutoff = 0.2f;
        width = 0.3f;
        path_data /= "lowpass_04.mrc";
    } else if (test_number == 5) {
        shape = {128, 128, 128};
        cutoff = 0;
        width = 0;
        path_data /= "lowpass_11.mrc";
    } else if (test_number == 6) {
        shape = {128, 128, 128};
        cutoff = 0.5f;
        width = 0;
        path_data /= "lowpass_12.mrc";
    } else if (test_number == 7) {
        shape = {64, 128, 128};
        cutoff = 0.2f;
        width = 0.3f;
        path_data /= "lowpass_13.mrc";
    }
    INFO("test number: " << test_number);

    size_t elements = getElementsFFT(shape);
    PtrHost<float> filter_expected(elements);
    file.open(path_data, IO::READ);
    file.readAll(filter_expected.get());

    uint batches = Test::IntRandomizer<uint>(1, 3).get();
    PtrHost<float> filter_result(elements * batches);
    PtrHost<float> input_result(elements * batches);
    PtrHost<float> input_expected(elements * batches);

    Test::initDataRandom(input_expected.get(), elements * batches, randomizer);
    std::memcpy(input_result.get(), input_expected.get(), elements * batches * sizeof(float));

    // Test saving the mask.
    Fourier::lowpass(filter_result.get(), shape, cutoff, width);
    float diff = Test::getAverageDifference(filter_expected.get(), filter_result.get(), elements);
    REQUIRE_THAT(diff, Test::isWithinAbs(0.f, 1e-7));

    // Test on-the-fly, in-place.
    Fourier::lowpass(input_result.get(), input_result.get(), shape, cutoff, width, batches);
    for (uint batch = 0; batch < batches; ++ batch)
        for (size_t idx = 0; idx < elements; ++idx)
            input_expected[elements * batch + idx] *= filter_expected[idx];
    diff = Test::getAverageDifference(input_result.get(), input_expected.get(), elements);
    REQUIRE_THAT(diff, Test::isWithinAbs(0.f, 1e-7));
}

TEST_CASE("Fourier: highpass filters", "[noa][cpu][fourier]") {
    Test::Randomizer<float> randomizer(-5, 5);
    path_t path_data = Test::PATH_TEST_DATA / "fourier";
    MRCFile file;

    size3_t shape;
    float cutoff{}, width{};

    int test_number = GENERATE(1, 2, 3, 4, 5, 6, 7);
    if (test_number == 1) {
        shape = {256, 256, 1};
        cutoff = 0;
        width = 0;
        path_data /= "highpass_01.mrc";
    } else if (test_number == 2) {
        shape = {256, 256, 1};
        cutoff = 0.5f;
        width = 0;
        path_data /= "highpass_02.mrc";
    } else if (test_number == 3) {
        shape = {256, 256, 1};
        cutoff = 0.35f;
        width = 0.1f;
        path_data /= "highpass_03.mrc";
    } else if (test_number == 4) {
        shape = {512, 256, 1};
        cutoff = 0.2f;
        width = 0.3f;
        path_data /= "highpass_04.mrc";
    } else if (test_number == 5) {
        shape = {128, 128, 128};
        cutoff = 0;
        width = 0;
        path_data /= "highpass_11.mrc";
    } else if (test_number == 6) {
        shape = {128, 128, 128};
        cutoff = 0.5f;
        width = 0;
        path_data /= "highpass_12.mrc";
    } else if (test_number == 7) {
        shape = {64, 128, 128};
        cutoff = 0.2f;
        width = 0.3f;
        path_data /= "highpass_13.mrc";
    }
    INFO("test number: " << test_number);

    size_t elements = getElementsFFT(shape);
    PtrHost<float> filter_expected(elements);
    file.open(path_data, IO::READ);
    file.readAll(filter_expected.get());

    uint batches = Test::IntRandomizer<uint>(1, 3).get();
    PtrHost<float> filter_result(elements * batches);
    PtrHost<float> input_result(elements * batches);
    PtrHost<float> input_expected(elements * batches);

    Test::initDataRandom(input_expected.get(), elements * batches, randomizer);
    std::memcpy(input_result.get(), input_expected.get(), elements * batches * sizeof(float));

    // Test saving the mask.
    Fourier::highpass(filter_result.get(), shape, cutoff, width);
    float diff = Test::getAverageDifference(filter_expected.get(), filter_result.get(), elements);
    REQUIRE_THAT(diff, Test::isWithinAbs(0.f, 1e-7));

    // Test on-the-fly, in-place.
    Fourier::highpass(input_result.get(), input_result.get(), shape, cutoff, width, batches);
    for (uint batch = 0; batch < batches; ++ batch)
        for (size_t idx = 0; idx < elements; ++idx)
            input_expected[elements * batch + idx] *= filter_expected[idx];
    diff = Test::getAverageDifference(input_result.get(), input_expected.get(), elements);
    REQUIRE_THAT(diff, Test::isWithinAbs(0.f, 1e-7));
}

TEST_CASE("Fourier: bandpass filters", "[noa][cpu][fourier]") {
    Test::Randomizer<float> randomizer(-5, 5);
    path_t path_data = Test::PATH_TEST_DATA / "fourier";
    MRCFile file;

    size3_t shape;
    float cutoff1{}, width1{};
    float cutoff2{}, width2{};

    int test_number = GENERATE(1, 2, 3, 4, 5);
    if (test_number == 1) {
        shape = {256, 256, 1};
        cutoff1 = 0.4f;
        cutoff2 = 0.5f;
        width1 = 0;
        width2 = 0;
        path_data /= "bandpass_01.mrc";
    } else if (test_number == 2) {
        shape = {256, 512, 1};
        cutoff1 = 0.3f;
        cutoff2 = 0.45f;
        width1 = 0.3f;
        width2 = 0.05f;
        path_data /= "bandpass_02.mrc";
    } else if (test_number == 3) {
        shape = {128, 128, 1};
        cutoff1 = 0.3f;
        cutoff2 = 0.4f;
        width1 = 0.05f;
        width2 = 0.05f;
        path_data /= "bandpass_03.mrc";
    } else if (test_number == 4) {
        shape = {128, 128, 128};
        cutoff1 = 0.2f;
        cutoff2 = 0.45f;
        width1 = 0.1f;
        width2 = 0.05f;
        path_data /= "bandpass_11.mrc";
    } else if (test_number == 5) {
        shape = {64, 128, 128};
        cutoff1 = 0.1f;
        cutoff2 = 0.3f;
        width1 = 0;
        width2 = 0.1f;
        path_data /= "bandpass_12.mrc";
    }
    INFO("test number: " << test_number);

    size_t elements = getElementsFFT(shape);
    PtrHost<float> filter_expected(elements);
    file.open(path_data, IO::READ);
    file.readAll(filter_expected.get());

    uint batches = Test::IntRandomizer<uint>(1, 3).get();
    PtrHost<float> filter_result(elements * batches);
    PtrHost<float> input_result(elements * batches);
    PtrHost<float> input_expected(elements * batches);

    Test::initDataRandom(input_expected.get(), elements * batches, randomizer);
    std::memcpy(input_result.get(), input_expected.get(), elements * batches * sizeof(float));

    // Test saving the mask.
    Fourier::bandpass(filter_result.get(), shape, cutoff1, cutoff2, width1, width2);
    float diff = Test::getAverageDifference(filter_expected.get(), filter_result.get(), elements);
    REQUIRE_THAT(diff, Test::isWithinAbs(0.f, 1e-7));

    // Test on-the-fly, in-place.
    Fourier::bandpass(input_result.get(), input_result.get(), shape, cutoff1, cutoff2, width1, width2, batches);
    for (uint batch = 0; batch < batches; ++ batch)
        for (size_t idx = 0; idx < elements; ++idx)
            input_expected[elements * batch + idx] *= filter_expected[idx];
    diff = Test::getAverageDifference(input_result.get(), input_expected.get(), elements);
    REQUIRE_THAT(diff, Test::isWithinAbs(0.f, 1e-7));
}
