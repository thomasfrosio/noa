#include <noa/common/files/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/fourier/Filters.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cpu::fourier::lowpass()", "[assets][noa][cpu][fourier]") {
    test::Randomizer<float> randomizer(-5, 5);

    path_t path_base = test::PATH_TEST_DATA / "fourier";
    YAML::Node tests = YAML::LoadFile(path_base / "param.yaml")["lowpass"];
    MRCFile file;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto shape = test["shape"].as<size3_t>();
        auto cutoff = test["cutoff"].as<float>();
        auto width = test["width"].as<float>();
        auto filename_expected = path_base / test["path"].as<path_t>();

        size_t elements = getElementsFFT(shape);
        cpu::memory::PtrHost<float> filter_expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(filter_expected.get());

        uint batches = test::IntRandomizer<uint>(1, 3).get();
        cpu::memory::PtrHost<float> filter_result(elements * batches);
        cpu::memory::PtrHost<float> input_result(elements * batches);
        cpu::memory::PtrHost<float> input_expected(elements * batches);

        test::initDataRandom(input_expected.get(), elements * batches, randomizer);
        std::memcpy(input_result.get(), input_expected.get(), elements * batches * sizeof(float));

        // Test saving the mask.
        cpu::fourier::lowpass(filter_result.get(), shape, cutoff, width);
        float diff = test::getAverageDifference(filter_expected.get(), filter_result.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));

        // Test on-the-fly, in-place.
        cpu::fourier::lowpass(input_result.get(), input_result.get(), shape, cutoff, width, batches);
        for (uint batch = 0; batch < batches; ++batch)
            for (size_t idx = 0; idx < elements; ++idx)
                input_expected[elements * batch + idx] *= filter_expected[idx];
        diff = test::getAverageDifference(input_result.get(), input_expected.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));
    }
}

TEST_CASE("cpu::fourier::highpass()", "[noa][cpu][fourier]") {
    test::Randomizer<float> randomizer(-5, 5);

    path_t path_base = test::PATH_TEST_DATA / "fourier";
    YAML::Node tests = YAML::LoadFile(path_base / "param.yaml")["highpass"];
    MRCFile file;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto shape = test["shape"].as<size3_t>();
        auto cutoff = test["cutoff"].as<float>();
        auto width = test["width"].as<float>();
        auto filename_expected = path_base / test["path"].as<path_t>();

        size_t elements = getElementsFFT(shape);
        cpu::memory::PtrHost<float> filter_expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(filter_expected.get());

        uint batches = test::IntRandomizer<uint>(1, 3).get();
        cpu::memory::PtrHost<float> filter_result(elements * batches);
        cpu::memory::PtrHost<float> input_result(elements * batches);
        cpu::memory::PtrHost<float> input_expected(elements * batches);

        test::initDataRandom(input_expected.get(), elements * batches, randomizer);
        std::memcpy(input_result.get(), input_expected.get(), elements * batches * sizeof(float));

        // Test saving the mask.
        cpu::fourier::highpass(filter_result.get(), shape, cutoff, width);
        float diff = test::getAverageDifference(filter_expected.get(), filter_result.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));

        // Test on-the-fly, in-place.
        cpu::fourier::highpass(input_result.get(), input_result.get(), shape, cutoff, width, batches);
        for (uint batch = 0; batch < batches; ++batch)
            for (size_t idx = 0; idx < elements; ++idx)
                input_expected[elements * batch + idx] *= filter_expected[idx];
        diff = test::getAverageDifference(input_result.get(), input_expected.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));
    }
}

TEST_CASE("cpu::fourier::bandpass()", "[noa][cpu][fourier]") {
    test::Randomizer<float> randomizer(-5, 5);

    path_t path_base = test::PATH_TEST_DATA / "fourier";
    YAML::Node tests = YAML::LoadFile(path_base / "param.yaml")["bandpass"];
    MRCFile file;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto shape = test["shape"].as<size3_t>();
        auto cutoff = test["cutoff"].as<std::vector<float>>();
        auto width = test["width"].as<std::vector<float>>();
        auto filename_expected = path_base / test["path"].as<path_t>();

        size_t elements = getElementsFFT(shape);
        cpu::memory::PtrHost<float> filter_expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(filter_expected.get());

        uint batches = test::IntRandomizer<uint>(1, 3).get();
        cpu::memory::PtrHost<float> filter_result(elements * batches);
        cpu::memory::PtrHost<float> input_result(elements * batches);
        cpu::memory::PtrHost<float> input_expected(elements * batches);

        test::initDataRandom(input_expected.get(), elements * batches, randomizer);
        std::memcpy(input_result.get(), input_expected.get(), elements * batches * sizeof(float));

        // Test saving the mask.
        cpu::fourier::bandpass(filter_result.get(), shape, cutoff[0], cutoff[1], width[0], width[1]);
        float diff = test::getAverageDifference(filter_expected.get(), filter_result.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));

        // Test on-the-fly, in-place.
        cpu::fourier::bandpass(input_result.get(), input_result.get(), shape,
                               cutoff[0], cutoff[1], width[0], width[1], batches);
        for (uint batch = 0; batch < batches; ++batch)
            for (size_t idx = 0; idx < elements; ++idx)
                input_expected[elements * batch + idx] *= filter_expected[idx];
        diff = test::getAverageDifference(input_result.get(), input_expected.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));
    }
}
