#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/fft/Filters.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cpu::fft::lowpass()", "[assets][noa][cpu][fft]") {
    test::Randomizer<float> randomizer(-5, 5);

    path_t path_base = test::PATH_TEST_DATA / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "param.yaml")["lowpass"];
    io::ImageFile file;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto shape = test["shape"].as<size3_t>();
        auto cutoff = test["cutoff"].as<float>();
        auto width = test["width"].as<float>();
        auto filename_expected = path_base / test["path"].as<path_t>();

        size_t elements = elementsFFT(shape);
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
        cpu::fft::lowpass(filter_result.get(), shape, cutoff, width);
        float diff = test::getAverageDifference(filter_expected.get(), filter_result.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));

        // Test on-the-fly, in-place.
        cpu::fft::lowpass(input_result.get(), input_result.get(), shape, cutoff, width, batches);
        for (uint batch = 0; batch < batches; ++batch)
            for (size_t idx = 0; idx < elements; ++idx)
                input_expected[elements * batch + idx] *= filter_expected[idx];
        diff = test::getAverageDifference(input_result.get(), input_expected.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));
    }
}

TEST_CASE("cpu::fft::highpass()", "[noa][cpu][fft]") {
    test::Randomizer<float> randomizer(-5, 5);

    path_t path_base = test::PATH_TEST_DATA / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "param.yaml")["highpass"];
    io::ImageFile file;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto shape = test["shape"].as<size3_t>();
        auto cutoff = test["cutoff"].as<float>();
        auto width = test["width"].as<float>();
        auto filename_expected = path_base / test["path"].as<path_t>();

        size_t size = elementsFFT(shape);
        cpu::memory::PtrHost<float> filter_expected(size);
        file.open(filename_expected, io::READ);
        file.readAll(filter_expected.get());

        uint batches = test::IntRandomizer<uint>(1, 3).get();
        cpu::memory::PtrHost<float> filter_result(size * batches);
        cpu::memory::PtrHost<float> input_result(size * batches);
        cpu::memory::PtrHost<float> input_expected(size * batches);

        test::initDataRandom(input_expected.get(), size * batches, randomizer);
        std::memcpy(input_result.get(), input_expected.get(), size * batches * sizeof(float));

        // Test saving the mask.
        cpu::fft::highpass(filter_result.get(), shape, cutoff, width);
        float diff = test::getAverageDifference(filter_expected.get(), filter_result.get(), size);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));

        // Test on-the-fly, in-place.
        cpu::fft::highpass(input_result.get(), input_result.get(), shape, cutoff, width, batches);
        for (uint batch = 0; batch < batches; ++batch)
            for (size_t idx = 0; idx < size; ++idx)
                input_expected[size * batch + idx] *= filter_expected[idx];
        diff = test::getAverageDifference(input_result.get(), input_expected.get(), size);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));
    }
}

TEST_CASE("cpu::fft::bandpass()", "[noa][cpu][fft]") {
    test::Randomizer<float> randomizer(-5, 5);

    path_t path_base = test::PATH_TEST_DATA / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "param.yaml")["bandpass"];
    io::ImageFile file;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto shape = test["shape"].as<size3_t>();
        auto cutoff = test["cutoff"].as<std::vector<float>>();
        auto width = test["width"].as<std::vector<float>>();
        auto filename_expected = path_base / test["path"].as<path_t>();

        size_t elements = elementsFFT(shape);
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
        cpu::fft::bandpass(filter_result.get(), shape, cutoff[0], cutoff[1], width[0], width[1]);
        float diff = test::getAverageDifference(filter_expected.get(), filter_result.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));

        // Test on-the-fly, in-place.
        cpu::fft::bandpass(input_result.get(), input_result.get(), shape,
                           cutoff[0], cutoff[1], width[0], width[1], batches);
        for (uint batch = 0; batch < batches; ++batch)
            for (size_t idx = 0; idx < elements; ++idx)
                input_expected[elements * batch + idx] *= filter_expected[idx];
        diff = test::getAverageDifference(input_result.get(), input_expected.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));
    }
}
