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

        size_t elements = noa::elementsFFT(shape);
        cpu::memory::PtrHost<float> filter_expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(filter_expected.get());

        size_t batches = test::Randomizer<size_t>(1, 3).get();
        cpu::memory::PtrHost<float> filter_result(elements * batches);
        cpu::memory::PtrHost<float> input_result(elements * batches);
        cpu::memory::PtrHost<float> input_expected(elements * batches);

        test::randomize(input_expected.get(), elements * batches, randomizer);
        std::memcpy(input_result.get(), input_expected.get(), elements * batches * sizeof(float));

        // Test saving the mask.
        cpu::fft::lowpass<float>(nullptr, filter_result.get(), shape, 1, cutoff, width);
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));

        // Test on-the-fly, in-place.
        cpu::fft::lowpass(input_result.get(), input_result.get(), shape, batches, cutoff, width);
        for (size_t batch = 0; batch < batches; ++batch)
            for (size_t idx = 0; idx < elements; ++idx)
                input_expected[elements * batch + idx] *= filter_expected[idx];
        REQUIRE(test::Matcher(test::MATCH_ABS, input_result.get(), input_expected.get(), elements, 1e-6));
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

        size_t size = noa::elementsFFT(shape);
        cpu::memory::PtrHost<float> filter_expected(size);
        file.open(filename_expected, io::READ);
        file.readAll(filter_expected.get());

        size_t batches = test::Randomizer<size_t>(1, 3).get();
        cpu::memory::PtrHost<float> filter_result(size * batches);
        cpu::memory::PtrHost<float> input_result(size * batches);
        cpu::memory::PtrHost<float> input_expected(size * batches);

        test::randomize(input_expected.get(), size * batches, randomizer);
        std::memcpy(input_result.get(), input_expected.get(), size * batches * sizeof(float));

        // Test saving the mask.
        cpu::fft::highpass<float>(nullptr, filter_result.get(), shape, 1, cutoff, width);
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), size, 1e-6));

        // Test on-the-fly, in-place.
        cpu::fft::highpass(input_result.get(), input_result.get(), shape, batches, cutoff, width);
        for (size_t batch = 0; batch < batches; ++batch)
            for (size_t idx = 0; idx < size; ++idx)
                input_expected[size * batch + idx] *= filter_expected[idx];
        REQUIRE(test::Matcher(test::MATCH_ABS, input_result.get(), input_expected.get(), size, 1e-6));
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

        size_t elements = noa::elementsFFT(shape);
        cpu::memory::PtrHost<float> filter_expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(filter_expected.get());

        size_t batches = test::Randomizer<size_t>(1, 3).get();
        cpu::memory::PtrHost<float> filter_result(elements * batches);
        cpu::memory::PtrHost<float> input_result(elements * batches);
        cpu::memory::PtrHost<float> input_expected(elements * batches);

        test::randomize(input_expected.get(), elements * batches, randomizer);
        std::memcpy(input_result.get(), input_expected.get(), elements * batches * sizeof(float));

        // Test saving the mask.
        cpu::fft::bandpass<float>(nullptr, filter_result.get(), shape, 1, cutoff[0], cutoff[1], width[0], width[1]);
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));

        // Test on-the-fly, in-place.
        cpu::fft::bandpass(input_result.get(), input_result.get(), shape, batches,
                           cutoff[0], cutoff[1], width[0], width[1]);
        for (size_t batch = 0; batch < batches; ++batch)
            for (size_t idx = 0; idx < elements; ++idx)
                input_expected[elements * batch + idx] *= filter_expected[idx];
        REQUIRE(test::Matcher(test::MATCH_ABS, input_result.get(), input_expected.get(), elements, 1e-6));
    }
}
