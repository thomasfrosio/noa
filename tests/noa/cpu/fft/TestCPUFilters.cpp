#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/fft/Filters.h>
#include <noa/cpu/fft/Remap.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cpu::fft::lowpass()", "[assets][noa][cpu][fft]") {
    test::Randomizer<float> randomizer(-5, 5);

    path_t path_base = test::PATH_NOA_DATA / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["lowpass"];
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
        cpu::Stream stream;
        size3_t pitch = shapeFFT(shape);
        cpu::fft::lowpass<fft::H2H, float>(nullptr, pitch, filter_result.get(), pitch, shape, 1, cutoff, width, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));

        // Test on-the-fly, in-place.
        cpu::fft::lowpass<fft::H2H>(
                input_result.get(), pitch, input_result.get(), pitch, shape, batches, cutoff, width, stream);
        for (size_t batch = 0; batch < batches; ++batch)
            for (size_t idx = 0; idx < elements; ++idx)
                input_expected[elements * batch + idx] *= filter_expected[idx];
        REQUIRE(test::Matcher(test::MATCH_ABS, input_result.get(), input_expected.get(), elements, 1e-6));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::lowpass(), remap", "[assets][noa][cpu][fft]", half_t, float) {
    size3_t shape = test::getRandomShape(3);
    float cutoff = 0.4f;
    float width = 0.1f;

    size_t elements = noa::elementsFFT(shape);
    size3_t pitch = shapeFFT(shape);

    cpu::memory::PtrHost<float> filter_expected(elements);
    cpu::memory::PtrHost<float> filter_result(elements);
    cpu::memory::PtrHost<float> filter_remapped(elements);
    test::memset(filter_expected.get(), elements, 1);
    test::memset(filter_result.get(), elements, 1);

    // H2HC
    cpu::Stream stream;
    cpu::fft::lowpass<fft::H2H, float>(nullptr, pitch, filter_expected.get(), pitch, shape, 1, cutoff, width, stream);
    cpu::fft::lowpass<fft::H2HC, float>(nullptr, pitch, filter_result.get(), pitch, shape, 1, cutoff, width, stream);
    cpu::fft::remap(fft::HC2H, filter_result.get(), pitch, filter_remapped.get(), pitch, shape, 1, stream);
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2HC
    cpu::fft::lowpass<fft::H2H, float>(nullptr, pitch, filter_expected.get(), pitch, shape, 1, cutoff, width, stream);
    cpu::fft::lowpass<fft::HC2HC, float>(nullptr, pitch, filter_result.get(), pitch, shape, 1, cutoff, width, stream);
    cpu::fft::remap(fft::HC2H, filter_result.get(), pitch, filter_remapped.get(), pitch, shape, 1, stream);
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2H
    cpu::fft::lowpass<fft::H2H, float>(nullptr, pitch, filter_expected.get(), pitch, shape, 1, cutoff, width, stream);
    cpu::fft::lowpass<fft::HC2H, float>(nullptr, pitch, filter_result.get(), pitch, shape, 1, cutoff, width, stream);
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));
}

TEST_CASE("cpu::fft::highpass()", "[noa][cpu][fft]") {
    test::Randomizer<float> randomizer(-5, 5);

    path_t path_base = test::PATH_NOA_DATA / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["highpass"];
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
        cpu::Stream stream;
        size3_t pitch = shapeFFT(shape);
        cpu::fft::highpass<fft::H2H, float>(nullptr, pitch, filter_result.get(), pitch, shape, 1, cutoff, width,
                                            stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), size, 1e-6));

        // Test on-the-fly, in-place.
        cpu::fft::highpass<fft::H2H>(
                input_result.get(), pitch, input_result.get(), pitch, shape, batches, cutoff, width, stream);
        for (size_t batch = 0; batch < batches; ++batch)
            for (size_t idx = 0; idx < size; ++idx)
                input_expected[size * batch + idx] *= filter_expected[idx];
        REQUIRE(test::Matcher(test::MATCH_ABS, input_result.get(), input_expected.get(), size, 1e-6));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::highpass(), remap", "[assets][noa][cpu][fft]", half_t, float) {
    size3_t shape = test::getRandomShape(3);
    float cutoff = 0.4f;
    float width = 0.1f;

    size_t elements = noa::elementsFFT(shape);
    size3_t pitch = shapeFFT(shape);

    cpu::memory::PtrHost<float> filter_expected(elements);
    cpu::memory::PtrHost<float> filter_result(elements);
    cpu::memory::PtrHost<float> filter_remapped(elements);
    test::memset(filter_expected.get(), elements, 1);
    test::memset(filter_result.get(), elements, 1);

    // H2HC
    cpu::Stream stream;
    cpu::fft::highpass<fft::H2H, float>(nullptr, pitch, filter_expected.get(), pitch, shape, 1, cutoff, width, stream);
    cpu::fft::highpass<fft::H2HC, float>(nullptr, pitch, filter_result.get(), pitch, shape, 1, cutoff, width, stream);
    cpu::fft::remap(fft::HC2H, filter_result.get(), pitch, filter_remapped.get(), pitch, shape, 1, stream);
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2HC
    cpu::fft::highpass<fft::H2H, float>(nullptr, pitch, filter_expected.get(), pitch, shape, 1, cutoff, width, stream);
    cpu::fft::highpass<fft::HC2HC, float>(nullptr, pitch, filter_result.get(), pitch, shape, 1, cutoff, width, stream);
    cpu::fft::remap(fft::HC2H, filter_result.get(), pitch, filter_remapped.get(), pitch, shape, 1, stream);
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2H
    cpu::fft::highpass<fft::H2H, float>(nullptr, pitch, filter_expected.get(), pitch, shape, 1, cutoff, width, stream);
    cpu::fft::highpass<fft::HC2H, float>(nullptr, pitch, filter_result.get(), pitch, shape, 1, cutoff, width, stream);
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));
}

TEST_CASE("cpu::fft::bandpass()", "[noa][cpu][fft]") {
    test::Randomizer<float> randomizer(-5, 5);

    path_t path_base = test::PATH_NOA_DATA / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["bandpass"];
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
        cpu::Stream stream;
        size3_t pitch = shapeFFT(shape);
        cpu::fft::bandpass<fft::H2H, float>(nullptr, pitch, filter_result.get(), pitch, shape, 1,
                                            cutoff[0], cutoff[1], width[0], width[1], stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));

        // Test on-the-fly, in-place.
        cpu::fft::bandpass<fft::H2H>(input_result.get(), pitch, input_result.get(), pitch, shape, batches,
                                     cutoff[0], cutoff[1], width[0], width[1], stream);
        for (size_t batch = 0; batch < batches; ++batch)
            for (size_t idx = 0; idx < elements; ++idx)
                input_expected[elements * batch + idx] *= filter_expected[idx];
        REQUIRE(test::Matcher(test::MATCH_ABS, input_result.get(), input_expected.get(), elements, 1e-6));
    }
}

TEMPLATE_TEST_CASE("cpu::fft::bandpass(), remap", "[assets][noa][cpu][fft]", half_t, float) {
    size3_t shape = test::getRandomShape(3);
    float cutoff1 = 0.3f, cutoff2 = 0.4f;
    float width = 0.1f;

    size_t elements = noa::elementsFFT(shape);
    size3_t pitch = shapeFFT(shape);

    cpu::memory::PtrHost<float> filter_expected(elements);
    cpu::memory::PtrHost<float> filter_result(elements);
    cpu::memory::PtrHost<float> filter_remapped(elements);
    test::memset(filter_expected.get(), elements, 1);
    test::memset(filter_result.get(), elements, 1);

    // H2HC
    cpu::Stream stream;
    cpu::fft::bandpass<fft::H2H, float>(nullptr, pitch, filter_expected.get(), pitch, shape, 1,
                                        cutoff1, cutoff2, width, width, stream);
    cpu::fft::bandpass<fft::H2HC, float>(nullptr, pitch, filter_result.get(), pitch, shape, 1,
                                         cutoff1, cutoff2, width, width, stream);
    cpu::fft::remap(fft::HC2H, filter_result.get(), pitch, filter_remapped.get(), pitch, shape, 1, stream);
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2HC
    cpu::fft::bandpass<fft::H2H, float>(nullptr, pitch, filter_expected.get(), pitch, shape, 1,
                                        cutoff1, cutoff2, width, width, stream);
    cpu::fft::bandpass<fft::HC2HC, float>(nullptr, pitch, filter_result.get(), pitch, shape, 1,
                                          cutoff1, cutoff2, width, width, stream);
    cpu::fft::remap(fft::HC2H, filter_result.get(), pitch, filter_remapped.get(), pitch, shape, 1, stream);
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2H
    cpu::fft::bandpass<fft::H2H, float>(nullptr, pitch, filter_expected.get(), pitch, shape, 1,
                                        cutoff1, cutoff2, width, width, stream);
    cpu::fft::bandpass<fft::HC2H, float>(nullptr, pitch, filter_result.get(), pitch, shape, 1,
                                         cutoff1, cutoff2, width, width, stream);
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));
}
