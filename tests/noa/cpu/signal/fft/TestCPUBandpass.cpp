#include <noa/common/io/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/fft/Remap.h>
#include <noa/cpu/signal/fft/Bandpass.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cpu::signal::fft::lowpass()", "[assets][noa][cpu][fft]") {
    test::Randomizer<float> randomizer(-5, 5);

    const path_t path_base = test::NOA_DATA_PATH / "signal" / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["lowpass"];
    io::MRCFile file;
    cpu::Stream stream;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto shape = test["shape"].as<size4_t>();
        const auto cutoff = test["cutoff"].as<float>();
        const auto width = test["width"].as<float>();
        const auto filename_expected = path_base / test["path"].as<path_t>();
        const size_t elements = shape.fft().elements();
        const size4_t stride = shape.fft().strides();
        const size_t elements_per_batch = size3_t{shape.get() + 1}.fft().elements();

        // Get expected filter. Asset is not batched so copy to all batches.
        cpu::memory::PtrHost<float> filter_expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(filter_expected.get());
        for (size_t batch = 1; batch < shape[0]; ++batch)
            test::copy(filter_expected.get(), filter_expected.get() + batch * stride[0], elements_per_batch);

        // Test saving the mask.
        cpu::memory::PtrHost<float> filter_result(elements);
        cpu::signal::fft::lowpass<fft::H2H, float>(nullptr, {}, filter_result.share(), stride, shape, cutoff, width, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));

        // Test on-the-fly, in-place.
        cpu::memory::PtrHost<float> i_expected(elements);
        cpu::memory::PtrHost<float> i_result(elements);
        test::randomize(i_expected.get(), elements, randomizer);
        std::copy(i_expected.begin(), i_expected.end(), i_result.get());
        cpu::signal::fft::lowpass<fft::H2H, float>(
                i_result.share(), stride, i_result.share(), stride, shape, cutoff, width, stream);
        for (size_t idx = 0; idx < elements; ++idx)
            i_expected[idx] *= filter_expected[idx];

        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, i_result.get(), i_expected.get(), elements, 1e-6));
    }
}

TEMPLATE_TEST_CASE("cpu::signal::fft::lowpass(), remap", "[noa][cpu][fft]", half_t, float) {
    const size4_t shape = test::getRandomShapeBatched(3);
    const float cutoff = 0.4f;
    const float width = 0.1f;

    const size_t elements = shape.fft().elements();
    const size4_t stride = shape.fft().strides();

    cpu::memory::PtrHost<float> filter_expected(elements);
    cpu::memory::PtrHost<float> filter_result(elements);
    cpu::memory::PtrHost<float> filter_remapped(elements);
    test::memset(filter_expected.get(), elements, 1);
    test::memset(filter_result.get(), elements, 1);

    // H2HC
    cpu::Stream stream;
    cpu::signal::fft::lowpass<fft::H2H, float>(nullptr, {}, filter_expected.share(), stride, shape, cutoff, width, stream);
    cpu::signal::fft::lowpass<fft::H2HC, float>(nullptr, {}, filter_result.share(), stride, shape, cutoff, width, stream);
    cpu::fft::remap<float>(fft::HC2H, filter_result.share(), stride, filter_remapped.share(), stride, shape, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2HC
    cpu::signal::fft::lowpass<fft::H2H, float>(nullptr, {}, filter_expected.share(), stride, shape, cutoff, width, stream);
    cpu::signal::fft::lowpass<fft::HC2HC, float>(nullptr, {}, filter_result.share(), stride, shape, cutoff, width, stream);
    cpu::fft::remap<float>(fft::HC2H, filter_result.share(), stride, filter_remapped.share(), stride, shape, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2H
    cpu::signal::fft::lowpass<fft::H2H, float>(nullptr, {}, filter_expected.share(), stride, shape, cutoff, width, stream);
    cpu::signal::fft::lowpass<fft::HC2H, float>(nullptr, {}, filter_result.share(), stride, shape, cutoff, width, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));
}

TEST_CASE("cpu::signal::fft::highpass()", "[assets][noa][cpu][fft]") {
    test::Randomizer<float> randomizer(-5, 5);

    const path_t path_base = test::NOA_DATA_PATH / "signal" / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["highpass"];
    io::MRCFile file;
    cpu::Stream stream;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto shape = test["shape"].as<size4_t>();
        const auto cutoff = test["cutoff"].as<float>();
        const auto width = test["width"].as<float>();
        const auto filename_expected = path_base / test["path"].as<path_t>();
        const size_t elements = shape.fft().elements();
        const size4_t stride = shape.fft().strides();
        const size_t elements_per_batch = size3_t{shape.get() + 1}.fft().elements();

        // Get expected filter. Asset is not batched so copy to all batches.
        cpu::memory::PtrHost<float> filter_expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(filter_expected.get());
        for (size_t batch = 1; batch < shape[0]; ++batch)
            test::copy(filter_expected.get(), filter_expected.get() + batch * stride[0], elements_per_batch);

        // Test saving the mask.
        cpu::memory::PtrHost<float> filter_result(elements);
        cpu::signal::fft::highpass<fft::H2H, float>(nullptr, {}, filter_result.share(), stride, shape, cutoff, width, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));

        // Test on-the-fly, in-place.
        cpu::memory::PtrHost<float> i_expected(elements);
        cpu::memory::PtrHost<float> i_result(elements);
        test::randomize(i_expected.get(), elements, randomizer);
        std::copy(i_expected.begin(), i_expected.end(), i_result.get());
        cpu::signal::fft::highpass<fft::H2H, float>(
                i_result.share(), stride, i_result.share(), stride, shape, cutoff, width, stream);
        for (size_t idx = 0; idx < elements; ++idx)
            i_expected[idx] *= filter_expected[idx];

        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, i_result.get(), i_expected.get(), elements, 1e-6));
    }
}

TEMPLATE_TEST_CASE("cpu::signal::fft::highpass(), remap", "[noa][cpu][fft]", half_t, float) {
    const size4_t shape = test::getRandomShapeBatched(3);
    const float cutoff = 0.4f;
    const float width = 0.1f;

    const size_t elements = shape.fft().elements();
    const size4_t stride = shape.fft().strides();

    cpu::memory::PtrHost<float> filter_expected(elements);
    cpu::memory::PtrHost<float> filter_result(elements);
    cpu::memory::PtrHost<float> filter_remapped(elements);
    test::memset(filter_expected.get(), elements, 1);
    test::memset(filter_result.get(), elements, 1);

    // H2HC
    cpu::Stream stream;
    cpu::signal::fft::highpass<fft::H2H, float>(nullptr, {}, filter_expected.share(), stride, shape, cutoff, width, stream);
    cpu::signal::fft::highpass<fft::H2HC, float>(nullptr, {}, filter_result.share(), stride, shape, cutoff, width, stream);
    cpu::fft::remap<float>(fft::HC2H, filter_result.share(), stride, filter_remapped.share(), stride, shape, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2HC
    cpu::signal::fft::highpass<fft::H2H, float>(nullptr, {}, filter_expected.share(), stride, shape, cutoff, width, stream);
    cpu::signal::fft::highpass<fft::HC2HC, float>(nullptr, {}, filter_result.share(), stride, shape, cutoff, width, stream);
    cpu::fft::remap<float>(fft::HC2H, filter_result.share(), stride, filter_remapped.share(), stride, shape, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2H
    cpu::signal::fft::highpass<fft::H2H, float>(nullptr, {}, filter_expected.share(), stride, shape, cutoff, width, stream);
    cpu::signal::fft::highpass<fft::HC2H, float>(nullptr, {}, filter_result.share(), stride, shape, cutoff, width, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));
}

TEST_CASE("cpu::signal::fft::bandpass()", "[assets][noa][cpu][fft]") {
    test::Randomizer<float> randomizer(-5, 5);

    const path_t path_base = test::NOA_DATA_PATH / "signal" / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["bandpass"];
    io::MRCFile file;
    cpu::Stream stream;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto shape = test["shape"].as<size4_t>();
        const auto cutoff = test["cutoff"].as<std::vector<float>>();
        const auto width = test["width"].as<std::vector<float>>();
        const auto filename_expected = path_base / test["path"].as<path_t>();
        const size_t elements = shape.fft().elements();
        const size4_t stride = shape.fft().strides();
        const size_t elements_per_batch = size3_t{shape.get() + 1}.fft().elements();

        // Get expected filter. Asset is not batched so copy to all batches.
        cpu::memory::PtrHost<float> filter_expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(filter_expected.get());
        for (size_t batch = 1; batch < shape[0]; ++batch)
            test::copy(filter_expected.get(), filter_expected.get() + batch * stride[0], elements_per_batch);

        // Test saving the mask.
        cpu::memory::PtrHost<float> filter_result(elements);
        cpu::signal::fft::bandpass<fft::H2H, float>(nullptr, {}, filter_result.share(), stride, shape,
                                                    cutoff[0], cutoff[1], width[0], width[1], stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));

        // Test on-the-fly, in-place.
        cpu::memory::PtrHost<float> i_expected(elements);
        cpu::memory::PtrHost<float> i_result(elements);
        test::randomize(i_expected.get(), elements, randomizer);
        std::copy(i_expected.begin(), i_expected.end(), i_result.get());
        cpu::signal::fft::bandpass<fft::H2H, float>(i_result.share(), stride, i_result.share(), stride, shape,
                                                    cutoff[0], cutoff[1], width[0], width[1], stream);
        for (size_t idx = 0; idx < elements; ++idx)
            i_expected[idx] *= filter_expected[idx];

        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, i_result.get(), i_expected.get(), elements, 1e-6));
    }
}

TEMPLATE_TEST_CASE("cpu::signal::fft::bandpass(), remap", "[noa][cpu][fft]", half_t, float) {
    const size4_t shape = test::getRandomShapeBatched(3);
    const float cutoff1 = 0.3f, cutoff2 = 0.4f;
    const float width = 0.1f;

    const size_t elements = shape.fft().elements();
    const size4_t stride = shape.fft().strides();

    cpu::memory::PtrHost<float> filter_expected(elements);
    cpu::memory::PtrHost<float> filter_result(elements);
    cpu::memory::PtrHost<float> filter_remapped(elements);
    test::memset(filter_expected.get(), elements, 1);
    test::memset(filter_result.get(), elements, 1);

    // H2HC
    cpu::Stream stream;
    cpu::signal::fft::bandpass<fft::H2H, float>(
            nullptr, {}, filter_expected.share(), stride, shape, cutoff1, cutoff2, width, width, stream);
    cpu::signal::fft::bandpass<fft::H2HC, float>(
            nullptr, {}, filter_result.share(), stride, shape, cutoff1, cutoff2, width, width, stream);
    cpu::fft::remap<float>(fft::HC2H, filter_result.share(), stride, filter_remapped.share(), stride, shape, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2HC
    cpu::signal::fft::bandpass<fft::H2H, float>(
            nullptr, {}, filter_expected.share(), stride, shape, cutoff1, cutoff2, width, width, stream);
    cpu::signal::fft::bandpass<fft::HC2HC, float>(
            nullptr, {}, filter_result.share(), stride, shape, cutoff1, cutoff2, width, width, stream);
    cpu::fft::remap<float>(fft::HC2H, filter_result.share(), stride, filter_remapped.share(), stride, shape, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_remapped.get(), elements, 1e-6));

    // HC2H
    cpu::signal::fft::bandpass<fft::H2H, float>(
            nullptr, {}, filter_expected.share(), stride, shape, cutoff1, cutoff2, width, width, stream);
    cpu::signal::fft::bandpass<fft::HC2H, float>(
            nullptr, {}, filter_result.share(), stride, shape, cutoff1, cutoff2, width, width, stream);
    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected.get(), filter_result.get(), elements, 1e-6));
}
