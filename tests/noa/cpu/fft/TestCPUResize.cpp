#include <noa/common/io/ImageFile.h>
#include <noa/cpu/fft/Resize.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// These tests use that fact cropping after padding cancels the padding and returns the input array.
// These are not very good tests but it is better than nothing.
TEMPLATE_TEST_CASE("cpu::fft::resize()", "[noa][cpu][fft]", float, cfloat_t, double, cdouble_t) {
    test::Randomizer<TestType> randomizer(1., 5.);
    test::Randomizer<size_t> randomizer_int(0, 32);
    uint ndim = GENERATE(1U, 2U, 3U);
    size_t batches = test::Randomizer<size_t>(1, 3).get();
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_padded(shape);
    if (ndim > 2) shape_padded.z += randomizer_int.get();
    if (ndim > 1) shape_padded.y += randomizer_int.get();
    shape_padded.x += randomizer_int.get();


    INFO(shape);
    INFO(shape_padded);
    cpu::Stream stream;

    AND_THEN("pad then crop") {
        const size3_t pitch = shapeFFT(shape);
        const size3_t pitch_padded = shapeFFT(shape_padded);

        cpu::memory::PtrHost<TestType> original(elements(pitch) * batches);
        cpu::memory::PtrHost<TestType> padded(elements(pitch_padded) * batches);
        cpu::memory::PtrHost<TestType> cropped(original.size());

        test::randomize(original.get(), original.elements(), randomizer);
        cpu::fft::resize<fft::H2H>(original.get(), pitch, shape,
                                   padded.get(), pitch_padded, shape_padded, batches, stream);
        cpu::fft::resize<fft::H2H>(padded.get(), pitch_padded, shape_padded,
                                   cropped.get(), pitch, shape, batches, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, original.get(), cropped.get(), original.elements(), 1e-10));
    }

    AND_THEN("padFull then cropFull") {
        cpu::memory::PtrHost<TestType> original(elements(shape) * batches);
        cpu::memory::PtrHost<TestType> padded(elements(shape_padded) * batches);
        cpu::memory::PtrHost<TestType> cropped(original.size());

        test::randomize(original.get(), original.elements(), randomizer);
        cpu::fft::resize<fft::F2F>(original.get(), shape, shape,
                                   padded.get(), shape_padded, shape_padded, batches, stream);
        cpu::fft::resize<fft::F2F>(padded.get(), shape_padded, shape_padded,
                                   cropped.get(), shape, shape, batches, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, original.get(), cropped.get(), original.elements(), 1e-10));
    }
}

TEST_CASE("cpu::fft::resize(), assets", "[assets][noa][cpu][fft]") {
    fs::path path = test::PATH_NOA_DATA / "fft";
    YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["resize"];
    io::ImageFile file;

    constexpr bool GENERATE_ASSETS = false;
    if constexpr (GENERATE_ASSETS) {
        test::Randomizer<float> randomizer(-128., 128.);

        for (const YAML::Node& node : tests["input"]) {
            auto shape = node["shape"].as<size3_t>();
            auto path_input = path / node["path"].as<path_t>();
            cpu::memory::PtrHost<float> input(elementsFFT(shape));
            test::randomize(input.get(), input.elements(), randomizer);
            file.open(path_input, io::WRITE);
            file.shape(shapeFFT(shape));
            file.writeAll(input.get(), false);
        }
    }

    for (size_t i = 0; i < tests["tests"].size(); ++i) {
        const YAML::Node& test = tests["tests"][i];
        auto filename_input = path / test["input"].as<path_t>();
        auto filename_expected = path / test["expected"].as<path_t>();
        auto shape_input = test["shape_input"].as<size3_t>();
        auto shape_expected = test["shape_expected"].as<size3_t>();

        file.open(filename_input, io::READ);
        cpu::memory::PtrHost<float> input(elementsFFT(shape_input));
        file.readAll(input.get(), false);

        cpu::memory::PtrHost<float> output(elementsFFT(shape_expected));
        cpu::Stream stream;
        cpu::fft::resize<fft::H2H>(input.get(), shapeFFT(shape_input), shape_input,
                                   output.get(), shapeFFT(shape_expected), shape_expected, 1, stream);

        if constexpr (GENERATE_ASSETS) {
            file.open(filename_expected, io::WRITE);
            file.shape(shapeFFT(shape_expected));
            file.writeAll(output.get(), false);
        } else {
            file.open(filename_expected, io::READ);
            cpu::memory::PtrHost<float> expected(elements(file.shape()));
            file.readAll(expected.get(), false);
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), expected.elements(), 1e-10));
        }
    }
}
