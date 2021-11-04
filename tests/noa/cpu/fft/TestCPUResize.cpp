#include <noa/common/io/ImageFile.h>
#include <noa/cpu/fft/Resize.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// These tests use that fact cropping after padding cancels the padding and returns the input array.
// These are not very good tests but it is better than nothing.
TEMPLATE_TEST_CASE("cpu::fft::pad(), crop()", "[noa][cpu][fft]", float, cfloat_t, double, cdouble_t) {
    test::RealRandomizer<TestType> randomizer(1., 5.);
    test::IntRandomizer<size_t> randomizer_int(0, 32);
    uint ndim = GENERATE(1U, 2U, 3U);
    size_t batches = test::IntRandomizer<size_t>(1, 3).get();
    size3_t shape = test::getRandomShape(ndim);
    size3_t shape_padded(shape);
    if (ndim > 2) shape_padded.z += randomizer_int.get();
    if (ndim > 1) shape_padded.y += randomizer_int.get();
    shape_padded.x += randomizer_int.get();

    INFO(shape);
    INFO(shape_padded);

    AND_THEN("pad then crop") {
        size_t elements_fft = elementsFFT(shape);
        size_t elements_fft_padded = elementsFFT(shape_padded);
        cpu::memory::PtrHost<TestType> original(elements_fft * batches);
        cpu::memory::PtrHost<TestType> padded(elements_fft_padded * batches);
        cpu::memory::PtrHost<TestType> cropped(elements_fft * batches);

        test::initDataRandom(original.get(), original.elements(), randomizer);
        cpu::fft::pad(original.get(), shape, padded.get(), shape_padded, batches);
        cpu::fft::crop(padded.get(), shape_padded, cropped.get(), shape, batches);

        TestType diff = test::getDifference(original.get(), cropped.get(), original.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }

    AND_THEN("padFull then cropFull") {
        size_t elements = noa::elements(shape);
        size_t elements_padded = noa::elements(shape_padded);
        cpu::memory::PtrHost<TestType> original(elements * batches);
        cpu::memory::PtrHost<TestType> padded(elements_padded * batches);
        cpu::memory::PtrHost<TestType> cropped(elements * batches);

        test::initDataRandom(original.get(), original.elements(), randomizer);
        cpu::fft::padFull(original.get(), shape, padded.get(), shape_padded, batches);
        cpu::fft::cropFull(padded.get(), shape_padded, cropped.get(), shape, batches);

        TestType diff = test::getDifference(original.get(), cropped.get(), original.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 1e-13));
    }
}

TEST_CASE("cpu::fft::pad(), crop(), assets", "[assets][noa][cpu][fft]") {
    fs::path path = test::PATH_TEST_DATA / "fft";
    YAML::Node tests = YAML::LoadFile(path / "param.yaml")["resize"];
    io::ImageFile file;

    constexpr bool GENERATE_ASSETS = false;
    if constexpr (GENERATE_ASSETS) {
        test::RealRandomizer<float> randomizer(-128., 128.);

        for (const YAML::Node& node : tests["input"]) {
            auto shape = node["shape"].as<size3_t>();
            auto path_input = path / node["path"].as<path_t>();
            cpu::memory::PtrHost<float> input(elementsFFT(shape));
            test::initDataRandom(input.get(), input.elements(), randomizer);
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
        if (shape_input.x < shape_expected.x)
            cpu::fft::pad(input.get(), shape_input, output.get(), shape_expected, 1);
        else
            cpu::fft::crop(input.get(), shape_input, output.get(), shape_expected, 1);

        if constexpr (GENERATE_ASSETS) {
            file.open(filename_expected, io::WRITE);
            file.shape(shapeFFT(shape_expected));
            file.writeAll(output.get(), false);
        } else {
            file.open(filename_expected, io::READ);
            cpu::memory::PtrHost<float> expected(elements(file.shape()));
            file.readAll(expected.get(), false);
            float diff = test::getDifference(expected.get(), output.get(), expected.elements());
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-14));
        }
    }
}
