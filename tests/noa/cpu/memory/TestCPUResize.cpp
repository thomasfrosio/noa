#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Resize.h>
#include <noa/cpu/memory/Set.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cpu::memory::resize() - centered", "[assets][noa][cpu][memory]") {
    constexpr bool COMPUTE_ASSETS = false;
    const path_t path_base = test::NOA_DATA_PATH / "memory";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["resize"];
    io::ImageFile file;
    cpu::Stream stream(cpu::Stream::DEFAULT);

    size4_t output_shape;
    int4_t left, right;
    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto expected_filename = path_base / test["expected"].as<path_t>();
        const auto is_centered = test["is_centered"].as<bool>();
        const auto input_shape = test["shape"].as<size4_t>();
        const auto border_mode = test["border"].as<BorderMode>();
        const auto border_value = test["border_value"].as<float>();

        if (is_centered) {
            output_shape = test["o_shape"].as<size4_t>();
        } else {
            left = test["left"].as<int4_t>();
            right = test["right"].as<int4_t>();
            output_shape = size4_t(int4_t(input_shape) + left + right);
        }

        // Initialize input and output:
        cpu::memory::PtrHost<float> input(input_shape.elements());
        cpu::memory::PtrHost<float> output(output_shape.elements());
        for (size_t i = 0; i < input.elements(); ++i)
            input[i] = float(i); // the inputs are a range from 0 to N
        if (is_centered) { // with central pixel (N//2) set to 0
            const size4_t center{input_shape / 2};
            for (uint batch = 0; batch < input_shape[0]; ++batch)
                input[indexing::at(batch, center[1], center[2], center[3], input_shape.strides())] = 0;
        }
        if (border_mode == BORDER_NOTHING)
            cpu::memory::set(output.begin(), output.end(), 2.f); // OOB (if any) elements are set to 2

        // Test:
        if (is_centered)
            cpu::memory::resize<float>(input.share(), input_shape.strides(), input_shape,
                                       output.share(), output_shape.strides(), output_shape,
                                       border_mode, border_value, stream);
        else
            cpu::memory::resize<float>(input.share(), input_shape.strides(), input_shape, left, right,
                                       output.share(), output_shape.strides(),
                                       border_mode, border_value, stream);

        if (COMPUTE_ASSETS) {
            file.open(expected_filename, io::WRITE);
            file.shape(output_shape);
            file.writeAll(output.get());
            file.close();
        } else {
            cpu::memory::PtrHost<float> expected(output.size());
            file.open(expected_filename, io::READ);
            file.readAll(expected.get());
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), output.size(), 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("cpu::memory::resize() - edge cases", "[noa][cpu]",
                   int32_t, uint32_t, int64_t, uint64_t, float, double) {
    const uint ndim = GENERATE(2U, 3U);
    cpu::Stream stream(cpu::Stream::DEFAULT);

    AND_THEN("copy") {
        const size4_t shape = test::getRandomShapeBatched(ndim);
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<TestType> input(elements);
        cpu::memory::PtrHost<TestType> output(elements);
        test::Randomizer<TestType> randomizer(0, 50);
        test::randomize(input.get(), elements, randomizer);
        cpu::memory::resize<TestType>(input.share(), shape.strides(), shape,
                                      output.share(), shape.strides(), shape,
                                      BORDER_VALUE, TestType{0}, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, input.get(), output.get(), output.size(), 1e-8));
    }
}
