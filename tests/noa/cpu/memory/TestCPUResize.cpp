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
    path_t path_base = test::PATH_NOA_DATA / "memory";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["resize"];
    io::ImageFile file;
    cpu::Stream stream;

    size3_t output_shape;
    int3_t left, right;
    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto expected_filename = path_base / test["expected"].as<path_t>();
        const auto is_centered = test["is_centered"].as<bool>();
        const auto input_shape = test["shape"].as<size3_t>();
        const auto border_mode = test["border"].as<BorderMode>();
        const auto border_value = test["border_value"].as<float>();
        const auto batches = test["batches"].as<uint>();

        if (is_centered) {
            output_shape = test["o_shape"].as<size3_t>();
        } else {
            left = test["left"].as<int3_t>();
            right = test["right"].as<int3_t>();
            output_shape = size3_t(int3_t(input_shape) + left + right);
        }

        // Initialize input and output:
        cpu::memory::PtrHost<float> input(noa::elements(input_shape) * batches);
        cpu::memory::PtrHost<float> output(noa::elements(output_shape) * batches);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = float(i); // the inputs are a range from 0 to N
        if (is_centered) { // with central pixel (N//2) set to 0
            size3_t center(input_shape / size_t{2});
            for (uint batch = 0; batch < batches; ++batch)
                input[batch * noa::elements(input_shape) + index(center, input_shape)] = 0;
        }
        if (border_mode == BORDER_NOTHING)
            cpu::memory::set(output.begin(), output.end(), 2.f);  // OOB elements are set to 2

        // Test:
        if (is_centered)
            cpu::memory::resize(input.get(), input_shape, input_shape,
                                output.get(), output_shape, output_shape, batches,
                                border_mode, border_value, stream);
        else
            cpu::memory::resize(input.get(), input_shape, input_shape, left, right,
                                output.get(), output_shape, batches,
                                border_mode, border_value, stream);

        if (COMPUTE_ASSETS) {
            file.open(expected_filename, io::WRITE);
            file.shape({output_shape.x, output_shape.y, output_shape.z * batches});
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
                   int, uint, long long, unsigned long long, float, double) {
    const uint ndim = GENERATE(2U, 3U);
    const size_t batches = test::Randomizer<size_t>(1, 3).get();
    cpu::Stream stream;

    AND_THEN("copy") {
        const size3_t shape = test::getRandomShape(ndim);
        const size_t elements = noa::elements(shape) * batches;
        cpu::memory::PtrHost<TestType> input(elements);
        cpu::memory::PtrHost<TestType> output(elements);
        test::Randomizer<TestType> randomizer(0, 50);
        test::randomize(input.get(), elements, randomizer);
        cpu::memory::resize(input.get(), shape, shape,
                            output.get(), shape, shape, batches,
                            BORDER_VALUE, TestType{0}, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, input.get(), output.get(), output.size(), 1e-8));
    }
}
