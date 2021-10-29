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
    path_t path_base = test::PATH_TEST_DATA / "memory";
    YAML::Node tests = YAML::LoadFile(path_base / "param.yaml")["resize"];
    io::ImageFile file;

    size3_t output_shape;
    int3_t left, right;
    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto expected_filename = path_base / test["expected"].as<path_t>();
        auto is_centered = test["is_centered"].as<bool>();
        auto input_shape = test["shape"].as<size3_t>();
        auto border_mode = test["border"].as<BorderMode>();
        auto border_value = test["border_value"].as<float>();
        auto batches = test["batches"].as<uint>();

        if (is_centered) {
            output_shape = test["o_shape"].as<size3_t>();
        } else {
            left = test["left"].as<int3_t>();
            right = test["right"].as<int3_t>();
            output_shape = int3_t(input_shape) + left + right;
        }

        // Initialize input and output:
        cpu::memory::PtrHost<float> input(noa::elements(input_shape) * batches);
        cpu::memory::PtrHost<float> output(noa::elements(output_shape) * batches);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = float(i); // the inputs are a range from 0 to N
        if (is_centered) { // with central pixel (N//2) set to 0
            size3_t center(input_shape / size_t{2});
            for (uint batch = 0; batch < batches; ++batch)
                input[batch * noa::elements(input_shape) + getIdx(center, input_shape)] = 0;
        }
        if (border_mode == BORDER_NOTHING)
            cpu::memory::set(output.begin(), output.end(), 2.f);  // OOB elements are set to 2

        // Test:
        if (is_centered)
            cpu::memory::resize(input.get(), input_shape, output.get(), output_shape,
                                border_mode, border_value, batches);
        else
            cpu::memory::resize(input.get(), input_shape, left, right, output.get(),
                                border_mode, border_value, batches);

        if (COMPUTE_ASSETS) {
            file.open(expected_filename, io::WRITE);
            file.shape({output_shape.x, output_shape.y, output_shape.z * batches});
            file.writeAll(output.get());
            file.close();
        } else {
            cpu::memory::PtrHost<float> expected(output.size());
            file.open(expected_filename, io::READ);
            file.readAll(expected.get());
            float diff = test::getDifference(expected.get(), output.get(), output.size());
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("cpu::memory::resize() - edge cases", "[noa][cpu]",
                   int, uint, long long, unsigned long long, float, double) {
    uint ndim = GENERATE(2U, 3U);
    uint batches = test::IntRandomizer<uint>(1, 3).get();

    AND_THEN("copy") {
        size3_t shape = test::getRandomShape(ndim);
        size_t elements = noa::elements(shape) * batches;
        cpu::memory::PtrHost<TestType> input(elements);
        cpu::memory::PtrHost<TestType> output(elements);
        test::Randomizer<TestType> randomizer(0, 50);
        test::initDataRandom(input.get(), elements, randomizer);
        cpu::memory::resize(input.get(), shape, output.get(), shape, BORDER_VALUE, TestType{0}, batches);
        TestType diff = test::getDifference(input.get(), output.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0, 1e-6));
    }
}
