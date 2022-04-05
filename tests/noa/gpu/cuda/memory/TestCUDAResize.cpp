#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Resize.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cuda::memory::resize()", "[assets][noa][cuda][memory]") {
    const path_t path_base = test::NOA_DATA_PATH / "memory";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["resize"];
    io::ImageFile file;

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
        cpu::memory::PtrHost<float> h_input(input_shape.elements());
        cpu::memory::PtrHost<float> h_output(output_shape.elements());
        for (size_t i = 0; i < h_input.size(); ++i)
            h_input[i] = float(i); // the inputs are a range from 0 to N
        if (is_centered) { // with central pixel (N//2) set to 0
            size4_t center{input_shape / 2};
            for (uint batch = 0; batch < input_shape[0]; ++batch)
                h_input[indexing::at(batch, center[1], center[2], center[3], input_shape.stride())] = 0;
        }
        if (border_mode == BORDER_NOTHING)
            cpu::memory::set(h_output.begin(), h_output.end(), 2.f); // OOB elements are set to 2

        cpu::memory::PtrHost<float> expected(h_output.elements());
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cuda::Stream stream;
        {
            cuda::memory::PtrDevice<float> d_input(h_input.elements());
            cuda::memory::PtrDevice<float> d_output(h_output.elements());
            cuda::memory::copy<float>(h_input.share(), d_input.share(), d_input.elements(), stream);
            cuda::memory::copy<float>(h_output.share(), d_output.share(), d_output.elements(), stream);

            if (is_centered) {
                cuda::memory::resize<float>(d_input.share(), input_shape.stride(), input_shape,
                                            d_output.share(), output_shape.stride(), output_shape,
                                            border_mode, border_value, stream);
            } else {
                cuda::memory::resize<float>(d_input.share(), input_shape.stride(), input_shape, left, right,
                                            d_output.share(), output_shape.stride(),
                                            border_mode, border_value, stream);
            }
            cuda::memory::copy<float>(d_output.share(), h_output.share(), h_output.elements(), stream);
            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), h_output.get(), h_output.size(), 1e-6));
        }

        cpu::memory::set(h_output.begin(), h_output.end(), 2.f);
        {
            cuda::memory::PtrDevicePadded<float> d_input(input_shape);
            cuda::memory::PtrDevicePadded<float> d_output(output_shape);
            cuda::memory::copy<float>(h_input.share(), input_shape.stride(),
                                      d_input.share(), d_input.stride(),
                                      d_input.shape(), stream);
            cuda::memory::copy<float>(h_output.share(), output_shape.stride(),
                                      d_output.share(), d_output.stride(),
                                      d_output.shape(), stream);

            if (is_centered) {
                cuda::memory::resize<float>(d_input.share(), d_input.stride(), input_shape,
                                            d_output.share(), d_output.stride(), output_shape,
                                            border_mode, border_value, stream);
            } else {
                cuda::memory::resize<float>(d_input.share(), d_input.stride(), input_shape, left, right,
                                            d_output.share(), d_output.stride(), border_mode, border_value, stream);
            }
            cuda::memory::copy<float>(d_output.share(), d_output.stride(),
                                      h_output.share(), output_shape.stride(), d_output.shape(), stream);
            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), h_output.get(), h_output.size(), 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::memory::resize() - edge cases", "[noa][cuda][memory]",
                   int32_t, uint64_t, float, double, chalf_t) {
    uint ndim = GENERATE(2U, 3U);
    cuda::Stream stream;

    AND_THEN("copy") {
        const size4_t shape = test::getRandomShapeBatched(ndim);
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<TestType> input(elements);
        test::Randomizer<TestType> randomizer(0, 50);
        test::randomize(input.get(), elements, randomizer);

        cuda::memory::PtrDevice<TestType> d_input(elements);
        cuda::memory::PtrDevice<TestType> d_output(elements);
        cpu::memory::PtrHost<TestType> output(elements);

        cuda::memory::copy<TestType>(input.share(), d_input.share(), elements, stream);
        cuda::memory::resize<TestType>(d_input.share(), shape.stride(), shape,
                                       d_output.share(), shape.stride(), shape,
                                       BORDER_VALUE, TestType{0}, stream);
        cuda::memory::copy<TestType>(d_output.share(), output.share(), elements, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, input.get(), output.get(), output.size(), 1e-6));
    }
}
