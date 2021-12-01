#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Resize.h>
#include <noa/cpu/memory/Set.h>

#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Resize.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include <noa/common/io/ImageFile.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cuda::memory::resize()", "[assets][noa][cuda][memory]") {
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
            output_shape = size3_t(int3_t(input_shape) + left + right);
        }

        // Initialize input and output:
        size_t i_elements = noa::elements(input_shape);
        size_t o_elements = noa::elements(output_shape);
        cpu::memory::PtrHost<float> expected(o_elements * batches);
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cpu::memory::PtrHost<float> h_input(i_elements * batches);
        cpu::memory::PtrHost<float> h_output(o_elements * batches);
        for (size_t i = 0; i < h_input.size(); ++i)
            h_input[i] = float(i); // the inputs are a range from 0 to N
        if (is_centered) { // with central pixel (N//2) set to 0
            size3_t center(input_shape / 2);
            for (uint batch = 0; batch < batches; ++batch)
                h_input[batch * noa::elements(input_shape) + index(center, input_shape)] = 0;
        }
        if (border_mode == BORDER_NOTHING)
            cpu::memory::set(h_output.begin(), h_output.end(), 2.f);  // OOB elements are set to 2

        cuda::Stream stream;
        AND_THEN("Contiguous") {
            cuda::memory::PtrDevice<float> d_input(i_elements * batches);
            cuda::memory::PtrDevice<float> d_output(o_elements * batches);
            cuda::memory::copy(h_input.get(), d_input.get(), d_input.elements(), stream);
            cuda::memory::copy(h_output.get(), d_output.get(), d_output.elements(), stream);

            if (is_centered)
                cuda::memory::resize(d_input.get(), input_shape.x, input_shape,
                                     d_output.get(), output_shape.x, output_shape,
                                     border_mode, border_value, batches, stream);
            else
                cuda::memory::resize(d_input.get(), input_shape, left, right, d_output.get(),
                                     border_mode, border_value, batches, stream);
            cuda::memory::copy(d_output.get(), h_output.get(), h_output.elements(), stream);
            cuda::Stream::synchronize(stream);
            float diff = test::getAverageNormalizedDifference(expected.get(), h_output.get(), h_output.elements());
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
        }

        cpu::memory::set(h_output.begin(), h_output.end(), 2.f);
        AND_THEN("Padded") {
            cuda::memory::PtrDevicePadded<float> d_input({input_shape.x, input_shape.y * input_shape.z, batches});
            cuda::memory::PtrDevicePadded<float> d_output({output_shape.x, output_shape.y * output_shape.z, batches});
            cuda::memory::copy(h_input.get(), input_shape.x,
                               d_input.get(), d_input.pitch(),
                               d_input.shape(), stream);
            cuda::memory::copy(h_output.get(), output_shape.x,
                               d_output.get(), d_output.pitch(),
                               d_output.shape(), stream);

            if (is_centered)
                cuda::memory::resize(d_input.get(), d_input.pitch(), input_shape,
                                     d_output.get(), d_output.pitch(), output_shape,
                                     border_mode, border_value, batches, stream);
            else
                cuda::memory::resize(d_input.get(), d_input.pitch(), input_shape, left, right,
                                     d_output.get(), d_output.pitch(), border_mode, border_value, batches, stream);
            cuda::memory::copy(d_output.get(), d_output.pitch(),
                               h_output.get(), output_shape.x, d_output.shape(), stream);
            cuda::Stream::synchronize(stream);
            float diff = test::getAverageNormalizedDifference(expected.get(), h_output.get(), h_output.elements());
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::memory::resize() - edge cases", "[noa][cuda][memory]",
                   int, uint, long long, unsigned long long, float, double) {
    uint ndim = GENERATE(2U, 3U);
    size_t batches = test::Randomizer<size_t>(1, 3).get();
    cuda::Stream stream;

    AND_THEN("copy") {
        size3_t shape = test::getRandomShape(ndim);
        size_t elements = noa::elements(shape) * batches;
        cpu::memory::PtrHost<TestType> input(elements);
        test::Randomizer<TestType> randomizer(0, 50);
        test::randomize(input.get(), elements, randomizer);

        cuda::memory::PtrDevice<TestType> d_input(elements);
        cuda::memory::PtrDevice<TestType> d_output(elements);
        cpu::memory::PtrHost<TestType> output(elements);

        cuda::memory::copy(input.get(), d_input.get(), elements);
        cuda::memory::resize(d_input.get(), shape.x, shape,
                             d_output.get(), shape.x, shape,
                             BORDER_VALUE, TestType{0}, batches, stream);
        cuda::memory::copy(d_output.get(), output.get(), elements);
        TestType diff = test::getDifference(input.get(), output.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0, 1e-6));
    }
}
