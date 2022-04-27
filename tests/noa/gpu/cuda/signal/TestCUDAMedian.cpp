#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/PtrManaged.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include <noa/cpu/signal/Median.h>
#include <noa/gpu/cuda/signal/Median.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cuda::signal::median()", "[assets][noa][cuda][filter]") {
    const path_t path_base = test::NOA_DATA_PATH / "signal";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["median"]["tests"];
    io::ImageFile file;
    cuda::Stream stream;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto filename_input = path_base / test["input"].as<path_t>();
        const auto window = test["window"].as<size_t>();
        const auto dim = test["dim"].as<int>();
        const auto border = test["border"].as<BorderMode>();
        const auto filename_expected = path_base / test["expected"].as<path_t>();

        file.open(filename_input, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.stride();
        const size_t elements = shape.elements();
        cuda::memory::PtrManaged<float> input(elements, stream);
        file.readAll(input.get());

        cpu::memory::PtrHost<float> expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        cuda::memory::PtrManaged<float> result(elements, stream);
        if (dim == 1)
            cuda::signal::median1<float>(input.share(), stride, result.share(), stride, shape, border, window, stream);
        else if (dim == 2)
            cuda::signal::median2<float>(input.share(), stride, result.share(), stride, shape, border, window, stream);
        else if (dim == 3)
            cuda::signal::median3<float>(input.share(), stride, result.share(), stride, shape, border, window, stream);
        else
            FAIL("dim is not correct");

        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, result.get(), expected.get(), result.size(), 1e-5));
    }
}

TEMPLATE_TEST_CASE("cuda::signal::median(), random", "[noa][cuda][filter]", int, half_t, float, double) {
    const int ndim = GENERATE(1, 2, 3);
    const BorderMode mode = GENERATE(BORDER_ZERO, BORDER_REFLECT);
    size_t window = test::Randomizer<uint>(2, 11).get();
    if (!(window % 2))
        window -= 1;
    if (ndim == 3 && window > 5)
        window = 3;

    test::Randomizer<size_t> random_size(16, 100);
    size4_t shape = test::getRandomShapeBatched(3);
    if (ndim != 3 && random_size.get() % 2)
        shape[1] = 1; // randomly switch to 2D
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();
    INFO(string::format("ndim:{}, mode:{}, window:{}, shape:{}", ndim, mode, window, shape));

    test::Randomizer<TestType> randomizer(-128, 128);
    cpu::memory::PtrHost<TestType> data(elements);
    test::randomize(data.get(), data.elements(), randomizer);

    cuda::Stream gpu_stream;
    cpu::Stream cpu_stream;

    cuda::memory::PtrDevicePadded<TestType> d_data(shape);
    cuda::memory::PtrDevicePadded<TestType> d_result(shape);
    cpu::memory::PtrHost<TestType> cuda_result(elements);
    cpu::memory::PtrHost<TestType> h_result(elements);
    cuda::memory::copy<TestType>(data.share(), stride, d_data.share(), d_data.stride(), shape, gpu_stream);

    if (ndim == 1) {
        cuda::signal::median1<TestType>(d_data.share(), d_data.stride(),
                                        d_result.share(), d_result.stride(),
                                        shape, mode, window, gpu_stream);
        cpu::signal::median1<TestType>(data.share(), stride,
                                       h_result.share(), stride,
                                       shape, mode, window, cpu_stream);
    } else if (ndim == 2) {
        cuda::signal::median2<TestType>(d_data.share(), d_data.stride(),
                                        d_result.share(), d_result.stride(),
                                        shape, mode, window, gpu_stream);
        cpu::signal::median2<TestType>(data.share(), stride,
                                       h_result.share(), stride,
                                       shape, mode, window, cpu_stream);
    } else {
        cuda::signal::median3<TestType>(d_data.share(), d_data.stride(),
                                        d_result.share(), d_result.stride(),
                                        shape, mode, window, gpu_stream);
        cpu::signal::median3<TestType>(data.share(), stride,
                                       h_result.share(), stride,
                                       shape, mode, window, cpu_stream);
    }
    cuda::memory::copy<TestType>(d_result.share(), d_result.stride(), cuda_result.share(), stride, shape, gpu_stream);
    cpu_stream.synchronize();
    gpu_stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, h_result.get(), cuda_result.get(), h_result.size(), 1e-5));
}