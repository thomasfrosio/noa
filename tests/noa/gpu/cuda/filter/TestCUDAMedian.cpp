#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/PtrManaged.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include <noa/cpu/filter/Median.h>
#include <noa/gpu/cuda/filter/Median.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cuda::filter::median()", "[assets][noa][cuda][filter]") {
    const path_t path_base = test::PATH_NOA_DATA / "filter";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["median"]["tests"];
    io::ImageFile file;
    cuda::Stream stream(cuda::Stream::CONCURRENT);

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
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cuda::memory::PtrManaged<float> input(elements, stream);
        file.readAll(input.get());

        cpu::memory::PtrHost<float> expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        cuda::memory::PtrManaged<float> result(elements, stream);
        if (dim == 1)
            cuda::filter::median1(input.get(), stride, result.get(), stride, shape, border, window, stream);
        else if (dim == 2)
            cuda::filter::median2(input.get(), stride, result.get(), stride, shape, border, window, stream);
        else if (dim == 3)
            cuda::filter::median3(input.get(), stride, result.get(), stride, shape, border, window, stream);
        else
            FAIL("dim is not correct");

        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, result.get(), expected.get(), result.size(), 1e-5));
    }
}

TEMPLATE_TEST_CASE("cuda::filter::median(), random", "[noa][cuda][filter]", int, half_t, float, double) {
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
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    INFO(string::format("ndim:{}, mode:{}, window:{}, shape:{}", ndim, mode, window, shape));

    test::Randomizer<TestType> randomizer(-128, 128);
    cpu::memory::PtrHost<TestType> data(elements);
    test::randomize(data.get(), data.elements(), randomizer);

    cuda::Stream gpu_stream(cuda::Stream::SERIAL);
    cpu::Stream cpu_stream(cpu::Stream::SERIAL);

    cuda::memory::PtrDevicePadded<TestType> d_data(shape);
    cuda::memory::PtrDevicePadded<TestType> d_result(shape);
    cpu::memory::PtrHost<TestType> cuda_result(elements);
    cpu::memory::PtrHost<TestType> h_result(elements);
    cuda::memory::copy(data.get(), stride, d_data.get(), d_data.strides(), shape, gpu_stream);

    if (ndim == 1) {
        cuda::filter::median1(d_data.get(), d_data.strides(), d_result.get(), d_result.strides(),
                              shape, mode, window, gpu_stream);
        cpu::filter::median1(data.get(), stride, h_result.get(), stride, shape, mode, window, cpu_stream);
    } else if (ndim == 2) {
        cuda::filter::median2(d_data.get(), d_data.strides(), d_result.get(), d_result.strides(),
                              shape, mode, window, gpu_stream);
        cpu::filter::median2(data.get(), stride, h_result.get(), stride, shape, mode, window, cpu_stream);
    } else {
        cuda::filter::median3(d_data.get(), d_data.strides(), d_result.get(), d_result.strides(),
                              shape, mode, window, gpu_stream);
        cpu::filter::median3(data.get(), stride, h_result.get(), stride, shape, mode, window, cpu_stream);
    }
    cuda::memory::copy(d_result.get(), d_result.strides(), cuda_result.get(), stride, shape, gpu_stream);
    cpu_stream.synchronize();
    gpu_stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, h_result.get(), cuda_result.get(), h_result.size(), 1e-5));
}
