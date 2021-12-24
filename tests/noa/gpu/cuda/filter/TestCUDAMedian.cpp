#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include <noa/cpu/filter/Median.h>
#include <noa/gpu/cuda/filter/Median.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

TEST_CASE("cuda::filter::median()", "[assets][noa][cuda][filter]") {
    using namespace noa;

    path_t path_base = test::PATH_NOA_DATA / "filter";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["median"]["tests"];
    io::ImageFile file;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto filename_input = path_base / test["input"].as<path_t>();
        auto window = test["window"].as<size_t>();
        auto dim = test["dim"].as<int>();
        auto border = test["border"].as<BorderMode>();
        auto filename_expected = path_base / test["expected"].as<path_t>();

        file.open(filename_input, io::READ);
        size3_t shape = file.shape();
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        cpu::memory::PtrHost<float> expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        cuda::memory::PtrDevicePadded<float> d_data(shape);
        cuda::memory::PtrDevicePadded<float> d_result(shape);
        cuda::Stream stream;

        cuda::memory::copy(input.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);

        if (dim == 1)
            cuda::filter::median1(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                                  shape, 1, border, window, stream);
        else if (dim == 2)
            cuda::filter::median2(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                                  shape, 1, border, window, stream);
        else if (dim == 3)
            cuda::filter::median3(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                                  shape, 1, border, window, stream);
        else
            FAIL("dim is not correct");

        cpu::memory::PtrHost<float> result(elements);
        cuda::memory::copy(d_result.get(), d_result.pitch(), result.get(), shape.x, shape, stream);
        stream.synchronize();

        REQUIRE(test::Matcher(test::MATCH_ABS, result.get(), expected.get(), result.size(), 1e-5));
    }
}

TEMPLATE_TEST_CASE("cuda::filter::median(), random", "[noa][cuda][filter]", int, float, double) {
    using namespace noa;

    int ndim = GENERATE(1, 2, 3);
    BorderMode mode = GENERATE(BORDER_ZERO, BORDER_REFLECT);
    size_t window = test::Randomizer<uint>(2, 11).get();
    if (!(window % 2))
        window -= 1;
    if (ndim == 3 && window > 5)
        window = 3;

    test::Randomizer<size_t> random_size(16, 100);
    size3_t shape{random_size.get(), random_size.get(), random_size.get()};
    if (ndim != 3 && random_size.get() % 2)
        shape.z = 1; // randomly switch to 2D
    size_t elements = noa::elements(shape);

    size_t batches = test::Randomizer<size_t>(1, 3).get();
    size_t elements_batched = elements * batches;
    size3_t shape_batched = {shape.x, rows(shape), batches};

    INFO(string::format("ndim:{}, mode:{}, window:{}, shape:{}, batches:{}",
                        ndim, mode, window, shape, batches));

    test::Randomizer<TestType> randomizer(-128, 128);
    cpu::memory::PtrHost<TestType> data(elements_batched);
    test::randomize(data.get(), data.elements(), randomizer);

    cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
    cuda::memory::PtrDevicePadded<TestType> d_result(shape_batched);
    cpu::memory::PtrHost<TestType> cuda_result(elements_batched);
    cpu::memory::PtrHost<TestType> h_result(elements_batched);
    cuda::Stream stream;

    cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);

    if (ndim == 1) {
        cuda::filter::median1(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                              shape, batches, mode, window, stream);
        cpu::filter::median1(data.get(), h_result.get(), shape, batches, mode, window);
    } else if (ndim == 2) {
        cuda::filter::median2(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                              shape, batches, mode, window, stream);
        cpu::filter::median2(data.get(), h_result.get(), shape, batches, mode, window);
    } else {
        cuda::filter::median3(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(),
                              shape, batches, mode, window, stream);
        cpu::filter::median3(data.get(), h_result.get(), shape, batches, mode, window);
    }
    cuda::memory::copy(d_result.get(), d_result.pitch(), cuda_result.get(), shape.x, shape_batched, stream);
    cuda::Stream::synchronize(stream);

    REQUIRE(test::Matcher(test::MATCH_ABS, h_result.get(), cuda_result.get(), h_result.size(), 1e-5));
}
