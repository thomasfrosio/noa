#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>

#include <noa/gpu/cuda/memory/PtrManaged.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/filter/Convolve.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

TEST_CASE("cuda::filter::convolve()", "[assets][noa][cuda][filter]") {
    using namespace noa;

    const path_t path_base = test::NOA_DATA_PATH / "filter";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["convolve"]["tests"];
    io::ImageFile file;
    cuda::Stream stream(cuda::Stream::CONCURRENT);

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto filename_input = path_base / test["input"].as<path_t>();
        const auto filename_filter = path_base / test["filter"].as<path_t>();
        const auto filename_expected = path_base / test["expected"].as<path_t>();

        // Input:
        file.open(filename_input, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.strides();
        cuda::memory::PtrManaged<float> data(shape.elements(), stream);
        file.readAll(data.get());

        // Filter:
        file.open(filename_filter, io::READ);
        size3_t filter_shape(file.shape().get() + 1);
        if (filter_shape[0] == 1 && filter_shape[1] == 2)
            filter_shape[1] = 1; // for 1D case, the MRC file as an extra row to make it 2D.
        cpu::memory::PtrHost<float> filter(filter_shape.elements());
        test::memset(filter.get(), filter.elements(), 1);
        file.read(filter.get(), 0, filter.elements());

        // Expected:
        cpu::memory::PtrHost<float> expected(data.elements());
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        cuda::memory::PtrManaged<float> result(data.elements(), stream);
        cuda::filter::convolve(data.get(), stride, result.get(), stride, shape, filter.get(), filter_shape, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), result.get(), result.size(), 1e-5));
    }
}

TEST_CASE("cuda::filter::convolve() - separable", "[assets][noa][cuda][filter]") {
    using namespace noa;

    const path_t path_base = test::NOA_DATA_PATH / "filter";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["convolve_separable"]["tests"];
    io::ImageFile file;
    cuda::Stream stream(cuda::Stream::CONCURRENT);

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto filename_input = path_base / test["input"].as<path_t>();
        const auto filename_filter = path_base / test["filter"].as<path_t>();
        const auto filename_expected = path_base / test["expected"].as<path_t>();
        const auto dim = test["dim"].as<std::vector<int>>();

        // Input
        file.open(filename_input, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.strides();
        cuda::memory::PtrManaged<float> data(shape.elements(), stream);
        file.readAll(data.get());

        // Filter:
        file.open(filename_filter, io::READ);
        const size_t filter_size = file.shape()[3];
        cpu::memory::PtrHost<float> filter(filter_size);
        file.read(filter.get(), 0, filter_size);

        // Expected:
        cpu::memory::PtrHost<float> expected(data.elements());
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        float* filter0 = nullptr;
        float* filter1 = nullptr;
        float* filter2 = nullptr;
        for (int i: dim) {
            if (i == 0)
                filter0 = filter.get();
            if (i == 1)
                filter1 = filter.get();
            if (i == 2)
                filter2 = filter.get();
        }

        cuda::memory::PtrManaged<float> result(data.elements(), stream);
        cuda::filter::convolve(data.get(), stride, result.get(), stride, shape,
                               filter0, filter_size, filter1, filter_size, filter2, filter_size, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), result.get(), result.size(), 1e-5));
    }
}
