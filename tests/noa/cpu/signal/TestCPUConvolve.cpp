#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/signal/Convolve.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

TEST_CASE("cpu::signal::convolve()", "[assets][noa][cpu][filter]") {
    using namespace noa;

    const path_t path_base = test::NOA_DATA_PATH / "signal";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["convolve"]["tests"];
    io::ImageFile file;

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
        cpu::memory::PtrHost<float> data(shape.elements());
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
        cpu::memory::PtrHost<float> result(data.elements());
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        cpu::Stream stream(cpu::Stream::DEFAULT);
        cpu::signal::convolve<float, float>(data.share(), stride,
                                            result.share(), stride, shape,
                                            filter.share(), filter_shape, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), result.get(), result.size(), 1e-5));
    }
}

TEST_CASE("cpu::signal::convolve() - separable", "[assets][noa][cpu][filter]") {
    using namespace noa;

    const path_t path_base = test::NOA_DATA_PATH / "signal";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["convolve_separable"]["tests"];
    io::ImageFile file;

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
        cpu::memory::PtrHost<float> data(shape.elements());
        file.readAll(data.get());

        // Filter:
        file.open(filename_filter, io::READ);
        const size_t filter_size = file.shape()[3];
        cpu::memory::PtrHost<float> filter(filter_size);
        file.read(filter.get(), 0, filter_size);

        // Expected:
        cpu::memory::PtrHost<float> expected(data.elements());
        cpu::memory::PtrHost<float> result(data.elements());
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        std::shared_ptr<float[]> filter0 = nullptr;
        std::shared_ptr<float[]> filter1 = nullptr;
        std::shared_ptr<float[]> filter2 = nullptr;
        for (int i: dim) {
            if (i == 0)
                filter0 = filter.share();
            if (i == 1)
                filter1 = filter.share();
            if (i == 2)
                filter2 = filter.share();
        }

        cpu::Stream stream(cpu::Stream::DEFAULT);
        cpu::signal::convolve<float, float>(data.share(), stride, result.share(), stride, shape,
                                            filter0, filter_size, filter1, filter_size, filter2, filter_size, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), result.get(), result.size(), 1e-5));
    }
}
