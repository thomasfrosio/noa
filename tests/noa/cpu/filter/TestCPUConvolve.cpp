#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/filter/Convolve.h>
#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/math/Reductions.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

TEST_CASE("cpu::filter::convolve()", "[assets][noa][cpu][filter]") {
    using namespace noa;

    path_t path_base = test::PATH_TEST_DATA / "filter";
    YAML::Node tests = YAML::LoadFile(path_base / "param.yaml")["convolve"]["tests"];
    io::ImageFile file;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto filename_input = path_base / test["input"].as<path_t>();
        auto filename_filter = path_base / test["filter"].as<path_t>();
        auto filename_expected = path_base / test["expected"].as<path_t>();

        // Input:
        file.open(filename_input, io::READ);
        size3_t shape = file.shape();
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<float> data(elements);
        file.readAll(data.get());

        // Filter:
        file.open(filename_filter, io::READ);
        size3_t filter_shape = file.shape();
        cpu::memory::PtrHost<float> filter(noa::elements(filter_shape));
        file.readAll(filter.get());
        if (filter_shape.y == 2 && filter_shape.z == 1)
            filter_shape.y = 1; // for 1D case, the MRC file as an extra row to make it 2D.

        // Expected:
        cpu::memory::PtrHost<float> expected(elements);
        cpu::memory::PtrHost<float> result(elements);
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        cpu::filter::convolve(data.get(), result.get(), shape, 1, filter.get(), filter_shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), result.get(), result.size(), 1e-5));
    }
}

TEST_CASE("cpu::filter::convolve() - separable", "[assets][noa][cpu][filter]") {
    using namespace noa;

    path_t path_base = test::PATH_TEST_DATA / "filter";
    YAML::Node tests = YAML::LoadFile(path_base / "param.yaml")["convolve_separable"]["tests"];
    io::ImageFile file;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto filename_input = path_base / test["input"].as<path_t>();
        auto filename_filter = path_base / test["filter"].as<path_t>();
        auto filename_expected = path_base / test["expected"].as<path_t>();
        auto dim = test["dim"].as<std::vector<int>>();

        // Input
        file.open(filename_input, io::READ);
        size3_t shape = file.shape();
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<float> data(elements);
        file.readAll(data.get());

        // Filter:
        file.open(filename_filter, io::READ);
        uint filter_size = uint(file.shape().x);
        cpu::memory::PtrHost<float> filter(filter_size * 2); // the MRC file as an extra row to make it 2D.
        file.readAll(filter.get());

        // Expected:
        cpu::memory::PtrHost<float> expected(elements);
        cpu::memory::PtrHost<float> result(elements);
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

        cpu::filter::convolve(data.get(), result.get(), shape, 1,
                              filter0, filter_size, filter1, filter_size, filter2, filter_size);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), result.get(), result.size(), 1e-5));
    }
}
