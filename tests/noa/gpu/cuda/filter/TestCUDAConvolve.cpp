#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Arithmetics.h>

#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/filter/Convolve.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

TEST_CASE("cuda::filter::convolve()", "[assets][noa][cuda][filter]") {
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
        cpu::memory::PtrHost<float> filter(noa::elements(filter_shape)); // filter can be on the host
        file.readAll(filter.get());
        if (filter_shape.y == 2 && filter_shape.z == 1)
            filter_shape.y = 1; // for 1D case, the MRC file as an extra row to make it 2D

        // Expected:
        cpu::memory::PtrHost<float> expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        cuda::Stream stream;
        cuda::memory::PtrDevicePadded<float> d_data(shape);
        cuda::memory::PtrDevicePadded<float> d_result(shape);
        cpu::memory::PtrHost<float> result(elements);

        cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
        cuda::memory::copy(filter.get(), filter.get(), filter.elements());
        cuda::filter::convolve(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(), shape, 1,
                               filter.get(), filter_shape, stream);
        cuda::memory::copy(d_result.get(), d_result.pitch(), result.get(), shape.x, shape, stream);
        cuda::Stream::synchronize(stream);

        // it's around 2e-5 and 6e-5
        REQUIRE(test::Matcher(test::MATCH_ABS, result.get(), expected.get(), result.size(), 1e-4));
    }
}

TEST_CASE("cuda::filter::convolve() - separable", "[assets][noa][cuda][filter]") {
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

        cuda::Stream stream;
        cuda::memory::PtrDevicePadded<float> d_data(shape);
        cuda::memory::PtrDevicePadded<float> d_result(shape);
        cpu::memory::PtrHost<float> result(elements);

        cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
        cuda::filter::convolve(d_data.get(), d_data.pitch(), d_result.get(), d_result.pitch(), shape, 1,
                               filter0, filter_size, filter1, filter_size, filter2, filter_size, stream);
        cuda::memory::copy(d_result.get(), d_result.pitch(), result.get(), shape.x, shape, stream);
        cuda::Stream::synchronize(stream);

        REQUIRE(test::Matcher(test::MATCH_ABS, result.get(), expected.get(), result.size(), 1e-4));
    }
}
