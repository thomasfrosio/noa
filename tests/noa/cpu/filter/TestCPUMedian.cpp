#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/filter/Median.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

TEST_CASE("cpu::filter::median()", "[assets][noa][cpu][filter]") {
    using namespace noa;

    path_t path_base = test::PATH_NOA_DATA / "filter";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["median"]["tests"];
    io::ImageFile file;
    cpu::Stream stream;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto filename_input = path_base / test["input"].as<path_t>();
        auto window = test["window"].as<size_t>();
        auto dim = test["dim"].as<int>();
        auto border = test["border"].as<BorderMode>();
        auto filename_expected = path_base / test["expected"].as<path_t>();
        INFO(dim);

        file.open(filename_input, io::READ);
        size3_t shape = file.shape();
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        cpu::memory::PtrHost<float> expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        cpu::memory::PtrHost<float> result(elements);
        if (dim == 1)
            cpu::filter::median1(input.get(), shape, result.get(), shape, shape, 1, border, window, stream);
        else if (dim == 2)
            cpu::filter::median2(input.get(), shape, result.get(), shape, shape, 1, border, window, stream);
        else if (dim == 3)
            cpu::filter::median3(input.get(), shape, result.get(), shape, shape, 1, border, window, stream);
        else
            FAIL("dim is not correct");

        REQUIRE(test::Matcher(test::MATCH_ABS, result.get(), expected.get(), result.size(), 1e-5));
    }
}
