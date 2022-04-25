#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/signal/Median.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

TEST_CASE("cpu::signal::median()", "[assets][noa][cpu][filter]") {
    using namespace noa;

    const path_t path_base = test::NOA_DATA_PATH / "signal";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["median"]["tests"];
    io::ImageFile file;
    cpu::Stream stream(cpu::Stream::DEFAULT);

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
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        cpu::memory::PtrHost<float> expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        cpu::memory::PtrHost<float> result(elements);
        if (dim == 1)
            cpu::signal::median1<float>(input.share(), stride, result.share(), stride, shape, border, window, stream);
        else if (dim == 2)
            cpu::signal::median2<float>(input.share(), stride, result.share(), stride, shape, border, window, stream);
        else if (dim == 3)
            cpu::signal::median3<float>(input.share(), stride, result.share(), stride, shape, border, window, stream);
        else
            FAIL("dim is not correct");

        REQUIRE(test::Matcher(test::MATCH_ABS, result.get(), expected.get(), result.size(), 1e-5));
    }
}
