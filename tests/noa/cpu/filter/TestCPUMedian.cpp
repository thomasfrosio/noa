#include <noa/common/files/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/filter/Median.h>
#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/math/Reductions.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

TEST_CASE("cpu::filter::median()", "[assets][noa][cpu][filter]") {
    using namespace noa;

    path_t path_base = test::PATH_TEST_DATA / "filter";
    YAML::Node tests = YAML::LoadFile(path_base / "param.yaml")["median"]["tests"];
    MRCFile file;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto filename_input = path_base / test["input"].as<path_t>();
        auto window = test["window"].as<uint>();
        auto dim = test["dim"].as<int>();
        auto border = test["border"].as<BorderMode>();
        auto filename_expected = path_base / test["expected"].as<path_t>();

        file.open(filename_input, io::READ);
        size3_t shape = file.getShape();
        size_t elements = getElements(shape);
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        cpu::memory::PtrHost<float> expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        cpu::memory::PtrHost<float> result(elements);
        if (dim == 1)
            cpu::filter::median1(input.get(), result.get(), shape, 1, border, window);
        else if (dim == 2)
            cpu::filter::median2(input.get(), result.get(), shape, 1, border, window);
        else if (dim == 3)
            cpu::filter::median3(input.get(), result.get(), shape, 1, border, window);
        else
            FAIL("dim is not correct");

        float min, max, mean;
        cpu::math::subtractArray(result.get(), expected.get(), result.get(), result.size(), 1);
        cpu::math::minMaxSumMean<float>(result.get(), &min, &max, nullptr, &mean, result.size(), 1);
        REQUIRE_THAT(math::abs(min), test::isWithinAbs(0.f, 1e-5));
        REQUIRE_THAT(math::abs(max), test::isWithinAbs(0.f, 1e-5));
        REQUIRE_THAT(math::abs(mean), test::isWithinAbs(0.f, 1e-6));
    }
}
