#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Transpose.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::memory::transpose()", "[assets][noa][cpu][memory]") {
    path_t path_base = test::PATH_NOA_DATA / "memory";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["transpose"]["tests"];
    io::ImageFile file;
    cpu::Stream stream;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto filename_input = path_base / test["input"].as<path_t>();
        const auto filename_expected = path_base / test["expected"].as<path_t>();
        const auto permutation = test["permutation"].as<uint3_t>();
        const auto inplace = test["inplace"].as<bool>();

        file.open(filename_input, io::READ);
        const size3_t shape = file.shape();
        const size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<float> data(elements);
        cpu::memory::PtrHost<float> result(elements);
        cpu::memory::PtrHost<float> expected(elements);
        file.readAll(data.get());
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        if (inplace) {
            cpu::memory::transpose(data.get(), shape, shape, data.get(), shape, permutation, 1, stream);
            float diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == 0);
        } else {
            cpu::memory::transpose(data.get(), shape, shape, result.get(), shape, permutation, 1, stream);
            float diff = test::getDifference(expected.get(), result.get(), elements);
            REQUIRE(diff == 0);
        }
    }
}
