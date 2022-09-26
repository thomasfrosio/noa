#include <noa/common/io/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Permute.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::memory::permute()", "[assets][noa][cpu][memory]") {
    const path_t path_base = test::NOA_DATA_PATH / "memory";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["transpose"]["tests"];
    io::MRCFile file;
    cpu::Stream stream(cpu::Stream::DEFAULT);

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto filename_input = path_base / test["input"].as<path_t>();
        const auto filename_expected = path_base / test["expected"].as<path_t>();
        const auto permutation = test["permutation"].as<dim4_t>();
        const auto inplace = test["inplace"].as<bool>();

        file.open(filename_input, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> data(elements);
        cpu::memory::PtrHost<float> result(elements);
        cpu::memory::PtrHost<float> expected(elements);
        file.readAll(data.get());
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        const size4_t output_shape = indexing::reorder(shape, permutation);
        const size4_t output_stride = output_shape.strides();

        if (inplace) {
            cpu::memory::permute<float>(data.share(), stride, shape,
                                        data.share(), output_stride, permutation, stream);
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), data.get(), elements, 1e-8));
        } else {
            cpu::memory::permute<float>(data.share(), stride, shape,
                                        result.share(), output_stride, permutation, stream);
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), result.get(), elements, 1e-8));
        }
    }
}
