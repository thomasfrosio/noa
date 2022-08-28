#include <noa/common/io/MRCFile.h>
#include <noa/gpu/cuda/memory/PtrManaged.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/math/Sort.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cuda::math::sort()", "[assets][noa][cuda][math]") {
    const path_t path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["sort"];

    cuda::Stream stream;
    for (size_t nb = 0; nb < tests["tests"].size(); ++nb) {
        INFO(nb);
        const YAML::Node test = tests["tests"][nb];

        const auto input_filename = path / test["input"].as<path_t>();
        const auto output_filename = path / test["output"].as<path_t>();
        const auto shape = test["shape"].as<size4_t>();
        const auto axis = test["axis"].as<int>();
        const auto ascending = test["ascending"].as<bool>();
        const size4_t strides = shape.strides();
        const size_t elements = shape.elements();

        io::MRCFile file(input_filename, io::READ);
        cuda::memory::PtrManaged<float> data(elements, stream);
        file.readAll(data.get());

        cuda::memory::PtrManaged<float> expected(elements, stream);
        file.open(output_filename, io::READ);
        file.readAll(expected.get());

        cuda::memory::PtrManaged<float> results(elements, stream);
        cuda::memory::copy(data.get(), results.get(), data.elements());

        cuda::math::sort(results.share(), strides, shape, ascending, axis, stream);

        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), results.get(), data.elements(), 1e-7));
    }
}
