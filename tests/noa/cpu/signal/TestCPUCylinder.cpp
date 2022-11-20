#include <noa/common/io/MRCFile.h>
#include <noa/common/geometry/Euler.h>
#include <noa/common/geometry/Transform.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/signal/Shape.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// Just compare against manually checked data.

TEST_CASE("cpu::signal::cylinder(), 3D", "[assets][noa][cpu]") {
    test::Randomizer<float> randomizer(-5, 5);

    const path_t path_base = test::NOA_DATA_PATH / "signal";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["cylinder"]["tests3D"];
    io::MRCFile file;
    cpu::Stream stream;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<size4_t>();
        const size4_t strides = shape.strides();
        const size_t elements = shape.elements();

        const auto center = test["center"].as<float3_t>();
        const auto radius = test["radius"].as<float>();
        const auto length = test["length"].as<float>();
        const auto taper = test["taper"].as<float>();
        const auto fwd_transform = geometry::euler2matrix(math::deg2rad(test["angle"].as<float3_t>()));

        const auto filename_expected = path_base / test["expected"].as<path_t>();
        file.open(filename_expected, io::READ);
        if (all(file.shape() != shape))
            FAIL("asset shape is not correct");
        cpu::memory::PtrHost<float> mask_expected(elements);
        file.readAll(mask_expected.get());

        cpu::memory::PtrHost<float> input_expected(elements);
        cpu::memory::PtrHost<float> input_result(elements);
        cpu::memory::PtrHost<float> mask_result(elements);

        // Test saving the mask.
        cpu::signal::cylinder<float>(nullptr, {}, mask_result.share(), strides, shape,
                                     center, radius, length, taper, math::inverse(fwd_transform),
                                     math::multiply_t{}, invert, stream);
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, mask_expected.get(), mask_result.get(), elements, 1e-6));

        AND_THEN("invert = true") {
            test::randomize(input_expected.get(), elements, randomizer);
            test::copy(input_expected.get(), input_result.get(), elements);

            cpu::signal::cylinder(input_result.share(), strides, input_result.share(), strides, shape,
                                  center, radius, length, taper, math::inverse(fwd_transform),
                                  math::multiply_t{}, true, stream);
            for (size_t idx = 0; idx < elements; ++idx)
                input_expected[idx] *= invert ? mask_expected[idx] : 1 - mask_expected[idx];

            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input_result.get(), input_expected.get(), elements, 1e-6));
        }

        AND_THEN("invert = false") {
            test::randomize(input_expected.get(), elements, randomizer);
            test::copy(input_expected.get(), input_result.get(), elements);

            cpu::signal::cylinder(input_result.share(), strides, input_result.share(), strides, shape,
                                  center, radius, length, taper, math::inverse(fwd_transform),
                                  math::multiply_t{}, false, stream);
            for (size_t idx = 0; idx < elements; ++idx)
                input_expected[idx] *= invert ? 1 - mask_expected[idx] : mask_expected[idx];

            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input_result.get(), input_expected.get(), elements, 1e-6));
        }
    }
}
