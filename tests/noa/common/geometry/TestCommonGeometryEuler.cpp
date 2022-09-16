#include <noa/common/geometry/Euler.h>
#include <noa/common/io/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("geometry::euler2matrix()", "[noa][geometry]") {
    const path_t path_base = test::NOA_DATA_PATH / "common" / "geometry";
    YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["euler2matrix"];
    const auto path_expected = path_base / param["file"].as<path_t>();
    const auto valid_axes = param["axes"].as<std::vector<std::string>>();
    const auto angles = math::deg2rad(param["angles"].as<float3_t>());

    // Get expected:
    io::MRCFile file(path_expected, io::READ);
    const size4_t shape = file.shape();
    const size_t elements = shape.elements();
    cpu::memory::PtrHost<float> expected(elements);
    file.readAll(expected.get());

    // Compute matrices:
    cpu::memory::PtrHost<float33_t> buffer(shape[0]);
    size_t count = 0;
    for (auto& axes: valid_axes)
        for (int intrinsic = 0; intrinsic < 2; ++intrinsic)
            for (int right_handed = 0; right_handed < 2; ++right_handed)
                buffer[count++] = geometry::euler2matrix(angles, axes, !intrinsic, !right_handed);
    const auto* output = reinterpret_cast<const float*>(buffer.get());

    // Switch to ZYX matrices:
    for (size_t i = 0; i < shape[0]; ++i) {
        float33_t& m = buffer[i];
        m = {m[2].flip(), m[1].flip(), m[0].flip()};
    }

    REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output, elements, 1e-6));
}

TEST_CASE("geometry::matrix2euler()", "[noa][geometry]") {
    const path_t path_base = test::NOA_DATA_PATH / "common" / "geometry";
    YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["euler2matrix"];
    const auto valid_axes = param["axes"].as<std::vector<std::string>>();
    const auto eulers = math::deg2rad(float3_t{10, 20, 30});
    for (auto& axes: valid_axes) {
        for (int intrinsic = 0; intrinsic < 2; ++intrinsic) {
            for (int right_handed = 0; right_handed < 2; ++right_handed) {
                const float33_t matrix = geometry::euler2matrix(eulers, axes, !intrinsic, !right_handed);

                const float3_t result_eulers = geometry::matrix2euler(matrix, axes, !intrinsic, !right_handed);
                const float33_t result_matrix = geometry::euler2matrix(result_eulers, axes, !intrinsic, !right_handed);

                REQUIRE(test::Matcher(test::MATCH_ABS, matrix.get(), result_matrix.get(), 9, 1e-6));
            }
        }
    }
}
