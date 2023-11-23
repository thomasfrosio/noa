#include <noa/core/geometry/Euler.hpp>
#include <noa/core/io/MRCFile.hpp>
#include <noa/core/indexing/Layout.hpp>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>
#include <memory>

using namespace ::noa;

TEST_CASE("core::geometry::euler2matrix()", "[noa][core]") {
    const Path path_base = test::NOA_DATA_PATH / "common" / "geometry";
    YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["euler2matrix"];
    const auto path_expected = path_base / param["file"].as<Path>();
    const auto valid_axes = param["axes"].as<std::vector<std::string>>();
    const auto angles = deg2rad(param["angles"].as<Vec3<f32>>());

    // Get expected:
    io::MRCFile file(path_expected, io::READ);
    const auto shape = file.shape().as<u64>();
    const auto matrices = shape[0];
    const auto elements = shape.elements();
    const auto expected = std::make_unique<f32[]>(elements);
    file.read_all(expected.get());

    // Compute matrices:
    const auto buffer = std::make_unique<Mat33<f32>[]>(matrices);
    const auto* result = reinterpret_cast<const f32*>(buffer.get()); // :-(
    u64 count = 0;
    for (const auto& axes: valid_axes)
        for (i32 intrinsic = 1; intrinsic >=0; --intrinsic) // in Python, I do (True, False)...
            for (i32 right_handed = 1; right_handed >=0; --right_handed)
                buffer[count++] = noa::geometry::euler2matrix(angles, axes, intrinsic, right_handed);

    // Switch the axes. In Python, I use xyz matrices...
    for (u64 i = 0; i < matrices; ++i)
        buffer[i] = noa::reorder(buffer[i], Vec3<i64>{2, 1, 0}); // zyx -> xyz

    REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), result, static_cast<i64>(elements), 1e-6));
}

TEST_CASE("core::matrix2euler()", "[noa][core]") {
    const Path path_base = test::NOA_DATA_PATH / "common" / "geometry";
    YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["euler2matrix"];
    const auto valid_axes = param["axes"].as<std::vector<std::string>>();
    const auto eulers_to_test = noa::deg2rad(Vec3<f32>{10, 20, 30});
    for (const auto& axes: valid_axes) {
        for (i32 intrinsic = 1; intrinsic >=0; --intrinsic) { // in Python, I do (True, False)...
            for (i32 right_handed = 1; right_handed >=0; --right_handed) {
                const auto expected_matrix = noa::geometry::euler2matrix(eulers_to_test, axes, intrinsic, right_handed);

                const Vec3<f32> result_eulers = noa::geometry::matrix2euler(expected_matrix, axes, intrinsic, right_handed);
                const Mat33<f32> result_matrix = noa::geometry::euler2matrix(result_eulers, axes, intrinsic, right_handed);

                REQUIRE(test::Matcher(test::MATCH_ABS, expected_matrix.data(), result_matrix.data(), 9, 1e-6));
            }
        }
    }
}
