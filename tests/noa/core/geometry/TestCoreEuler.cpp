#include <noa/core/geometry/Euler.hpp>
#include <noa/core/io/ImageFile.hpp>
#include <noa/core/indexing/Layout.hpp>
#include <catch2/catch.hpp>
#include <memory>

#include "Assets.h"
#include "Utils.hpp"

using namespace ::noa::types;
namespace nio = ::noa::io;


TEST_CASE("core::geometry::euler2matrix()", "[noa]") {
    const Path path_base = test::NOA_DATA_PATH / "common" / "geometry";
    YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["euler2matrix"];
    const auto path_expected = path_base / param["file"].as<Path>();
    const auto valid_axes = param["axes"].as<std::vector<std::string>>();
    const auto angles = deg2rad(param["angles"].as<Vec3<f32>>());

    // Get expected:
    auto file = nio::ImageFile(path_expected, {.read = true});
    const auto shape = file.shape();
    const auto n_matrices = shape[0];
    const auto n_elements = shape.n_elements();
    const auto buffer_expected = test::make_unique<Mat33<f32>[]>(n_elements);
    const auto expected = Span(buffer_expected.get(), n_matrices);
    {
        const auto tmp = test::make_unique<f32[]>(n_elements);
        file.read_all(Span(tmp.get(), shape));
        for (size_t i{}; auto& e: expected) {
            e = Mat33<f32>::from_pointer(tmp.get() + i);
            i += 9;
        }
    }

    // Compute matrices:
    const auto buffer = test::make_unique<Mat33<f32>[]>(n_matrices);
    const auto matrices = Span(buffer.get(), n_matrices);
    i64 count{};
    for (const std::string& axes: valid_axes)
        for (bool is_intrinsic: std::array{true, false})
            for (bool is_right_handed: std::array{true, false})
                matrices.at(count++) = noa::geometry::euler2matrix(angles, {axes, is_intrinsic, is_right_handed});

    // Switch the axes. In Python, I use xyz matrices...
    for (auto& matrix: matrices)
        matrix = noa::indexing::reorder(matrix, Vec{2, 1, 0}); // zyx -> xyz

    bool allclose{true};
    for (i64 i: noa::irange(n_matrices)) {
        if (not noa::allclose(matrices[i], expected[i], 1e-6f)) {
            allclose = false;
            break;
        }
    }
    REQUIRE(allclose);
}

TEST_CASE("core::geometry::matrix2euler()", "[noa]") {
    const Path path_base = test::NOA_DATA_PATH / "common" / "geometry";
    YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["euler2matrix"];
    const auto valid_axes = param["axes"].as<std::vector<std::string>>();
    const auto eulers_to_test = noa::deg2rad(Vec{10., 20., 30.});

    for (const std::string& axes: valid_axes) {
        for (bool is_intrinsic: std::array{true, false}) {
            for (bool is_right_handed: std::array{true, false}) {
                const auto options = noa::geometry::EulerOptions{axes, is_intrinsic, is_right_handed};
                const Mat expected_matrix = noa::geometry::euler2matrix(eulers_to_test, options);

                const Vec result_eulers = noa::geometry::matrix2euler(expected_matrix, options);
                const Mat result_matrix = noa::geometry::euler2matrix(result_eulers, options);

                REQUIRE(noa::allclose(expected_matrix, result_matrix, 1e-6));
            }
        }
    }
}
