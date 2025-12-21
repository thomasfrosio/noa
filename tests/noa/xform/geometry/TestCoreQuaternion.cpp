#include <random>

#include <../../../../src/noa/xform/core/Euler.hpp>
#include <../../../../src/noa/xform/core/Quaternion.hpp>
#include <noa/core/geometry/Transform.hpp>

#include "Catch.hpp"

TEST_CASE("core::geometry::Quaternion") {
    using namespace ::noa::types;
    using namespace ::noa::geometry;

    AND_THEN("new") {
        auto q0 = Quaternion<f64>::from_coefficients(1, 2, 3, 4);
        REQUIRE(q0.coefficients() == Vec{1., 2., 3., 4.});
        REQUIRE((q0.z == 1 and q0.y == 2 and q0.x == 3 and q0.w == 4));
        REQUIRE((q0 == Quaternion<f64>{.z=1, .y=2, .x=3, .w=4}));
    }

    AND_THEN("rotate") {
        auto m = noa::geometry::rotate_z(noa::deg2rad(90.));
        auto v = Vec{0., 0., 1.};

        auto a0 = m * v;
        REQUIRE(noa::allclose(a0, Vec{0., 1., 0.}));

        auto q1 = Quaternion<f64>::from_matrix(m);
        auto a1 = q1 * Quaternion<f64>::from_coefficients(v.push_back(0)) * q1.conj();
        auto a2 = q1.rotate(v);

        REQUIRE(noa::allclose(a1.coefficients(), Vec{0., 1., 0., 0.}));
        REQUIRE(noa::allclose(a2, Vec{0., 1., 0.}));
    }

    AND_THEN("rotations") {
        auto dev = std::random_device{};
        auto rng = std::mt19937(dev());
        auto distribution = std::uniform_real_distribution(-1., 1.);

        for (int i = 0; i < 1000; ++i) {
            const auto euler_angles = Vec{distribution(rng), distribution(rng), distribution(rng)};
            const auto rotation_matrix = noa::geometry::euler2matrix(euler_angles);
            const auto quaternion = Quaternion<f64>::from_matrix(rotation_matrix);

            const auto v = Vec{distribution(rng), distribution(rng), distribution(rng)};
            const auto v0 = rotation_matrix * v;
            const auto v1 = quaternion.rotate(v);
            const auto v2 = (quaternion * Quaternion<f64>::from_coefficients(v.push_back(0)) * quaternion.conj())
                            .coefficients().pop_back();
            const auto v3 = quaternion.to_matrix() * v;

            REQUIRE(noa::allclose(v0, v1));
            REQUIRE(noa::allclose(v0, v2));
            REQUIRE(noa::allclose(v0, v3));
        }
    }
}
