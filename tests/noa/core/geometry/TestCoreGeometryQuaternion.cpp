#include <noa/core/geometry/Euler.hpp>
#include <noa/core/geometry/Quaternion.hpp>
#include <noa/core/geometry/Transform.hpp>

#include <random>

#include <catch2/catch.hpp>

TEST_CASE("core::geometry::Quaternion") {
    using namespace ::noa;

    AND_THEN("") {
        auto q0 = Quaternion<f64>::from_coefficients(1, 2, 3, 4);
        REQUIRE(noa::all(q0.vec() == Vec4<f64>{1, 2, 3, 4}));
        REQUIRE((q0.z() == 1 && q0.y() == 2 && q0.x() == 3 && q0.w() == 4));

    }

    AND_THEN("rotate") {
        auto m = noa::geometry::rotate_z(noa::math::deg2rad(90.));
        auto v = Vec3<f64>{0, 0, 1};

        auto a0 = m * v;
        REQUIRE(noa::all(noa::math::are_almost_equal(a0, Vec3<f64>{0, 1, 0})));

        auto q1 = Quaternion<f64>::from_matrix(m);
        auto a1 = q1 * Quaternion<f64>(v.push_back(0)) * q1.conj();
        auto a2 = q1.rotate(v);

        REQUIRE(noa::all(noa::math::are_almost_equal(a1.vec(), Vec4<f64>{0, 1, 0, 0})));
        REQUIRE(noa::all(noa::math::are_almost_equal(a2, Vec3<f64>{0, 1, 0})));
    }

    AND_THEN("rotations") {
        auto dev = std::random_device{};
        auto rng = std::mt19937(dev());
        auto distribution = std::uniform_real_distribution<f64>(-1., 1);

        for (int i = 0; i < 1000; ++i) {
            const auto euler_angles = Vec3<f64>{distribution(rng), distribution(rng), distribution(rng)};
            const auto rotation_matrix = noa::geometry::euler2matrix(euler_angles);
            const auto quaternion = Quaternion<f64>::from_matrix(rotation_matrix);

            const auto v = Vec3<f64>{distribution(rng), distribution(rng), distribution(rng)};
            const auto v0 = rotation_matrix * v;
            const auto v1 = quaternion.rotate(v);
            const auto v2 = (quaternion * Quaternion<f64>{v.push_back(0)} * quaternion.conj()).vec().pop_back();
            const auto v3 = quaternion.to_matrix() * v;

            REQUIRE(noa::all(noa::math::are_almost_equal(v0, v1)));
            REQUIRE(noa::all(noa::math::are_almost_equal(v0, v2)));
            REQUIRE(noa::all(noa::math::are_almost_equal(v0, v3)));
        }
    }
}
