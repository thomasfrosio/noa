#include <noa/unified/math/LinAlg.hpp>
#include <catch2/catch.hpp>

TEST_CASE("unified::math::lstsq - scipy example()", "[noa][unified]") {
    // We want to fit a quadratic polynomial of the form ``data_y = c0 + c1 * data_x^2``
    constexpr size_t N = 7;
    std::array<double, N> data_x{1.0, 2.5, 3.5, 4.0, 5.0, 7.0, 8.5};
    std::array<double, N> data_y{0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6};

    // We first form the "design matrix" M, with a constant
    // column of 1s and a column containing ``data_x^2``.
    const noa::Array<double> a({1, 1, N, 2});
    const noa::Array<double> b({1, 1, N, 1});
    const noa::Array<double> x({1, 1, 2, 1});
    for (size_t i = 0; i < N; ++i) {
        a(0, 0, i, 0) = 1;
        a(0, 0, i, 1) = data_x[i] * data_x[i];
        b(0, 0, i, 0) = data_y[i];
    }

    // We want to find the least-squares solution to ``a.dot(x) = b``,
    // where ``x`` is a vector with length 2 that holds the parameters
    // ``a`` and ``b``.
    noa::math::lstsq(a, b, x);
    x.eval();

    REQUIRE_THAT(x(0, 0, 0, 0), Catch::WithinAbs(0.20925829, 1e-7));
    REQUIRE_THAT(x(0, 0, 1, 0), Catch::WithinAbs(0.12013861, 1e-7));
}
