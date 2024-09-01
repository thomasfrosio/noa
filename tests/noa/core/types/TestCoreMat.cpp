#include <noa/core/types/Mat.hpp>
#include <catch2/catch.hpp>


using namespace ::noa::types;

TEMPLATE_TEST_CASE("core::Mat", "[noa][core]", f32, f64) {
    // Loading matrices onto the GPU can be quite expensive, especially for matrices made of vectors
    // with an odd number of elements. To fix this, force to a 16-bytes alignment.
    using real_t = TestType;
    static_assert(alignof(Mat<real_t, 2, 2>) == 16);
    static_assert(alignof(Mat<real_t, 2, 3>) == 16);
    static_assert(alignof(Mat<real_t, 4, 4>) == 16);

    using mat22_t = Mat<real_t, 2, 2>;
    using mat23_t = Mat<real_t, 2, 3>;
    using mat33_t = Mat<real_t, 3, 3>;
    using mat44_t = Mat<real_t, 4, 4>;
    using mat42_t = Mat<real_t, 4, 2>;
    using vec2_t = Vec<real_t, 2>;

    AND_THEN("(Conversion) Constructors") {
        // Mat is an aggregate...
        auto a = mat23_t{{{1, 2, 3},
                          {4, 5, 6}}};
        REQUIRE((a[0][0] == 1 and a[0][1] == 2 and a[0][2] == 3 and
                 a[1][0] == 4 and a[1][1] == 5 and a[1][2] == 6));

        // ... with a bunch of factory functions
        REQUIRE(mat22_t::from_values(0, 0, 0, 0) == mat22_t{});
        REQUIRE(mat22_t::from_values(3, 0, 0, 3) == mat22_t::eye(3));
        REQUIRE(mat22_t::from_values(1, 0, 0, 2) == mat22_t::eye(Vec{1, 2}));

        using r_t = mat22_t::row_type;
        static_assert(std::is_same_v<r_t, Vec<real_t, 2>>);
        REQUIRE(mat22_t::from_values(1, 2, 3, 4) == mat22_t::from_rows(r_t{1, 2}, r_t{3, 4}));

        using c_t = mat42_t::column_type;
        static_assert(std::is_same_v<c_t, Vec<real_t, 4>>);
        REQUIRE(mat42_t::from_values(1, 5, 2, 6, 3, 7, 4, 8) ==
                mat42_t::from_columns(c_t{1, 2, 3, 4}, c_t{5, 6, 7, 8}));

        auto b = Mat<f64, 4, 4>::from_values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        auto c = mat44_t::from_matrix(b);
        REQUIRE(b.as<real_t>() == c);

        auto d = std::array<real_t, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        auto e = mat44_t::from_pointer(d.data());
        REQUIRE(e == c);
    }

    AND_THEN("Component accesses") {
        REQUIRE((mat23_t::ROWS == 2 and mat23_t::COLS == 3 and mat23_t::DIAG == 2 and mat23_t::SIZE == 2));
        static_assert(std::is_same_v<typename mat23_t::row_type, Vec<real_t, 3>>);
        static_assert(std::is_same_v<typename mat23_t::column_type, Vec<real_t, 2>>);

        auto a = mat23_t::from_values(1, 2, 3, 4, 5, 6);
        auto b = mat23_t{{{1, 2, 3}, {4, 5, 6}}};

        for (i32 i{0}; auto& e: a) {
            REQUIRE((e[0] == static_cast<real_t>(1 + i * 3) and
                     e[1] == static_cast<real_t>(2 + i * 3) and
                     e[2] == static_cast<real_t>(3 + i * 3)));
            i++;
        }
        REQUIRE((a[0][0] == b[0][0] and a[0][1] == b[0][1] and a[0][2] == b[0][2] and
                 a[1][0] == b[1][0] and a[1][1] == b[1][1] and a[1][2] == b[1][2]));
    }

    AND_THEN("Assignment operators") {
        mat33_t test = mat33_t::eye(1);
        test += 1;
        REQUIRE(test == mat33_t::from_values(2, 1, 1, 1, 2, 1, 1, 1, 2));
        test -= 3.5;
        REQUIRE(noa::allclose(test, mat33_t::from_values(-1.5, -2.5, -2.5, -2.5, -1.5, -2.5, -2.5, -2.5, -1.5)));
        test *= -2;
        REQUIRE(noa::allclose(test, mat33_t::from_values(3, 5, 5, 5, 3, 5, 5, 5, 3)));
        test /= 2;
        REQUIRE(noa::allclose(test, mat33_t::from_values(1.5, 2.5, 2.5, 2.5, 1.5, 2.5, 2.5, 2.5, 1.5)));

        mat33_t test1 = mat33_t::from_values(1.23, 3.2, -1.23, 4.2, -0.32, 0.4, -2.34, 3.14, 0.2);
        mat33_t test2(test1);
        mat33_t test3 = mat33_t::from_values(1.13, 1.4, 1.29, -1.2, -0.52, 0.3, 2.04, -3.15, 0.25);
        test2 += test3;
        REQUIRE(noa::allclose(test2, mat33_t::from_values(2.36, 4.6, 0.06, 3., -0.84, 0.7, -0.3, -0.01, 0.45)));
        test2 = test1;
        test2 -= test3;
        REQUIRE(noa::allclose(test2, mat33_t::from_values(0.1, 1.8, -2.52, 5.4, 0.2, 0.1, -4.38, 6.29, -0.05)));
        test2 = test1;
        test2 *= test3;
        REQUIRE(noa::allclose(test2, mat33_t::from_values(
                -4.9593, 3.9325, 2.2392,
                5.946, 4.7864, 5.422,
                -6.0042, -5.5388, -2.0266)));
        test2 = test1;
        test2 /= test3;
        REQUIRE(noa::allclose(test2, mat33_t::from_values(
                -0.240848, -2.468117, -0.715481,
                0.600243, -1.802665, 0.665945,
                0.227985, 0.501578, -0.978299)));
    }

    AND_THEN("Math functions") {
        {
            auto m0 = mat22_t::from_values(2, 3, 1, 5);
            auto m1 = mat22_t::from_values(7, 6, 8, 4);
            REQUIRE(noa::allclose(noa::ewise_multiply(m0, m1), mat22_t::from_values(14, 18, 8, 20)));
            REQUIRE(noa::allclose(noa::outer_product(vec2_t{1, 2}, vec2_t{3, 4}), mat22_t::from_values(3, 4, 6, 8)));
            REQUIRE(noa::allclose(noa::transpose(m0), mat22_t::from_values(2, 1, 3, 5)));
            REQUIRE(noa::allclose(noa::determinant(m0), static_cast<TestType>(7)));
            REQUIRE(noa::allclose(noa::inverse(m1), mat22_t::from_values(-0.2, 0.3, 0.4, -0.35)));

            auto m2 = mat22_t::from_values(0, -1, 1, 0);
            REQUIRE(noa::allclose(noa::inverse(m2), noa::transpose(m2))); // orthogonal: A-1=A.T
            REQUIRE(noa::allclose(noa::inverse(m2), mat22_t::from_values(0, 1, -1, 0))); // orthogonal: A-1=A.T
        }
        {
            mat44_t m0 = mat44_t::from_values(2, 3, 1, 5, 6, 11, 8, 9, 4, 12, 13, 16, 3, 4, 2, 5);
            mat44_t m1 = mat44_t::from_values(1, 3, 2, 5, 7, 6, 8, 12, 4, 3, 7, 13, 15, 2, 1, 11);
            REQUIRE(noa::allclose(noa::ewise_multiply(m0, m1),
                                  mat44_t::from_values(2, 9, 2, 25,
                                                       42, 66, 64, 108,
                                                       16, 36, 91, 208,
                                                       45, 8, 2, 55)));
            REQUIRE(noa::allclose(noa::outer_product(Vec4<TestType>{1, 2, 3, 4}, Vec4<TestType>{5, 6, 7, 8}),
                                  mat44_t::from_values(5, 6, 7, 8, 10, 12, 14, 16, 15, 18, 21, 24, 20, 24, 28, 32)));
            REQUIRE(noa::allclose(noa::transpose(m0),
                                  mat44_t::from_values(2, 6, 4, 3,
                                                       3, 11, 12, 4,
                                                       1, 8, 13, 2,
                                                       5, 9, 16, 5)));
            REQUIRE(noa::allclose(noa::determinant(m0), static_cast<TestType>(-103.99999999999999)));
            REQUIRE(noa::allclose(noa::inverse(m1),
                                  mat44_t::from_values(-0.153245, 0.096169, -0.073495, 0.051603,
                                                       0.347146, 0.108679, -0.221267, -0.014855,
                                                       -0.399531, 0.215012, 0.022674, -0.07975,
                                                       0.182174, -0.170446, 0.138389, 0.030493)));

            mat44_t m2 = mat44_t::from_values(0, 1, 0, 10, -1, 0, 0, 20, 0, 0, 1, 30, 0, 0, 0, 1);
            REQUIRE(noa::allclose(noa::transpose(m2),
                                  mat44_t::from_values(0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1)));
            REQUIRE(noa::allclose(noa::inverse(m2),
                                  mat44_t::from_values(0, -1, 0, 20, 1, 0, 0, -10, 0, 0, 1, -30, 0, 0, 0, 1)));

            auto a0 = std::array<real_t, 5 * 6>{};
            for (real_t i{}; auto& e: a0)
                e = ++i;
            auto a1 = std::array<real_t, 6 * 7>{};
            for (real_t i{}; auto& e: a1)
                e = ++i;

            auto m3 = Mat<real_t, 5, 6>::from_pointer(a0.data());
            auto m4 = Mat<real_t, 6, 7>::from_pointer(a1.data());
            auto a = m3 * m4;
            auto b = matmul(m3, m4);
            REQUIRE(noa::allclose(a, b));
        }
    }

    AND_THEN("Unary operators") {
        auto test = mat22_t::from_values(1, 2, 3, 4);
        REQUIRE(+test == test);
        REQUIRE(-test == test * static_cast<real_t>(-1));
    }

    AND_THEN("Binary operators") {
        // Test against numpy.
        auto tmp2 = mat22_t::from_values(1, 2, 9, 8);
        auto tmp3 = mat22_t::from_values(1, 3, 4, 6);
        REQUIRE(noa::allclose(tmp2 + tmp3, mat22_t::from_values(2, 5, 13, 14)));
        REQUIRE(noa::allclose(tmp2 + real_t(3.4), mat22_t::from_values(4.4, 5.4, 12.4, 11.4)));
        REQUIRE(noa::allclose(real_t(3.4) + tmp2, mat22_t::from_values(4.4, 5.4, 12.4, 11.4)));

        REQUIRE(noa::allclose(tmp3 - tmp2, mat22_t::from_values(0, 1, -5, -2)));
        REQUIRE(noa::allclose(tmp3 - real_t(2.2), mat22_t::from_values(-1.2, 0.8, 1.8, 3.8)));
        REQUIRE(noa::allclose(real_t(-1.3) - tmp3, mat22_t::from_values(-2.3, -4.3, -5.3, -7.3)));

        REQUIRE(noa::allclose(tmp2 * tmp3, mat22_t::from_values(9, 15, 41, 75)));
        REQUIRE(noa::allclose(tmp2 * real_t(2.7), mat22_t::from_values(2.7, 5.4, 24.3, 21.6)));
        REQUIRE(noa::allclose(real_t(2.7) * tmp2, mat22_t::from_values(2.7, 5.4, 24.3, 21.6)));
        REQUIRE(all(noa::allclose(tmp2 * vec2_t::from_values(1.4, 1.5), vec2_t::from_values(4.4, 24.6))));
        REQUIRE(all(noa::allclose(vec2_t::from_values(1.4, 1.5) * tmp2, vec2_t::from_values(14.9, 14.8))));

        REQUIRE(noa::allclose(tmp2 / tmp3, mat22_t::from_values(0.33333333, 0.16666667, -3.66666667, 3.16666667)));
        REQUIRE(noa::allclose(tmp2 / real_t(2.8), mat22_t::from_values(0.35714286, 0.71428571, 3.21428571, 2.85714286)));
        REQUIRE(noa::allclose(real_t(-2.8) / tmp2, mat22_t::from_values(-2.8, -1.4, -0.31111111, -0.35)));
        REQUIRE(all(noa::allclose(tmp3 / vec2_t::from_values(1.4, 1.5), vec2_t::from_values(-0.65, 0.68333333))));
        REQUIRE(all(noa::allclose(vec2_t::from_values(1.4, 1.5) / tmp3, vec2_t::from_values(-0.4, 0.45))));
    }

    AND_THEN("Boolean operators") {
        auto test = mat22_t::from_values(1, 2, 3, 4);
        mat22_t test1(test);
        REQUIRE(test == test1);
        REQUIRE_FALSE(test != test1);
    }

    AND_THEN("Others") {
        REQUIRE(noa::string::stringify<Mat23<f32>>() == "Mat<f32,2,3>");
        REQUIRE(noa::string::stringify<Mat44<f64>>() == "Mat<f64,4,4>");
    }
}
