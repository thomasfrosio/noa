#include <noa/core/Types.hpp>
#include <noa/core/geometry/Transform.hpp>

#include <catch2/catch.hpp>

using namespace ::noa;

TEMPLATE_TEST_CASE("core::Mat22", "[noa][core]", double, float) {
    using Mat = Mat22<TestType>;

    AND_THEN("Component accesses") {
        REQUIRE(Mat::ROWS == 2);
        REQUIRE(Mat::COLS == 2);
        REQUIRE(Mat::COUNT == 4);

        Mat test(1, 2, 3, 4);
        REQUIRE((test[0][0] == 1 && test[0][1] == 2 && test[1][0] == 3 && test[1][1] == 4));
    }

    AND_THEN("(Conversion) Constructors") {
        Mat test;
        REQUIRE(Mat(1, 0, 0, 1) == test);
        REQUIRE(Mat(3, 0, 0, 3) == Mat(3));
        REQUIRE(Mat(1, 0, 0, 2) == Mat(Vec2<TestType>(1, 2)));

        Mat tmp0(1, 2, 3, 4);
        Mat33<TestType> tmp1(1, 2, 0, 3, 4, 0, 0, 0, 0);
        REQUIRE(tmp0 == noa::geometry::affine2linear(tmp1));
        REQUIRE(Mat(Vec2<TestType>(1, 2),
                    Vec2<TestType>(3, 4)) == tmp0);
        REQUIRE(Mat(Vec2<int>(1, 2),
                    Vec2<int>(3, 4)) == tmp0);
    }

    AND_THEN("Assignment operators") {
        Mat test;
        test += 1;
        REQUIRE(test == Mat(2, 1, 1, 2));
        test -= 3.5;
        REQUIRE(math::are_almost_equal(test, Mat(-1.5, -2.5, -2.5, -1.5)));
        test *= -2;
        REQUIRE(math::are_almost_equal(test, Mat(3, 5, 5, 3)));
        test /= 2;
        REQUIRE(math::are_almost_equal(test, Mat(1.5, 2.5, 2.5, 1.5)));

        Mat test1(1.23, 3.2, -1.23, 4.2);
        Mat test2(test1);
        Mat test3(1.13, 1.4, 1.29, -1.2);
        test2 += test3;
        REQUIRE(math::are_almost_equal(test2, Mat(2.36, 4.6, 0.06, 3.)));
        test2 = test1;
        test2 -= test3;
        REQUIRE(math::are_almost_equal(test2, Mat(0.1, 1.8, -2.52, 5.4)));
        test2 = test1;
        test2 *= test3;
        REQUIRE(math::are_almost_equal(test2, Mat(5.5179, -2.118, 4.0281, -6.762)));
        test2 = test1;
        test2 = test2 / test3;
        REQUIRE(math::are_almost_equal(test2, Mat(1.77229602, -0.59898798, 1.24667932, -2.0455408)));
    }

    AND_THEN("Math functions") {
        Mat tmp1(2, 3, 1, 5);
        Mat tmp2(7, 6, 8, 4);
        REQUIRE(math::are_almost_equal(math::element_multiply(tmp1, tmp2), Mat(14, 18, 8, 20)));
        REQUIRE(math::are_almost_equal(math::outer_product(Vec2<TestType>(1, 2), Vec2<TestType>(3, 4)), Mat(3, 4, 6, 8)));
        REQUIRE(math::are_almost_equal(math::transpose(tmp1), Mat(2, 1, 3, 5)));
        REQUIRE(math::are_almost_equal(math::determinant(tmp1), static_cast<TestType>(7)));
        REQUIRE(math::are_almost_equal(math::inverse(tmp2), Mat(-0.2, 0.3, 0.4, -0.35)));

        Mat test1(0, -1, 1, 0);
        REQUIRE(math::are_almost_equal(math::inverse(test1), math::transpose(test1))); // orthogonal: A-1=A.T
        REQUIRE(math::are_almost_equal(math::inverse(test1), Mat(0, 1, -1, 0))); // orthogonal: A-1=A.T
    }

    AND_THEN("Unary operators") {
        Mat test(1, 2, 3, 4);
        REQUIRE(+test == test);
        REQUIRE(-test == test * static_cast<TestType>(-1));
    }

    AND_THEN("Binary operators") {
        // Test against numpy.
        Mat tmp2(1, 2, 9, 8);
        Mat tmp3(1, 3, 4, 6);
        REQUIRE(math::are_almost_equal(tmp2 + tmp3, Mat(2, 5, 13, 14)));
        REQUIRE(math::are_almost_equal(tmp2 + TestType(3.4), Mat(4.4, 5.4, 12.4, 11.4)));
        REQUIRE(math::are_almost_equal(TestType(3.4) + tmp2, Mat(4.4, 5.4, 12.4, 11.4)));

        REQUIRE(math::are_almost_equal(tmp3 - tmp2, Mat(0, 1, -5, -2)));
        REQUIRE(math::are_almost_equal(tmp3 - TestType(2.2), Mat(-1.2, 0.8, 1.8, 3.8)));
        REQUIRE(math::are_almost_equal(TestType(-1.3) - tmp3, Mat(-2.3, -4.3, -5.3, -7.3)));

        REQUIRE(math::are_almost_equal(tmp2 * tmp3, Mat(9, 15, 41, 75)));
        REQUIRE(math::are_almost_equal(tmp2 * TestType(2.7), Mat(2.7, 5.4, 24.3, 21.6)));
        REQUIRE(math::are_almost_equal(TestType(2.7) * tmp2, Mat(2.7, 5.4, 24.3, 21.6)));
        REQUIRE(all(math::are_almost_equal(tmp2 * Vec2<TestType>(1.4, 1.5), Vec2<TestType>(4.4, 24.6))));
        REQUIRE(all(math::are_almost_equal(Vec2<TestType>(1.4, 1.5) * tmp2, Vec2<TestType>(14.9, 14.8))));

        REQUIRE(math::are_almost_equal(tmp2 / tmp3, Mat(0.33333333, 0.16666667, -3.66666667, 3.16666667)));
        REQUIRE(math::are_almost_equal(tmp2 / TestType(2.8), Mat(0.35714286, 0.71428571, 3.21428571, 2.85714286)));
        REQUIRE(math::are_almost_equal(TestType(-2.8) / tmp2, Mat(-2.8, -1.4, -0.31111111, -0.35)));
        REQUIRE(all(math::are_almost_equal(tmp3 / Vec2<TestType>(1.4, 1.5), Vec2<TestType>(-0.65, 0.68333333))));
        REQUIRE(all(math::are_almost_equal(Vec2<TestType>(1.4, 1.5) / tmp3, Vec2<TestType>(-0.4, 0.45))));
    }

    AND_THEN("Boolean operators") {
        Mat test(1, 2, 3, 4);
        Mat test1(test);
        REQUIRE(test == test1);
        REQUIRE_FALSE(test != test1);
    }

    AND_THEN("Others") {
        REQUIRE(string::human<Float22>() == "Float22");
        REQUIRE(string::human<Double22>() == "Double22");
    }
}

TEMPLATE_TEST_CASE("core::Mat23", "[noa][core]", double, float) {
    using Mat = Mat23<TestType>;

    AND_THEN("Component accesses") {
        REQUIRE(Mat::ROWS == 2);
        REQUIRE(Mat::COLS == 3);
        REQUIRE(Mat::COUNT == 6);

        Mat test(1, 2, 3, 4, 5, 6);
        REQUIRE((test[0][0] == 1 && test[0][1] == 2 && test[0][2] == 3 &&
                 test[1][0] == 4 && test[1][1] == 5 && test[1][2] == 6));
    }

    AND_THEN("(Conversion) Constructors") {
        Mat test;
        REQUIRE(Mat(1, 0, 0, 0, 1, 0) == test);
        REQUIRE(Mat(3, 0, 0, 0, 3, 0) == Mat(3));
        REQUIRE(Mat(1, 0, 0, 0, 2, 0) == Mat(Vec2<TestType>(1, 2)));

        Mat tmp0(1, 2, 3, 4, 5, 6);
        Mat33<TestType> tmp1(1, 2, 3, 4, 5, 6, 7, 8, 9);
        REQUIRE(tmp0 == noa::geometry::affine2truncated(tmp1));
        Mat22<TestType> tmp2(1, 2, 3, 4);
        REQUIRE(Mat(1, 2, 0, 3, 4, 0) == noa::geometry::linear2truncated(tmp2));
        REQUIRE(Mat(1, 2, -1, 3, 4, -2) == noa::geometry::linear2truncated(tmp2, Vec2<TestType>(-1, -2)));
        REQUIRE(Mat(Vec3<TestType>(1, 2, 3),
                    Vec3<TestType>(4, 5, 6)) == tmp0);
        REQUIRE(Mat(Vec3<int>(1, 2, 3),
                    Vec3<int>(4, 5, 6)) == tmp0);
    }

    AND_THEN("Assignment operators") {
        Mat test;
        test += 1;
        REQUIRE(test == Mat(2, 1, 1, 1, 2, 1));
        test -= 3.5;
        REQUIRE(math::are_almost_equal(test, Mat(-1.5, -2.5, -2.5, -2.5, -1.5, -2.5)));
        test *= -2;
        REQUIRE(math::are_almost_equal(test, Mat(3, 5, 5, 5, 3, 5)));
        test /= 2;
        REQUIRE(math::are_almost_equal(test, Mat(1.5, 2.5, 2.5, 2.5, 1.5, 2.5)));

        Mat test1(1.23, 3.2, -1.23, 4.2, -0.32, 0.4);
        Mat test2(test1);
        Mat test3(1.13, 1.4, 1.29, -1.2, -0.52, 0.3);
        test2 += test3;
        REQUIRE(math::are_almost_equal(test2, Mat(2.36, 4.6, 0.06, 3., -0.84, 0.7)));
        test2 = test1;
        test2 -= test3;
        REQUIRE(math::are_almost_equal(test2, Mat(0.1, 1.8, -2.52, 5.4, 0.2, 0.1)));
    }

    AND_THEN("Math functions") {
        Mat tmp1(2, 3, 1, 5, 6, 11);
        Mat tmp2(1, 3, 2, 5, 7, 6);
        REQUIRE(math::are_almost_equal(math::element_multiply(tmp1, tmp2), Mat(2, 9, 2, 25, 42, 66)));
        REQUIRE(math::are_almost_equal(tmp1, Mat(2, 3, 1, 5, 6, 11)));
    }

    AND_THEN("Unary operators") {
        Mat test(1, 2, 3, 4, 5, 6);
        REQUIRE(+test == test);
        REQUIRE(-test == test * static_cast<TestType>(-1));
    }

    AND_THEN("Binary operators") {
        // Test against numpy.
        Mat tmp2(1, 2, 3, 9, 8, 7);
        Mat tmp3(1, 2, 3, 6, 5, 4);
        REQUIRE(math::are_almost_equal(tmp2 + tmp3, Mat(2, 4, 6, 15, 13, 11)));
        REQUIRE(math::are_almost_equal(tmp2 + TestType(3.4), Mat(4.4, 5.4, 6.4, 12.4, 11.4, 10.4)));
        REQUIRE(math::are_almost_equal(TestType(3.4) + tmp2, Mat(4.4, 5.4, 6.4, 12.4, 11.4, 10.4)));

        REQUIRE(math::are_almost_equal(tmp3 - tmp2, Mat(0, 0, 0, -3, -3, -3)));
        REQUIRE(math::are_almost_equal(tmp3 - TestType(2.2), Mat(-1.2, -0.2, 0.8, 3.8, 2.8, 1.8)));
        REQUIRE(math::are_almost_equal(TestType(-1.3) - tmp3, Mat(-2.3, -3.3, -4.3, -7.3, -6.3, -5.3)));

        REQUIRE(math::are_almost_equal(tmp2 * TestType(2.7), Mat(2.7, 5.4, 8.1, 24.3, 21.6, 18.9)));
        REQUIRE(math::are_almost_equal(TestType(2.7) * tmp2, Mat(2.7, 5.4, 8.1, 24.3, 21.6, 18.9)));
        REQUIRE(all(math::are_almost_equal(tmp2 * Vec3<TestType>(1.4, 1.5, 1.6), Vec2<TestType>(9.2, 35.8))));
        REQUIRE(all(math::are_almost_equal(Vec2<TestType>(1.4, 1.5) * tmp2, Vec3<TestType>(14.9, 14.8, 14.7))));

        REQUIRE(math::are_almost_equal(tmp2 / TestType(2.8), Mat(0.35714286, 0.71428571, 1.07142857,
                                                        3.21428571, 2.85714286, 2.5)));
        REQUIRE(math::are_almost_equal(TestType(-2.8) / tmp2, Mat(-2.8, -1.4, -0.93333333,
                                                         -0.31111111, -0.35, -0.4)));
    }

    AND_THEN("Boolean operators") {
        Mat test(1, 2, 3, 4, 5, 6);
        Mat test1(test);
        REQUIRE(test == test1);
        REQUIRE_FALSE(test != test1);
    }

    AND_THEN("Others") {
        REQUIRE(string::human<Float23>() == "Float23");
        REQUIRE(string::human<Double23>() == "Double23");
    }
}

TEMPLATE_TEST_CASE("core::Mat33", "[noa][core]", double, float) {
    using Mat = Mat33<TestType>;

    AND_THEN("Component accesses") {
        REQUIRE(Mat::ROWS == 3);
        REQUIRE(Mat::COLS == 3);
        REQUIRE(Mat::COUNT == 9);

        Mat test(1, 2, 3, 4, 5, 6, 7, 8, 10);
        REQUIRE((test[0][0] == 1 && test[0][1] == 2 && test[0][2] == 3 &&
                 test[1][0] == 4 && test[1][1] == 5 && test[1][2] == 6 &&
                 test[2][0] == 7 && test[2][1] == 8 && test[2][2] == 10));
    }

    AND_THEN("(Conversion) Constructors") {
        Mat test;
        REQUIRE(Mat(1, 0, 0, 0, 1, 0, 0, 0, 1) == test);
        REQUIRE(Mat(3, 0, 0, 0, 3, 0, 0, 0, 3) == Mat(3));
        REQUIRE(Mat(1, 0, 0, 0, 2, 0, 0, 0, 3) == Mat(Vec3<TestType>(1, 2, 3)));

        Mat tmp0(1, 2, 3, 4, 5, 6, 7, 8, 9);
        Mat44<TestType> tmp1(1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 0, 0, 0, 0);
        REQUIRE(tmp0 == noa::geometry::affine2linear(tmp1));
        Mat22<TestType> tmp2(1, 2, 3, 4);
        REQUIRE(Mat(1, 2, 0, 3, 4, 0, 0, 0, 1) == noa::geometry::linear2affine(tmp2));
        REQUIRE(Mat(1, 2, -1, 3, 4, -2, 0, 0, 1) == noa::geometry::linear2affine(tmp2, Vec2<TestType>(-1, -2)));
        REQUIRE(Mat(Vec3<TestType>(1, 2, 3),
                    Vec3<TestType>(4, 5, 6),
                    Vec3<TestType>(7, 8, 9)) == tmp0);
        REQUIRE(Mat(Vec3<int>(1, 2, 3),
                    Vec3<int>(4, 5, 6),
                    Vec3<int>(7, 8, 9)) == tmp0);
    }

    AND_THEN("Assignment operators") {
        Mat test;
        test += 1;
        REQUIRE(test == Mat(2, 1, 1, 1, 2, 1, 1, 1, 2));
        test -= 3.5;
        REQUIRE(math::are_almost_equal(test, Mat(-1.5, -2.5, -2.5, -2.5, -1.5, -2.5, -2.5, -2.5, -1.5)));
        test *= -2;
        REQUIRE(math::are_almost_equal(test, Mat(3, 5, 5, 5, 3, 5, 5, 5, 3)));
        test /= 2;
        REQUIRE(math::are_almost_equal(test, Mat(1.5, 2.5, 2.5, 2.5, 1.5, 2.5, 2.5, 2.5, 1.5)));

        Mat test1(1.23, 3.2, -1.23, 4.2, -0.32, 0.4, -2.34, 3.14, 0.2);
        Mat test2(test1);
        Mat test3(1.13, 1.4, 1.29, -1.2, -0.52, 0.3, 2.04, -3.15, 0.25);
        test2 += test3;
        REQUIRE(math::are_almost_equal(test2, Mat(2.36, 4.6, 0.06, 3., -0.84, 0.7, -0.3, -0.01, 0.45)));
        test2 = test1;
        test2 -= test3;
        REQUIRE(math::are_almost_equal(test2, Mat(0.1, 1.8, -2.52, 5.4, 0.2, 0.1, -4.38, 6.29, -0.05)));
        test2 = test1;
        test2 *= test3;
        REQUIRE(math::are_almost_equal(test2, Mat(-4.9593, 3.9325, 2.2392,
                                         5.946, 4.7864, 5.422,
                                         -6.0042, -5.5388, -2.0266)));
        test2 = test1;
        test2 /= test3;
        REQUIRE(math::are_almost_equal(test2, Mat(-0.240848, -2.468117, -0.715481,
                                         0.600243, -1.802665, 0.665945,
                                         0.227985, 0.501578, -0.978299)));
    }

    AND_THEN("Math functions") {
        Mat tmp1(2, 3, 1, 5, 6, 11, 8, 9, 4);
        Mat tmp2(1, 3, 2, 5, 7, 6, 8, 12, 4);
        REQUIRE(math::are_almost_equal(math::element_multiply(tmp1, tmp2), Mat(2, 9, 2, 25, 42, 66, 64, 108, 16)));
        REQUIRE(math::are_almost_equal(math::outer_product(Vec3<TestType>(1, 2, 3), Vec3<TestType>(4, 5, 6)),
                              Mat(4, 5, 6, 8, 10, 12, 12, 15, 18)));
        REQUIRE(math::are_almost_equal(math::transpose(tmp1), Mat(2, 5, 8, 3, 6, 9, 1, 11, 4)));
        REQUIRE(math::are_almost_equal(math::determinant(tmp1), static_cast<TestType>(51)));
        REQUIRE(math::are_almost_equal(math::inverse(tmp2), Mat(-0.916667, 0.25, 0.083333,
                                                       0.583333, -0.25, 0.083333,
                                                       0.083333, 0.25, -0.166667)));

        Mat test1(1, 0, 0, 0, 0, -1, 0, 1, 0); // rotate X by 90deg
        REQUIRE(math::are_almost_equal(math::inverse(test1), math::transpose(test1))); // orthogonal, rotate X by -90deg
        REQUIRE(math::are_almost_equal(math::inverse(test1), Mat(1, 0, 0, 0, 0, 1, 0, -1, 0))); // orthogonal: A-1=A.T
    }

    AND_THEN("Unary operators") {
        Mat test(1, 2, 3, 4, 5, 6, 7, 8, 9);
        REQUIRE(+test == test);
        REQUIRE(-test == test * static_cast<TestType>(-1));
    }

    AND_THEN("Binary operators") {
        // Test against numpy.
        Mat tmp2(1, 2, 3, 9, 8, 7, 4, 5, 6);
        Mat tmp3(1, 2, 3, 6, 5, 4, 9, 8, 7);
        REQUIRE(math::are_almost_equal(tmp2 + tmp3, Mat(2, 4, 6, 15, 13, 11, 13, 13, 13)));
        REQUIRE(math::are_almost_equal(tmp2 + TestType(3.4), Mat(4.4, 5.4, 6.4, 12.4, 11.4, 10.4, 7.4, 8.4, 9.4)));
        REQUIRE(math::are_almost_equal(TestType(3.4) + tmp2, Mat(4.4, 5.4, 6.4, 12.4, 11.4, 10.4, 7.4, 8.4, 9.4)));

        REQUIRE(math::are_almost_equal(tmp3 - tmp2, Mat(0, 0, 0, -3, -3, -3, 5, 3, 1)));
        REQUIRE(math::are_almost_equal(tmp3 - TestType(2.2), Mat(-1.2, -0.2, 0.8, 3.8, 2.8, 1.8, 6.8, 5.8, 4.8)));
        REQUIRE(math::are_almost_equal(TestType(-1.3) - tmp3, Mat(-2.3, -3.3, -4.3, -7.3, -6.3, -5.3, -10.3, -9.3, -8.3)));

        REQUIRE(math::are_almost_equal(tmp2 * tmp3, Mat(40, 36, 32, 120, 114, 108, 88, 81, 74)));
        REQUIRE(math::are_almost_equal(tmp2 * TestType(2.7), Mat(2.7, 5.4, 8.1, 24.3, 21.6, 18.9, 10.8, 13.5, 16.2)));
        REQUIRE(math::are_almost_equal(TestType(2.7) * tmp2, Mat(2.7, 5.4, 8.1, 24.3, 21.6, 18.9, 10.8, 13.5, 16.2)));
        REQUIRE(all(math::are_almost_equal(tmp2 * Vec3<TestType>(1.4, 1.5, 1.6), Vec3<TestType>(9.2, 35.8, 22.7))));
        REQUIRE(all(math::are_almost_equal(Vec3<TestType>(1.4, 1.5, 1.6) * tmp2, Vec3<TestType>(21.3, 22.8, 24.3))));

        Mat tmp4(1, 2, 3, 2, 5, 4, 9, 2, 7); // not singular
        REQUIRE(math::are_almost_equal(tmp2 / tmp4, Mat(1.00000000, 2.22044605e-16, -1.04083409e-17,
                                               -2.53846154, 2.30769231, 7.69230769e-01,
                                               5.38461538e-01, 6.92307692e-01, 2.30769231e-01)));
        REQUIRE(math::are_almost_equal(tmp2 / TestType(2.8), Mat(0.35714286, 0.71428571, 1.07142857,
                                                        3.21428571, 2.85714286, 2.5,
                                                        1.42857143, 1.78571429, 2.14285714)));
        REQUIRE(math::are_almost_equal(TestType(-2.8) / tmp2, Mat(-2.8, -1.4, -0.93333333,
                                                         -0.31111111, -0.35, -0.4,
                                                         -0.7, -0.56, -0.46666667)));
        REQUIRE(all(math::are_almost_equal(tmp4 / Vec3<TestType>(1.4, 1.5, 1.6), Vec3<TestType>(-0.28076923,
                                                                                           -0.07692308,
                                                                                           0.61153846))));
        REQUIRE(all(math::are_almost_equal(Vec3<TestType>(1.4, 1.5, 1.6) / tmp4, Vec3<TestType>(-0.1, 0.3, 0.1))));
    }

    AND_THEN("Boolean operators") {
        Mat test(1, 2, 3, 4, 5, 6, 7, 8, 9);
        Mat test1(test);
        REQUIRE(test == test1);
        REQUIRE_FALSE(test != test1);
    }

    AND_THEN("Others") {
        REQUIRE(string::human<Float33>() == "Float33");
        REQUIRE(string::human<Double33>() == "Double33");
    }
}

TEMPLATE_TEST_CASE("core::Mat34", "[noa][core]", double, float) {
    using Mat = Mat34<TestType>;

    AND_THEN("Component accesses") {
        REQUIRE(Mat::ROWS == 3);
        REQUIRE(Mat::COLS == 4);
        REQUIRE(Mat::COUNT == 12);

        Mat test(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        REQUIRE((test[0][0] == 1 && test[0][1] == 2 && test[0][2] == 3 && test[0][3] == 4 &&
                 test[1][0] == 5 && test[1][1] == 6 && test[1][2] == 7 && test[1][3] == 8 &&
                 test[2][0] == 9 && test[2][1] == 10 && test[2][2] == 11 && test[2][3] == 12));
    }

    AND_THEN("(Conversion) Constructors") {
        Mat test;
        REQUIRE(Mat(1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0) == test);
        REQUIRE(Mat(3, 0, 0, 0,
                    0, 3, 0, 0,
                    0, 0, 3, 0) == Mat(3));
        REQUIRE(Mat(1, 0, 0, 0,
                    0, 2, 0, 0,
                    0, 0, 3, 0) == Mat(Vec3<TestType>(1, 2, 3)));

        Mat33<TestType> tmp0(1, 2, 3, 4, 5, 6, 7, 8, 9);
        REQUIRE(Mat(1, 2, 3, 0,
                    4, 5, 6, 0,
                    7, 8, 9, 0) == noa::geometry::linear2truncated(tmp0));
        REQUIRE(Mat(1, 2, 3, -1,
                    4, 5, 6, -2,
                    7, 8, 9, -3) == noa::geometry::linear2truncated(tmp0, Vec3<TestType>(-1, -2, -3)));
        Mat tmp1(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        REQUIRE(Mat(Vec4<TestType>(1, 2, 3, 4),
                    Vec4<TestType>(5, 6, 7, 8),
                    Vec4<TestType>(9, 10, 11, 12)) == tmp1);
        REQUIRE(Mat(Vec4<int>(1, 2, 3, 4),
                    Vec4<int>(5, 6, 7, 8),
                    Vec4<int>(9, 10, 11, 12)) == tmp1);
    }

    AND_THEN("Assignment operators") {
        Mat test;
        test += 1;
        REQUIRE(test == Mat(2, 1, 1, 1,
                            1, 2, 1, 1,
                            1, 1, 2, 1));
        test -= 3.5;
        REQUIRE(math::are_almost_equal(test, Mat(-1.5, -2.5, -2.5, -2.5,
                                        -2.5, -1.5, -2.5, -2.5,
                                        -2.5, -2.5, -1.5, -2.5)));
        test *= -2;
        REQUIRE(math::are_almost_equal(test, Mat(3, 5, 5, 5,
                                        5, 3, 5, 5,
                                        5, 5, 3, 5)));
        test /= 2;
        REQUIRE(math::are_almost_equal(test, Mat(1.5, 2.5, 2.5, 2.5,
                                        2.5, 1.5, 2.5, 2.5,
                                        2.5, 2.5, 1.5, 2.5)));

        Mat test1(1.23, 3.2, -1.23, 4.2,
                  -0.32, 0.4, -2.34, 3.14,
                  0.2, 3.86, -0.91, 1.23);
        Mat test2(test1);
        Mat test3(1.13, 1.4, 1.29, -1.2,
                  -0.52, 0.3, 2.04, -3.15,
                  0.25, 3.86, -0.91, 3.4);
        test2 += test3;
        REQUIRE(math::are_almost_equal(test2, Mat(2.36, 4.6, 0.06, 3.,
                                         -0.84, 0.7, -0.3, -0.01,
                                         0.45, 7.72, -1.82, 4.63)));
        test2 = test1;
        test2 -= test3;
        REQUIRE(math::are_almost_equal(test2, Mat(0.1, 1.8, -2.52, 5.4,
                                         0.2, 0.1, -4.38, 6.29,
                                         -0.05, 0., 0., -2.17)));
    }

    AND_THEN("Math functions") {
        Mat tmp1(2, 3, 1, 5, 6, 11, 8, 9, 4, 12, 13, 16);
        Mat tmp2(1, 3, 2, 5, 7, 6, 8, 12, 4, 3, 7, 13);
        REQUIRE(math::are_almost_equal(math::element_multiply(tmp1, tmp2), Mat(2, 9, 2, 25,
                                                                     42, 66, 64, 108,
                                                                     16, 36, 91, 208)));
        REQUIRE(math::are_almost_equal(tmp1, Mat(2, 3, 1, 5, 6, 11, 8, 9, 4, 12, 13, 16)));
    }

    AND_THEN("Unary operators") {
        Mat test(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        REQUIRE(+test == test);
        REQUIRE(-test == test * static_cast<TestType>(-1));
    }

    AND_THEN("Binary operators") {
        // Test against numpy.
        Mat tmp2(1, 2, 3, 9,
                 8, 7, 4, 5,
                 6, 12, 9, 15);
        Mat tmp3(1, 2, 3, 6, 5, 4, 9, 8, 7, 1, 9, 11);
        REQUIRE(math::are_almost_equal(tmp2 + tmp3, Mat(2, 4, 6, 15,
                                               13, 11, 13, 13,
                                               13, 13, 18, 26)));
        REQUIRE(math::are_almost_equal(tmp2 + TestType(3.4), Mat(4.4, 5.4, 6.4, 12.4,
                                                        11.4, 10.4, 7.4, 8.4,
                                                        9.4, 15.4, 12.4, 18.4)));
        REQUIRE(math::are_almost_equal(TestType(3.4) + tmp2, Mat(4.4, 5.4, 6.4, 12.4,
                                                        11.4, 10.4, 7.4, 8.4,
                                                        9.4, 15.4, 12.4, 18.4)));

        REQUIRE(math::are_almost_equal(tmp3 - tmp2, Mat(0, 0, 0, -3,
                                               -3, -3, 5, 3,
                                               1, -11, 0, -4)));
        REQUIRE(math::are_almost_equal(tmp3 - TestType(2.2), Mat(-1.2, -0.2, 0.8, 3.8,
                                                        2.8, 1.8, 6.8, 5.8,
                                                        4.8, -1.2, 6.8, 8.8)));
        REQUIRE(math::are_almost_equal(TestType(-1.3) - tmp3, Mat(-2.3, -3.3, -4.3, -7.3,
                                                         -6.3, -5.3, -10.3, -9.3,
                                                         -8.3, -2.3, -10.3, -12.3)));

        REQUIRE(math::are_almost_equal(tmp2 * TestType(2.7), Mat(2.7, 5.4, 8.1, 24.3,
                                                        21.6, 18.9, 10.8, 13.5,
                                                        16.2, 32.4, 24.3, 40.5)));
        REQUIRE(math::are_almost_equal(TestType(2.7) * tmp2, Mat(2.7, 5.4, 8.1, 24.3,
                                                        21.6, 18.9, 10.8, 13.5,
                                                        16.2, 32.4, 24.3, 40.5)));
        REQUIRE(all(math::are_almost_equal(tmp2 * Vec4<TestType>(1.4, 1.5, 1.6, 1.7),
                                  Vec3<TestType>(24.5, 36.6, 66.3))));
        REQUIRE(all(math::are_almost_equal(Vec3<TestType>(1.4, 1.5, 1.6) * tmp2,
                                  Vec4<TestType>(23, 32.5, 24.6, 44.1))));

        REQUIRE(math::are_almost_equal(tmp2 / TestType(2.8), Mat(0.357143, 0.714286, 1.071429, 3.214286,
                                                        2.857143, 2.5, 1.428571, 1.785714,
                                                        2.142857, 4.285714, 3.214286, 5.357143)));
        REQUIRE(math::are_almost_equal(TestType(-2.8) / tmp2, Mat(-2.8, -1.4, -0.933333, -0.311111,
                                                         -0.35, -0.4, -0.7, -0.56,
                                                         -0.466667, -0.233333, -0.311111, -0.186667)));
    }

    AND_THEN("Boolean operators") {
        Mat test(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        Mat test1(test);
        REQUIRE(test == test1);
        REQUIRE_FALSE(test != test1);
    }

    AND_THEN("Others") {
        REQUIRE(string::human<Float34>() == "Float34");
        REQUIRE(string::human<Double34>() == "Double34");
    }
}


TEMPLATE_TEST_CASE("core::Mat44", "[noa][core]", double, float) {
    using Mat = Mat44<TestType>;

    AND_THEN("Component accesses") {
        REQUIRE(Mat::ROWS == 4);
        REQUIRE(Mat::COLS == 4);
        REQUIRE(Mat::COUNT == 16);

        Mat test(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        REQUIRE((test[0][0] == 1 && test[0][1] == 2 && test[0][2] == 3 && test[0][3] == 4 &&
                 test[1][0] == 5 && test[1][1] == 6 && test[1][2] == 7 && test[1][3] == 8 &&
                 test[2][0] == 9 && test[2][1] == 10 && test[2][2] == 11 && test[2][3] == 12 &&
                 test[3][0] == 13 && test[3][1] == 14 && test[3][2] == 15 && test[3][3] == 16));
    }

    AND_THEN("(Conversion) Constructors") {
        Mat test;
        REQUIRE(Mat(1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1) == test);
        REQUIRE(Mat(3, 0, 0, 0,
                    0, 3, 0, 0,
                    0, 0, 3, 0,
                    0, 0, 0, 3) == Mat(3));
        REQUIRE(Mat(1, 0, 0, 0,
                    0, 2, 0, 0,
                    0, 0, 3, 0,
                    0, 0, 0, 4) == Mat(Vec4<TestType>(1, 2, 3, 4)));

        Mat33<TestType> tmp0(1, 2, 3, 4, 5, 6, 7, 8, 9);
        REQUIRE(Mat(1, 2, 3, 0,
                    4, 5, 6, 0,
                    7, 8, 9, 0,
                    0, 0, 0, 1) == noa::geometry::linear2affine(tmp0));
        REQUIRE(Mat(1, 2, 3, -1,
                    4, 5, 6, -2,
                    7, 8, 9, -3,
                    0, 0, 0, 1) == noa::geometry::linear2affine(tmp0, Vec3<TestType>(-1, -2, -3)));
        Mat tmp1(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        REQUIRE(Mat(Vec4<TestType>(1, 2, 3, 4),
                    Vec4<TestType>(5, 6, 7, 8),
                    Vec4<TestType>(9, 10, 11, 12),
                    Vec4<TestType>(13, 14, 15, 16)) == tmp1);
        REQUIRE(Mat(Vec4<int>(1, 2, 3, 4),
                    Vec4<int>(5, 6, 7, 8),
                    Vec4<int>(9, 10, 11, 12),
                    Vec4<int>(13, 14, 15, 16)) == tmp1);
    }

    AND_THEN("Assignment operators") {
        Mat test;
        test += 1;
        REQUIRE(test == Mat(2, 1, 1, 1,
                            1, 2, 1, 1,
                            1, 1, 2, 1,
                            1, 1, 1, 2));
        test -= 3.5;
        REQUIRE(math::are_almost_equal(test, Mat(-1.5, -2.5, -2.5, -2.5,
                                        -2.5, -1.5, -2.5, -2.5,
                                        -2.5, -2.5, -1.5, -2.5,
                                        -2.5, -2.5, -2.5, -1.5)));
        test *= -2;
        REQUIRE(math::are_almost_equal(test, Mat(3, 5, 5, 5,
                                        5, 3, 5, 5,
                                        5, 5, 3, 5,
                                        5, 5, 5, 3)));
        test /= 2;
        REQUIRE(math::are_almost_equal(test, Mat(1.5, 2.5, 2.5, 2.5,
                                        2.5, 1.5, 2.5, 2.5,
                                        2.5, 2.5, 1.5, 2.5,
                                        2.5, 2.5, 2.5, 1.5)));

        Mat test1(1.23, 3.2, -1.23, 4.2,
                  -0.32, 0.4, -2.34, 3.14,
                  0.2, 3.86, -0.91, 1.23,
                  6.32, -2., -0.5, 1.1);
        Mat test2(test1);
        Mat test3(1.13, 1.4, 1.29, -1.2,
                  -0.52, 0.3, 2.04, -3.15,
                  0.25, 3.86, -0.91, 3.4,
                  2.34, 9, -2.14, -8.34);
        test2 += test3;
        REQUIRE(math::are_almost_equal(test2, Mat(2.36, 4.6, 0.06, 3.,
                                         -0.84, 0.7, -0.3, -0.01,
                                         0.45, 7.72, -1.82, 4.63,
                                         8.66, 7., -2.64, -7.24)));
        test2 = test1;
        test2 -= test3;
        REQUIRE(math::are_almost_equal(test2, Mat(0.1, 1.8, -2.52, 5.4,
                                         0.2, 0.1, -4.38, 6.29,
                                         -0.05, 0., 0., -2.17,
                                         3.98, -11., 1.64, 9.44)));
        test2 = test1;
        test2 *= test3;
        REQUIRE(math::are_almost_equal(test2, Mat(9.2464, 35.7342, 0.246, -50.766,
                                         6.193, 18.8996, -4.187, -35.0196,
                                         0.8695, 8.9954, 6.3283, -25.7512,
                                         10.6306, 16.218, 2.1738, -12.158)));
        test2 = test1;
        test2 /= test3;
        REQUIRE(math::are_almost_equal(test2, Mat(0.656266, -0.729289, 0.717642, -0.030009,
                                         -0.685905, -0.537112, 0.291653, 0.043958,
                                         -0.192292, 0.148722, 0.752878, 0.130943,
                                         4.30265, -3.762953, -1.71699, -0.029694)));
    }

    AND_THEN("Math functions") {
        Mat tmp1(2, 3, 1, 5, 6, 11, 8, 9, 4, 12, 13, 16, 3, 4, 2, 5);
        Mat tmp2(1, 3, 2, 5, 7, 6, 8, 12, 4, 3, 7, 13, 15, 2, 1, 11);
        REQUIRE(math::are_almost_equal(math::element_multiply(tmp1, tmp2), Mat(2, 9, 2, 25,
                                                                               42, 66, 64, 108,
                                                                               16, 36, 91, 208,
                                                                               45, 8, 2, 55)));
        REQUIRE(math::are_almost_equal(math::outer_product(Vec4<TestType>(1, 2, 3, 4), Vec4<TestType>(5, 6, 7, 8)),
                                       Mat(5, 6, 7, 8, 10, 12, 14, 16, 15, 18, 21, 24, 20, 24, 28, 32)));
        REQUIRE(math::are_almost_equal(math::transpose(tmp1), Mat(2, 6, 4, 3,
                                                                  3, 11, 12, 4,
                                                                  1, 8, 13, 2,
                                                                  5, 9, 16, 5)));
        REQUIRE(math::are_almost_equal(math::determinant(tmp1), static_cast<TestType>(-103.99999999999999)));
        REQUIRE(math::are_almost_equal(math::inverse(tmp2), Mat(-0.153245, 0.096169, -0.073495, 0.051603,
                                                                0.347146, 0.108679, -0.221267, -0.014855,
                                                                -0.399531, 0.215012, 0.022674, -0.07975,
                                                                0.182174, -0.170446, 0.138389, 0.030493)));

        Mat test1(0, 1, 0, 10, -1, 0, 0, 20, 0, 0, 1, 30, 0, 0, 0, 1);
        REQUIRE(math::are_almost_equal(math::transpose(test1), Mat(0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1)));
        REQUIRE(math::are_almost_equal(math::inverse(test1), Mat(0, -1, 0, 20, 1, 0, 0, -10, 0, 0, 1, -30, 0, 0, 0, 1)));
    }

    AND_THEN("Unary operators") {
        Mat test(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        REQUIRE(+test == test);
        REQUIRE(-test == test * static_cast<TestType>(-1));
    }

    AND_THEN("Binary operators") {
        // Test against numpy.
        Mat tmp2(1, 2, 3, 9, 8, 7, 4, 5, 6, 12, 9, 15, 14, 11, 10, 4);
        Mat tmp3(1, 2, 3, 6, 5, 4, 9, 8, 7, 1, 9, 11, 13, 14, 5, 9);
        REQUIRE(math::are_almost_equal(tmp2 + tmp3, Mat(2, 4, 6, 15,
                                                        13, 11, 13, 13,
                                                        13, 13, 18, 26,
                                                        27, 25, 15, 13)));
        REQUIRE(math::are_almost_equal(tmp2 + TestType(3.4), Mat(4.4, 5.4, 6.4, 12.4,
                                                                 11.4, 10.4, 7.4, 8.4,
                                                                 9.4, 15.4, 12.4, 18.4,
                                                                 17.4, 14.4, 13.4, 7.4)));
        REQUIRE(math::are_almost_equal(TestType(3.4) + tmp2, Mat(4.4, 5.4, 6.4, 12.4,
                                                                 11.4, 10.4, 7.4, 8.4,
                                                                 9.4, 15.4, 12.4, 18.4,
                                                                 17.4, 14.4, 13.4, 7.4)));

        REQUIRE(math::are_almost_equal(tmp3 - tmp2, Mat(0, 0, 0, -3,
                                                        -3, -3, 5, 3,
                                                        1, -11, 0, -4,
                                                        -1, 3, -5, 5)));
        REQUIRE(math::are_almost_equal(tmp3 - TestType(2.2), Mat(-1.2, -0.2, 0.8, 3.8,
                                                                 2.8, 1.8, 6.8, 5.8,
                                                                 4.8, -1.2, 6.8, 8.8,
                                                                 10.8, 11.8, 2.8, 6.8)));
        REQUIRE(math::are_almost_equal(TestType(-1.3) - tmp3, Mat(-2.3, -3.3, -4.3, -7.3,
                                                                  -6.3, -5.3, -10.3, -9.3,
                                                                  -8.3, -2.3, -10.3, -12.3,
                                                                  -14.3, -15.3, -6.3, -10.3)));

        REQUIRE(math::are_almost_equal(tmp2 * tmp3, Mat(149, 139, 93, 136,
                                                        136, 118, 148, 193,
                                                        324, 279, 282, 366,
                                                        191, 138, 251, 318)));
        REQUIRE(math::are_almost_equal(tmp2 * TestType(2.7), Mat(2.7, 5.4, 8.1, 24.3,
                                                                 21.6, 18.9, 10.8, 13.5,
                                                                 16.2, 32.4, 24.3, 40.5,
                                                                 37.8, 29.7, 27., 10.8)));
        REQUIRE(math::are_almost_equal(TestType(2.7) * tmp2, Mat(2.7, 5.4, 8.1, 24.3,
                                                                 21.6, 18.9, 10.8, 13.5,
                                                                 16.2, 32.4, 24.3, 40.5,
                                                                 37.8, 29.7, 27., 10.8)));
        REQUIRE(all(math::are_almost_equal(tmp2 * Vec4<TestType>(1.4, 1.5, 1.6, 1.7),
                                           Vec4<TestType>(24.5, 36.6, 66.3, 58.9))));
        REQUIRE(all(math::are_almost_equal(Vec4<TestType>(1.4, 1.5, 1.6, 1.7) * tmp2,
                                           Vec4<TestType>(46.8, 51.2, 41.6, 50.9))));

        Mat tmp4(1, 2, 3, 2, 5, 4, 9, 2, 7, 4, 9, 8, 7, 1, 9, 11); // not singular
        REQUIRE(math::are_almost_equal(tmp2 / tmp4, Mat(2.75, -1.166667, 0.194444, 0.388889,
                                                        -2., -1.111111, 4.407407, -2.185185,
                                                        4.5, -2.5, 3.666667, -1.666667,
                                                        -4.25, -0.277778, 5.935185, -3.12963)));
        REQUIRE(math::are_almost_equal(tmp2 / TestType(2.8), Mat(0.357143, 0.714286, 1.071429, 3.214286,
                                                                 2.857143, 2.5, 1.428571, 1.785714,
                                                                 2.142857, 4.285714, 3.214286, 5.357143,
                                                                 5., 3.928571, 3.571429, 1.428571)));
        REQUIRE(math::are_almost_equal(TestType(-2.8) / tmp2, Mat(-2.8, -1.4, -0.933333, -0.311111,
                                                                  -0.35, -0.4, -0.7, -0.56,
                                                                  -0.466667, -0.233333, -0.311111, -0.186667,
                                                                  -0.2, -0.254545, -0.28, -0.7)));
        REQUIRE(all(math::are_almost_equal(tmp4 / Vec4<TestType>(1.4, 1.5, 1.6, 1.7),
                                           Vec4<TestType>(-0.666667, 0.205556, 0.392593, 0.238889))));
        REQUIRE(all(math::are_almost_equal(Vec4<TestType>(1.4, 1.5, 1.6, 1.7) / tmp4,
                                           Vec4<TestType>(0.15, -0.177778, 0.535185, -0.22963))));
    }

    AND_THEN("Boolean operators") {
        Mat test(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        Mat test1(test);
        REQUIRE(test == test1);
        REQUIRE_FALSE(test != test1);
    }

    AND_THEN("Others") {
        REQUIRE(string::human<Float44>() == "Float44");
        REQUIRE(string::human<Double44>() == "Double44");
    }
}
