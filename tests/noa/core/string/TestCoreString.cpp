#include <noa/core/string/Format.hpp>
#include <noa/core/string/Parse.hpp>
#include <noa/core/string/Split.hpp>
#include <noa/core/Exception.hpp>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;
using namespace ::noa::string;

#define REQUIRE_FOR_ALL(range, predicate) for (auto& e: (range)) REQUIRE((predicate(e)))

// -------------------------------------------------------------------------------------------------
// Trim
// -------------------------------------------------------------------------------------------------
TEST_CASE("core::string::trim_left", "[noa][core]") {
    std::string result;
    std::vector<std::string> tests =
            {"", "   ", "  foo ", "  \tfoo", "  \n foo", "  \rfoo",
             " foo bar ", "\t  \n 123; \n", " , 123 "};
    std::vector<std::string> expected =
            {"", "", "foo ", "foo", "foo", "foo", "foo bar ", "123; \n", ", 123 "};
    for (size_t i = 0; i < tests.size(); ++i) {
        result = trim_left(tests[i]);
        REQUIRE(result == expected[i]);
        result = trim_left(std::move(tests[i]));
        REQUIRE(result == expected[i]);
    }
}

TEST_CASE("core::string::trim_right", "[noa][core]") {
    std::string result;
    std::vector<std::string> tests =
            {"", "   ", "  foo ", "  \tfoo", " \n foo\n ", "foo \r",
             " foo bar ", "\t  \n 123; \n", " , 123 "};
    std::vector<std::string> expected =
            {"", "", "  foo", "  \tfoo", " \n foo", "foo",
             " foo bar", "\t  \n 123;", " , 123"};
    for (size_t i = 0; i < tests.size(); ++i) {
        result = trim_right(tests[i]);
        REQUIRE(result == expected[i]);
        result = trim_right(std::move(tests[i]));
        REQUIRE(result == expected[i]);
    }
}

TEST_CASE("core::string::trim", "[noa][core]") {
    std::string result;
    std::vector<std::string> tests =
            {"", "  ", "  foo ", "  \tfoo", " \n foo\n ", "foo \r", " foo bar ",
             "\t  \n 123; \n", " , 123 ", "foo \n  \n", "  foo bar \n foo "};
    std::vector<std::string> expected =
            {"", "", "foo", "foo", "foo", "foo", "foo bar",
             "123;", ", 123", "foo", "foo bar \n foo"};
    for (size_t i = 0; i < tests.size(); ++i) {
        result = trim(tests[i]);
        REQUIRE(result == expected[i]);
        result = trim(std::move(tests[i]));
        REQUIRE(result == expected[i]);
    }
}

TEMPLATE_TEST_CASE("core::string::parse(), to integer", "[noa][core]", u8, u16, u32, u64, i8, i16, i32, i64) {
    TestType result;
    TestType min = std::numeric_limits<TestType>::min();
    TestType max = std::numeric_limits<TestType>::max();

    GIVEN("a string that can be converted to an integer") {
        std::vector<std::string> tests =
                {"1", " 6", "\t 7", "9.", "56", "011", "0 ", "10.3", "10e3",
                 " 123  ", "0123", "0x9999910", fmt::format("  {},,", min),
                 fmt::format("  {}  ", max)};
        std::vector<TestType> expected =
                {1, 6, 7, 9, 56, 11, 0, 10, 10, 123, 123, 0, min, max};

        for (size_t i{0}; i < tests.size(); ++i) {
            result = parse<TestType>(tests[i]);
            REQUIRE(result == expected[i]);
        }

        auto test1 = parse<i8>(" -43");
        REQUIRE(test1 == -43);
        auto test2 = parse<i16>("\t  -194");
        REQUIRE(test2 == -194);
        int test3 = parse<i32>("-54052");
        REQUIRE(test3 == -54052);
        auto test4 = parse<i64>("   -525107745");
        REQUIRE(test4 == -525107745);
        auto test5 = parse<u64>("11111111155488");
        REQUIRE(test5 == 11111111155488);
    }

    GIVEN("a string that cannot be converted to an integer") {
        auto to_test = GENERATE("     ", "", ".", " n10", "--10", "e10");
        REQUIRE_THROWS_AS(parse<TestType>(to_test), noa::Exception);
    }

    GIVEN("a string that falls out of range") {
        std::string tmp;
        if (std::is_signed_v<TestType>)
            REQUIRE_THROWS_AS(parse<TestType>(fmt::format("  {}1,,", min)), noa::Exception);
        else
            REQUIRE_THROWS_AS(parse<TestType>(fmt::format("  -{}1,,", min)), noa::Exception);

        REQUIRE_THROWS_AS(parse<TestType>(fmt::format("  {}1  ", max)), noa::Exception);
        REQUIRE_THROWS_AS(parse<u8>("-1"), noa::Exception);
        REQUIRE_THROWS_AS(parse<u16>("-1"), noa::Exception);
        REQUIRE_THROWS_AS(parse<u32>("-1"), noa::Exception);
        REQUIRE_THROWS_AS(parse<u64>("-1"), noa::Exception);
    }
}

TEMPLATE_TEST_CASE("core::string::parse(), to floating-point", "[noa][core]", f32, f64) {
    TestType result;

    GIVEN("a string that can be converted to a floating point") {
        WHEN("should return a number") {
            std::vector<std::string> tests =
                    {"1", " 6", "\t7", "9.", ".5", "011", "-1", "123.123", ".0",
                     "10x", "-10.3", "10e3", "10e-04", "0E-12", "09999910",
                     "0x1273", "-0x1273"};
            std::vector<float> expected =
                    {1, 6, 7, 9, .5, 11, -1, 123.123f, 0, 10, -10.3f,
                     10e3, 10e-04f, 0e-12f, 9999910., 4723., -4723.};
            for (size_t i{0}; i < tests.size(); ++i) {
                result = parse<TestType>(tests[i]);
                REQUIRE_THAT(result, Catch::WithinULP(expected[i], 2));
            }
        }

        WHEN("should return NaN") {
            auto tests = GENERATE("nan", "Nan", "-NaN" "-nan");
            REQUIRE(std::isnan(parse<TestType>(tests)) == true);
        }

        WHEN("should return Inf") {
            auto tests = GENERATE("inf", "-inf", "INFINITY" "-INFINITY", "-Inf");
            REQUIRE(std::isinf(parse<TestType>(tests)) == true);
        }
    }

    GIVEN("a string that can be converted to a floating point") {
        auto tests = GENERATE("     ", "", ".", " n10", "--10", "e10");
        REQUIRE_THROWS_AS(parse<TestType>(tests), noa::Exception);
    }

    GIVEN("a string that falls out of the floating point range") {
        TestType min = std::numeric_limits<TestType>::min();
        TestType max = std::numeric_limits<TestType>::max();
        TestType lowest = std::numeric_limits<TestType>::lowest();
        std::vector<std::string> tests = {fmt::format("  {}1,,", min),
                                          fmt::format("  {}1  ", max),
                                          fmt::format("  {}1  ", lowest)};
        for (auto& test: tests)
            REQUIRE_THROWS_AS(parse<TestType>(test), noa::Exception);
    }
}

TEST_CASE("core::string::parse(), to boolean", "[noa][core]") {
    GIVEN("a string that can be converted to a bool") {
        WHEN("should return true") {
            const auto* to_test = GENERATE("1", "true", "TRUE", "y", "yes", "YES", "on", "ON");
            REQUIRE(parse<bool>(to_test) == true);
        }

        WHEN("should return false") {
            const auto* to_test = GENERATE("0", "false", "FALSE", "n", "no", "NO", "off", "OFF");
            REQUIRE(parse<bool>(to_test) == false);
        }
    }

    GIVEN("a string that cannot be converted to a bool") {
        const auto* to_test = GENERATE(" y", "yes please", ".", "", " 0", "wrong");
        REQUIRE_THROWS_AS(parse<bool>(to_test), noa::Exception);
    }
}

TEST_CASE("core::string::split(), to strings", "[noa][core]") {
    std::vector<std::string> tests =
            {",1,2,3,4,5,",
             "1,2,3,4,5",
             "1,2, 3 ,4\n ,5 ,",
             "1 , 2 3\t   ,4 ,  5 6  7, ",
             " 1, 2,  ,  4 5",
             " ",
             "",
             " ,\n   ",
             "   1,2,3",
             " 1 , 2 , 3 , 4 5 67  "};
    std::vector<std::vector<std::string>> expected =
            {{"",  "1",   "2", "3",      "4", "5", ""},
             {"1", "2",   "3", "4",      "5"},
             {"1", "2",   "3", "4",      "5", ""},
             {"1", "2 3", "4", "5 6  7", ""},
             {"1", "2",   "",  "4 5"},
             {""},
             {""},
             {"",  ""},
             {"1", "2",   "3"},
             {"1", "2",   "3", "4 5 67"}};

    WHEN("output is a vector") {
        for (size_t i = 0; i < tests.size(); ++i)
            REQUIRE_THAT(split<std::string>(tests[i]), Catch::Equals(expected[i]));
        REQUIRE(split<std::string>("123,foo,,dd2").size() == 4);
    }

    WHEN("output is an array") {
        auto result_0 = split<std::string, 7>(tests[0]);
        REQUIRE_THAT(test::array2vector(result_0), Catch::Equals(expected[0]));
        REQUIRE_THROWS_AS((split<std::string, 6>(tests[0])), noa::Exception);

        auto result_1 = split<std::string, 5>(tests[1]);
        REQUIRE_THAT(test::array2vector(result_1), Catch::Equals(expected[1]));
        REQUIRE_THROWS_AS((split<std::string, 6>(tests[0])), noa::Exception);

        auto result_2 = split<std::string, 6>(tests[2]);
        REQUIRE_THAT(test::array2vector(result_2), Catch::Equals(expected[2]));
        REQUIRE_THROWS_AS((split<std::string, 1>(tests[0])), noa::Exception);

        auto result_3 = split<std::string, 5>(tests[3]);
        REQUIRE_THAT(test::array2vector(result_3), Catch::Equals(expected[3]));
        REQUIRE_THROWS_AS((split<std::string, 6>(tests[0])), noa::Exception);

        auto result_4 = split<std::string, 4>(tests[4]);
        REQUIRE_THAT(test::array2vector(result_4), Catch::Equals(expected[4]));
        REQUIRE_THROWS_AS((split<std::string, 10>(tests[0])), noa::Exception);

        auto result_5 = split<std::string, 1>(tests[5]);
        REQUIRE_THAT(test::array2vector(result_5), Catch::Equals(expected[5]));
        REQUIRE_THROWS_AS((split<std::string, 2>(tests[0])), noa::Exception);

        auto result_6 = split<std::string, 1>(tests[6]);
        REQUIRE_THAT(test::array2vector(result_6), Catch::Equals(expected[6]));
        REQUIRE_THROWS_AS((split<std::string, 0>(tests[0])), noa::Exception);

        auto result_7 = split<std::string, 2>(tests[7]);
        REQUIRE_THAT(test::array2vector(result_7), Catch::Equals(expected[7]));
        REQUIRE_THROWS_AS((split<std::string, 0>(tests[0])), noa::Exception);

        auto result_8 = split<std::string, 3>(tests[8]);
        REQUIRE_THAT(test::array2vector(result_8), Catch::Equals(expected[8]));
        REQUIRE_THROWS_AS((split<std::string, 6>(tests[0])), noa::Exception);

        auto result_9 = split<std::string, 4>(tests[9]);
        REQUIRE_THAT(test::array2vector(result_9), Catch::Equals(expected[9]));
        REQUIRE_THROWS_AS((split<std::string, 1>(tests[0])), noa::Exception);
    }
}

TEMPLATE_TEST_CASE("core::string::split(), to integers", "[noa][core]", u8, u16, u32, u64, i8, i16, i32, i64) {
    TestType min = std::numeric_limits<TestType>::min();
    TestType max = std::numeric_limits<TestType>::max();
    std::string str_min_max = fmt::format("  {}, {}", min, max);

    std::vector<std::string> tests_valid =
            {" 1, 6, \t7, 9. , 56, 011  \t, 0,10.3,10e3    ",
             "123  ,0123, 0x9999910,1 1", str_min_max};
    std::vector<std::vector<TestType>> expected_valid =
            {{1,   6,   7, 9, 56, 11, 0, 10, 10},
             {123, 123, 0, 1},
             {min, max}};
    std::vector<std::string> tests_invalid = {" ", "120, , 23", "120, 1, 23,", ",120, 1, 23,"};

    GIVEN("a vector") {
        std::vector<TestType> result;

        WHEN("the input string can be converted") {
            for (size_t i{0}; i < tests_valid.size(); ++i)
                REQUIRE_THAT(split<TestType>(tests_valid[i]), Catch::Equals(expected_valid[i]));
        }

        WHEN("the input string cannot be converted") {
            for (size_t i{0}; i < tests_invalid.size(); ++i)
                REQUIRE_THROWS_AS(split<TestType>(tests_invalid[i]), noa::Exception);
        }
    }

    GIVEN("an array") {
        WHEN("the input string can be converted") {
            auto result_0 = split<TestType, 9>(tests_valid[0]);
            REQUIRE_THAT(test::array2vector(result_0), Catch::Equals(expected_valid[0]));

            auto result_1 = split<TestType, 4>(tests_valid[1]);
            REQUIRE_THAT(test::array2vector(result_1), Catch::Equals(expected_valid[1]));

            auto result_2 = split<TestType, 2>(tests_valid[2]);
            REQUIRE_THAT(test::array2vector(result_2), Catch::Equals(expected_valid[2]));
        }

        WHEN("the input string cannot be converted") {
            REQUIRE_THROWS_AS((split<TestType, 1>(tests_invalid[0])), noa::Exception);
            REQUIRE_THROWS_AS((split<TestType, 3>(tests_invalid[1])), noa::Exception);
            REQUIRE_THROWS_AS((split<TestType, 4>(tests_invalid[2])), noa::Exception);
            REQUIRE_THROWS_AS((split<TestType, 5>(tests_invalid[3])), noa::Exception);
        }
    }
}

TEMPLATE_TEST_CASE("core::string::split(), to floating-points", "[noa][core]", f32, f64) {
    std::vector<std::string> tests =
            {" 1, 6., \t7, 9. , .56, 123.123, 011, -1, .0",
             "10x,-10.3  , 10e3  , 10e-04,0E-12    , 09999910"};
    std::vector<std::vector<f32>> expected =
            {{1,  6,      7,     9,       .56f,   123.123f, 11, -1, .0},
             {10, -10.3f, 10e3f, 10e-04f, 0e-12f, 9999910.f}};

    GIVEN("a vector") {
        std::vector<TestType> result{};
        WHEN("the input string can be converted") {
            for (size_t nb{0}; nb < tests.size(); ++nb) {
                result = split<TestType>(tests[nb]);
                REQUIRE(result.size() == expected[nb].size());
                for (size_t idx = 0; idx < expected[nb].size(); ++idx)
                    REQUIRE_THAT(result[idx], Catch::WithinULP(expected[nb][idx], 2));
            }

            WHEN("should return NaN") {
                std::string test = {"nan, Nan  , -NaN,-nan"};
                result = split<TestType>(test);
                REQUIRE(result.size() == 4);
                REQUIRE_FOR_ALL(result, std::isnan);
            }

            WHEN("should return Inf") {
                std::string test = {"inf, -inf , INFINITY ,-INFINITY,-Inf"};
                result = split<TestType>(test);
                REQUIRE(result.size() == 5);
                REQUIRE_FOR_ALL(result, std::isinf);
            }
        }

        WHEN("the input cannot be converted") {
            auto test = GENERATE("", "  ", ". ,10", "1, 2., n10", "3, --10", "0, e10");
            REQUIRE_THROWS_AS(split<TestType>(test), noa::Exception);
        }
    }

    GIVEN("an array") {
        WHEN("the input string can be converted") {
            auto result_0 = split<TestType, 9>(tests[0]);
            for (size_t idx = 0; idx < expected[0].size(); ++idx)
                REQUIRE_THAT(result_0[idx], Catch::WithinULP(expected[0][idx], 2));

            auto result_1 = split<TestType, 6>(tests[1]);
            for (size_t idx = 0; idx < expected[1].size(); ++idx)
                REQUIRE_THAT(result_1[idx], Catch::WithinULP(expected[1][idx], 2));

            REQUIRE_THROWS_AS((split<TestType, 3>(tests[0])), noa::Exception);
            REQUIRE_THROWS_AS((split<TestType, 7>(tests[1])), noa::Exception);

            WHEN("should return NaN") {
                std::string test = {"nan, Nan  , -NaN,-nan"};
                auto result = split<TestType, 4>(test);
                REQUIRE_FOR_ALL(result, std::isnan);
            }

            WHEN("should return Inf") {
                std::string test = {"inf, -inf , INFINITY ,-INFINITY,-Inf"};
                auto result = split<TestType, 5>(test);
                REQUIRE_FOR_ALL(result, std::isinf);
            }
        }

        WHEN("the input cannot be converted") {
            auto test = GENERATE("", "  ", ". ", "n10", "--10", "e10");
            REQUIRE_THROWS_AS((split<TestType, 1>(test)), noa::Exception);
        }
    }
}

TEST_CASE("core::string::split(), to boolean", "[noa][core]") {
    std::string test = "1,true,   TRUE, y,yes, YES,on, ON,0,false,False  ,n,no, NO, ofF , OFF";
    std::vector<bool> expected = {true, true, true, true, true, true, true, true,
                                  false, false, false, false, false, false, false, false};

    GIVEN("a vector") {
        REQUIRE_THAT(split<bool>(test), Catch::Equals(expected));
    }

    GIVEN("an array") {
        REQUIRE_THAT(test::array2vector(split<bool, 16>(test)), Catch::Equals(expected));
    }
}

TEMPLATE_TEST_CASE("core::string::split(), with default values", "[noa][core]",
                   u8, u16, u32, u64, i8, i16, i32, i64, f32, f64) {
    using vector = std::vector<TestType>;

    GIVEN("a vector") {
        std::string test1 = "123,,12, 0, \t,, 8";
        std::string test2 = ",1,2,3,4,5,";
        REQUIRE_THAT(split<TestType>(test1, test2),
                     Catch::Equals(vector{123, 1, 12, 0, 4, 5, 8}));

        test1 = "12,,12, 0, \t,, 8";
        test2 = ",1,2,3,4,5,";
        REQUIRE_THAT(split<std::string>(test1, test2),
                     Catch::Equals(std::vector<std::string>{"12", "1", "12", "0", "4", "5", "8"}));

        test1 = "1,,0, 0, \t,, 1";
        test2 = ",1,1,3,1,1,12";
        REQUIRE_THAT(split<bool>(test1, test2),
                     Catch::Equals(std::vector<bool>{1, 1, 0, 0, 1, 1, 1}));

        test1 = "1,,0, 0, \t,, 1";
        test2 = ",1,1,3,1,  ,12";
        REQUIRE_THROWS_AS(split<TestType>(test1, test2), noa::Exception);
    }

    GIVEN("an array") {
        std::string test1 = "123,,12, 0, \t,, 8";
        std::string test2 = ",1,2,3,4,5,";
        auto result1 = split<TestType, 7>(test1, test2);
        std::vector<double> expected1 = {123, 1, 12, 0, 4, 5, 8};
        REQUIRE(expected1.size() == result1.size());
        for (size_t idx = 0; idx < expected1.size(); ++idx)
            REQUIRE_THAT(result1[idx], Catch::WithinULP(expected1[idx], 2));

        test1 = "123,,12, 0, \t,, 8";
        test2 = ",1,2,3,4,5,";
        auto result2 = split<std::string, 7>(test1, test2);
        std::vector<std::string> expected2{"123", "1", "12", "0", "4", "5", "8"};
        REQUIRE_THAT(test::array2vector(result2), Catch::Equals(expected2));

        test1 = "1,,0, 0, \t,, 1";
        test2 = ",1,1,3,1,1,12";
        auto result3 = split<bool, 7>(test1, test2);
        std::vector<bool> expected3{true, true, false, false, true, true, true};
        REQUIRE_THAT(test::array2vector(result3), Catch::Equals(expected3));

        WHEN("the inputs are invalid") {
            test1 = "1,,0, 0, \t,, 1";
            test2 = ",1,1,3,1,  ,12";
            REQUIRE_THROWS_AS((split<TestType, 7>(test1, test2)), noa::Exception);

            test1 = "1,,0, 0, \t,, 1";
            test2 = ",1,1,3,1, 1 ,12";
            REQUIRE_THROWS_AS((split<TestType, 10>(test1, test2)), noa::Exception);
            REQUIRE_THROWS_AS((split<TestType, 4>(test1, test2)), noa::Exception);
        }
    }
}
