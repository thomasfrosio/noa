#include <noa/common/String.h>
#include <noa/common/Exception.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;
using string_t = std::string;

#define REQUIRE_FOR_ALL(range, predicate) for (auto& e: (range)) REQUIRE((predicate(e)))

// -------------------------------------------------------------------------------------------------
// Trim
// -------------------------------------------------------------------------------------------------
TEST_CASE("string::leftTrim(Copy)", "[noa][common][string]") {
    string_t result;
    std::vector<string_t> tests = {"", "   ", "  foo ", "  \tfoo", "  \n foo", "  \rfoo", " foo bar ",
                                   "\t  \n 123; \n", " , 123 "};
    std::vector<string_t> expected = {"", "", "foo ", "foo", "foo", "foo", "foo bar ", "123; \n",
                                      ", 123 "};
    for (size_t i = 0; i < tests.size(); ++i) {
        result = string::leftTrim(tests[i]);
        REQUIRE(result == expected[i]);
        result = string::leftTrim(std::move(tests[i]));
        REQUIRE(result == expected[i]);
    }
}

TEST_CASE("string::rightTrim(Copy)", "[noa][common][string]") {
    string_t result;
    std::vector<string_t> tests = {"", "   ", "  foo ", "  \tfoo", " \n foo\n ", "foo \r",
                                   " foo bar ", "\t  \n 123; \n", " , 123 "};
    std::vector<string_t> expected = {"", "", "  foo", "  \tfoo", " \n foo", "foo",
                                      " foo bar", "\t  \n 123;", " , 123"};
    for (size_t i = 0; i < tests.size(); ++i) {
        result = string::rightTrim(tests[i]);
        REQUIRE(result == expected[i]);
        result = string::rightTrim(std::move(tests[i]));
        REQUIRE(result == expected[i]);
    }
}

TEST_CASE("string::trim(Copy)", "[noa][common][string]") {
    string_t result;
    std::vector<string_t> tests = {"", "  ", "  foo ", "  \tfoo", " \n foo\n ", "foo \r", " foo bar ",
                                   "\t  \n 123; \n", " , 123 ", "foo \n  \n", "  foo bar \n foo "};
    std::vector<string_t> expected = {"", "", "foo", "foo", "foo", "foo", "foo bar",
                                      "123;", ", 123", "foo", "foo bar \n foo"};
    for (size_t i = 0; i < tests.size(); ++i) {
        result = string::trim(tests[i]);
        REQUIRE(result == expected[i]);
        result = string::trim(std::move(tests[i]));
        REQUIRE(result == expected[i]);
    }
}


// -------------------------------------------------------------------------------------------------
// String to scalar (integers, floating points, bool)
// -------------------------------------------------------------------------------------------------
TEMPLATE_TEST_CASE("string::toInt()", "[noa][common][string]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t) {
    TestType result;
    TestType min = std::numeric_limits<TestType>::min();
    TestType max = std::numeric_limits<TestType>::max();

    GIVEN("a string that can be converted to an integer") {
        std::vector<string_t> tests = {"1", " 6", "\t 7", "9.", "56", "011", "0 ", "10.3", "10e3",
                                       " 123  ", "0123", "0x9999910", fmt::format("  {},,", min),
                                       fmt::format("  {}  ", max)};
        std::vector<TestType> expected = {1, 6, 7, 9, 56, 11, 0, 10, 10, 123, 123, 0, min, max};

        for (size_t i{0}; i < tests.size(); ++i) {
            result = string::toInt<TestType>(tests[i]);
            REQUIRE(result == expected[i]);
        }

        auto test1 = string::toInt<int8_t>(" -43");
        REQUIRE(test1 == -43);
        auto test2 = string::toInt<short>("\t  -194");
        REQUIRE(test2 == -194);
        int test3 = string::toInt("-54052");
        REQUIRE(test3 == -54052);
        auto test4 = string::toInt<long>("   -525107745");
        REQUIRE(test4 == -525107745);
        auto test5 = string::toInt<uint64_t>("11111111155488");
        REQUIRE(test5 == 11111111155488);
    }

    GIVEN("a string that cannot be converted to an integer") {
        auto to_test = GENERATE("     ", "", ".", " n10", "--10", "e10");
        REQUIRE_THROWS_AS(string::toInt<TestType>(to_test), noa::Exception);
    }

    GIVEN("a string that falls out of range") {
        std::string tmp;
        if (std::is_signed_v<TestType>)
            REQUIRE_THROWS_AS(string::toInt<TestType>(fmt::format("  {}1,,", min)), noa::Exception);
        else
            REQUIRE_THROWS_AS(string::toInt<TestType>(fmt::format("  -{}1,,", min)), noa::Exception);

        REQUIRE_THROWS_AS(string::toInt<TestType>(fmt::format("  {}1  ", max)), noa::Exception);
        REQUIRE_THROWS_AS(string::toInt<uint8_t>("-1"), noa::Exception);
        REQUIRE_THROWS_AS(string::toInt<unsigned short>("-1"), noa::Exception);
        REQUIRE_THROWS_AS(string::toInt<unsigned int>("-1"), noa::Exception);
        REQUIRE_THROWS_AS(string::toInt<unsigned long>("-1"), noa::Exception);
        REQUIRE_THROWS_AS(string::toInt<unsigned long long>("-1"), noa::Exception);
    }
}

TEMPLATE_TEST_CASE("string::toFloat()", "[noa][common][string]", float, double) {
    TestType result;

    GIVEN("a string that can be converted to a floating point") {
        WHEN("should return a number") {
            std::vector<string_t> tests = {"1", " 6", "\t7", "9.", ".5", "011", "-1", "123.123", ".0",
                                           "10x", "-10.3", "10e3", "10e-04", "0E-12", "09999910",
                                           "0x1273", "-0x1273"};
            std::vector<float> expected = {1, 6, 7, 9, .5, 11, -1, 123.123f, 0, 10, -10.3f,
                                           10e3, 10e-04f, 0e-12f, 9999910., 4723., -4723.};
            for (size_t i{0}; i < tests.size(); ++i) {
                result = string::toFloat<TestType>(tests[i]);
                REQUIRE_THAT(result, Catch::WithinULP(expected[i], 2));
            }
        }

        WHEN("should return NaN") {
            auto tests = GENERATE("nan", "Nan", "-NaN" "-nan");
            REQUIRE(std::isnan(string::toFloat<TestType>(tests)) == true);
        }

        WHEN("should return Inf") {
            auto tests = GENERATE("inf", "-inf", "INFINITY" "-INFINITY", "-Inf");
            REQUIRE(std::isinf(string::toFloat<TestType>(tests)) == true);
        }
    }

    GIVEN("a string that can be converted to a floating point") {
        auto tests = GENERATE("     ", "", ".", " n10", "--10", "e10");
        REQUIRE_THROWS_AS(string::toFloat<TestType>(tests), noa::Exception);
    }

    GIVEN("a string that falls out of the floating point range") {
        TestType min = std::numeric_limits<TestType>::min();
        TestType max = std::numeric_limits<TestType>::max();
        TestType lowest = std::numeric_limits<TestType>::lowest();
        std::vector<string_t> tests = {fmt::format("  {}1,,", min),
                                       fmt::format("  {}1  ", max),
                                       fmt::format("  {}1  ", lowest)};
        for (auto& test: tests)
            REQUIRE_THROWS_AS(string::toFloat<TestType>(test), noa::Exception);
    }
}

TEST_CASE("string::toBool()", "[noa][common][string]") {
    GIVEN("a string that can be converted to a bool") {
        WHEN("should return true") {
            auto to_test = GENERATE("1", "true", "TRUE", "y", "yes", "YES", "on", "ON");
            REQUIRE(string::toBool(to_test) == true);
        }

        WHEN("should return false") {
            auto to_test = GENERATE("0", "false", "FALSE", "n", "no", "NO", "off", "OFF");
            REQUIRE(string::toBool(to_test) == false);
        }
    }

    GIVEN("a string that cannot be converted to a bool") {
        auto to_test = GENERATE(" y", "yes please", ".", "", " 0", "wrong");
        REQUIRE_THROWS_AS(string::toBool(to_test), noa::Exception);
    }
}

// -------------------------------------------------------------------------------------------------
// parse
// -------------------------------------------------------------------------------------------------
TEST_CASE("string::parse(), to strings", "[noa][common][string]") {
    std::vector<std::string> tests = {",1,2,3,4,5,",
                                      "1,2,3,4,5",
                                      "1,2, 3 ,4\n ,5 ,",
                                      "1 , 2 3\t   ,4 ,  5 6  7, ",
                                      " 1, 2,  ,  4 5",
                                      " ",
                                      "",
                                      " ,\n   ",
                                      "   1,2,3",
                                      " 1 , 2 , 3 , 4 5 67  "};
    std::vector<std::vector<std::string>> expected = {{"",  "1",   "2", "3",      "4", "5", ""},
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
            REQUIRE_THAT(string::parse<std::string>(tests[i]), Catch::Equals(expected[i]));
        REQUIRE(string::parse<std::string>("123,foo,,dd2").size() == 4);
    }

    WHEN("output is an array") {
        auto result_0 = string::parse<std::string, 7>(tests[0]);
        REQUIRE_THAT(test::toVector(result_0), Catch::Equals(expected[0]));
        REQUIRE_THROWS_AS((string::parse<std::string, 6>(tests[0])), noa::Exception);

        auto result_1 = string::parse<std::string, 5>(tests[1]);
        REQUIRE_THAT(test::toVector(result_1), Catch::Equals(expected[1]));
        REQUIRE_THROWS_AS((string::parse<std::string, 6>(tests[0])), noa::Exception);

        auto result_2 = string::parse<std::string, 6>(tests[2]);
        REQUIRE_THAT(test::toVector(result_2), Catch::Equals(expected[2]));
        REQUIRE_THROWS_AS((string::parse<std::string, 1>(tests[0])), noa::Exception);

        auto result_3 = string::parse<std::string, 5>(tests[3]);
        REQUIRE_THAT(test::toVector(result_3), Catch::Equals(expected[3]));
        REQUIRE_THROWS_AS((string::parse<std::string, 6>(tests[0])), noa::Exception);

        auto result_4 = string::parse<std::string, 4>(tests[4]);
        REQUIRE_THAT(test::toVector(result_4), Catch::Equals(expected[4]));
        REQUIRE_THROWS_AS((string::parse<std::string, 10>(tests[0])), noa::Exception);

        auto result_5 = string::parse<std::string, 1>(tests[5]);
        REQUIRE_THAT(test::toVector(result_5), Catch::Equals(expected[5]));
        REQUIRE_THROWS_AS((string::parse<std::string, 2>(tests[0])), noa::Exception);

        auto result_6 = string::parse<std::string, 1>(tests[6]);
        REQUIRE_THAT(test::toVector(result_6), Catch::Equals(expected[6]));
        REQUIRE_THROWS_AS((string::parse<std::string, 0>(tests[0])), noa::Exception);

        auto result_7 = string::parse<std::string, 2>(tests[7]);
        REQUIRE_THAT(test::toVector(result_7), Catch::Equals(expected[7]));
        REQUIRE_THROWS_AS((string::parse<std::string, 0>(tests[0])), noa::Exception);

        auto result_8 = string::parse<std::string, 3>(tests[8]);
        REQUIRE_THAT(test::toVector(result_8), Catch::Equals(expected[8]));
        REQUIRE_THROWS_AS((string::parse<std::string, 6>(tests[0])), noa::Exception);

        auto result_9 = string::parse<std::string, 4>(tests[9]);
        REQUIRE_THAT(test::toVector(result_9), Catch::Equals(expected[9]));
        REQUIRE_THROWS_AS((string::parse<std::string, 1>(tests[0])), noa::Exception);
    }
}

TEMPLATE_TEST_CASE("string::parse(), to int", "[noa][common][string]",
                   uint8_t, unsigned short, unsigned int, unsigned long, unsigned long long,
                   int8_t, short, int, long, long long) {
    TestType min = std::numeric_limits<TestType>::min();
    TestType max = std::numeric_limits<TestType>::max();
    string_t str_min_max = fmt::format("  {}, {}", min, max);

    std::vector<std::string> tests_valid = {" 1, 6, \t7, 9. , 56, 011  \t, 0,10.3,10e3    ",
                                            "123  ,0123, 0x9999910,1 1", str_min_max};
    std::vector<std::vector<TestType>> expected_valid = {{1,   6,   7, 9, 56, 11, 0, 10, 10},
                                                         {123, 123, 0, 1},
                                                         {min, max}};
    std::vector<std::string> tests_invalid = {" ", "120, , 23", "120, 1, 23,", ",120, 1, 23,"};

    GIVEN("a vector") {
        std::vector<TestType> result;

        WHEN("the input string can be converted") {
            for (size_t i{0}; i < tests_valid.size(); ++i)
                REQUIRE_THAT(string::parse<TestType>(tests_valid[i]), Catch::Equals(expected_valid[i]));
        }

        WHEN("the input string cannot be converted") {
            for (size_t i{0}; i < tests_invalid.size(); ++i)
                REQUIRE_THROWS_AS(string::parse<TestType>(tests_invalid[i]), noa::Exception);
        }
    }

    GIVEN("an array") {
        WHEN("the input string can be converted") {
            auto result_0 = string::parse<TestType, 9>(tests_valid[0]);
            REQUIRE_THAT(test::toVector(result_0), Catch::Equals(expected_valid[0]));

            auto result_1 = string::parse<TestType, 4>(tests_valid[1]);
            REQUIRE_THAT(test::toVector(result_1), Catch::Equals(expected_valid[1]));

            auto result_2 = string::parse<TestType, 2>(tests_valid[2]);
            REQUIRE_THAT(test::toVector(result_2), Catch::Equals(expected_valid[2]));
        }

        WHEN("the input string cannot be converted") {
            REQUIRE_THROWS_AS((string::parse<TestType, 1>(tests_invalid[0])), noa::Exception);
            REQUIRE_THROWS_AS((string::parse<TestType, 3>(tests_invalid[1])), noa::Exception);
            REQUIRE_THROWS_AS((string::parse<TestType, 4>(tests_invalid[2])), noa::Exception);
            REQUIRE_THROWS_AS((string::parse<TestType, 5>(tests_invalid[3])), noa::Exception);
        }
    }
}

TEMPLATE_TEST_CASE("string::parse(), to float", "[noa][common][string]", float, double) {
    std::vector<string_t> tests = {" 1, 6., \t7, 9. , .56, 123.123, 011, -1, .0",
                                   "10x,-10.3  , 10e3  , 10e-04,0E-12    , 09999910"};
    std::vector<std::vector<float>> expected = {{1,  6,      7,     9,       .56f,   123.123f, 11, -1, .0},
                                                {10, -10.3f, 10e3f, 10e-04f, 0e-12f, 9999910.f}};

    GIVEN("a vector") {
        std::vector<TestType> result{};
        WHEN("the input string can be converted") {
            for (size_t nb{0}; nb < tests.size(); ++nb) {
                result = string::parse<TestType>(tests[nb]);
                REQUIRE(result.size() == expected[nb].size());
                for (size_t idx = 0; idx < expected[nb].size(); ++idx)
                    REQUIRE_THAT(result[idx], Catch::WithinULP(expected[nb][idx], 2));
            }

            WHEN("should return NaN") {
                string_t test = {"nan, Nan  , -NaN,-nan"};
                result = string::parse<TestType>(test);
                REQUIRE(result.size() == 4);
                REQUIRE_FOR_ALL(result, std::isnan);
            }

            WHEN("should return Inf") {
                string_t test = {"inf, -inf , INFINITY ,-INFINITY,-Inf"};
                result = string::parse<TestType>(test);
                REQUIRE(result.size() == 5);
                REQUIRE_FOR_ALL(result, std::isinf);
            }
        }

        WHEN("the input cannot be converted") {
            auto test = GENERATE("", "  ", ". ,10", "1, 2., n10", "3, --10", "0, e10");
            REQUIRE_THROWS_AS(string::parse<TestType>(test), noa::Exception);
        }
    }

    GIVEN("an array") {
        WHEN("the input string can be converted") {
            auto result_0 = string::parse<TestType, 9>(tests[0]);
            for (size_t idx = 0; idx < expected[0].size(); ++idx)
                REQUIRE_THAT(result_0[idx], Catch::WithinULP(expected[0][idx], 2));

            auto result_1 = string::parse<TestType, 6>(tests[1]);
            for (size_t idx = 0; idx < expected[1].size(); ++idx)
                REQUIRE_THAT(result_1[idx], Catch::WithinULP(expected[1][idx], 2));

            REQUIRE_THROWS_AS((string::parse<TestType, 3>(tests[0])), noa::Exception);
            REQUIRE_THROWS_AS((string::parse<TestType, 7>(tests[1])), noa::Exception);

            WHEN("should return NaN") {
                string_t test = {"nan, Nan  , -NaN,-nan"};
                auto result = string::parse<TestType, 4>(test);
                REQUIRE_FOR_ALL(result, std::isnan);
            }

            WHEN("should return Inf") {
                string_t test = {"inf, -inf , INFINITY ,-INFINITY,-Inf"};
                auto result = string::parse<TestType, 5>(test);
                REQUIRE_FOR_ALL(result, std::isinf);
            }
        }

        WHEN("the input cannot be converted") {
            auto test = GENERATE("", "  ", ". ", "n10", "--10", "e10");
            REQUIRE_THROWS_AS((string::parse<TestType, 1>(test)), noa::Exception);
        }
    }
}

TEST_CASE("string::parse(), to bool", "[noa][common][string]") {
    string_t test = "1,true,   TRUE, y,yes, YES,on, ON,0,false,False  ,n,no, NO, ofF , OFF";
    std::vector<bool> expected = {true, true, true, true, true, true, true, true,
                                  false, false, false, false, false, false, false, false};

    GIVEN("a vector") {
        REQUIRE_THAT(string::parse<bool>(test), Catch::Equals(expected));
    }

    GIVEN("an array") {
        REQUIRE_THAT(test::toVector(string::parse<bool, 16>(test)), Catch::Equals(expected));
    }
}

TEMPLATE_TEST_CASE("string::parse with default values", "[noa][common][string]",
                   uint8_t, unsigned short, unsigned int, unsigned long, unsigned long long,
                   int8_t, short, int, long, long long,
                   float, double) {
    using vector = std::vector<TestType>;

    GIVEN("a vector") {
        string_t test1 = "123,,12, 0, \t,, 8";
        string_t test2 = ",1,2,3,4,5,";
        REQUIRE_THAT(string::parse<TestType>(test1, test2), Catch::Equals(vector{123, 1, 12, 0, 4, 5, 8}));

        test1 = "12,,12, 0, \t,, 8";
        test2 = ",1,2,3,4,5,";
        REQUIRE_THAT(string::parse<std::string>(test1, test2),
                     Catch::Equals(std::vector<string_t>{"12", "1", "12", "0", "4", "5", "8"}));

        test1 = "1,,0, 0, \t,, 1";
        test2 = ",1,1,3,1,1,12";
        REQUIRE_THAT(string::parse<bool>(test1, test2), Catch::Equals(std::vector<bool>{1, 1, 0, 0, 1, 1, 1}));

        test1 = "1,,0, 0, \t,, 1";
        test2 = ",1,1,3,1,  ,12";
        REQUIRE_THROWS_AS(string::parse<TestType>(test1, test2), noa::Exception);
    }

    GIVEN("an array") {
        string_t test1 = "123,,12, 0, \t,, 8";
        string_t test2 = ",1,2,3,4,5,";
        auto result1 = string::parse<TestType, 7>(test1, test2);
        std::vector<double> expected1 = {123, 1, 12, 0, 4, 5, 8};
        REQUIRE(expected1.size() == result1.size());
        for (size_t idx = 0; idx < expected1.size(); ++idx)
            REQUIRE_THAT(result1[idx], Catch::WithinULP(expected1[idx], 2));

        test1 = "123,,12, 0, \t,, 8";
        test2 = ",1,2,3,4,5,";
        auto result2 = string::parse<std::string, 7>(test1, test2);
        std::vector<std::string> expected2{"123", "1", "12", "0", "4", "5", "8"};
        REQUIRE_THAT(test::toVector(result2), Catch::Equals(expected2));

        test1 = "1,,0, 0, \t,, 1";
        test2 = ",1,1,3,1,1,12";
        auto result3 = string::parse<bool, 7>(test1, test2);
        std::vector<bool> expected3{true, true, false, false, true, true, true};
        REQUIRE_THAT(test::toVector(result3), Catch::Equals(expected3));

        WHEN("the inputs are invalid") {
            test1 = "1,,0, 0, \t,, 1";
            test2 = ",1,1,3,1,  ,12";
            REQUIRE_THROWS_AS((string::parse<TestType, 7>(test1, test2)), noa::Exception);

            test1 = "1,,0, 0, \t,, 1";
            test2 = ",1,1,3,1, 1 ,12";
            REQUIRE_THROWS_AS((string::parse<TestType, 10>(test1, test2)), noa::Exception);
            REQUIRE_THROWS_AS((string::parse<TestType, 4>(test1, test2)), noa::Exception);
        }
    }
}
