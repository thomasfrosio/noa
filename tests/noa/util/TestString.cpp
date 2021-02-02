#include <catch2/catch.hpp>
#include "../../Helpers.h"

#include "noa/util/String.h"

using namespace ::Noa;
using string = std::string;

// -------------------------------------------------------------------------------------------------
// Trim
// -------------------------------------------------------------------------------------------------
TEST_CASE("String::leftTrim(Copy)", "[noa][string]") {
    string result;
    std::vector<string> tests = {"", "   ", "  foo ", "  \tfoo", "  \n foo", "  \rfoo", " foo bar ",
                                 "\t  \n 123; \n", " , 123 "};
    std::vector<string> expected = {"", "", "foo ", "foo", "foo", "foo", "foo bar ", "123; \n",
                                    ", 123 "};
    for (size_t i = 0; i < tests.size(); ++i) {
        result = tests[i];
        String::leftTrim(result);
        REQUIRE(result == expected[i]);
        result = String::leftTrimCopy(tests[i]);
        REQUIRE(result == expected[i]);
    }
}

TEST_CASE("String::rightTrim(Copy)", "[noa][string]") {
    string result;
    std::vector<string> tests = {"", "   ", "  foo ", "  \tfoo", " \n foo\n ", "foo \r",
                                 " foo bar ", "\t  \n 123; \n", " , 123 "};
    std::vector<string> expected = {"", "", "  foo", "  \tfoo", " \n foo", "foo",
                                    " foo bar", "\t  \n 123;", " , 123"};
    for (size_t i = 0; i < tests.size(); ++i) {
        result = tests[i];
        String::rightTrim(result);
        REQUIRE(result == expected[i]);
        result = String::rightTrimCopy(tests[i]);
        REQUIRE(result == expected[i]);
    }
}

TEST_CASE("String::trim(Copy)", "[noa][string]") {
    string result;
    std::vector<string> tests = {"", "  ", "  foo ", "  \tfoo", " \n foo\n ", "foo \r", " foo bar ",
                                 "\t  \n 123; \n", " , 123 ", "foo \n  \n", "  foo bar \n foo "};
    std::vector<string> expected = {"", "", "foo", "foo", "foo", "foo", "foo bar",
                                    "123;", ", 123", "foo", "foo bar \n foo"};
    for (size_t i = 0; i < tests.size(); ++i) {
        result = tests[i];
        String::trim(result);
        REQUIRE(result == expected[i]);
        result = String::trimCopy(tests[i]);
        REQUIRE(result == expected[i]);
    }
}


// -------------------------------------------------------------------------------------------------
// String to scalar (integers, floating points, bool)
// -------------------------------------------------------------------------------------------------
TEMPLATE_TEST_CASE("String::toInt", "[noa][string]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t) {
    Errno err;
    TestType result;
    TestType min = std::numeric_limits<TestType>::min();
    TestType max = std::numeric_limits<TestType>::max();

    GIVEN("a string that can be converted to an integer") {
        std::vector<string> tests = {"1", " 6", "\t 7", "9.", "56", "011", "0 ", "10.3", "10e3",
                                     " 123  ", "0123", "0x9999910", fmt::format("  {},,", min),
                                     fmt::format("  {}  ", max)};
        std::vector<TestType> expected = {1, 6, 7, 9, 56, 11, 0, 10, 10, 123, 123, 0, min, max};

        for (size_t i{0}; i < tests.size(); ++i) {
            result = String::toInt<TestType>(tests[i], err);
            REQUIRE((result == expected[i] && err == Errno::good));
            err = Errno::good;
        }

        int8_t test1 = String::toInt<int8_t>(string{" -43"}, err);
        REQUIRE((test1 == -43 && err == Errno::good));
        short test2 = String::toInt<short>(string{"\t  -194"}, err);
        REQUIRE((test2 == -194 && err == Errno::good));
        int test3 = String::toInt(string{"-54052"}, err);
        REQUIRE((test3 == -54052 && err == Errno::good));
        long test4 = String::toInt<long>(string{"   -525107745"}, err);
        REQUIRE((test4 == -525107745 && err == Errno::good));
        uint64_t test5 = String::toInt<uint64_t>(string{"11111111155488"}, err);
        REQUIRE((test5 == 11111111155488 && err == Errno::good));
    }

    GIVEN("a string that cannot be converted to an integer") {
        auto to_test = GENERATE("     ", "", ".", " n10", "--10", "e10");
        String::toInt<TestType>(string{to_test}, err);
        REQUIRE(err == Errno::invalid_argument);
    }

    GIVEN("a string that falls out of range") {
        if (std::is_signed_v<TestType>)
            String::toInt<TestType>(fmt::format("  {}1,,", min), err);
        else
            String::toInt<TestType>(fmt::format("  -{}1,,", min), err);
        REQUIRE(err == Errno::out_of_range);

        err = Errno::good;
        String::toInt<TestType>(fmt::format("  {}1  ", max), err);
        REQUIRE(err == Errno::out_of_range);

        err = Errno::good;
        String::toInt<uint8_t>(string{"-1"}, err);
        REQUIRE(err == Errno::out_of_range);

        err = Errno::good;
        String::toInt<uint16_t>(string{"-1"}, err);
        REQUIRE(err == Errno::out_of_range);

        err = Errno::good;
        String::toInt<uint32_t>(string{"-1"}, err);
        REQUIRE(err == Errno::out_of_range);

        err = Errno::good;
        String::toInt<uint64_t>(string{"-1"}, err);
        REQUIRE(err == Errno::out_of_range);
    }
}

TEMPLATE_TEST_CASE("String::toFloat", "[noa][string]", float, double, long double) {
    Errno err;
    TestType result;

    GIVEN("a string that can be converted to a floating point") {
        WHEN("should return a number") {
            std::vector<string> tests = {"1", " 6", "\t7", "9.", ".5", "011", "-1", "123.123", ".0",
                                         "10x", "-10.3", "10e3", "10e-04", "0E-12", "09999910",
                                         "0x1273", "-0x1273"};
            std::vector<float> expected = {1, 6, 7, 9, .5, 11, -1, 123.123f, 0, 10, -10.3f,
                                           10e3, 10e-04f, 0e-12f, 9999910., 4723., -4723.};
            for (size_t i{0}; i < tests.size(); ++i) {
                result = String::toFloat<TestType>(tests[i], err);
                REQUIRE_THAT(result, Catch::WithinULP(expected[i], 2));
                REQUIRE(err == Errno::good);
            }
        }

        WHEN("should return NaN") {
            auto tests = GENERATE("nan", "Nan", "-NaN" "-nan");
            result = std::isnan(String::toFloat<TestType>(string{tests}, err));
            REQUIRE((result == true && err == Errno::good));
        }

        WHEN("should return Inf") {
            auto tests = GENERATE("inf", "-inf", "INFINITY" "-INFINITY", "-Inf");
            result = std::isinf(String::toFloat<TestType>(string{tests}, err));
            REQUIRE((result == true && err == Errno::good));
        }
    }

    GIVEN("a string that can be converted to a floating point") {
        auto tests = GENERATE("     ", "", ".", " n10", "--10", "e10");
        String::toFloat<TestType>(string{tests}, err);
        REQUIRE(err == Errno::invalid_argument);
    }

    GIVEN("a string that falls out of the floating point range") {
        TestType min = std::numeric_limits<TestType>::min();
        TestType max = std::numeric_limits<TestType>::max();
        TestType lowest = std::numeric_limits<TestType>::lowest();
        std::vector<string> tests = {fmt::format("  {}1,,", min),
                                     fmt::format("  {}1  ", max),
                                     fmt::format("  {}1  ", lowest)};
        for (auto& test: tests) {
            String::toFloat<TestType>(test, err);
            REQUIRE(err == Errno::out_of_range);
            err = Errno::good;
        }
    }
}

TEST_CASE("String::toBool should convert a string into a bool", "[noa][string]") {
    Errno err;
    bool result;

    GIVEN("a string that can be converted to a bool") {
        WHEN("should return true") {
            auto to_test = GENERATE("1", "true", "TRUE", "y", "yes", "YES", "on", "ON");
            result = String::toBool(string{to_test}, err);
            REQUIRE(result == true);
            REQUIRE_ERRNO_GOOD(err);
        }

        WHEN("should return false") {
            auto to_test = GENERATE("0", "false", "FALSE", "n", "no", "NO", "off", "OFF");
            result = String::toBool(string{to_test}, err);
            REQUIRE(result == false);
            REQUIRE_ERRNO_GOOD(err);
        }
    }

    GIVEN("a string that cannot be converted to a bool") {
        auto to_test = GENERATE(" y", "yes please", ".", "", " 0", "wrong");
        String::toBool(string{to_test}, err);
        REQUIRE(err == Errno::invalid_argument);
    }
}


// -------------------------------------------------------------------------------------------------
// parse
// -------------------------------------------------------------------------------------------------
TEMPLATE_TEST_CASE("String::parse to strings", "[noa][string]", std::string, std::string_view) {
    Errno err;
    std::vector<TestType> tests = {",1,2,3,4,5,",
                                   "1,2,3,4,5",
                                   "1,2, 3 ,4\n ,5 ,",
                                   "1 , 2 3\t   ,4 ,  5 6  7, ",
                                   " 1, 2,  ,  4 5",
                                   " ",
                                   "",
                                   " ,\n   ",
                                   "   1,2,3", " 1 , 2 , 3 , 4 5 67  "};
    std::vector<std::vector<string>> expected = {{"",  "1",   "2", "3",      "4", "5", ""},
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
        std::vector<string> result;
        for (size_t i = 0; i < tests.size(); ++i) {
            INFO(tests[i])
            err = String::parse(tests[i], result);
            REQUIRE_THAT(result, Catch::Equals(expected[i]));
            REQUIRE(err == Errno::good);
            result.clear();
        }
        err = String::parse(string{"123,foo,,dd2"}, result, 3);
        REQUIRE(err == Errno::invalid_size);
        err = String::parse(string{"123,foo,,dd2"}, result, 5);
        REQUIRE(err == Errno::invalid_size);
    }

    WHEN("output is an array") {
        for (size_t i = 0; i < tests.size(); ++i) {
            INFO(tests[i]);
            std::array<string, 1> arr1{};
            err = String::parse(tests[i], arr1);
            REQUIRE_RANGE_EQUALS_OR_INVALID_SIZE(arr1, expected[i], err)

            std::array<string, 2> arr2{};
            err = String::parse(tests[i], arr2);
            REQUIRE_RANGE_EQUALS_OR_INVALID_SIZE(arr2, expected[i], err)

            std::array<string, 3> arr3{};
            err = String::parse(tests[i], arr3);
            REQUIRE_RANGE_EQUALS_OR_INVALID_SIZE(arr3, expected[i], err)

            std::array<string, 4> arr4{};
            err = String::parse(tests[i], arr4);
            REQUIRE_RANGE_EQUALS_OR_INVALID_SIZE(arr4, expected[i], err)

            std::array<string, 5> arr5{};
            err = String::parse(tests[i], arr5);
            REQUIRE_RANGE_EQUALS_OR_INVALID_SIZE(arr5, expected[i], err)

            std::array<string, 6> arr6{};
            err = String::parse(tests[i], arr6);
            REQUIRE_RANGE_EQUALS_OR_INVALID_SIZE(arr6, expected[i], err)

            std::array<string, 7> arr7{};
            err = String::parse(tests[i], arr7);
            REQUIRE_RANGE_EQUALS_OR_INVALID_SIZE(arr7, expected[i], err)
        }
        std::array<string, 10> test1{};
        err = String::parse(string{"123,foo,,dd2"}, test1, 4);
        REQUIRE_ERRNO_GOOD(err);
        std::vector<string> expected1{"123", "foo", "", "dd2"};
        for (size_t i{0}; i < expected1.size(); ++i)
            REQUIRE(expected1[i] == test1[i]);
        err = String::parse(string{"123,foo,,dd2"}, test1, 3);
        REQUIRE(err == Errno::invalid_size);
        err = String::parse(string{"123,foo,,dd2"}, test1, 5);
        REQUIRE(err == Errno::invalid_size);
    }
}

TEMPLATE_TEST_CASE("String::parse to int", "[noa][string]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t) {
    Errno err;
    TestType min = std::numeric_limits<TestType>::min();
    TestType max = std::numeric_limits<TestType>::max();
    string str_min_max = fmt::format("  {}, {}", min, max);

    std::vector<string> tests_valid = {" 1, 6, \t7, 9. , 56, 011  \t, 0,10.3,10e3    ",
                                       "123  ,0123, 0x9999910,1 1", str_min_max};
    std::vector<std::vector<TestType>> expected_valid = {{1,   6,   7, 9, 56, 11, 0, 10, 10},
                                                         {123, 123, 0, 1},
                                                         {min, max}};
    std::vector<string> tests_invalid = {" ", "120, , 23", "120, 1, 23,", ",120, 1, 23,"};
    std::vector<std::vector<TestType>> expected_invalid = {{0},
                                                           {120, 0},
                                                           {120, 1, 23, 0},
                                                           {0}};
    GIVEN("a vector") {
        std::vector<TestType> result;

        WHEN("the input string can be converted") {
            for (size_t i{0}; i < tests_valid.size(); ++i) {
                err = String::parse(tests_valid[i], result);
                REQUIRE_THAT(result, Catch::Equals(expected_valid[i]));
                REQUIRE_ERRNO_GOOD(err);
                result.clear();
            }
        }

        WHEN("the input string cannot be converted") {
            for (size_t i{0}; i < tests_invalid.size(); ++i) {
                err = String::parse(tests_invalid[i], result);
                REQUIRE_THAT(result, Catch::Equals(expected_invalid[i]));
                REQUIRE(err == Errno::invalid_argument);
                result.clear();
            }
        }
    }

    GIVEN("an array") {
        WHEN("the input string can be converted") {
            for (size_t i{0}; i < tests_valid.size(); ++i) {
                INFO("idx: " << i);
                std::array<TestType, 2> arr2{};
                err = String::parse(tests_valid[i], arr2);
                REQUIRE_RANGE_EQUALS_OR_INVALID_SIZE(arr2, expected_valid[i], err)

                std::array<TestType, 4> arr4{};
                err = String::parse(tests_valid[i], arr4);
                REQUIRE_RANGE_EQUALS_OR_INVALID_SIZE(arr4, expected_valid[i], err)

                std::array<TestType, 9> arr9{};
                err = String::parse(tests_valid[i], arr9);
                REQUIRE_RANGE_EQUALS_OR_INVALID_SIZE(arr9, expected_valid[i], err)
            }
        }

        WHEN("the input string cannot be converted") {
            std::array<TestType, 4> result{};
            for (auto& i : tests_invalid) {
                err = String::parse(i, result);
                REQUIRE((err == Errno::invalid_argument || err == Errno::invalid_size));
            }
        }
    }
}

TEMPLATE_TEST_CASE("String::parse to float", "[noa][string]", float, double, long double) {
    Errno err;
    std::vector<string> tests = {" 1, 6., \t7, 9. , .56, 123.123, 011, -1, .0",
                                 "10x,-10.3  , 10e3  , 10e-04,0E-12    , 09999910"};
    std::vector<std::vector<float>> expected = {{1,  6,      7,     9,       .56f,   123.123f, 11, -1, .0},
                                                {10, -10.3f, 10e3f, 10e-04f, 0e-12f, 9999910.f}};

    GIVEN("a vector") {
        std::vector<TestType> result{};
        WHEN("the input string can be converted") {
            for (size_t i{0}; i < tests.size(); ++i) {
                err = String::parse(tests[i], result);
                REQUIRE_ERRNO_GOOD(err);
                REQUIRE_RANGE_EQUALS_ULP(result, expected[i], 2)
                result.clear();
            }

            WHEN("should return NaN") {
                string test = {"nan, Nan  , -NaN,-nan"};
                err = String::parse(test, result);
                REQUIRE(result.size() == 4);
                REQUIRE_ERRNO_GOOD(err);
                REQUIRE_FOR_ALL(result, std::isnan);
            }

            WHEN("should return Inf") {
                string test = {"inf, -inf , INFINITY ,-INFINITY,-Inf"};
                err = String::parse(test, result);
                REQUIRE(result.size() == 5);
                REQUIRE_ERRNO_GOOD(err);
                REQUIRE_FOR_ALL(result, std::isinf);
            }
        }

        WHEN("the input cannot be converted") {
            auto test = GENERATE("", "  ", ". ,10", "1, 2., n10", "3, --10", "0, e10");
            err = String::parse(string{test}, result);
            REQUIRE(err == Errno::invalid_argument);
        }
    }

    GIVEN("an array") {
        WHEN("the input string can be converted") {
            for (size_t i = 0; i < tests.size(); ++i) {
                std::array<TestType, 2> arr2{};
                err = String::parse(tests[i], arr2);
                REQUIRE_RANGE_EQUALS_ULP_OR_INVALID_SIZE(arr2, expected[i], 2, err)

                std::array<TestType, 4> arr4{};
                err = String::parse(tests[i], arr4);
                REQUIRE_RANGE_EQUALS_ULP_OR_INVALID_SIZE(arr4, expected[i], 2, err)

                std::array<TestType, 6> arr6{};
                err = String::parse(tests[i], arr6);
                REQUIRE_RANGE_EQUALS_ULP_OR_INVALID_SIZE(arr6, expected[i], 2, err)

                std::array<TestType, 9> arr9{};
                err = String::parse(tests[i], arr9);
                REQUIRE_RANGE_EQUALS_ULP_OR_INVALID_SIZE(arr9, expected[i], 2, err)
            }

            WHEN("should return NaN") {
                string test = {"nan, Nan  , -NaN,-nan"};
                std::array<TestType, 4> result{};
                err = String::parse(test, result);
                REQUIRE_FOR_ALL(result, std::isnan);
                REQUIRE_ERRNO_GOOD(err);
            }

            WHEN("should return Inf") {
                string test = {"inf, -inf , INFINITY ,-INFINITY,-Inf"};
                std::array<TestType, 5> result{};
                err = String::parse(test, result);
                REQUIRE_FOR_ALL(result, std::isinf);
                REQUIRE_ERRNO_GOOD(err);
            }
        }

        WHEN("the input cannot be converted") {
            std::array<TestType, 1> result{};
            auto test = GENERATE("", "  ", ". ", "n10", "--10", "e10");
            err = String::parse(string{test}, result);
            REQUIRE(err == Errno::invalid_argument);
        }
    }
}

TEST_CASE("String::parse to bool", "[noa][string]") {
    Errno err;
    string test = "1,true,   TRUE, y,yes, YES,on, ON,0,false,False  ,n,no, NO, ofF , OFF";
    std::vector<bool> expected = {true, true, true, true, true, true, true, true,
                                  false, false, false, false, false, false, false, false};

    GIVEN("a vector") {
        std::vector<bool> vec;
        err = String::parse(test, vec);
        REQUIRE(err == Errno::good);
        REQUIRE_THAT(vec, Catch::Equals(expected));
    }

    GIVEN("an array") {
        std::array<bool, 16> arr{};
        err = String::parse(test, arr);
        REQUIRE(err == Errno::good);
        REQUIRE_THAT(Test::toVector(arr), Catch::Equals(expected));
    }
}

TEMPLATE_TEST_CASE("String::parse with default values", "[noa][string]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                   float, double, long double) {
    using vector = std::vector<TestType>;

    GIVEN("a vector") {
        string test1 = "123,,12, 0, \t,, 8";
        string test2 = ",1,2,3,4,5,";
        vector vec;
        Errno err = String::parse(test1, test2, vec);
        REQUIRE(err == Errno::good);
        REQUIRE_THAT(vec, Catch::Equals(vector{123, 1, 12, 0, 4, 5, 8}));

        test1 = "12,,12, 0, \t,, 8";
        test2 = ",1,2,3,4,5,";
        std::vector<string> vec1;
        err = String::parse(test1, test2, vec1);
        REQUIRE(err == Errno::good);
        REQUIRE_THAT(vec1, Catch::Equals(std::vector<string>{"12", "1", "12", "0", "4", "5", "8"}));

        test1 = "1,,0, 0, \t,, 1";
        test2 = ",1,1,3,1,1,12";
        std::vector<bool> vec2;
        err = String::parse(test1, test2, vec2);
        REQUIRE(err == Errno::good);
        REQUIRE_THAT(vec2, Catch::Equals(std::vector<bool>{1, 1, 0, 0, 1, 1, 1}));

        test1 = "1,,0, 0, \t,, 1";
        test2 = ",1,1,3,1,  ,12";
        vector vec3;
        err = String::parse(test1, test2, vec3);
        REQUIRE(err == Errno::invalid_argument);
    }

    GIVEN("an array") {
        string test1 = "123,,12, 0, \t,, 8";
        string test2 = ",1,2,3,4,5,";
        std::array<TestType, 7> result1{};
        std::vector<float> expected1{123, 1, 12, 0, 4, 5, 8};
        Errno err = String::parse(test1, test2, result1);
        REQUIRE_RANGE_EQUALS_ULP(result1, expected1, 2)
        REQUIRE_ERRNO_GOOD(err);

        test1 = "123,,12, 0, \t,, 8";
        test2 = ",1,2,3,4,5,";
        std::array<string, 7> result2{};
        std::vector<string> expected2{"123", "1", "12", "0", "4", "5", "8"};
        err = String::parse(test1, test2, result2);
        REQUIRE_RANGE_EQUALS(result2, expected2)
        REQUIRE_ERRNO_GOOD(err);

        test1 = "1,,0, 0, \t,, 1";
        test2 = ",1,1,3,1,1,12";
        std::array<bool, 7> result3{};
        std::vector<bool> expected3{true, true, false, false, true, true, true};
        err = String::parse(test1, test2, result3);
        REQUIRE_RANGE_EQUALS(result3, expected3)
        REQUIRE_ERRNO_GOOD(err);

        WHEN("the inputs are invalid") {
            test1 = "1,,0, 0, \t,, 1";
            test2 = ",1,1,3,1,  ,12";
            std::array<TestType, 7> result4{};
            err = String::parse(test1, test2, result4);
            REQUIRE(err == Errno::invalid_argument);

            test1 = "1,,0, 0, \t,, 1";
            test2 = ",1,1,3,1, 1 ,12";
            std::array<TestType, 10> result5{};
            err = String::parse(test1, test2, result5);
            REQUIRE(err == Errno::invalid_size);

            std::array<TestType, 4> result6{};
            err = String::parse(test1, test2, result6);
            REQUIRE(err == Errno::invalid_size);
        }
    }
}

TEMPLATE_TEST_CASE("String::parse to IntX", "[noa][string]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t) {
    Errno err;

    //@CLION-formatter:off
    GIVEN("Int2") {
        std::vector<string> tests = {"1,2", "  45, 23", "0, 3 "};
        std::vector<Int2<TestType>> expected = {{1, 2}, {45, 23}, {0, 3}};
        for (size_t idx{0}; idx < tests.size(); ++idx) {
            Int2<TestType> result2;
            err = String::parse(tests[idx], result2);
            REQUIRE_ERRNO_GOOD(err);
            REQUIRE(result2 == expected[idx]);

            err = String::parse(string{"123, "}, result2);
            REQUIRE(err == Errno::invalid_argument);

            Int3<TestType> result3;
            err = String::parse(tests[idx], result3);
            REQUIRE(err == Errno::invalid_size);

            Int4<TestType> result4;
            err = String::parse(tests[idx], result4);
            REQUIRE(err == Errno::invalid_size);
        }
    }

    GIVEN("Int3") {
        std::vector<string> tests = {"1,2, 23", " 2, 45, 23", "0, 3   \n, 127"};
        std::vector<Int3<TestType>> expected = {{1, 2, 23}, {2, 45, 23}, {0, 3, 127}};
        for (size_t idx{0}; idx < tests.size(); ++idx) {
            Int3<TestType> result3;
            err = String::parse(tests[idx], result3);
            REQUIRE_ERRNO_GOOD(err);
            REQUIRE(result3 == expected[idx]);

            err = String::parse(string{", ,"}, result3);
            REQUIRE(err == Errno::invalid_argument);

            Int2<TestType> result2;
            err = String::parse(tests[idx], result2);
            REQUIRE(err == Errno::invalid_size);

            Int4<TestType> result4;
            err = String::parse(tests[idx], result4);
            REQUIRE(err == Errno::invalid_size);
        }
    }

    GIVEN("Int4") {
        std::vector<string> tests = {"1,2, 23, 34", "\t2, 2, 45, 23", "0, 3  \n \n, 127, 3 "};
        std::vector<Int4<TestType>> expected = {{1, 2, 23, 34}, {2, 2, 45, 23}, {0, 3, 127, 3}};
        for (size_t idx{0}; idx < tests.size(); ++idx) {
            Int4<TestType> result4;
            err = String::parse(tests[idx], result4);
            REQUIRE_ERRNO_GOOD(err);
            REQUIRE(result4 == expected[idx]);

            err = String::parse(string{"12, 1,2,"}, result4);
            REQUIRE(err == Errno::invalid_argument);

            Int2<TestType> result2;
            err = String::parse(tests[idx], result2);
            REQUIRE(err == Errno::invalid_size);

            Int3<TestType> result3;
            err = String::parse(tests[idx], result3);
            REQUIRE(err == Errno::invalid_size);
        }
    }
    //@CLION-formatter:on
}

#define F(x) static_cast<TestType>(x)

TEMPLATE_TEST_CASE("String::parse to FloatX", "[noa][string]", float, double) {
    Errno err;

    //@CLION-formatter:off
    GIVEN("Float2") {
        std::vector<string> tests = {"1.32,-223.234f", "  452, 23.", ".0, 1e-3 "};
        std::vector<Float2<TestType>> expected = {{F(1.32), F(-223.234)}, {452, 23}, {0, F(1e-3)}};
        for (size_t idx{0}; idx < tests.size(); ++idx) {
            Float2<TestType> result2;
            err = String::parse(tests[idx], result2);
            REQUIRE_ERRNO_GOOD(err);
            REQUIRE(Math::isEqual(result2, expected[idx]));

            err = String::parse(string{"123, "}, result2);
            REQUIRE(err == Errno::invalid_argument);

            Float3<TestType> result3;
            err = String::parse(tests[idx], result3);
            REQUIRE(err == Errno::invalid_size);

            Float4<TestType> result4;
            err = String::parse(tests[idx], result4);
            REQUIRE(err == Errno::invalid_size);
        }
    }

    GIVEN("Float3") {
        std::vector<string> tests = {"-1,2, 23.2", " 2, 45, 23", "0.01, 3.   \n, -127.234"};
        std::vector<Float3<TestType>> expected = {{-1, 2, F(23.2)}, {2, 45, 23}, {F(0.01), 3, F(-127.234)}};
        for (size_t idx{0}; idx < tests.size(); ++idx) {
            Float3<TestType> result3;
            err = String::parse(tests[idx], result3);
            REQUIRE_ERRNO_GOOD(err);
            REQUIRE(Math::isEqual(result3, expected[idx]));

            err = String::parse(string{", ,"}, result3);
            REQUIRE(err == Errno::invalid_argument);

            Float2<TestType> result2;
            err = String::parse(tests[idx], result2);
            REQUIRE(err == Errno::invalid_size);

            Float4<TestType> result4;
            err = String::parse(tests[idx], result4);
            REQUIRE(err == Errno::invalid_size);
        }
    }

    GIVEN("Float4") {
        std::vector<string> tests = {"1,2, 23, 34", "\t2e2, 2.f, 45d, 23.99", "0, 3  \n \n, 127, 3 "};
        std::vector<Float4<TestType>> expected = {{1, 2, 23, 34}, {200, 2, 45, F(23.99)}, {0, 3, 127, 3}};
        for (size_t idx{0}; idx < tests.size(); ++idx) {
            Float4<TestType> result4;
            err = String::parse(tests[idx], result4);
            REQUIRE_ERRNO_GOOD(err);
            REQUIRE(Math::isEqual(result4, expected[idx]));

            err = String::parse(string{"12, 1,2,"}, result4);
            REQUIRE(err == Errno::invalid_argument);

            Float2<TestType> result2;
            err = String::parse(tests[idx], result2);
            REQUIRE(err == Errno::invalid_size);

            Float3<TestType> result3;
            err = String::parse(tests[idx], result3);
            REQUIRE(err == Errno::invalid_size);
        }
    }
    //@CLION-formatter:on
}

#undef F
