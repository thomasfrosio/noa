/*
 * Test noa/util/String.h
 */

#include <catch2/catch.hpp>

#include "noa/util/String.h"


// -------------------------------------------------------------------------------------------------
// Trim
// -------------------------------------------------------------------------------------------------
SCENARIO("String::leftTrim(Copy) should remove spaces on the left", "[noa][string]") {
    using namespace ::Noa::String;
    GIVEN("a rvalue") {
        REQUIRE(leftTrim("").empty());
        REQUIRE(leftTrim("  ").empty());
        REQUIRE(leftTrim("  foo ") == "foo ");
        REQUIRE(leftTrim("  \tfoo") == "foo");
        REQUIRE(leftTrim("  \n foo") == "foo");
        REQUIRE(leftTrim("  \rfoo") == "foo");
        REQUIRE(leftTrim("foo bar ") == "foo bar ");
    }

    GIVEN("a lvalue") {
        std::string str;
        REQUIRE(leftTrim(str).empty());
        str = "  ";
        REQUIRE(leftTrim(str).empty());
        str = "  foo ";
        REQUIRE(leftTrim(str) == "foo ");
        str = "  \tfoo";
        REQUIRE(leftTrim(str) == "foo");
        str = "  \n foo";
        REQUIRE(leftTrim(str) == "foo");
        str = "  \rfoo";
        REQUIRE(leftTrim(str) == "foo");
        str = "foo bar ";
        REQUIRE(leftTrim(str) == "foo bar ");

        THEN("the input string is modified") {
            str = "   foo  bar ";
            leftTrim(str);
            REQUIRE(str == "foo  bar ");
        }

        THEN("a lvalue should be returned") {
            str = "\t  \n 123; \n";
            std::string& out = leftTrim(str);
            REQUIRE(str == "123; \n");
            REQUIRE(out == "123; \n");
        }
    }

    GIVEN("a value") {
        REQUIRE(leftTrimCopy("").empty());
        REQUIRE(leftTrimCopy("  ").empty());
        REQUIRE(leftTrimCopy("  foo ") == "foo ");
        REQUIRE(leftTrimCopy("  \tfoo  ") == "foo  ");
        REQUIRE(leftTrimCopy("  \n foo") == "foo");
        REQUIRE(leftTrimCopy("  \rfoo ") == "foo ");
        REQUIRE(leftTrimCopy("foo bar ") == "foo bar ");

        std::string str;
        REQUIRE(leftTrimCopy(str).empty());
        str = "  ";
        REQUIRE(leftTrimCopy(str).empty());
        str = "  foo\t ";
        REQUIRE(leftTrimCopy(str) == "foo\t ");
        str = "  \tfoo";
        REQUIRE(leftTrimCopy(str) == "foo");
        str = "  \n foo";
        REQUIRE(leftTrimCopy(str) == "foo");
        str = ".  \rfoo ";
        REQUIRE(leftTrimCopy(str) == ".  \rfoo ");
        str = "foo bar ";
        REQUIRE(leftTrimCopy(str) == "foo bar ");

        THEN("the input string isn't modified") {
            str = "   foo  bar ";
            std::string out = leftTrimCopy(str);
            REQUIRE(str == "   foo  bar ");
            REQUIRE(out == "foo  bar ");
        }
    }
}


SCENARIO("Noa::String::rightTrim should remove spaces on the right", "[noa][string]") {
    using namespace ::Noa::String;
    GIVEN("a rvalue") {
        REQUIRE(rightTrim("").empty());
        REQUIRE(rightTrim("  ").empty());
        REQUIRE(rightTrim(" foo  ") == " foo");
        REQUIRE(rightTrim(" foo \t") == " foo");
        REQUIRE(rightTrim("  \nfoo  \n") == "  \nfoo");
        REQUIRE(rightTrim(" foo\r ") == " foo");
        REQUIRE(rightTrim(" foo bar ") == " foo bar");
    }

    GIVEN("a lvalue") {
        std::string str;
        REQUIRE(rightTrim(str).empty());
        str = "  ";
        REQUIRE(rightTrim(str).empty());
        str = "  foo  ";
        REQUIRE(rightTrim(str) == "  foo");
        str = "  \tfoo \t";
        REQUIRE(rightTrim(str) == "  \tfoo");
        str = "foo \n  \n";
        REQUIRE(rightTrim(str) == "foo");
        str = "  \rfoo\r";
        REQUIRE(rightTrim(str) == "  \rfoo");
        str = "foo bar \n foo ";
        REQUIRE(rightTrim(str) == "foo bar \n foo");

        THEN("the input string is modified") {
            str = "   foo  bar ";
            rightTrim(str);
            REQUIRE(str == "   foo  bar");
        }

        THEN("a lvalue should be returned") {
            str = "\t  \n 123; \n";
            std::string& out = rightTrim(str);
            REQUIRE(str == "\t  \n 123;");
            REQUIRE(out == "\t  \n 123;");
        }
    }

    GIVEN("a value") {
        REQUIRE(rightTrimCopy("").empty());
        REQUIRE(rightTrimCopy("  ").empty());
        REQUIRE(rightTrimCopy("  foo ") == "  foo");
        REQUIRE(rightTrimCopy(". foo \t ") == ". foo");
        REQUIRE(rightTrimCopy("  \n foo") == "  \n foo");
        REQUIRE(rightTrimCopy("  \rfoo , ") == "  \rfoo ,");
        REQUIRE(rightTrimCopy("foo bar ") == "foo bar");

        std::string str;
        REQUIRE(rightTrimCopy(str).empty());
        str = "  ";
        REQUIRE(rightTrimCopy(str).empty());
        str = "  foo\t ";
        REQUIRE(rightTrimCopy(str) == "  foo");
        str = "foo  \t";
        REQUIRE(rightTrimCopy(str) == "foo");
        str = "foo \n  ";
        REQUIRE(rightTrimCopy(str) == "foo");
        str = ".  \rfoo ";
        REQUIRE(rightTrimCopy(str) == ".  \rfoo");
        str = "foo bar";
        REQUIRE(rightTrimCopy(str) == "foo bar");

        THEN("the input string isn't modified") {
            str = "   foo  bar ";
            std::string out = rightTrimCopy(str);
            REQUIRE(str == "   foo  bar ");
            REQUIRE(out == "   foo  bar");
        }
    }
}


SCENARIO("Noa::String::trim should remove spaces on the right _and_ left", "[noa][string]") {
    using namespace ::Noa::String;
    using vec_t = std::vector<std::string>;

    GIVEN("a string as a rvalue (in-place)") {
        REQUIRE(trim("").empty());
        REQUIRE(trim("  ").empty());
        REQUIRE(trim(" foo  ") == "foo");
        REQUIRE(trim(" foo \t") == "foo");
        REQUIRE(trim("  \nfoo  \n") == "foo");
        REQUIRE(trim(" foo\r ") == "foo");
        REQUIRE(trim(" foo bar ") == "foo bar");
    }

    GIVEN("a string as a lvalue (in-place)") {
        std::string str;
        REQUIRE(trim(str).empty());
        str = "  ";
        REQUIRE(trim(str).empty());
        str = "  foo  ";
        REQUIRE(trim(str) == "foo");
        str = "  \tfoo \t";
        REQUIRE(trim(str) == "foo");
        str = "foo \n  \n";
        REQUIRE(trim(str) == "foo");
        str = "  \rfoo\r";
        REQUIRE(trim(str) == "foo");
        str = "  foo bar \n foo ";
        REQUIRE(trim(str) == "foo bar \n foo");

        THEN("the input string is modified") {
            str = "   foo  bar ";
            trim(str);
            REQUIRE(str == "foo  bar");
        }

        THEN("a lvalue should be returned") {
            str = "\t  \n 123; \n";
            std::string& out = trim(str);
            REQUIRE(str == "123;");
            REQUIRE(out == "123;");
        }
    }

    GIVEN("a string as a value (copy)") {
        REQUIRE(trimCopy("").empty());
        REQUIRE(trimCopy("  ").empty());
        REQUIRE(trimCopy("  foo ") == "foo");
        REQUIRE(trimCopy(". foo \t ") == ". foo");
        REQUIRE(trimCopy("  \n foo") == "foo");
        REQUIRE(trimCopy("  \rfoo , ") == "foo ,");
        REQUIRE(trimCopy("foo bar ") == "foo bar");

        std::string str;
        REQUIRE(trimCopy(str).empty());
        str = "  ";
        REQUIRE(trimCopy(str).empty());
        str = "  foo\t ";
        REQUIRE(trimCopy(str) == "foo");
        str = "foo  \t";
        REQUIRE(trimCopy(str) == "foo");
        str = "foo \n  ";
        REQUIRE(trimCopy(str) == "foo");
        str = ".  \rfoo ";
        REQUIRE(trimCopy(str) == ".  \rfoo");
        str = "foo bar";
        REQUIRE(trimCopy(str) == "foo bar");

        THEN("the input string isn't modified") {
            str = "   foo  bar ";
            std::string out = trimCopy(str);
            REQUIRE(str == "   foo  bar ");
            REQUIRE(out == "foo  bar");
        }
    }

    vec_t input;
    GIVEN("a vector of string(s) as a lvalue (in-place)") {
        REQUIRE_THAT(trim(input), Catch::Equals(vec_t({})));
        input = {""};
        REQUIRE_THAT(trim(input), Catch::Equals(vec_t({""})));
        input = {"foo", "  foo\t", " bar   ", "", "\n", " "};
        REQUIRE_THAT(trim(input), Catch::Equals(vec_t({"foo", "foo", "bar", "", "", ""})));
        input = {"", "", ". ", ". foo. "};
        REQUIRE_THAT(trim(input), Catch::Equals(vec_t({"", "", ".", ". foo."})));

        WHEN("the input vector is modified") {
            input = {"foo  ", "  foo\t ", ". ", ""};
            trim(input);
            REQUIRE_THAT(input, Catch::Equals(vec_t({"foo", "foo", ".", ""})));
        }
    }

    input.clear();
    GIVEN("a vector of string(s) as a value (copy)") {
        REQUIRE_THAT(trimCopy(input), Catch::Equals(vec_t({})));
        input = {""};
        REQUIRE_THAT(trimCopy(input), Catch::Equals(vec_t({""})));
        input = {"foo", "  foo\t", " bar   ", "", "\n", " "};
        REQUIRE_THAT(trimCopy(input), Catch::Equals(vec_t({"foo", "foo", "bar", "", "", ""})));
        input = {"", "", ". ", ". foo. "};
        REQUIRE_THAT(trimCopy(input), Catch::Equals(vec_t({"", "", ".", ". foo."})));

        WHEN("the input vector isn't modified") {
            input = {" foo  ", "  foo \t ", ". ", "  .   "};
            vec_t out = trimCopy(input);
            REQUIRE_THAT(input, Catch::Equals(vec_t({" foo  ", "  foo \t ", ". ", "  .   "})));
            REQUIRE_THAT(out, Catch::Equals(vec_t({"foo", "foo", ".", "."})));
        }
    }
}


// -------------------------------------------------------------------------------------------------
// String to scalar (integers, floating points, bool)
// -------------------------------------------------------------------------------------------------
TEMPLATE_TEST_CASE("Noa::String::toInt should convert a string into an int", "[noa][string]",
                   int8_t, short, int, long, long long, uint8_t, unsigned short, unsigned int,
                   unsigned long, unsigned long long) {
    using namespace ::Noa;
    using str = std::string;

    WHEN("asking for a signed integer") {
        uint8_t err = 0;
        TestType test = String::toInt<TestType>(str{"1"}, err);
        REQUIRE((test == 1 && err == 0));
        test = String::toInt<TestType>(str{" 6"}, err);
        REQUIRE((test == 6 && err == 0));
        test = String::toInt<TestType>(str{"\t 7"}, err);
        REQUIRE((test == 7 && err == 0));
        test = String::toInt<TestType>(str{"9."}, err);
        REQUIRE((test == 9 && err == 0));
        test = String::toInt<TestType>(str{"56"}, err);
        REQUIRE((test == 56 && err == 0));
        test = String::toInt<TestType>(str{"011"}, err);
        REQUIRE((test == 11 && err == 0));
        test = String::toInt<TestType>(str{"0 "}, err);
        REQUIRE((test == 0 && err == 0));
        test = String::toInt<TestType>(str{"10.3"}, err);
        REQUIRE((test == 10 && err == 0));
        test = String::toInt<TestType>(str{"10e3"}, err);
        REQUIRE((test == 10 && err == 0));
        test = String::toInt<TestType>(str{" 123  "}, err);
        REQUIRE((test == 123 && err == 0));
        test = String::toInt<TestType>(str{"0123"}, err);
        REQUIRE((test == 123 && err == 0));
        test = String::toInt<TestType>(str{"0x9999910"}, err);
        REQUIRE((test == 0 && err == 0));

        TestType min = std::numeric_limits<TestType>::min();
        TestType max = std::numeric_limits<TestType>::max();
        std::string str_min{fmt::format("  {},,", min)};
        std::string str_max{fmt::format("  {}  ", max)};
        test = String::toInt<TestType>(str_min, err);
        REQUIRE((test == min && err == 0));
        test = String::toInt<TestType>(str_max, err);
        REQUIRE((test == max && err == 0));

        int8_t test1 = String::toInt<int8_t>(str{" -43"}, err);
        REQUIRE((test1 == -43 && err == 0));
        short test2 = String::toInt<short>(str{"\t  -194"}, err);
        REQUIRE((test2 == -194 && err == 0));
        int test3 = String::toInt(str{"-54052"}, err);
        REQUIRE((test3 == -54052 && err == 0));
        long test4 = String::toInt<long>(str{"   -525107745"}, err);
        REQUIRE((test4 == -525107745 && err == 0));
        long long test5 = String::toInt<long long>(str{"-11111111155488"}, err);
        REQUIRE((test5 == -11111111155488 && err == 0));
    }

    WHEN("should raise an invalid argument error") {
        uint8_t err = 0;
        String::toInt<TestType>(str{""}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
        String::toInt<TestType>(str{"    "}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
        String::toInt<TestType>(str{"."}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
        String::toInt<TestType>(str{" n10"}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
        String::toInt<TestType>(str{"--10"}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
        String::toInt<TestType>(str{"e10"}, err);
        REQUIRE(err == Errno::invalid_argument);
    }

    GIVEN("should raise an out of range error") {
        uint8_t err = 0;
        TestType min = std::numeric_limits<TestType>::min();
        TestType max = std::numeric_limits<TestType>::max();
        if (std::is_signed_v<TestType>) {
            String::toInt<TestType>(fmt::format("  {}1,,", min), err);
        } else {
            String::toInt<TestType>(fmt::format("  -{}1,,", min), err);
        }
        REQUIRE(err == Errno::out_of_range);
        err = 0;
        String::toInt<TestType>(fmt::format("  {}1  ", max), err);
        REQUIRE(err == Errno::out_of_range);
        err = 0;

        String::toInt<uint8_t>(str{"-1"}, err);
        REQUIRE(err == Errno::out_of_range);
        err = 0;
        String::toInt<unsigned short>(str{"-1"}, err);
        REQUIRE(err == Errno::out_of_range);
        err = 0;
        String::toInt<unsigned int>(str{"-1"}, err);
        REQUIRE(err == Errno::out_of_range);
        err = 0;
        String::toInt<unsigned long>(str{"-1"}, err);
        REQUIRE(err == Errno::out_of_range);
        err = 0;
        String::toInt<unsigned long long>(str{"-1"}, err);
        REQUIRE(err == Errno::out_of_range);
    }
}


TEMPLATE_TEST_CASE("Noa::String::toFloat should convert a string into a float", "[noa][string]",
                   float, double, long double) {
    using namespace ::Noa;
    using str = std::string;

    WHEN("asking for a floating point") {
        uint8_t err = 0;
        TestType test;
        test = String::toFloat<TestType>(str{"1"}, err);
        REQUIRE((test == static_cast<TestType>(1.) && err == 0));
        test = String::toFloat<TestType>(str{" 6"}, err);
        REQUIRE((test == static_cast<TestType>(6.) && err == 0));
        test = String::toFloat<TestType>(str{"\t7"}, err);
        REQUIRE((test == static_cast<TestType>(7.) && err == 0));
        test = String::toFloat<TestType>(str{"9."}, err);
        REQUIRE((test == static_cast<TestType>(9.) && err == 0));
        test = String::toFloat<TestType>(str{".5"}, err);
        REQUIRE((test == static_cast<TestType>(.5) && err == 0));
        test = String::toFloat<TestType>(str{"123.123"}, err);
        REQUIRE_THAT(test, Catch::WithinAbs(123.123, 0.00005));
        REQUIRE(err == 0);
        test = String::toFloat<TestType>(str{"011"}, err);
        REQUIRE((test == static_cast<TestType>(11.) && err == 0));
        test = String::toFloat<TestType>(str{"-1"}, err);
        REQUIRE((test == static_cast<TestType>(-1.) && err == 0));
        test = String::toFloat<TestType>(str{".0"}, err);
        REQUIRE((test == static_cast<TestType>(0.) && err == 0));
        test = String::toFloat<TestType>(str{"10x"}, err);
        REQUIRE((test == static_cast<TestType>(10.) && err == 0));
        test = String::toFloat<TestType>(str{"-10.3"}, err);
        REQUIRE_THAT(test, Catch::WithinAbs(-10.3, 0.00005));
        REQUIRE(err == 0);
        test = String::toFloat<TestType>(str{"10e3"}, err);
        REQUIRE((test == static_cast<TestType>(10e3) && err == 0));
        test = String::toFloat<TestType>(str{"10e-04"}, err);
        REQUIRE_THAT(test, Catch::WithinAbs(10e-04, 0.000005));
        REQUIRE(err == 0);
        test = String::toFloat<TestType>(str{"0E-12"}, err);
        REQUIRE((test == static_cast<TestType>(0e-12) && err == 0));
        test = String::toFloat<TestType>(str{"09999910"}, err);
        REQUIRE((test == static_cast<TestType>(9999910.) && err == 0));

        REQUIRE((std::isnan(String::toFloat<TestType>(str{"nan"}, err)) && err == 0));
        REQUIRE((std::isnan(String::toFloat<TestType>(str{"NaN"}, err)) && err == 0));
        REQUIRE((std::isnan(String::toFloat<TestType>(str{"-NaN"}, err)) && err == 0));
        REQUIRE((std::isinf(String::toFloat<TestType>(str{"INFINITY"}, err)) && err == 0));
        REQUIRE((std::isinf(String::toFloat<TestType>(str{"inf"}, err)) && err == 0));
        REQUIRE((std::isinf(String::toFloat<TestType>(str{"-Inf"}, err)) && err == 0));

        test = String::toFloat<TestType>(str{"0x1273"}, err);
        REQUIRE((test == static_cast<TestType>(4723.) && err == 0));
        test = String::toFloat<TestType>(str{"-0x1273"}, err);
        REQUIRE((test == static_cast<TestType>(-4723.) && err == 0));
    }

    WHEN("should raise an invalid argument error") {
        uint8_t err = 0;
        String::toFloat<TestType>(str{""}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
        String::toFloat<TestType>(str{"    "}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
        String::toFloat<TestType>(str{"."}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
        String::toFloat<TestType>(str{" n10"}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
        String::toFloat<TestType>(str{"--10"}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
        String::toFloat<TestType>(str{"e10"}, err);
        REQUIRE(err == Errno::invalid_argument);
    }

    GIVEN("out of range string") {
        uint8_t err = 0;
        TestType min = std::numeric_limits<TestType>::min();
        TestType max = std::numeric_limits<TestType>::max();
        TestType lowest = std::numeric_limits<TestType>::lowest();
        String::toFloat<TestType>(fmt::format("  {}1,,", min), err);
        REQUIRE(err == Errno::out_of_range);
        err = 0;
        String::toFloat<TestType>(fmt::format("  {}1  ", max), err);
        REQUIRE(err == Errno::out_of_range);
        err = 0;
        String::toFloat<TestType>(fmt::format("  {}1  ", lowest), err);
        REQUIRE(err == Errno::out_of_range);
    }
}


SCENARIO("Noa::String::toBool should convert a string into a bool", "[noa][string]") {
    using namespace ::Noa;
    using str = std::string;

    GIVEN("a valid string") {
        uint8_t err = 0;
        bool test;
        test = String::toBool(str{"1"}, err);
        REQUIRE((test == true && err == 0));
        test = String::toBool(str{"true"}, err);
        REQUIRE((test == true && err == 0));
        test = String::toBool(str{"TRUE"}, err);
        REQUIRE((test == true && err == 0));
        test = String::toBool(str{"y"}, err);
        REQUIRE((test == true && err == 0));
        test = String::toBool(str{"yes"}, err);
        REQUIRE((test == true && err == 0));
        test = String::toBool(str{"YES"}, err);
        REQUIRE((test == true && err == 0));
        test = String::toBool(str{"on"}, err);
        REQUIRE((test == true && err == 0));
        test = String::toBool(str{"ON"}, err);
        REQUIRE((test == true && err == 0));

        test = String::toBool(str{"0"}, err);
        REQUIRE((test == false && err == 0));
        test = String::toBool(str{"false"}, err);
        REQUIRE((test == false && err == 0));
        test = String::toBool(str{"FALSE"}, err);
        REQUIRE((test == false && err == 0));
        test = String::toBool(str{"n"}, err);
        REQUIRE((test == false && err == 0));
        test = String::toBool(str{"no"}, err);
        REQUIRE((test == false && err == 0));
        test = String::toBool(str{"NO"}, err);
        REQUIRE((test == false && err == 0));
        test = String::toBool(str{"off"}, err);
        REQUIRE((test == false && err == 0));
        test = String::toBool(str{"OFF"}, err);
        REQUIRE((test == false && err == 0));
    }

    GIVEN("a invalid string") {
        uint8_t err = 0;
        String::toBool(str{" y"}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
        String::toBool(str{"yes please"}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
        String::toBool(str{"."}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
        String::toBool(str{""}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
        String::toBool(str{" 0"}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
        String::toBool(str{"wrong"}, err);
        REQUIRE(err == Errno::invalid_argument);
        err = 0;
    }
}


// -------------------------------------------------------------------------------------------------
// parse
// -------------------------------------------------------------------------------------------------
TEMPLATE_TEST_CASE("Noa::String::parse should parse strings", "[noa][string]",
                   std::string, std::string_view) {
    using namespace ::Noa;
    using vec_t = std::vector<std::string>;

    WHEN("output is a vector of string") {
        uint8_t err = 0;
        std::vector<std::string> vec;
        err = String::parse<TestType>("1,2,3,4,5", vec);
        REQUIRE_THAT(vec, Catch::Equals(vec_t{"1", "2", "3", "4", "5"}));
        REQUIRE(err == 0);
        vec.clear();
        err = String::parse<TestType>(",1,2,3,4,5,", vec);
        REQUIRE_THAT(vec, Catch::Equals(vec_t{"", "1", "2", "3", "4", "5", ""}));
        REQUIRE(err == 0);
        vec.clear();
        err = String::parse<TestType>(" ,\n1,2 ,3,4 ,5,", vec);
        REQUIRE_THAT(vec, Catch::Equals(vec_t{"", "1", "2", "3", "4", "5", ""}));
        REQUIRE(err == 0);
        vec.clear();
        err = String::parse<TestType>("1,2, 3 ,4\n ,5 ,", vec);
        REQUIRE_THAT(vec, Catch::Equals(vec_t{"1", "2", "3", "4", "5", ""}));
        REQUIRE(err == 0);
        vec.clear();
        err = String::parse<TestType>("1  2 3\t   4 ,  5 , ", vec);
        REQUIRE_THAT(vec, Catch::Equals(vec_t{"1  2 3\t   4", "5", ""}));
        REQUIRE(err == 0);
        vec.clear();
        err = String::parse<TestType>(" 1, 2,  ,  4 5", vec);
        REQUIRE_THAT(vec, Catch::Equals(vec_t{"1", "2", "", "4 5"}));
        REQUIRE(err == 0);
        vec.clear();
        err = String::parse<TestType>(" ", vec);
        REQUIRE_THAT(vec, Catch::Equals(vec_t{""}));
        REQUIRE(err == 0);
        vec.clear();
        err = String::parse<TestType>("", vec);
        REQUIRE_THAT(vec, Catch::Equals(vec_t{""}));
        REQUIRE(err == 0);
        vec.clear();
        err = String::parse<TestType>("  ", vec);
        REQUIRE_THAT(vec, Catch::Equals(vec_t{""}));
        REQUIRE(err == 0);
        vec.clear();
        err = String::parse<TestType>(" ,\n   ", vec);
        REQUIRE_THAT(vec, Catch::Equals(vec_t{"", ""}));
        REQUIRE(err == 0);
        vec.clear();

        TestType s0{"   1,2,3"};
        err = String::parse(s0, vec);
        REQUIRE_THAT(vec, Catch::Equals(vec_t{"1", "2", "3"}));
        REQUIRE(err == 0);
        vec.clear();
        TestType s1{" 1 , 2 , 3 , 4 5 67  "};
        err = String::parse(s1, vec);
        REQUIRE_THAT(vec, Catch::Equals(vec_t{"1", "2", "3", "4 5 67"}));
        REQUIRE(err == 0);
        vec.clear();
    }

    WHEN("output is a array of string") {
        uint8_t err = 0;
        std::array<std::string, 4> arr0;
        err = String::parse<TestType>("1,2,3,4", arr0);
        REQUIRE((arr0[0] == "1" && arr0[1] == "2" && arr0[2] == "3" && arr0[3] == "4" && err == 0));
        std::array<std::string, 7> arr1;
        err = String::parse<TestType>(",1,2,3,4,5,", arr1);
        REQUIRE((arr1[0].empty() && arr1[1] == "1" && arr1[2] == "2" && arr1[3] == "3" &&
                 arr1[4] == "4" && arr1[5] == "5" && arr1[6].empty() && err == 0));
        std::array<std::string, 7> arr2;
        err = String::parse<TestType>(" ,\n1,2 ,3,4 ,5,", arr2);
        REQUIRE((arr2[0].empty() && arr2[1] == "1" && arr2[2] == "2" && arr2[3] == "3" &&
                 arr2[4] == "4" && arr2[5] == "5" && arr2[6].empty() && err == 0));
        std::array<std::string, 4> arr3;
        err = String::parse<TestType>(" 1, 2,  ,  4 5", arr3);
        REQUIRE((arr3[0] == "1" && arr3[1] == "2" && arr3[2].empty() && arr3[3] == "4 5" &&
                 err == 0));

        std::array<std::string, 1> arr4;
        err = String::parse<TestType>(" ", arr4);
        REQUIRE((arr4[0].empty() && err == 0));
        std::array<std::string, 1> arr5;
        err = String::parse<TestType>("", arr5);
        REQUIRE((arr5[0].empty() && err == 0));
        std::array<std::string, 2> arr6;
        err = String::parse<TestType>(" ,\n   ", arr6);
        REQUIRE((arr6[0].empty() && arr6[1].empty() && err == 0));

        std::array<std::string, 3> arr7;
        TestType str1{"   1,2,3"};
        err = String::parse(str1, arr7);
        REQUIRE((arr7[0] == "1" && arr7[1] == "2" && arr7[2] == "3" && err == 0));

        std::array<std::string, 4> arr8;
        TestType str2{" 1 , 2 , 3 , 4 5 67  "};
        err = String::parse(str2, arr8);
        REQUIRE((arr8[0] == "1" && arr8[1] == "2" && arr8[2] == "3" && arr8[3] == "4 5 67"
                 && err == 0));
    }
}


TEMPLATE_TEST_CASE("Noa::String::parse: with sequence of integer as output", "[noa][string]",
                   int8_t, short, int, long, long long, uint8_t, unsigned short, unsigned int,
                   unsigned long, unsigned long long) {
    using namespace ::Noa;
    using string = std::string;
    using vector = std::vector<TestType>;

    WHEN("sending a vector") {
        uint8_t err = 0;
        string test = " 1, 6, \t7, 9. , 56, 011  \t, 0,10.3,10e3  ,  123  ,0123, 0x9999910,1 1";
        vector vec;
        err = String::parse(test, vec);
        REQUIRE(err == 0);
        REQUIRE_THAT(vec, Catch::Equals(vector{1, 6, 7, 9, 56, 11, 0, 10, 10, 123, 123, 0, 1}));

        TestType min = std::numeric_limits<TestType>::min();
        TestType max = std::numeric_limits<TestType>::max();
        vec.clear();
        test = fmt::format(" {},{}", min, max);
        err = String::parse(test, vec);
        REQUIRE(err == 0);
        REQUIRE_THAT(vec, Catch::Equals(vector{min, max}));

        std::vector<int8_t> vec1;
        test = " -53, -1, \t  -09, -0";
        err = String::parse(test, vec1);
        REQUIRE(err == 0);
        REQUIRE_THAT(vec1, Catch::Equals(std::vector<int8_t>{-53, -1, -9, 0}));

        std::vector<short> vec2;
        test = " -194, -1, \t  -09, -0";
        err = String::parse(test, vec2);
        REQUIRE(err == 0);
        REQUIRE_THAT(vec2, Catch::Equals(std::vector<short>{-194, -1, -9, 0}));

        std::vector<int> vec3;
        test = " -54052, -1, \t  -09, -0";
        err = String::parse(test, vec3);
        REQUIRE(err == 0);
        REQUIRE_THAT(vec3, Catch::Equals(std::vector<int>{-54052, -1, -9, 0}));

        std::vector<long> vec4;
        test = " -525107745 , -1, \t  -09, -0";
        err = String::parse(test, vec4);
        REQUIRE(err == 0);
        REQUIRE_THAT(vec4, Catch::Equals(std::vector<long>{-525107745, -1, -9, 0}));

        std::vector<long long> vec5;
        test = " -11111111155488 , -1, \t  -09, -0";
        err = String::parse(test, vec5);
        REQUIRE(err == 0);
        REQUIRE_THAT(vec5, Catch::Equals(std::vector<long long>{-11111111155488, -1, -9, 0}));

        // When error should be raised.
        test = "";
        vec.clear();
        err = String::parse(test, vec);
        REQUIRE(err == Errno::invalid_argument);

        test = "120, , 23";
        vec.clear();
        err = String::parse(test, vec);
        REQUIRE(err == Errno::invalid_argument);
        REQUIRE_THAT(vec, Catch::Equals(vector{120}));

        test = "120, 1, 23,";
        vec.clear();
        err = String::parse(test, vec);
        REQUIRE(err == Errno::invalid_argument);
        REQUIRE_THAT(vec, Catch::Equals(vector{120, 1, 23}));

        test = ",120, 1, 23,";
        vec.clear();
        err = String::parse(test, vec);
        REQUIRE(err == Errno::invalid_argument);
        REQUIRE_THAT(vec, Catch::Equals(vector{}));
    }

    WHEN("sending an array") {
        uint8_t err = 0;
        string test = " 1, 6, \t7, 9. , 56, 011  \t, 0,10.3,10e3  ,  123  ,0123, 0x9999910,1 1";
        std::array<TestType, 13> arr1a{};
        std::array<TestType, 13> arr1b{1, 6, 7, 9, 56, 11, 0, 10, 10, 123, 123, 0, 1};
        err = String::parse(test, arr1a);
        REQUIRE(err == 0);
        for (size_t i{0}; i < arr1b.size(); ++i) {
            REQUIRE(arr1a[i] == arr1b[i]);
        }

        TestType min = std::numeric_limits<TestType>::min();
        TestType max = std::numeric_limits<TestType>::max();
        test = fmt::format(" {},{}", min, max);
        std::array<TestType, 2> arr2a{};
        std::array<TestType, 2> arr2b{min, max};
        err = String::parse(test, arr2a);
        REQUIRE(err == 0);
        for (size_t i{0}; i < arr2b.size(); ++i) {
            REQUIRE(arr2a[i] == arr2b[i]);
        }


        test = " -53, -1, \t  -09, -0";
        std::array<int8_t, 4> arr3a{};
        std::array<int8_t, 4> arr3b{-53, -1, -9, 0};
        err = String::parse(test, arr3a);
        REQUIRE(err == 0);
        for (size_t i{0}; i < arr3b.size(); ++i) {
            REQUIRE(arr3a[i] == arr3b[i]);
        }

        test = " -194, -1, \t  -09, -0";
        std::array<short, 4> arr4a{};
        std::array<short, 4> arr4b{-194, -1, -9, 0};
        err = String::parse(test, arr4a);
        REQUIRE(err == 0);
        for (size_t i{0}; i < arr4b.size(); ++i) {
            REQUIRE(arr4a[i] == arr4b[i]);
        }

        test = " -54052, -1, \t  -09, -0";
        std::array<int, 4> arr5a{};
        std::array<int, 4> arr5b{-54052, -1, -9, 0};
        err = String::parse(test, arr5a);
        REQUIRE(err == 0);
        for (size_t i{0}; i < arr5b.size(); ++i) {
            REQUIRE(arr5a[i] == arr5b[i]);
        }

        test = " -525107745 , -1, \t  -09, -0";
        std::array<long, 4> arr6a{};
        std::array<long, 4> arr6b{-525107745, -1, -9, 0};
        err = String::parse(test, arr6a);
        REQUIRE(err == 0);
        for (size_t i{0}; i < arr6b.size(); ++i) {
            REQUIRE(arr6a[i] == arr6b[i]);
        }

        test = " -11111111155488 , -1, \t  -09, -0";
        std::array<long long, 4> arr7a{};
        std::array<long long, 4> arr7b{-11111111155488, -1, -9, 0};
        err = String::parse(test, arr7a);
        REQUIRE(err == 0);
        for (size_t i{0}; i < arr7b.size(); ++i) {
            REQUIRE(arr7a[i] == arr7b[i]);
        }


        // When error should be raised.
        test = "";
        std::array<TestType, 1> arr8a{};
        err = String::parse(test, arr8a);
        REQUIRE(err == Errno::invalid_argument);

        test = "120, , 23";
        std::array<TestType, 3> arr9a{};
        err = String::parse(test, arr9a);
        REQUIRE(err == Errno::invalid_argument);

        test = "120, 1, 23,";
        std::array<TestType, 4> arr10a{};
        err = String::parse(test, arr10a);
        REQUIRE(err == Errno::invalid_argument);

        test = ",120, 1, 23,";
        std::array<TestType, 5> arr11a{};
        err = String::parse(test, arr11a);
        REQUIRE(err == Errno::invalid_argument);

        test = "1,120, 1, 23,1";
        std::array<TestType, 4> arr12a{};
        err = String::parse(test, arr12a);
        REQUIRE(err == Errno::invalid_size);

        test = "1,120, 1, 23,1";
        std::array<TestType, 10> arr13a{};
        err = String::parse(test, arr13a);
        REQUIRE(err == Errno::invalid_size);
    }
}


TEMPLATE_TEST_CASE("Noa::String::parse with sequence of floating point as output", "[noa][string]",
                   float, double, long double) {
    using namespace ::Noa;
    using string = std::string;
    using vector = std::vector<TestType>;

    WHEN("sending a vector") {
        uint8_t err = 0;
        string test = " 1, 6., \t7, 9. , .56, 123.123, 011, -1, .0,"
                      "10x,-10.3  , 10e3  , 10e-04,0E-12    , 09999910";
        std::vector<float> vec1;
        err = String::parse(test, vec1);
        REQUIRE(err == 0);
        REQUIRE_THAT(vec1, Catch::Equals(std::vector<float>{1, 6, 7, 9, .56f, 123.123f, 11, -1,
                                                            .0f, 10, -10.3f, 10e3f, 10e-04f,
                                                            0e-12f, 9999910.f}));
        std::vector<double> vec2;
        err = String::parse(test, vec2);
        REQUIRE(err == 0);
        REQUIRE_THAT(vec2, Catch::Equals(std::vector<double>{1, 6, 7, 9, .56, 123.123, 11, -1,
                                                             .0, 10, -10.3, 10e3, 10e-04,
                                                             0e-12, 9999910.}));
        std::vector<long double> vec3;
        err = String::parse(test, vec3);
        REQUIRE(err == 0);
        REQUIRE_THAT(vec3, Catch::Equals(std::vector<long double>{1, 6, 7, 9, .56l, 123.123l, 11,
                                                                  -1, .0l, 10, -10.3l, 10e3l,
                                                                  10e-04l, 0e-12l, 9999910.l}));

        vector vec4;
        test = "nan, NaN,-NaN,  INFINITY,inf, \t-Inf";
        err = String::parse(test, vec4);
        REQUIRE(err == 0);
        REQUIRE((std::isnan(vec4[0]) && std::isnan(vec4[1]) && std::isnan(vec4[2]) &&
                 std::isinf(vec4[3]) && std::isinf(vec4[4]) && std::isinf(vec4[5])));

        test = "0x1273,-0x1273";
        vector vec5;
        err = String::parse(test, vec5);
        REQUIRE(err == 0);
        REQUIRE_THAT(vec5, Catch::Equals(vector{static_cast<TestType>(4723.l),
                                                static_cast<TestType>(-4723.l),}));

        // Should raise an error
        vector vec6;
        test = "";
        err = String::parse(test, vec6);
        REQUIRE(err == Errno::invalid_argument);

        test = "    ";
        err = String::parse(test, vec6);
        REQUIRE(err == Errno::invalid_argument);

        err = 0;
        test = ". ,10";
        err = String::parse(test, vec6);
        REQUIRE(err == Errno::invalid_argument);

        err = 0;
        test = "1, 2., n10";
        err = String::parse(test, vec6);
        REQUIRE(err == Errno::invalid_argument);

        err = 0;
        test = "3, --10";
        err = String::parse(test, vec6);
        REQUIRE(err == Errno::invalid_argument);

        err = 0;
        test = "0, e10";
        err = String::parse(test, vec6);
        REQUIRE(err == Errno::invalid_argument);

        TestType min = std::numeric_limits<TestType>::min();
        TestType max = std::numeric_limits<TestType>::max();
        TestType lowest = std::numeric_limits<TestType>::lowest();
        err = 0;
        err = String::parse(fmt::format("  {}1,,", min), vec6);
        REQUIRE(err == Errno::out_of_range);
        err = 0;
        err = String::parse(fmt::format("  {}1  ", max), vec6);
        REQUIRE(err == Errno::out_of_range);
        err = 0;
        err = String::parse(fmt::format("  {}1  ", lowest), vec6);
        REQUIRE(err == Errno::out_of_range);
    }

    WHEN("sending a vector") {
        uint8_t err = 0;
        string test = " 1, 6., \t7, 9. , .56, 123.123, 011, -1, .0,"
                      "10x,-10.3  , 10e3  , 10e-04,0E-12    , 09999910";
        std::array<float, 15> arr1a{};
        std::array<float, 15> arr1b{1, 6, 7, 9, .56f, 123.123f, 11, -1,
                                    .0f, 10, -10.3f, 10e3f, 10e-04f,
                                    0e-12f, 9999910.f};
        err = String::parse(test, arr1a);
        REQUIRE(err == 0);
        for (size_t i{0}; i < arr1b.size(); ++i) {
            REQUIRE_THAT(arr1a[i], Catch::WithinAbs(static_cast<double>(arr1b[i]), 0.00005));
        }
        std::array<double, 15> arr2a{};
        std::array<double, 15> arr2b{1, 6, 7, 9, .56, 123.123, 11, -1,
                                     .0, 10, -10.3, 10e3, 10e-04,
                                     0e-12, 9999910.};
        err = String::parse(test, arr2a);
        REQUIRE(err == 0);
        for (size_t i{0}; i < arr2b.size(); ++i) {
            REQUIRE_THAT(arr2a[i], Catch::WithinAbs(arr2b[i], 0.00005));
        }
        std::array<long double, 15> arr3a{};
        std::array<long double, 15> arr3b{1, 6, 7, 9, .56l, 123.123l, 11, -1,
                                          .0l, 10, -10.3l, 10e3l, 10e-04l,
                                          0e-12l, 9999910.l};
        err = String::parse(test, arr3a);
        REQUIRE(err == 0);
        for (size_t i{0}; i < arr3b.size(); ++i) {
            REQUIRE_THAT(arr3a[i], Catch::WithinAbs(static_cast<double>(arr3b[i]), 0.00005));
        }


        std::array<TestType, 6> arr4{};
        test = "nan, NaN,-NaN,  INFINITY,inf, \t-Inf";
        err = String::parse(test, arr4);
        REQUIRE(err == 0);
        REQUIRE((std::isnan(arr4[0]) && std::isnan(arr4[1]) && std::isnan(arr4[2]) &&
                 std::isinf(arr4[3]) && std::isinf(arr4[4]) && std::isinf(arr4[5])));

        test = "0x1273,-0x1273";
        std::array<TestType, 2> arr5{};
        err = String::parse(test, arr5);
        REQUIRE(err == 0);
        REQUIRE_THAT(arr5[0], Catch::WithinAbs(4723., 0.0005));
        REQUIRE_THAT(arr5[1], Catch::WithinAbs(-4723., 0.0005));

        std::array<TestType, 4> arr6{};
        test = "";
        err = String::parse(test, arr6);
        REQUIRE(err == Errno::invalid_argument);

        test = "    ";
        err = String::parse(test, arr6);
        REQUIRE(err == Errno::invalid_argument);

        err = 0;
        test = ". ,10";
        err = String::parse(test, arr6);
        REQUIRE(err == Errno::invalid_argument);

        err = 0;
        test = "1, 2., n10";
        err = String::parse(test, arr6);
        REQUIRE(err == Errno::invalid_argument);

        err = 0;
        test = "3, --10";
        err = String::parse(test, arr6);
        REQUIRE(err == Errno::invalid_argument);

        err = 0;
        test = "0, e10";
        err = String::parse(test, arr6);
        REQUIRE(err == Errno::invalid_argument);

        test = "1,120, 1, 23,1";
        std::array<TestType, 4> arr7{};
        err = String::parse(test, arr7);
        REQUIRE(err == Errno::invalid_size);

        test = "1,120, 1, 23,1";
        std::array<TestType, 10> arr8{};
        err = String::parse(test, arr8);
        REQUIRE(err == Errno::invalid_size);

        TestType min = std::numeric_limits<TestType>::min();
        TestType max = std::numeric_limits<TestType>::max();
        TestType lowest = std::numeric_limits<TestType>::lowest();
        std::array<TestType, 1> arr9{};
        err = 0;
        err = String::parse(fmt::format("  {}1", min), arr9);
        REQUIRE(err == Errno::out_of_range);
        err = 0;
        err = String::parse(fmt::format("  {}1  ", max), arr9);
        REQUIRE(err == Errno::out_of_range);
        err = 0;
        err = String::parse(fmt::format("  {}1  ", lowest), arr9);
        REQUIRE(err == Errno::out_of_range);
    }
}


SCENARIO("Noa::String::parse with sequence of booleans as output", "[noa][string]") {
    using namespace ::Noa;
    using string = std::string;
    using vector = std::vector<bool>;

    WHEN("sending a vector") {
        uint8_t err = 0;
        string test = "1,true,   TRUE, y,yes, YES,on, ON,0,false,FALSE,n,no, NO, off , OFF";
        vector vec;
        err = String::parse(test, vec);
        REQUIRE(err == 0);
        REQUIRE_THAT(vec, Catch::Equals(vector{1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}));
    }

    WHEN("sending a vector") {
        uint8_t err = 0;
        string test = "1,true,   TRUE, y,yes, YES,on, ON,0,false,FALSE,n,no, NO, off , OFF";
        std::array<bool, 16> arr{};
        err = String::parse(test, arr);
        REQUIRE(err == 0);
        REQUIRE(err == 0);
        for (size_t i{0}; i < 8; ++i) {
            REQUIRE(arr[i] == true);
        }
        for (size_t i{8}; i < 16; ++i) {
            REQUIRE(arr[i] == false);
        }
    }
}


TEMPLATE_TEST_CASE("Noa::String::parse with default values", "[noa][string]",
                   int8_t, short, int, long, long long, uint8_t, unsigned short, unsigned int,
                   unsigned long, unsigned long long, float, double, long double) {

    using namespace ::Noa;
    using string = std::string;
    using vector = std::vector<TestType>;

    WHEN("sending a vector") {
        string test1 = "123,,12, 0, \t,, 8";
        string test2 = ",1,2,3,4,5,";
        vector vec;
        uint8_t err = String::parse(test1, test2, vec);
        REQUIRE(err == 0);
        REQUIRE_THAT(vec, Catch::Equals(vector{123, 1, 12, 0, 4, 5, 8}));

        test1 = "123,,12, 0, \t,, 8";
        test2 = ",1,2,3,4,5,";
        std::vector<std::string> vec1;
        err = String::parse(test1, test2, vec1);
        REQUIRE(err == 0);
        REQUIRE_THAT(vec1, Catch::Equals(std::vector<std::string>{"123", "1", "12",
                                                                  "0", "4", "5", "8"}));

        test1 = "1,,0, 0, \t,, 1";
        test2 = ",1,1,3,1,1,12";
        std::vector<bool> vec2;
        err = String::parse(test1, test2, vec2);
        REQUIRE(err == 0);
        REQUIRE_THAT(vec2, Catch::Equals(std::vector<bool>{1, 1, 0, 0, 1, 1, 1}));

        test1 = "1,,0, 0, \t,, 1";
        test2 = ",1,1,3,1,  ,12";
        vector vec3;
        err = String::parse(test1, test2, vec3);
        REQUIRE(err == Errno::invalid_argument);
    }

    WHEN("sending an array") {
        string test1 = "123,,12, 0, \t,, 8";
        string test2 = ",1,2,3,4,5,";
        std::array<TestType, 7> arr1a{};
        std::array<TestType, 7> arr1b{123, 1, 12, 0, 4, 5, 8};
        uint8_t err = String::parse(test1, test2, arr1a);
        REQUIRE(err == 0);
        for (size_t i{0}; i < 7; ++i) {
            REQUIRE_THAT(arr1a[i], Catch::WithinAbs(static_cast<double>(arr1b[i]), 0.000005));
        }

        test1 = "123,,12, 0, \t,, 8";
        test2 = ",1,2,3,4,5,";
        std::array<std::string, 7> arr2a{};
        std::array<std::string, 7> arr2b{"123", "1", "12", "0", "4", "5", "8"};
        err = String::parse(test1, test2, arr2a);
        REQUIRE(err == 0);
        for (size_t i{0}; i < 7; ++i) {
            REQUIRE(arr2a[i] == arr2b[i]);
        }

        test1 = "1,,0, 0, \t,, 1";
        test2 = ",1,1,3,1,1,12";
        std::array<bool, 7> arr3a{};
        std::array<bool, 7> arr3b{true, true, false, false, true, true, true};
        err = String::parse(test1, test2, arr3a);
        REQUIRE(err == 0);
        INFO(fmt::format("{}", arr3a))
        for (size_t i{0}; i < 7; ++i) {
            REQUIRE(arr3a[i] == arr3b[i]);
        }

        test1 = "1,,0, 0, \t,, 1";
        test2 = ",1,1,3,1,  ,12";
        std::array<TestType, 7> arr4{};
        err = String::parse(test1, test2, arr4);
        REQUIRE(err == Errno::invalid_argument);

        test1 = "1,,0, 0, \t,, 1";
        test2 = ",1,1,3,1, 1 ,12";
        std::array<TestType, 10> arr5{};
        err = String::parse(test1, test2, arr5);
        REQUIRE(err == Errno::invalid_size);

        test1 = "1,,0, 0, \t,, 1";
        test2 = ",1,1,3,1, 1 ,12";
        std::array<TestType, 4> arr6{};
        err = String::parse(test1, test2, arr6);
        REQUIRE(err == Errno::invalid_size);
    }
}
