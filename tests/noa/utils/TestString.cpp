/*
 * Test noa/utils/String.h
 */

#include <catch2/catch.hpp>

#include "noa/utils/String.h"


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
// parse
// -------------------------------------------------------------------------------------------------
TEMPLATE_TEST_CASE("Noa::String::parse should parse strings", "[noa][string]",
                   std::string, std::string_view) {
    using namespace ::Noa::String;
    using vec_t = std::vector<std::string>;

    GIVEN("a string") {
        WHEN("sending by rvalue") {
            vec_t v0 = parse<TestType>("1,2,3,4,5");
            REQUIRE_THAT(v0, Catch::Equals(vec_t({"1", "2", "3", "4", "5"})));
            vec_t v1 = parse<TestType>(",1,2,3,4,5,");
            REQUIRE_THAT(v1, Catch::Equals(vec_t({"", "1", "2", "3", "4", "5", ""})));
            vec_t v2 = parse<TestType>(" ,\n1,2 ,3,4 ,5,");
            REQUIRE_THAT(v2, Catch::Equals(vec_t({"", "1", "2", "3", "4", "5", ""})));
            vec_t v3 = parse<TestType>("1,2, 3 ,4\n ,5 ,");
            REQUIRE_THAT(v3, Catch::Equals(vec_t({"1", "2", "3", "4", "5", ""})));
            vec_t v4 = parse<TestType>("1  2 3\t   4 ,  5 , ");
            REQUIRE_THAT(v4, Catch::Equals(vec_t({"1  2 3\t   4", "5", ""})));
            vec_t v5 = parse<TestType>(" 1, 2,  ,  4 5");
            REQUIRE_THAT(v5, Catch::Equals(vec_t({"1", "2", "", "4 5"})));
            vec_t v6 = parse<TestType>(" ");
            REQUIRE_THAT(v6, Catch::Equals(vec_t({""})));
            vec_t v7 = parse<TestType>("");
            REQUIRE_THAT(v7, Catch::Equals(vec_t({""})));
            vec_t v8 = parse<TestType>("  ");
            REQUIRE_THAT(v8, Catch::Equals(vec_t({""})));
            vec_t v9 = parse<TestType>(" ,\n   ");
            REQUIRE_THAT(v9, Catch::Equals(vec_t({"", ""})));
        }

        WHEN("sending by lvalue") {
            TestType s0{"   1,2,3"};
            vec_t v0 = parse(s0);
            REQUIRE_THAT(v0, Catch::Equals(vec_t({"1", "2", "3"})));
            TestType s1{" 1 , 2 , 3 , 4 5 67  "};
            vec_t v1 = parse(s1);
            REQUIRE_THAT(v1, Catch::Equals(vec_t({"1", "2", "3", "4 5 67"})));
        }
    }

    vec_t out;
    GIVEN("a string and the output vector") {
        WHEN("sending by rvalue") {
            parse<TestType>("1,2,3,4,5", out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({"1", "2", "3", "4", "5"})));
            out.clear();
            parse<TestType>(",1,2,3,4,5,", out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({"", "1", "2", "3", "4", "5", ""})));
            out.clear();
            parse<TestType>(" ,1,2 ,3,4 ,5,", out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({"", "1", "2", "3", "4", "5", ""})));
            out.clear();
            parse<TestType>("1,2,,4,5,6,", out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({"1", "2", "", "4", "5", "6", ""})));
            out.clear();
            parse<TestType>("1//2/ 3 /4,//5 ,/", out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({"1//2/ 3 /4", "//5", "/"})));
            out.clear();
            parse<TestType>(" ", out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({""})));
            out.clear();
        }

        WHEN("sending lvalues") {
            TestType s0;
            parse(s0, out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({""})));
            out.clear();

            TestType s1{"my file.txt"};
            parse(s1, out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({"my file.txt"})));
            out.clear();

            TestType s2{"1  "};
            parse(s2, out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({"1"})));
            out.clear();
        }
    }
}


// -------------------------------------------------------------------------------------------------
// String to scalar (integers, floating points, bool)
// -------------------------------------------------------------------------------------------------
SCENARIO("Noa::String::toInt should convert a string into an int", "[noa][string]") {
    using namespace ::Noa::String;
    using vec_t = std::vector<int>;

    int int_min = std::numeric_limits<int>::min();
    int int_max = std::numeric_limits<int>::max();
    GIVEN("a valid string") {
        REQUIRE(toInt("1") == 1);
        REQUIRE(toInt(" 6") == 6);
        REQUIRE(toInt("\t7") == 7);
        REQUIRE(toInt("9.") == 9);
        REQUIRE(toInt("1234567") == 1234567);
        REQUIRE(toInt("011") == 11);
        REQUIRE(toInt("-1") == -1);
        REQUIRE(toInt("0") == 0);
        REQUIRE(toInt("10x") == 10);
        REQUIRE(toInt("10.3") == 10);
        REQUIRE(toInt("10e3") == 10);
        REQUIRE(toInt("09999910") == 9999910);
        REQUIRE(toInt("0x9999910") == 0);

        std::string str_min{fmt::format("  {},,", int_min)};
        std::string str_max{fmt::format("  {}  ", int_max)};
        REQUIRE(toInt(str_min) == int_min);
        REQUIRE(toInt(str_max) == int_max);
    }

    GIVEN("out of range string") {
        long int_min_out = static_cast<long>(int_min) - 1;
        long int_max_out = static_cast<long>(int_max) + 1;
        std::string str_min{fmt::format("  {},,", int_min_out)};
        std::string str_max{fmt::format("  {}  ", int_max_out)};
        REQUIRE_THROWS_AS(toInt(str_min), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toInt(str_max), Noa::ErrorCore);
    }

    GIVEN("an invalid string") {
        REQUIRE_THROWS_AS(toInt(""), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toInt("   "), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toInt("."), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toInt("  n10"), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toInt("e10"), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toInt("--10"), Noa::ErrorCore);
    }

    GIVEN("a valid vector") {
        std::vector<std::string> v0 = {};
        REQUIRE_THAT(toInt(v0), Catch::Equals(vec_t({})));
        vec_t v1 = toInt({"123", "  23.2", "8877   "});
        REQUIRE_THAT(v1, Catch::Equals(vec_t({123, 23, 8877})));
        vec_t v2 = toInt({"1", "2", "3", "5", "00000006"});
        REQUIRE_THAT(v2, Catch::Equals(vec_t({1, 2, 3, 5, 6})));
    }

    GIVEN("a out of range string in vector") {
        long int_min_out = static_cast<long>(int_min) - 1;
        long int_max_out = static_cast<long>(int_max) + 1;
        std::string str_min{fmt::format("  {},,", int_min_out)};
        std::string str_max{fmt::format("  {}  ", int_max_out)};
        REQUIRE_THROWS_AS(toInt({"1", "2", str_min}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toInt({"1", "2", str_max}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toInt({str_min}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toInt({str_max}), Noa::ErrorCore);
    }

    GIVEN("an invalid string in vector") {
        REQUIRE_THROWS_AS(toInt({"1234", "", "1"}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toInt({"1234", "   ", "1"}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toInt({"1234", ".", "1"}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toInt({"1234", "  n10", "1"}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toInt({"1234", "e10", "1"}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toInt({"1234", "--10", "1"}), Noa::ErrorCore);
    }

    uint8_t a;
    toInt<size_t>("", a);
}


SCENARIO("Noa::String::toFloat should convert a string into a float", "[noa][string]") {
    using namespace ::Noa::String;
    using vec_t = std::vector<float>;

    float float_min = std::numeric_limits<float>::min();
    float float_max = std::numeric_limits<float>::max();
    float float_low = std::numeric_limits<float>::lowest();
    GIVEN("a valid string") {
        REQUIRE(toFloat("1") == 1.f);
        REQUIRE(toFloat(" 6") == 6.f);
        REQUIRE(toFloat("\t7") == 7.f);
        REQUIRE(toFloat("9.") == 9.f);
        REQUIRE(toFloat(".5") == .5f);
        REQUIRE(toFloat("1234567.123") == 1234567.123f);
        REQUIRE(toFloat("011") == 11.f);
        REQUIRE(toFloat("-1") == -1.f);
        REQUIRE(toFloat(".0") == 0.f);
        REQUIRE(toFloat("10x") == 10.f);
        REQUIRE(toFloat("-10.3") == -10.3f);
        REQUIRE(toFloat("10e3") == 10e3f);
        REQUIRE(toFloat("10e-04") == 10e-04f);
        REQUIRE(toFloat("0E-12") == 0e-12f);
        REQUIRE(toFloat("09999910") == 9999910.f);

        REQUIRE(std::isnan(toFloat("nan")));
        REQUIRE(std::isnan(toFloat("NaN")));
        REQUIRE(std::isnan(toFloat("-NaN")));
        REQUIRE(std::isinf(toFloat("INFINITY")));
        REQUIRE(std::isinf(toFloat("inf")));
        REQUIRE(std::isinf(toFloat("-Inf")));

        REQUIRE(toFloat("0x1273") == 4723.f);
        REQUIRE(toFloat("-0x1273") == -4723.f);

        std::string str_min{fmt::format("  {},,", float_min)};
        std::string str_max{fmt::format("  {}  ", float_max)};
        std::string str_low{fmt::format("  {}  ", float_low)};
        REQUIRE(toFloat(str_min) == float_min);
        REQUIRE(toFloat(str_max) == float_max);
        REQUIRE(toFloat(str_low) == float_low);
    }

    GIVEN("out of range string") {
        double float_min_out = static_cast<double>(float_min) * 0.1;
        double float_low_out = static_cast<double>(float_low) * 10.;
        double float_max_out = static_cast<double>(float_max) * 10.;
        std::string str_low{fmt::format("  {},", float_low_out)};
        std::string str_min{fmt::format("  {},,", float_min_out)};
        std::string str_max{fmt::format("  {}  ", float_max_out)};
        REQUIRE_THROWS_AS(toFloat(str_low), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat(str_min), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat(str_max), Noa::ErrorCore);
    }

    GIVEN("an invalid string") {
        REQUIRE_THROWS_AS(toFloat(""), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat("   "), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat("."), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat("  n10"), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat(".e10"), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat("--10"), Noa::ErrorCore);
    }

    GIVEN("a valid vector") {
        std::vector<std::string> v0 = {};
        REQUIRE_THAT(toFloat(v0), Catch::Equals(vec_t({})));
        vec_t v1 = toFloat({"123", "  .23", "-88.77   "});
        REQUIRE_THAT(v1, Catch::Equals(vec_t({123.f, .23f, -88.77f})));
        vec_t v2 = toFloat({"-1", "2", "3", "-543.33", "-000006"});
        REQUIRE_THAT(v2, Catch::Equals(vec_t({-1.f, 2.f, 3.f, -543.33f, -6.f})));
    }

    GIVEN("a out of range string in vector") {
        double float_min_out = static_cast<double>(float_min) * 0.1;
        double float_max_out = static_cast<double>(float_max) * 10.;
        double float_low_out = static_cast<double>(float_low) * 10.;
        std::string str_min{fmt::format("  {},,", float_min_out)};
        std::string str_max{fmt::format("  {}  ", float_max_out)};
        std::string str_low{fmt::format("{},0 ", float_low_out)};
        REQUIRE_THROWS_AS(toFloat({"1.", "22222233", str_min}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat({".1", "-1232", str_max}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat({".1", "-1232", str_low}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat({str_min}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat({str_max}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat({str_low}), Noa::ErrorCore);
    }

    GIVEN("an invalid string in vector") {
        REQUIRE_THROWS_AS(toFloat({"1234", "", "1"}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat({"1234", "   ", "1"}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat({"1234", ".", "1"}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat({"1234", "  n10", "1"}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat({"1234", "e10", "1"}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toFloat({"1234", "--10", "1"}), Noa::ErrorCore);
    }
}


SCENARIO("Noa::String::toBool should convert a string into a bool", "[noa][string]") {
    using namespace ::Noa::String;
    using vec_t = std::vector<bool>;

    GIVEN("a valid string") {
        REQUIRE(toBool("1") == true);
        REQUIRE(toBool("true") == true);
        REQUIRE(toBool("TRUE") == true);
        REQUIRE(toBool("y") == true);
        REQUIRE(toBool("yes") == true);
        REQUIRE(toBool("YES") == true);
        REQUIRE(toBool("on") == true);
        REQUIRE(toBool("ON") == true);

        REQUIRE(toBool("0") == false);
        REQUIRE(toBool("false") == false);
        REQUIRE(toBool("FALSE") == false);
        REQUIRE(toBool("n") == false);
        REQUIRE(toBool("no") == false);
        REQUIRE(toBool("NO") == false);
        REQUIRE(toBool("off") == false);
        REQUIRE(toBool("OFF") == false);
    }

    GIVEN("a invalid string") {
        REQUIRE_THROWS_AS(toBool(" y"), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toBool("yes please"), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toBool("Yes"), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toBool("."), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toBool(""), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toBool(" 0"), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toBool("wrong"), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toBool("No"), Noa::ErrorCore);
    }

    GIVEN("a valid vector") {
        REQUIRE_THAT(toBool(std::vector<std::string>({})), Catch::Equals(vec_t({})));
        REQUIRE_THAT(toBool({"1", "false", "y"}), Catch::Equals(vec_t({true, false, true})));
        REQUIRE_THAT(toBool({"true", "1", "1"}), Catch::Equals(vec_t({true, true, true})));
        REQUIRE_THAT(toBool({"TRUE", "FALSE", "OFF"}), Catch::Equals(vec_t({true, false, false})));
        REQUIRE_THAT(toBool({"n", "off", "n"}), Catch::Equals(vec_t({false, false, false})));
    }

    GIVEN("a invalid vector") {
        REQUIRE_THROWS_AS(toBool({"y", "true", "Tru"}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toBool(std::vector<std::string>{""}), Noa::ErrorCore);
        REQUIRE_THROWS_AS(toBool({"1", "1", ""}), Noa::ErrorCore);
    }
}
