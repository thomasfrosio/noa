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
// Split and splitFirstOf
// -------------------------------------------------------------------------------------------------
TEMPLATE_TEST_CASE("Noa::String::split(FirstOf) should split strings", "[noa][string]",
                   std::string, std::string_view) {
    using namespace ::Noa::String;
    using vec_t = std::vector<std::string>;

    GIVEN("a string and a delim") {
        WHEN("using split") {
            WHEN("sending rvalues") {
                vec_t v0 = split<TestType>("1,2,3,4,5", ",");
                REQUIRE_THAT(v0, Catch::Equals(vec_t({"1", "2", "3", "4", "5"})));
                vec_t v1 = split<TestType>(",1,2,3,4,5,", ",");
                REQUIRE_THAT(v1, Catch::Equals(vec_t({"", "1", "2", "3", "4", "5", ""})));
                vec_t v2 = split<TestType>(" ,1,2 ,3,4 ,5,", " ,");
                REQUIRE_THAT(v2, Catch::Equals(vec_t({"", "1,2", "3,4", "5,"})));
                vec_t v3 = split<TestType>("1,2, 3 ,4,5 ,", " ,");
                REQUIRE_THAT(v3, Catch::Equals(vec_t({"1,2, 3", "4,5", ""})));
                vec_t v4 = split<TestType>("1//2/ 3 /4,//5 ,/", "/");
                REQUIRE_THAT(v4, Catch::Equals(vec_t({"1", "", "2", " 3 ", "4,", "", "5 ,", ""})));
                vec_t v5 = split<TestType>("1 / /2 / 3 /4, / /5 ,/ / ", " / ");
                REQUIRE_THAT(v5, Catch::Equals(vec_t({"1", "/2", "3 /4,", "/5 ,/", ""})));
            }

            WHEN("sending lvalues") {
                TestType s0{"1//2/ 3 /4,//5 ,/"};
                vec_t v0 = split(s0, "/");
                REQUIRE_THAT(v0, Catch::Equals(vec_t({"1", "", "2", " 3 ", "4,", "", "5 ,", ""})));
                TestType s1{"1 / /2 / 3 /4, / /5 ,/ / "};
                vec_t v1 = split(s1, " / ");
                REQUIRE_THAT(v1, Catch::Equals(vec_t({"1", "/2", "3 /4,", "/5 ,/", ""})));
            }
        }

        WHEN("using splitFirstOf") {
            WHEN("sending rvalues") {
                vec_t v0 = splitFirstOf<TestType>("1,2,3,4,5", ",");
                REQUIRE_THAT(v0, Catch::Equals(vec_t({"1", "2", "3", "4", "5"})));
                vec_t v1 = splitFirstOf<TestType>(",1,2,3,4,5,", ",");
                REQUIRE_THAT(v1, Catch::Equals(vec_t({"", "1", "2", "3", "4", "5", ""})));
                vec_t v2 = splitFirstOf<TestType>(" ,1,2 ,3,4 ", " ,");
                REQUIRE_THAT(v2, Catch::Equals(vec_t({"", "", "1", "2", "", "3", "4", ""})));
                vec_t v3 = splitFirstOf<TestType>("1,2, 3 ,4,5 ,", " ,");
                REQUIRE_THAT(v3, Catch::Equals(vec_t({"1", "2", "", "3", "", "4", "5", "", ""})));
                vec_t v4 = splitFirstOf<TestType>("1//2/ 3 /4,//5 ,/", "/");
                REQUIRE_THAT(v4, Catch::Equals(vec_t({"1", "", "2", " 3 ", "4,", "", "5 ,", ""})));

                WHEN("duplicated character in delim, the character is ignored") {
                    vec_t expected({"1", "", "", "2", "", "", "3", "", "4,",
                                    "", "", "", "5", ",", "", "", "", ""});
                    vec_t v5 = splitFirstOf<TestType>("1/ /2 / 3 /4, / /5 , / /", " /");
                    REQUIRE_THAT(v5, Catch::Equals(expected));
                    vec_t v6 = splitFirstOf<TestType>("1/ /2 / 3 /4, / /5 ,/ / ", " //");
                    REQUIRE_THAT(v5, Catch::Equals(expected));
                }
            }

            WHEN("sending lvalues") {
                TestType s0{"1//2/ 3 /4,//5 ,/"};
                vec_t v0 = splitFirstOf(s0, "/");
                REQUIRE_THAT(v0, Catch::Equals(vec_t({"1", "", "2", " 3 ", "4,", "", "5 ,", ""})));
                TestType s1{"1 /2 / 3 /4,"};
                vec_t v1 = splitFirstOf(s1, " /");
                REQUIRE_THAT(v1, Catch::Equals(vec_t({"1", "", "2", "", "", "3", "", "4,"})));

                WHEN("duplicated character in delim, the character is ignored") {
                    vec_t expected({"1", "", "", "2", "", "", "3", "", "4,",
                                    "", "", "", "5", ",", "", "", "", ""});
                    TestType s01{"1/ /2 / 3 /4, / /5 , / /"};
                    vec_t v01 = splitFirstOf(s01, " /");
                    REQUIRE_THAT(v01, Catch::Equals(expected));

                    TestType s02{"1/ /2 / 3 /4, / /5 ,/ / "};
                    vec_t v02 = splitFirstOf(s02, " //");
                    REQUIRE_THAT(v02, Catch::Equals(expected));
                }
            }
        }
    }

    vec_t out;
    GIVEN("a string, a delim _and_ the output vector") {
        WHEN("using split") {
            WHEN("sending rvalues") {
                split<TestType>("1,2,3,4,5", ",", out);
                REQUIRE_THAT(out, Catch::Equals(vec_t({"1", "2", "3", "4", "5"})));
                out.clear();
                split<TestType>(",1,2,3,4,5,", ",", out);
                REQUIRE_THAT(out, Catch::Equals(vec_t({"", "1", "2", "3", "4", "5", ""})));
                out.clear();
                split<TestType>(" ,1,2 ,3,4 ,5,", " ,", out);
                REQUIRE_THAT(out, Catch::Equals(vec_t({"", "1,2", "3,4", "5,"})));
                out.clear();
                split<TestType>("1,2, 3 ,4,5 ,", " ,", out);
                REQUIRE_THAT(out, Catch::Equals(vec_t({"1,2, 3", "4,5", ""})));
                out.clear();
                split<TestType>("1//2/ 3 /4,//5 ,/", "/", out);
                REQUIRE_THAT(out, Catch::Equals(vec_t({"1", "", "2", " 3 ", "4,", "", "5 ,", ""})));
                out.clear();
                split<TestType>("1 / /2 / 3 /4, / /5 ,/ / ", " / ", out);
                REQUIRE_THAT(out, Catch::Equals(vec_t({"1", "/2", "3 /4,", "/5 ,/", ""})));
                out.clear();
            }

            WHEN("sending lvalues") {
                TestType s0{"1//2/ 3 /4,//5 ,/"};
                split(s0, "/", out);
                REQUIRE_THAT(out, Catch::Equals(vec_t({"1", "", "2", " 3 ", "4,", "", "5 ,", ""})));
                out.clear();

                TestType s1{"1 / /2 / 3 /4, / /5 ,/ / "};
                split(s1, " / ", out);
                REQUIRE_THAT(out, Catch::Equals(vec_t({"1", "/2", "3 /4,", "/5 ,/", ""})));
                out.clear();
            }
        }

        WHEN("using splitFirstOf") {
            splitFirstOf<TestType>("1,2,3,4,5", ",", out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({"1", "2", "3", "4", "5"})));
            out.clear();
            splitFirstOf<TestType>(",1,2,3,4,5,", ",", out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({"", "1", "2", "3", "4", "5", ""})));
            out.clear();
            splitFirstOf<TestType>(" ,1,2 ,3,4 ,5,", " ,", out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({"", "", "1", "2", "", "3", "4", "", "5", ""})));
            out.clear();
            splitFirstOf<TestType>("1,2, 3 ,4,5 ,", " ,", out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({"1", "2", "", "3", "", "4", "5", "", ""})));
            out.clear();
            splitFirstOf<TestType>("1//2/ 3 /4,//5 ,//", "/", out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({"1", "", "2", " 3 ", "4,", "", "5 ,", "", ""})));
            out.clear();

            WHEN("duplicated character in delim, the character is ignored") {
                vec_t expected({"1", "", "", "2", "", "", "3", "", "4,",
                                "", "", "", "5", ",", "", "", "", ""});
                splitFirstOf<TestType>("1/ /2 / 3 /4, / /5 , / /", " /", out);
                REQUIRE_THAT(out, Catch::Equals(expected));
                out.clear();
                splitFirstOf<TestType>("1/ /2 / 3 /4, / /5 ,/ / ", " //", out);
                REQUIRE_THAT(out, Catch::Equals(expected));
                out.clear();
            }
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
            REQUIRE_THAT(v4, Catch::Equals(vec_t({"1", "2", "3", "4", "5", ""})));
            vec_t v5 = parse<TestType>(" 1, 2,  ,  4 5");
            REQUIRE_THAT(v5, Catch::Equals(vec_t({"1", "2", "", "4", "5"})));
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
            TestType s0{"--in --out foo bar , 1  "};
            vec_t v0 = parse(s0);
            REQUIRE_THAT(v0, Catch::Equals(vec_t({"--in", "--out", "foo", "bar", "1"})));
            TestType s1{" 1 , 2 , 3 , 4 5 67  "};
            vec_t v1 = parse(s1);
            REQUIRE_THAT(v1, Catch::Equals(vec_t({"1", "2", "3", "4", "5", "67"})));
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
            REQUIRE_THAT(out, Catch::Equals(vec_t({"1//2/", "3", "/4", "//5", "/"})));
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

            TestType s1{"1234"};
            parse(s1, out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({"1234"})));
            out.clear();

            TestType s2{"1  "};
            parse(s2, out);
            REQUIRE_THAT(out, Catch::Equals(vec_t({"1"})));
            out.clear();
        }
    }
}


// -------------------------------------------------------------------------------------------------
// Tokenize
// -------------------------------------------------------------------------------------------------


// -------------------------------------------------------------------------------------------------
// String to scalar (integers, floating points, bool)
// -------------------------------------------------------------------------------------------------
