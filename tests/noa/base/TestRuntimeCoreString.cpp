#include <noa/base/Strings.hpp>

#include "Catch.hpp"

using namespace ::noa::types;
namespace nd = ::noa::details;

TEST_CASE("base::string::trim_left", "[noa][core]") {
    std::string result;
    std::vector<std::string> tests{
        "", "   ", "  foo ", "  \tfoo", "  \n foo", "  \rfoo", " foo bar ", "\t  \n 123; \n", " , 123 "
    };
    std::vector<std::string> expected{
        "", "", "foo ", "foo", "foo", "foo", "foo bar ", "123; \n", ", 123 "
    };
    for (size_t i = 0; i < tests.size(); ++i) {
        result = nd::trim_left(tests[i]);
        REQUIRE(result == expected[i]);
        result = nd::trim_left(std::move(tests[i]));
        REQUIRE(result == expected[i]);
    }
}

TEST_CASE("base::string::trim_right") {
    std::string result;
    std::vector<std::string> tests{
        "", "   ", "  foo ", "  \tfoo", " \n foo\n ", "foo \r", " foo bar ", "\t  \n 123; \n", " , 123 "
    };
    std::vector<std::string> expected{
        "", "", "  foo", "  \tfoo", " \n foo", "foo", " foo bar", "\t  \n 123;", " , 123"
    };
    for (size_t i = 0; i < tests.size(); ++i) {
        result = nd::trim_right(tests[i]);
        REQUIRE(result == expected[i]);
        result = nd::trim_right(std::move(tests[i]));
        REQUIRE(result == expected[i]);
    }
}

TEST_CASE("base::string::trim") {
    std::string result;
    std::vector<std::string> tests{
        "", "  ", "  foo ", "  \tfoo", " \n foo\n ", "foo \r", " foo bar ",
        "\t  \n 123; \n", " , 123 ", "foo \n  \n", "  foo bar \n foo "
    };
    std::vector<std::string> expected{
        "", "", "foo", "foo", "foo", "foo", "foo bar",
        "123;", ", 123", "foo", "foo bar \n foo"
    };
    for (size_t i = 0; i < tests.size(); ++i) {
        result = nd::trim(tests[i]);
        REQUIRE(result == expected[i]);
        result = nd::trim(std::move(tests[i]));
        REQUIRE(result == expected[i]);
    }
}

TEMPLATE_TEST_CASE("base::string::parse(), to integer", "", u8, u16, u32, u64, i8, i16, i32, i64) {
    TestType min = std::numeric_limits<TestType>::min();
    TestType max = std::numeric_limits<TestType>::max();

    GIVEN("a string that can be converted to an integer") {
        std::vector<std::string> tests{
            "1", " 6", "\t 7", "9.", "56", "011", "0 ", "10.3", "10e3",
            " 123  ", "0123", "0x9999910", fmt::format("  {},,", min),
            fmt::format("  {}  ", max)
        };
        std::vector<TestType> expected{1, 6, 7, 9, 56, 11, 0, 10, 10, 123, 123, 0, min, max};

        for (size_t i{}; i < tests.size(); ++i)
            REQUIRE(nd::parse<TestType>(tests[i]) == expected[i]);
        REQUIRE(nd::parse<i8>(" -43") == -43);
        REQUIRE(nd::parse<i16>("\t  -194") == -194);
        REQUIRE(nd::parse<i32>("-54052") == -54052);
        REQUIRE(nd::parse<i64>("   -525107745") == -525107745);
        REQUIRE(nd::parse<u64>("11111111155488") == 11111111155488);
    }

    GIVEN("a string that cannot be converted to an integer") {
        for (std::string_view str: {"     ", "", ".", " n10", "--10", "e10"})
            REQUIRE(not nd::parse<TestType>(str));
    }

    GIVEN("a string that falls out of range") {
        std::string tmp;
        if (std::is_signed_v<TestType>)
            REQUIRE(not nd::parse<TestType>(fmt::format("  {}1,,", min)));
        else
            REQUIRE(not nd::parse<TestType>(fmt::format("  -{}1,,", min)));

        REQUIRE(not nd::parse<TestType>(fmt::format("  {}1  ", max)));
        REQUIRE(not nd::parse<u8>("-1"));
        REQUIRE(not nd::parse<u16>("-1"));
        REQUIRE(not nd::parse<u32>("-1"));
        REQUIRE(not nd::parse<u64>("-1"));
    }
}

TEMPLATE_TEST_CASE("base::string::parse(), to floating-point", "", f32, f64) {
    GIVEN("a string that can be converted to a floating point") {
        WHEN("should return a number") {
            std::vector<std::string> tests{
                "1", " 6", "\t7", "9.", ".5", "011", "-1", "123.123", ".0",
                "10x", "-10.3", "10e3", "10e-04", "0E-12", "09999910"
            };
            std::vector<f32> expected{
                1, 6, 7, 9, .5, 11, -1, 123.123f, 0, 10, -10.3f,
                10e3, 10e-04f, 0e-12f, 9999910.
            };
            for (size_t i{}; i < tests.size(); ++i) {
                INFO(i);
                const auto result = nd::parse<TestType>(tests[i]);
                REQUIRE(result.has_value());
                REQUIRE_THAT(result.value_or(0), Catch::Matchers::WithinULP(expected[i], 2));
            }
        }

        WHEN("should return NaN") {
            auto tests = GENERATE("nan", "Nan", "-NaN" "-nan");
            REQUIRE(std::isnan(nd::parse<TestType>(tests).value()) == true);
        }

        WHEN("should return Inf") {
            auto tests = GENERATE("inf", "-inf", "INFINITY" "-INFINITY", "-Inf");
            REQUIRE(std::isinf(nd::parse<TestType>(tests).value()) == true);
        }
    }

    GIVEN("a string that can be converted to a floating point") {
        for (std::string_view str: {"     ", "", ".", " n10", "--10", "e10"})
            REQUIRE(not nd::parse<TestType>(str));
    }

    GIVEN("a string that falls out of the floating point range") {
        TestType min = std::numeric_limits<TestType>::min();
        TestType max = std::numeric_limits<TestType>::max();
        TestType lowest = std::numeric_limits<TestType>::lowest();
        std::array tests{
            fmt::format("  {}1,,", min),
            fmt::format("  {}1  ", max),
            fmt::format("  {}1  ", lowest)
        };
        for (auto& test: tests)
            REQUIRE(not nd::parse<TestType>(test));
    }
}

TEST_CASE("base::string::parse(), to boolean") {
    GIVEN("a string that can be converted to a bool") {
        WHEN("should return true") {
            const auto* to_test = GENERATE("1", "true", "TRUE", "y", "yes", "YES");
            REQUIRE(nd::parse<bool>(to_test) == true);
        }

        WHEN("should return false") {
            const auto* to_test = GENERATE("0", "false", "FALSE", "n", "no", "NO");
            REQUIRE(nd::parse<bool>(to_test) == false);
        }
    }

    GIVEN("a string that cannot be converted to a bool") {
        const auto* to_test = GENERATE("yes please", ".", "", "wrong");
        REQUIRE(not nd::parse<bool>(to_test));
    }
}
