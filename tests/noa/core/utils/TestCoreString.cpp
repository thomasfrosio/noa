#include <noa/core/utils/Strings.hpp>
#include <catch2/catch.hpp>

using namespace ::noa::types;
namespace ns = ::noa::string;

TEST_CASE("core::string::trim_left", "[noa][core]") {
    std::string result;
    std::vector<std::string> tests{
        "", "   ", "  foo ", "  \tfoo", "  \n foo", "  \rfoo", " foo bar ", "\t  \n 123; \n", " , 123 "
    };
    std::vector<std::string> expected{
        "", "", "foo ", "foo", "foo", "foo", "foo bar ", "123; \n", ", 123 "
    };
    for (size_t i = 0; i < tests.size(); ++i) {
        result = ns::trim_left(tests[i]);
        REQUIRE(result == expected[i]);
        result = ns::trim_left(std::move(tests[i]));
        REQUIRE(result == expected[i]);
    }
}

TEST_CASE("core::string::trim_right", "[noa][core]") {
    std::string result;
    std::vector<std::string> tests{
        "", "   ", "  foo ", "  \tfoo", " \n foo\n ", "foo \r", " foo bar ", "\t  \n 123; \n", " , 123 "
    };
    std::vector<std::string> expected{
        "", "", "  foo", "  \tfoo", " \n foo", "foo", " foo bar", "\t  \n 123;", " , 123"
    };
    for (size_t i = 0; i < tests.size(); ++i) {
        result = ns::trim_right(tests[i]);
        REQUIRE(result == expected[i]);
        result = ns::trim_right(std::move(tests[i]));
        REQUIRE(result == expected[i]);
    }
}

TEST_CASE("core::string::trim", "[noa][core]") {
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
        result = ns::trim(tests[i]);
        REQUIRE(result == expected[i]);
        result = ns::trim(std::move(tests[i]));
        REQUIRE(result == expected[i]);
    }
}

TEMPLATE_TEST_CASE("core::string::parse(), to integer", "[noa][core]", u8, u16, u32, u64, i8, i16, i32, i64) {
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
            REQUIRE(ns::parse<TestType>(tests[i]) == expected[i]);
        REQUIRE(ns::parse<i8>(" -43") == -43);
        REQUIRE(ns::parse<i16>("\t  -194") == -194);
        REQUIRE(ns::parse<i32>("-54052") == -54052);
        REQUIRE(ns::parse<i64>("   -525107745") == -525107745);
        REQUIRE(ns::parse<u64>("11111111155488") == 11111111155488);
    }

    GIVEN("a string that cannot be converted to an integer") {
        for (std::string_view str: {"     ", "", ".", " n10", "--10", "e10"})
            REQUIRE(not ns::parse<TestType>(str));
    }

    GIVEN("a string that falls out of range") {
        std::string tmp;
        if (std::is_signed_v<TestType>)
            REQUIRE(not ns::parse<TestType>(fmt::format("  {}1,,", min)));
        else
            REQUIRE(not ns::parse<TestType>(fmt::format("  -{}1,,", min)));

        REQUIRE(not ns::parse<TestType>(fmt::format("  {}1  ", max)));
        REQUIRE(not ns::parse<u8>("-1"));
        REQUIRE(not ns::parse<u16>("-1"));
        REQUIRE(not ns::parse<u32>("-1"));
        REQUIRE(not ns::parse<u64>("-1"));
    }
}

TEMPLATE_TEST_CASE("core::string::parse(), to floating-point", "[noa][core]", f32, f64) {
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
                const auto result = ns::parse<TestType>(tests[i]);
                REQUIRE(result.has_value());
                REQUIRE_THAT(result.value_or(0), Catch::WithinULP(expected[i], 2));
            }
        }

        WHEN("should return NaN") {
            auto tests = GENERATE("nan", "Nan", "-NaN" "-nan");
            REQUIRE(std::isnan(ns::parse<TestType>(tests).value()) == true);
        }

        WHEN("should return Inf") {
            auto tests = GENERATE("inf", "-inf", "INFINITY" "-INFINITY", "-Inf");
            REQUIRE(std::isinf(ns::parse<TestType>(tests).value()) == true);
        }
    }

    GIVEN("a string that can be converted to a floating point") {
        for (std::string_view str: {"     ", "", ".", " n10", "--10", "e10"})
            REQUIRE(not ns::parse<TestType>(str));
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
            REQUIRE(not ns::parse<TestType>(test));
    }
}

TEST_CASE("core::string::parse(), to boolean", "[noa][core]") {
    GIVEN("a string that can be converted to a bool") {
        WHEN("should return true") {
            const auto* to_test = GENERATE("1", "true", "TRUE", "y", "yes", "YES");
            REQUIRE(ns::parse<bool>(to_test) == true);
        }

        WHEN("should return false") {
            const auto* to_test = GENERATE("0", "false", "FALSE", "n", "no", "NO");
            REQUIRE(ns::parse<bool>(to_test) == false);
        }
    }

    GIVEN("a string that cannot be converted to a bool") {
        const auto* to_test = GENERATE("yes please", ".", "", "wrong");
        REQUIRE(not ns::parse<bool>(to_test));
    }
}
