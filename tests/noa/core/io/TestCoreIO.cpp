#include <fstream>
#include <filesystem>

#include <noa/core/io/IO.hpp>
#include <noa/core/io/Encoding.hpp>
#include <noa/core/io/BinaryFile.hpp>
#include <noa/core/utils/Zip.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
namespace fs = std::filesystem;
namespace nio = noa::io;

TEMPLATE_TEST_CASE("core::io::encoding and decoding - real types", "", u8, i16, i32, u32, f16, f32, f64) {
    const fs::path directory = "test_encoding";
    const fs::path filename = directory / "data";
    nio::mkdir(directory);

    const nio::Encoding::Type dtype = GENERATE(
        nio::Encoding::U4,
        nio::Encoding::I8, nio::Encoding::U8,
        nio::Encoding::I16, nio::Encoding::U16,
        nio::Encoding::I32, nio::Encoding::U32,
        nio::Encoding::I64, nio::Encoding::U64,
        nio::Encoding::F16, nio::Encoding::F32, nio::Encoding::F64
    );
    const bool clamp = GENERATE(true, false);
    const bool swap = GENERATE(true, false);
    auto encoding = nio::Encoding{dtype, clamp, swap};

    const auto shape = test::random_shape<i64>(1, {
        .batch_range={1, 10},
        .only_even_sizes = dtype == nio::Encoding::U4
    });
    const auto ssize = shape.n_elements();
    const auto size = static_cast<size_t>(ssize);
    INFO("shape: " << shape << ", encoding: " << encoding);

    // Randomize data. No decimals, otherwise it makes everything more complicated.
    i64 range_min{}, range_max{};
    if (dtype == nio::Encoding::U4) {
        range_max = clamp ? 30 : 15;
    } else {
        if (clamp) {
            range_min = dtype == nio::Encoding::F16 ? -2048 : -30000;
            range_max = dtype == nio::Encoding::F16 ? 2048 : 30000;
        } else {
            range_min = 0;
            range_max = 127;
        }
    }
    auto randomizer = test::Randomizer<i64>(range_min, range_max);

    const auto data_ptr = std::make_unique<TestType[]>(size);
    const auto data_decoded_ptr = std::make_unique<TestType[]>(size);
    const auto data_span = Span(data_ptr.get(), shape);
    const auto data_decoded_span = Span(data_decoded_ptr.get(), shape);

    for (auto&& [v0, v1]: zip(data_span.as_1d(), data_decoded_span.as_1d())) {
        v0 = noa::clamp_cast<TestType>(randomizer.get());
        if (clamp) {
            // Encoded data is clamped to fit the data type, so clamp input data as well.
            auto[min, max] = encoding.value_range<TestType>();
            v1 = noa::clamp(v0, min, max);
        } else {
            v1 = v0;
        }
    }

    const auto n_bytes = encoding.encoded_size(ssize);
    const auto decoded_ptr = std::make_unique<TestType[]>(size);
    const auto decoded_span = Span(decoded_ptr.get(), shape);
    {
        const auto encoded_ptr = std::make_unique<std::byte[]>(static_cast<size_t>(n_bytes));
        const auto encoded_span = Span(encoded_ptr.get(), n_bytes);
        nio::encode(data_span.as_strided().as_const(), encoded_span, encoding);
        nio::decode(encoded_span, encoding, decoded_span.as_strided());
        REQUIRE(test::allclose_abs(data_decoded_span, decoded_span, 1e-6));
    }

    {
        auto file = nio::BinaryFile(filename, {.write = true}, {.new_size = n_bytes});
        nio::encode(data_span.as_strided().as_const(), file.stream(), encoding);
        file.open(filename, {.read = true});
        nio::decode(file.stream(), encoding, decoded_span.as_strided());
        REQUIRE(test::allclose_abs(data_decoded_span, decoded_span, 1e-6));
    }

    nio::remove_all(directory);
}

TEMPLATE_TEST_CASE("core::io::encoding and decoding - complex", "", c16, c32, c64) {
    const fs::path directory = "test_encoding";
    const fs::path filename = directory / "data";
    nio::mkdir(directory);

    const bool clamp = GENERATE(true, false);
    const bool swap = GENERATE(true, false);
    const nio::Encoding::Type dtype = GENERATE(nio::Encoding::C16, nio::Encoding::C32, nio::Encoding::C64);
    const auto encoding = nio::Encoding{.dtype=dtype, .clamp=clamp, .endian_swap=swap};
    const auto shape = test::random_shape<i64>(1, {.batch_range{1, 5}});

    const auto ssize = shape.n_elements();
    const auto size = static_cast<size_t>(ssize);
    INFO(size);
    INFO(encoding);

    const auto data_ptr = std::make_unique<TestType[]>(size);
    const auto data_span = Span(data_ptr.get(), shape);
    const auto data_decoded_ptr = std::make_unique<TestType[]>(size);
    const auto data_decoded_span = Span(data_decoded_ptr.get(), shape);
    auto randomizer = test::Randomizer<f32>(-10000, 10000);
    for (auto&& [v0, v1]: zip(data_span.as_1d(), data_decoded_span.as_1d())) {
        v0 = TestType::from_values(randomizer.get(), randomizer.get());
        if (dtype == nio::Encoding::C16) {
            // For f16, cast to have the same loss of precision than encoded/decoded data.
            v1 = static_cast<TestType>(static_cast<c16>(v0));
        } else {
            v1 = v0;
        }
    }

    const auto n_bytes = encoding.encoded_size(ssize);
    const auto decoded_ptr = std::make_unique<TestType[]>(size);
    const auto decoded_span = Span(decoded_ptr.get(), shape);

    {
        const auto encoded_ptr = std::make_unique<std::byte[]>(static_cast<size_t>(n_bytes));
        const auto encoded_span = Span(encoded_ptr.get(), n_bytes);
        nio::encode(data_span.as_strided().as_const(), encoded_span, encoding);
        nio::decode(encoded_span, encoding, decoded_span.as_strided());
        REQUIRE(test::allclose_abs(data_decoded_span, decoded_span, 1e-6));
    }

    {
        auto file = nio::BinaryFile(filename, {.write = true}, {.new_size = n_bytes});
        nio::encode(data_span.as_strided().as_const(), file.stream(), encoding);
        file.open(filename, {.read = true});
        nio::decode(file.stream(), encoding, decoded_span.as_strided());
        REQUIRE(test::allclose_abs(data_decoded_span, decoded_span, 1e-6));
    }

    nio::remove_all(directory);
}

TEST_CASE("core::io::encoding and decoding - many elements") {
    const fs::path directory = "test_encoding";
    const fs::path filename = directory / "data";
    nio::mkdir(directory);

    constexpr auto encoding = nio::Encoding{
        .dtype = nio::Encoding::I16,
        .clamp = false,
        .endian_swap = true
    };
    constexpr auto shape = Shape4<i64>{2, 256, 256, 256};
    constexpr auto ssize = shape.n_elements();
    constexpr auto size = static_cast<size_t>(ssize);

    const auto data_ptr = std::make_unique<f32[]>(size);
    const auto data_span = Span(data_ptr.get(), shape);
    const auto data_decoded_ptr = std::make_unique<f32[]>(size);
    const auto data_decoded_span = Span(data_decoded_ptr.get(), shape);

    test::Randomizer<f32> randomizer(-10000, 10000);
    for (auto&& [v0, v1]: zip(data_span.as_1d(), data_decoded_span.as_1d())) {
        v0 = randomizer.get();
        v1 = std::trunc(v0); // float/double -> i16 -> float/double
    }

    const auto n_bytes = encoding.encoded_size(ssize);
    const auto decoded_ptr = std::make_unique<f32[]>(size);
    const auto decoded_span = Span(decoded_ptr.get(), shape);

    {
        const auto encoded_ptr = std::make_unique<std::byte[]>(static_cast<size_t>(n_bytes));
        const auto encoded_span = Span(encoded_ptr.get(), n_bytes);
        nio::encode(data_span.as_strided().as_const(), encoded_span, encoding, 4);
        nio::decode(encoded_span, encoding, decoded_span.as_strided(), 4);
        REQUIRE(test::allclose_abs(data_decoded_span, decoded_span, 1e-6));
    }

    {
        auto file = nio::BinaryFile(filename, {.write = true}, {.new_size = n_bytes});
        nio::encode(data_span.as_strided().as_const(), file.stream(), encoding, 4);
        file.open(filename, {.read = true});
        nio::decode(file.stream(), encoding, decoded_span.as_strided(), 4);
        REQUIRE(test::allclose_abs(data_decoded_span, decoded_span, 1e-6));
    }
}

TEST_CASE("core::io::swap_endian") {
    constexpr size_t N = 100;
    auto randomizer = test::Randomizer<i32>(-1234434, 94321458);
    f32 diff{};
    for (size_t i{}; i < N; ++i) {
        auto a = static_cast<f32>(randomizer.get());
        auto b = nio::swap_endian(nio::swap_endian(a));
        diff += std::abs(a - b);
    }
    REQUIRE_THAT(diff, Catch::Matchers::WithinULP(0.f, 2));
}

TEST_CASE("core::io::expand_user") {
    const char* home_directory = std::getenv("HOME");
    if (not home_directory)
        return;

    for (std::string postfix: {"", "/dir/file", "/dir\\file"}) {
        fs::path raw = std::string("~") + postfix;
        nio::expand_user(raw);
        fs::path expanded = std::string(home_directory) + postfix;
        REQUIRE(raw == expanded);
    }
}
