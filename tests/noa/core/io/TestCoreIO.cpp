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

    const nio::DataType dtype = GENERATE(
        nio::DataType::U4,
        nio::DataType::I8, nio::DataType::U8,
        nio::DataType::I16, nio::DataType::U16,
        nio::DataType::I32, nio::DataType::U32,
        nio::DataType::I64, nio::DataType::U64,
        nio::DataType::F16, nio::DataType::F32, nio::DataType::F64
    );
    const bool clamp = GENERATE(true, false);
    const bool swap = GENERATE(true, false);
    auto options = nio::EncodeOptions{.clamp = clamp, .endian_swap = swap};

    const auto shape = test::random_shape<isize>(1, {
        .batch_range={1, 10},
        .only_even_sizes = dtype == nio::DataType::U4
    });
    const auto ssize = shape.n_elements();
    const auto size = static_cast<size_t>(ssize);
    INFO("shape: " << shape << ", clamp: " << clamp << ", swap: " << swap);

    // Randomize data. No decimals, otherwise it makes everything more complicated.
    i64 range_min{}, range_max{};
    if (dtype == nio::DataType::U4) {
        range_max = clamp ? 30 : 15;
    } else {
        if (clamp) {
            range_min = dtype == nio::DataType::F16 ? -2048 : -30000;
            range_max = dtype == nio::DataType::F16 ? 2048 : 30000;
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
            auto[min, max] = dtype.value_range<TestType>();
            v1 = noa::clamp(v0, min, max);
        } else {
            v1 = v0;
        }
    }

    const auto n_bytes = dtype.n_bytes(ssize);
    const auto decoded_ptr = std::make_unique<TestType[]>(size);
    const auto decoded_span = Span(decoded_ptr.get(), shape);
    {
        const auto encoded_ptr = std::make_unique<std::byte[]>(static_cast<size_t>(n_bytes));
        const auto encoded_span = Span(encoded_ptr.get(), n_bytes);
        nio::encode(data_span.as_strided().as_const(), encoded_span, dtype, options);
        nio::decode(encoded_span, dtype, decoded_span.as_strided(), options);
        REQUIRE(test::allclose_abs(data_decoded_span, decoded_span, 1e-6));
    }

    {
        auto file = nio::BinaryFile(filename, {.write = true}, {.new_size = n_bytes});
        nio::encode(data_span.as_strided().as_const(), file.stream(), dtype, options);
        file.open(filename, {.read = true});
        nio::decode(file.stream(), dtype, decoded_span.as_strided(), options);
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
    const nio::DataType dtype = GENERATE(nio::DataType::C16, nio::DataType::C32, nio::DataType::C64);
    const auto options = nio::EncodeOptions{.clamp=clamp, .endian_swap=swap};
    const auto shape = test::random_shape<isize>(1, {.batch_range{1, 5}});

    const auto ssize = shape.n_elements();
    const auto size = static_cast<size_t>(ssize);
    INFO(size);
    INFO(clamp);
    INFO(swap);

    const auto data_ptr = std::make_unique<TestType[]>(size);
    const auto data_span = Span(data_ptr.get(), shape);
    const auto data_decoded_ptr = std::make_unique<TestType[]>(size);
    const auto data_decoded_span = Span(data_decoded_ptr.get(), shape);
    auto randomizer = test::Randomizer<f32>(-10000, 10000);
    for (auto&& [v0, v1]: zip(data_span.as_1d(), data_decoded_span.as_1d())) {
        v0 = TestType::from_values(randomizer.get(), randomizer.get());
        if (dtype == nio::DataType::C16) {
            // For f16, cast to have the same loss of precision than encoded/decoded data.
            v1 = static_cast<TestType>(static_cast<c16>(v0));
        } else {
            v1 = v0;
        }
    }

    const auto n_bytes = dtype.n_bytes(ssize);
    const auto decoded_ptr = std::make_unique<TestType[]>(size);
    const auto decoded_span = Span(decoded_ptr.get(), shape);

    {
        const auto encoded_ptr = std::make_unique<std::byte[]>(static_cast<size_t>(n_bytes));
        const auto encoded_span = Span(encoded_ptr.get(), n_bytes);
        nio::encode(data_span.as_strided().as_const(), encoded_span, dtype, options);
        nio::decode(encoded_span, dtype, decoded_span.as_strided(), options);
        REQUIRE(test::allclose_abs(data_decoded_span, decoded_span, 1e-6));
    }

    {
        auto file = nio::BinaryFile(filename, {.write = true}, {.new_size = n_bytes});
        nio::encode(data_span.as_strided().as_const(), file.stream(), dtype, options);
        file.open(filename, {.read = true});
        nio::decode(file.stream(), dtype, decoded_span.as_strided(), options);
        REQUIRE(test::allclose_abs(data_decoded_span, decoded_span, 1e-6));
    }

    nio::remove_all(directory);
}

TEST_CASE("core::io::encoding and decoding - many elements") {
    const fs::path directory = "test_encoding";
    const fs::path filename = directory / "data";
    nio::mkdir(directory);

    constexpr auto dtype = nio::DataType{"i16"};
    constexpr auto options = nio::EncodeOptions{
        .clamp = false,
        .endian_swap = true,
        .n_threads = 4,
    };
    constexpr auto shape = Shape4{2, 256, 256, 256};
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

    const auto n_bytes = dtype.n_bytes(ssize);
    const auto decoded_ptr = std::make_unique<f32[]>(size);
    const auto decoded_span = Span(decoded_ptr.get(), shape);

    {
        const auto encoded_ptr = std::make_unique<std::byte[]>(static_cast<size_t>(n_bytes));
        const auto encoded_span = Span(encoded_ptr.get(), n_bytes);
        nio::encode(data_span.as_strided().as_const(), encoded_span, dtype, options);
        nio::decode(encoded_span, dtype, decoded_span.as_strided(), options);
        REQUIRE(test::allclose_abs(data_decoded_span, decoded_span, 1e-6));
    }

    {
        auto file = nio::BinaryFile(filename, {.write = true}, {.new_size = n_bytes});
        nio::encode(data_span.as_strided().as_const(), file.stream(), dtype, options);
        file.open(filename, {.read = true});
        nio::decode(file.stream(), dtype, decoded_span.as_strided(), options);
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
