#include <fstream>
#include <filesystem>
#include <noa/core/io/IO.hpp>
#include <noa/core/io/Encoding.hpp>
#include <catch2/catch.hpp>

#include "Utils.hpp"

using namespace ::noa::types;
namespace fs = std::filesystem;
namespace nio = noa::io;

TEMPLATE_TEST_CASE("core::io::encoding and decoding - real types", "[noa]", u8, i16, i32, u32, f16, f32, f64) {
    const nio::Encoding::Type dtype = GENERATE(
        nio::Encoding::I8, nio::Encoding::U8,
        nio::Encoding::I16, nio::Encoding::U16,
        nio::Encoding::I32, nio::Encoding::U32,
        nio::Encoding::I64, nio::Encoding::U64,
        nio::Encoding::F16, nio::Encoding::F32, nio::Encoding::F64
    );
    const bool clamp = GENERATE(true, false);
    const bool swap = GENERATE(true, false);
    auto encoding = nio::Encoding{dtype, clamp, swap};

    const auto shape = test::random_shape<i64>(1, {.batch_range={1, 10}});
    const auto ssize = shape.n_elements();
    const auto size = static_cast<size_t>(ssize);
    INFO("shape: " << shape << ", encoding: " << encoding);

    const auto data0 = std::make_unique<TestType[]>(size);
    const auto data1 = std::make_unique<TestType[]>(size);
    const auto s0 = Span(data0.get(), shape);
    const auto s1 = Span(data1.get(), shape);

    const auto n_bytes = encoding.encoded_size(ssize);
    const auto file = std::make_unique<std::byte[]>(static_cast<size_t>(n_bytes));
    const auto s2 = Span(file.get(), n_bytes);

    // Randomize data. No decimals, otherwise it makes everything more complicated.
    i64 range_min{}, range_max{};
    if (clamp) {
        range_min = dtype == nio::Encoding::F16 ? -2048 : -30000;
        range_max = dtype == nio::Encoding::F16 ? 2048 : 30000;
    } else {
        range_min = 0;
        range_max = 127;
    }
    auto randomizer = test::Randomizer<i64>(range_min, range_max);
    for (auto& e: s0.as_1d())
        e = noa::clamp_cast<TestType>(randomizer.get());

    nio::encode(s0.as_strided().as_const(), s2, encoding);
    nio::decode(s2, encoding, s1.as_strided());

    if (clamp) {
        // Encoded data was clamped to fit the data type, so clamp input data as well.
        auto[min, max] = encoding.value_range<TestType>();
        for (auto& e: s0.as_1d())
            e = noa::clamp(e, min, max);
    }

    REQUIRE(test::allclose_abs(s0, s1, 1e-6));
}

TEMPLATE_TEST_CASE("core::io::encoding and decoding - u4", "[noa]", u8, i16, i32, u32, f16, f32) {
    const bool clamp = GENERATE(true, false);
    const bool swap = GENERATE(true, false);
    const auto encoding = nio::Encoding{nio::Encoding::U4, clamp, swap};
    INFO(encoding);

    const auto shape = test::random_shape<i64>(1, {.batch_range{1, 4}, .only_even_sizes = true});
    const auto ssize = shape.n_elements();
    const auto size = static_cast<size_t>(ssize);
    INFO(size);

    const auto input = std::make_unique<TestType[]>(size);
    const auto output = std::make_unique<TestType[]>(size);
    const auto s0 = Span(input.get(), shape);
    const auto s1 = Span(output.get(), shape);

    const auto n_bytes = encoding.encoded_size(ssize);
    const auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(n_bytes));
    const auto s2 = Span(buffer.get(), n_bytes);

    // Randomize data. No decimals, otherwise it makes everything more complicated.
    auto randomizer = test::Randomizer<i64>(0, clamp ? 30 : 15);
    for (auto& e: s0.as_contiguous_1d())
        e = noa::clamp_cast<TestType>(randomizer.get());

    nio::encode(s0.as_strided().as_const(), s2, encoding);
    nio::decode(s2, encoding, s1.as_strided());

    if (clamp) {
        // Serialized data was clamped to fit the data type, so clamp input data as well.
        auto[min, max] = encoding.value_range<TestType>();
        for (auto& e: s0.as_contiguous_1d())
            e = noa::clamp(e, min, max);
    }

    REQUIRE(test::allclose_abs(s0, s1, 1e-6));
}

TEMPLATE_TEST_CASE("core::io::encoding and decoding - complex", "[noa][core]", c16, c32, c64) {
    const bool clamp = GENERATE(true, false);
    const bool swap = GENERATE(true, false);
    const nio::Encoding::Type dtype = GENERATE(nio::Encoding::C16, nio::Encoding::C32, nio::Encoding::C64);
    const auto encoding = nio::Encoding{.dtype=dtype, .clamp=false, .endian_swap=true};
    const auto shape = test::random_shape<i64>(1, {.batch_range{1, 5}});

    const auto ssize = shape.n_elements();
    const auto size = static_cast<size_t>(ssize);
    INFO("size: " << size << ", clamp:" << clamp << ", swap: " << swap);

    const auto input = std::make_unique<TestType[]>(size);
    const auto output = std::make_unique<TestType[]>(size);
    const auto s0 = Span(input.get(), shape);
    const auto s1 = Span(output.get(), shape);

    const auto n_bytes = encoding.encoded_size(ssize);
    const auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(n_bytes));
    const auto s2 = Span(buffer.get(), n_bytes);

    auto randomizer = test::Randomizer<f32>(-10000, 10000);
    for (auto& e: s0.as_contiguous_1d())
        e = TestType::from_real(randomizer.get());

    nio::encode(s0.as_const().as_strided(), s2, encoding);
    nio::decode(s2, encoding, s1.as_strided());

    if (dtype == nio::Encoding::C16)
        for (auto& e: s0.as_contiguous_1d())
            e = static_cast<TestType>(static_cast<c16>(e)); // for half, mimic conversion on raw data

    REQUIRE(test::allclose_abs(s0, s1, 1e-6));
}

TEST_CASE("core::io::encoding and decoding - many elements", "[noa][core]") {
    constexpr auto encoding = nio::Encoding{.dtype = nio::Encoding::I16, .clamp = false, .endian_swap = true};
    constexpr auto shape = Shape4<i64>{2, 256, 256, 256};
    constexpr auto ssize = shape.n_elements();
    constexpr auto size = static_cast<size_t>(ssize);

    const auto input = std::make_unique<f32[]>(size);
    const auto output = std::make_unique<f32[]>(size);
    const auto s0 = Span(input.get(), shape);
    const auto s1 = Span(output.get(), shape);

    const auto n_bytes = encoding.encoded_size(ssize);
    const auto buffer = std::make_unique<std::byte[]>(static_cast<size_t>(n_bytes));
    const auto s2 = Span(buffer.get(), n_bytes);

    test::Randomizer<f32> randomizer(-10000, 10000);
    for (auto& e: s0.as_contiguous_1d())
        e = randomizer.get();

    nio::encode(s0.as_strided<const f32>(), s2, encoding, 4);
    nio::decode(s2, encoding, s1.as_strided(), 4);

    for (auto& e: s0.as_contiguous_1d())
        e = std::trunc(e); // float/double -> int16 -> float/double
    REQUIRE(test::allclose_rel(s0, s1));
}

TEST_CASE("core::io::swap_endian", "[noa]") {
    constexpr size_t N = 100;
    auto randomizer = test::Randomizer<i32>(-1234434, 94321458);
    f32 diff{};
    for (size_t i{}; i < N; ++i) {
        auto a = static_cast<f32>(randomizer.get());
        auto b = nio::swap_endian(nio::swap_endian(a));
        diff += std::abs(a - b);
    }
    REQUIRE_THAT(diff, Catch::WithinULP(0.f, 2));
}
