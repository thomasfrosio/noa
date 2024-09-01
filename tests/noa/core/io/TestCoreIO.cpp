#include <fstream>
#include <filesystem>
#include <noa/core/io/IO.hpp>
#include <catch2/catch.hpp>

#include "Utils.hpp"

using namespace ::noa::types;
namespace fs = std::filesystem;
namespace nio = noa::io;

TEMPLATE_TEST_CASE("core::io::(de)serialize - real types", "[noa][core]",
                   u8, i16, i32, u32, f16, f32, f64) {
    const fs::path test_dir = "testIO";
    const fs::path test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    const auto shape = test::random_shape<i64>(1, {.batch_range={1, 10}});
    const auto ssize = shape.n_elements();
    const auto size = static_cast<size_t>(ssize);
    const nio::Encoding::Format dtype =
            GENERATE(nio::Encoding::I8, nio::Encoding::U8,
                     nio::Encoding::I16, nio::Encoding::U16,
                     nio::Encoding::I32, nio::Encoding::U32,
                     nio::Encoding::I64, nio::Encoding::U64,
                     nio::Encoding::F16, nio::Encoding::F32, nio::Encoding::F64);
    const bool clamp = GENERATE(true, false);
    const bool swap = GENERATE(true, false);

    INFO("shape: " << shape << ", dtype: " << dtype << ", clamp:" << clamp << ", swap: " << swap);

    const auto data = std::make_unique<TestType[]>(size);
    const auto read_data = std::make_unique<TestType[]>(size);
    const auto s0 = Span<TestType, 4>(data.get(), shape);
    const auto s1 = Span<TestType, 4>(read_data.get(), shape);

    // Randomize data. No decimals, otherwise it makes everything more complicated.
    i64 range_min{}, range_max{};
    if (clamp) {
        range_min = dtype == nio::Encoding::F16 ? -2048 : -30000;
        range_max = dtype == nio::Encoding::F16 ? 2048 : 30000;
    } else {
        range_min = 0;
        range_max = 127;
    }
    test::Randomizer<i64> randomizer(range_min, range_max);
    for (auto& e: s0.as_contiguous_1d())
        e = noa::clamp_cast<TestType>(randomizer.get());

    // Serialize:
    auto file = std::fstream(test_file, std::ios::out | std::ios::trunc);
    auto encoding = nio::Encoding{dtype, clamp, swap};
    nio::serialize(s0.as_const(), file, encoding);
    file.close();
    REQUIRE(static_cast<i64>(fs::file_size(test_file)) == encoding.encoded_size(ssize));

    // Deserialize:
    file.open(test_file, std::ios::in);
    nio::deserialize(file, encoding, s1);

    if (clamp) {
        // Serialized data was clamped to fit the data type, so clamp input data as well.
        auto[min, max] = encoding.value_range<TestType>();
        for (auto& e: s0.as_contiguous_1d())
            e = noa::clamp(e, min, max);
    }

    REQUIRE(test::allclose_abs(data.get(), read_data.get(), static_cast<i64>(size), 1e-6));
    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}

TEMPLATE_TEST_CASE("core::io::(de)serialize - U4", "[noa][core]", u8, i16, i32, u32, f16, f32) {
    const fs::path test_dir = "testIO";
    const fs::path test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    const bool clamp = GENERATE(true, false);
    const bool swap = GENERATE(true, false);
    const auto shape = test::random_shape<i64>(1, {.batch_range{1, 4}});
    const auto ssize = shape.n_elements();
    const auto size = static_cast<size_t>(ssize);

    const auto encoding = nio::Encoding{nio::Encoding::U4, clamp, swap};
    INFO("size: " << size << ", clamp:" << clamp << ", swap: " << swap);

    const auto data = std::make_unique<TestType[]>(size);
    const auto read_data = std::make_unique<TestType[]>(size);
    const auto s0 = Span<TestType, 4>(data.get(), shape);
    const auto s1 = Span<TestType, 4>(read_data.get(), shape);

    // Randomize data. No decimals, otherwise it makes everything more complicated.
    test::Randomizer<int64_t> randomizer(0, clamp ? 30 : 15);
    for (auto& e: s0.as_contiguous_1d())
        e = noa::clamp_cast<TestType>(randomizer.get());

    // Serialize:
    std::fstream file(test_file, std::ios::out | std::ios::trunc);
    nio::serialize(s0.as_const(), file, encoding);
    file.close();
    REQUIRE(static_cast<i64>(fs::file_size(test_file)) == encoding.encoded_size(ssize, shape[3]));

    // Deserialize:
    file.open(test_file, std::ios::in);
    nio::deserialize(file, encoding, s1);

    if (clamp) {
        // Serialized data was clamped to fit the data type, so clamp input data as well.
        auto[min, max] = encoding.value_range<TestType>();
        for (auto& e: s0.as_contiguous_1d())
            e = noa::clamp(e, min, max);
    }

    REQUIRE(test::allclose_abs(data.get(), read_data.get(), static_cast<i64>(size), 1e-6));
    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}

TEMPLATE_TEST_CASE("core::io::(de)serialize - complex", "[noa][core]", c16, c32, c64) {
    const fs::path test_dir = "testIO";
    const fs::path test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    const bool clamp = GENERATE(true, false);
    const bool swap = GENERATE(true, false);
    const nio::Encoding::Format dtype = GENERATE(nio::Encoding::C16, nio::Encoding::C32, nio::Encoding::C64);
    const auto encoding = nio::Encoding{.format=dtype, .clamp=false, .endian_swap=true};
    const auto shape = test::random_shape<i64>(1, {.batch_range{1, 5}});

    const auto ssize = shape.n_elements();
    const auto size = static_cast<size_t>(ssize);
    INFO("size: " << size << ", clamp:" << clamp << ", swap: " << swap);

    const auto data = std::make_unique<TestType[]>(size);
    const auto read_data = std::make_unique<TestType[]>(size);
    const auto s0 = Span<TestType, 4>(data.get(), shape);
    const auto s1 = Span<TestType, 4>(read_data.get(), shape);

    // Randomize data.
    test::Randomizer<f32> randomizer(-10000, 10000);
    for (auto& e: s0.as_contiguous_1d())
        e = TestType::from_real(randomizer.get());

    // Serialize:
    std::fstream file(test_file, std::ios::out | std::ios::trunc);
    nio::serialize(s0.as_const(), file, encoding);
    file.close();
    REQUIRE(static_cast<i64>(fs::file_size(test_file)) == encoding.encoded_size(ssize));

    // Deserialize:
    file.open(test_file, std::ios::in);
    nio::deserialize(file, encoding, s1);

    if (dtype == nio::Encoding::C16) {
        for (auto& e: s0.as_contiguous_1d())
            e = static_cast<TestType>(static_cast<c16>(e)); // for half, mimic conversion on raw data
    }
    REQUIRE(test::allclose_abs(data.get(), read_data.get(), static_cast<i64>(size), 1e-6));
    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}

TEST_CASE("core::io::(de)serialize - many elements", "[noa][core]") {
    const fs::path test_dir = "testIO";
    const fs::path test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    const auto encoding = nio::Encoding{.format=nio::Encoding::I16, .clamp=false, .endian_swap=true};
    const Shape4<i64> shape{2, 256, 256, 256};
    const auto ssize = shape.n_elements();
    const auto size = static_cast<size_t>(ssize);

    const auto data = std::make_unique<f32[]>(size);
    const auto read_data = std::make_unique<f32[]>(size);
    const auto s0 = Span<f32, 4>(data.get(), shape);
    const auto s1 = Span<f32, 4>(read_data.get(), shape);

    // Randomize data.
    test::Randomizer<f32> randomizer(-10000, 10000);
    for (auto& e: s0.as_contiguous_1d())
        e = randomizer.get();

    // Serialize:
    std::fstream file(test_file, std::ios::out | std::ios::trunc);
    nio::serialize(s0.as_const(), file, encoding);
    file.close();
    REQUIRE(static_cast<i64>(fs::file_size(test_file)) == encoding.encoded_size(ssize));

    // Deserialize:
    file.open(test_file, std::ios::in);
    nio::deserialize(file, encoding, s1);

    for (auto& e: s0.as_contiguous_1d())
        e = std::trunc(e); // float/double -> int16 -> float/double
    REQUIRE(test::allclose_rel(data.get(), read_data.get(), size));

    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}

TEST_CASE("core::io::swap_endian()", "[noa][core]") {
    constexpr size_t N = 100;
    const auto data1 = std::make_unique<float[]>(N);
    const auto data2 = std::make_unique<float[]>(N);
    for (size_t i{}; i < N; ++i) {
        auto t = static_cast<float>(test::Randomizer<int>(-1234434, 94321458).get());
        data1[i] = t;
        data2[i] = t;
    }
    nio::swap_endian(data1.get(), N);
    nio::swap_endian(data1.get(), N);
    f32 diff{0};
    for (size_t i{}; i < N; ++i)
        diff += data1[i] - data2[i];
    REQUIRE_THAT(diff, Catch::WithinULP(0.f, 2));
}
