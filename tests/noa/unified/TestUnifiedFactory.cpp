#include <noa/unified/Factory.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;

TEMPLATE_TEST_CASE("unified::arange(), cpu", "", i32, i64, u32, u64, f32, f64, c32, c64) {
    using real_t = noa::traits::value_type_t<TestType>;

    {
        constexpr i64 n_elements = 100;
        const Array<TestType> results(n_elements);
        const Array<TestType> expected(n_elements);

        c32 a;
        a = 1;

        noa::arange(results, noa::Arange{.start=TestType{0}, .step=TestType{1}});
        for (i64 i{}; auto& e: expected.span_1d_contiguous())
            e = static_cast<real_t>(i++);

        REQUIRE(test::allclose_rel(results, expected, 1e-7f));
    }

    {
        constexpr i64 n_elements = 55;
        const Array<TestType> results(n_elements);
        const Array<TestType> expected(n_elements);

        noa::arange(results, noa::Arange{.start=TestType{3}, .step=TestType{5}});
        for (i64 i{}; auto& e: expected.span_1d_contiguous())
            e = real_t(3) + static_cast<real_t>(i++) * real_t(5);

        REQUIRE(test::allclose_rel(results, expected, 1e-7f));
    }

    {
        const auto shape = test::random_shape_batched(3);
        const Array<TestType> results(shape);
        const Array<TestType> expected(shape);

        noa::arange(results, noa::Arange{.start=TestType{3}, .step=TestType{5}});
        for (i64 i{}; auto& e: expected.span_1d_contiguous())
            e = real_t(3) + static_cast<real_t>(i++) * real_t(5);

        REQUIRE(test::allclose_rel(results, expected, 1e-7f));
    }
}

TEST_CASE("unified::linspace()") {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for(const auto& device: devices) {
        const StreamGuard guard(device, Stream::DEFAULT);
        const ArrayOption options(device, Allocator::MANAGED);

        {
            constexpr i64 n_elements = 5;
            const auto results = noa::linspace(n_elements, noa::Linspace{0., 5., false}, options);
            constexpr std::array expected = {0., 1., 2., 3., 4.};
            REQUIRE(test::allclose_abs(results.eval().get(), expected.data(), n_elements, 1e-7));
        }

        {
            constexpr i64 n_elements = 5;
            const Array<f64> results(n_elements, options);
            const f64 step = noa::linspace(results.view(), noa::Linspace{0., 5.});
            constexpr std::array expected = {0., 1.25, 2.5, 3.75, 5.};
            REQUIRE(step == 1.25);
            REQUIRE(test::allclose_abs(results.eval().get(), expected.data(), n_elements, 1e-7));
        }

        {
            constexpr i64 n_elements = 8;
            const Array<f64> results(n_elements);
            const f64 step = noa::linspace(results, noa::Linspace{0., 1.});
            constexpr std::array expected = {
                0., 0.14285714, 0.28571429, 0.42857143,
                0.57142857, 0.71428571, 0.85714286, 1.
            };
            REQUIRE_THAT(step, Catch::Matchers::WithinAbs(0.14285714285714285, 1e-9));
            REQUIRE(test::allclose_abs(results.eval().get(), expected.data(), n_elements, 1e-7));
        }

        {
            constexpr i64 n_elements = 8;
            const Array<f64> results(n_elements, options);
            const f64 step = noa::linspace(results, noa::Linspace{0., 1., false});
            constexpr std::array expected = {0., 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875};
            REQUIRE(step == 0.125);
            REQUIRE(test::allclose_abs(results.eval().get(), expected.data(), n_elements, 1e-7));
        }

        {
            constexpr i64 n_elements = 9;
            const Array<f64> results(n_elements, options);
            const f64 step = noa::linspace(results, noa::Linspace{3., 40.});
            constexpr std::array expected = {3., 7.625, 12.25, 16.875, 21.5, 26.125, 30.75, 35.375, 40.};
            REQUIRE(step == 4.625);
            REQUIRE(test::allclose_abs(results.eval().get(), expected.data(), n_elements, 1e-7));
        }

        {
            constexpr i64 n_elements = 1;
            const Array<f64> results(n_elements, options);
            const f64 step = noa::linspace(results, noa::Linspace{0., 1., false});
            constexpr std::array expected = {0.};
            REQUIRE(step == 1.);
            REQUIRE(test::allclose_abs(results.eval().get(), expected.data(), n_elements, 1e-7));
        }
    }
}

TEMPLATE_TEST_CASE("unified:: factories, cpu vs gpu", "", i32, i64, u32, u64, f64, c64) {
    const bool pad = GENERATE(false, true);
    const auto type = GENERATE(Device::CPU, Device::GPU);
    INFO(pad);

    if (not Device::is_any(type))
        return;

    const auto subregion_shape = test::random_shape_batched(3);
    auto shape = subregion_shape;
    if (pad) {
        shape[1] += 10;
        shape[2] += 11;
        shape[3] += 12;
    }

    const auto options = ArrayOption(Device(type), Allocator::MANAGED);
    INFO(options.device);

    auto results = Array<TestType>(shape, options);
    results = results.subregion(
            noa::indexing::Ellipsis{},
            noa::indexing::Slice{0, subregion_shape[1]},
            noa::indexing::Slice{0, subregion_shape[2]},
            noa::indexing::Slice{0, subregion_shape[3]});

    AND_THEN("arange") {
        const TestType start = test::Randomizer<TestType>(0, 3).get();
        const TestType step = test::Randomizer<TestType>(1, 2).get();
        noa::arange(results, noa::Arange{start, step});

        const auto expected = Array<TestType>(results.shape());
        test::arange(expected.get(), expected.n_elements(), start, step);

        REQUIRE(test::allclose_abs_safe(results, expected, 1e-5));
    }

    AND_THEN("zeros/ones/fill") {
        const auto value = test::Randomizer<TestType>(0, 400).get();

        noa::fill(results, value);
        REQUIRE(test::allclose_abs_safe(results, value, 1e-6));

        const auto zeros = noa::zeros<TestType>(shape, options);
        REQUIRE(test::allclose_abs_safe(zeros, TestType{0}, 1e-6));

        const auto ones = noa::ones<TestType>(shape, options);
        REQUIRE(test::allclose_abs_safe(ones, TestType{1}, 1e-6));

        const auto fill = noa::fill(shape, value, options);
        REQUIRE(test::allclose_abs_safe(fill, value, 1e-6));
    }
}

TEST_CASE("unified:: Factory functions") {
    using namespace noa::types;

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    AND_THEN("arange") {
        Array expected = noa::empty<f64>(test::random_shape_batched(3));
        for (i64 i{}; auto& e: expected.span_1d_contiguous())
        e = static_cast<f64>(i++);

        for (auto device: devices) {
            auto a = noa::arange<f64>(expected.shape(), noa::Arange{0., 1.}, {.device=device, .allocator="unified"});
            REQUIRE(test::allclose_abs(a, expected));
        }
    }

    AND_THEN("linspace") {
        Array expected = noa::empty<f64>(test::random_shape_batched(3));
        auto linspace = noa::Linspace{.start=0., .stop=30., .endpoint=true}.for_size(expected.n_elements());
        for (i64 i{}; auto& e: expected.span_1d_contiguous())
            e = linspace(i++);

        for (auto device: devices) {
            auto a = noa::linspace(expected.shape(), noa::Linspace{0., 30.}, {.device=device, .allocator="unified"});
            REQUIRE(test::allclose_abs(a, expected));
        }
    }

    AND_THEN("iota") {
        Array expected = noa::empty<f64>({2, 20, 20, 20});
        auto tile = Vec{1, 5, 5, 5};

        Span span = expected.span_contiguous();
        for (i64 i{}; i < span.shape()[0]; ++i)
            for (i64 j{}; j < span.shape()[1]; ++j)
                for (i64 k{}; k < span.shape()[2]; ++k)
                    for (i64 l{}; l < span.shape()[3]; ++l)
                        span(i, j, k, l) = static_cast<f64>(
                                noa::indexing::offset_at(expected.strides(), Vec{i, j, k, l} % tile.as<i64>()));

        for (auto device: devices) {
            auto a = noa::iota<f64>(expected.shape(), tile, {.device=device, .allocator="unified"});
            REQUIRE(test::allclose_abs(a, expected));
        }
    }
}
