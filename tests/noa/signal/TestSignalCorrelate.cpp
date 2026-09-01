#include <noa/runtime/Factory.hpp>
#include <noa/runtime/Random.hpp>
#include <noa/runtime/Reduce.hpp>
#include <noa/runtime/Blas.hpp>
#include <noa/base/Zip.hpp>

#include <noa/fft/Transform.hpp>
#include <noa/fft/Factory.hpp>
#include <noa/fft/Remap.hpp>
#include <noa/io/IO.hpp>
#include <noa/signal/Correlate.hpp>
#include <noa/xform/Draw.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
namespace nf = ::noa::fft;
namespace ng = ::noa::xform;
namespace ns = ::noa::signal;

namespace {
    // DOUBLE_PHASE seem to be the more accurate, then CONVENTIONAL, MUTUAL and PHASE.
    constexpr auto cross_correlation_modes = std::array{
        ns::Correlation::CONVENTIONAL,
        ns::Correlation::MUTUAL,
        ns::Correlation::PHASE,
        ns::Correlation::DOUBLE_PHASE,
    };

    template<size_t N>
    struct TestData {
        Vec<f64, N> radius;
        f64 smoothness;
        Vec<f64, N> lhs_center;
        Vec<f64, N> rhs_center;
        Vec<f64, N> expected_shift;
    };

    template<size_t N>
    auto generate_data(const Shape<isize, N>& shape) {
        auto data = TestData<N>{};
        data.radius = Vec<f64, N>::from_value(N == 2 ? 25. : 20.);
        data.smoothness = 7.;
        data.lhs_center = (shape.vec / 2).template as<f64>();

        constexpr auto max_shift = N == 2 ? 30.: 10.;
        test::Randomizer<f64> randomizer{-max_shift, max_shift};
        test::randomize(data.expected_shift.data(), N, randomizer);
        data.rhs_center = data.lhs_center + data.expected_shift;
        return data;
    }
}

TEMPLATE_TEST_CASE("signal::cross_correlation_map/peak", "", (Vec<f32, 2>), (Vec<f64, 2>), (Vec<f32, 3>), (Vec<f64, 3>)) {
    using value_t = TestType::value_type;
    constexpr size_t N = TestType::SIZE;

    auto devices = std::vector<Device>{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    auto shape = test::random_shape<isize, N>(N) + Shape<isize, N>::from_value(N == 2 ? 200 : 50);
    auto data = generate_data(shape);

    for (auto correlation_mode: cross_correlation_modes) {
        const auto xmap_options = ns::CrossCorrelationMapOptions{
            .mode = correlation_mode,
            .rank = N,
            .ifft_norm = nf::NORM_DEFAULT,
        };

        for (auto device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);

            auto [lhs, lhs_rfft] = nf::empty<value_t>(shape, options);
            auto [rhs, rhs_rfft] = nf::empty<value_t>(shape, options);
            const auto xmap = Array<value_t, N>(shape, options);
            const auto buffer = noa::empty_like(lhs_rfft);

            ng::draw<N>({}, lhs, ng::Rectangle{data.lhs_center, data.radius, data.smoothness}.get());
            ng::draw<N>({}, rhs, ng::Rectangle{data.rhs_center, data.radius, data.smoothness}.get());
            nf::r2c(lhs, lhs_rfft, {.rank = N});
            nf::r2c(rhs, rhs_rfft, {.rank = N});

            auto run = [&]<nf::Layout REMAP>(auto xpeak_options) {
                ns::cross_correlation_map<REMAP>(lhs_rfft, rhs_rfft, xmap, xmap_options, buffer);

                constexpr auto REMAP_ = REMAP.flip().erase_output();
                auto [peak_coordinate, peak_value] = ns::cross_correlation_peak<REMAP_>(xmap, xpeak_options);
                auto computed_shift = -(peak_coordinate - data.lhs_center);
                if (correlation_mode == ns::Correlation::DOUBLE_PHASE)
                    computed_shift /= 2;

                const auto max_value = noa::max(xmap);
                REQUIRE(max_value <= peak_value);

                for (size_t i: noa::irange(N))
                    REQUIRE_THAT(computed_shift[i], Catch::Matchers::WithinAbs(data.expected_shift[i], 5e-2));
                return computed_shift;
            };

            const auto xpeak_options = ns::CrossCorrelationPeakOptions<N>{};
            const auto shift_centered = run.template operator()<"H2FC">(xpeak_options);
            const auto shift_not_centered = run.template operator()<"H2F">(xpeak_options);

            auto xpeak_options_max = ns::CrossCorrelationPeakOptions{
                .maximum_lag = abs(data.expected_shift) * 2
            };
            if (correlation_mode == ns::Correlation::DOUBLE_PHASE)
                xpeak_options_max.maximum_lag *= 2;

            // fmt::print("shape={}, maximum_lag={}\n", shape, xpeak_options_max.maximum_lag);
            const auto shift_centered_max = run.template operator()<"H2FC">(xpeak_options_max);
            const auto shift_not_centered_max = run.template operator()<"H2F">(xpeak_options_max);

            for (size_t i: noa::irange(N)) {
                REQUIRE_THAT(shift_not_centered[i], Catch::Matchers::WithinAbs(shift_centered[i], 1e-4));
                REQUIRE_THAT(shift_not_centered[i], Catch::Matchers::WithinAbs(shift_not_centered_max[i], 1e-6));
                REQUIRE_THAT(shift_centered[i], Catch::Matchers::WithinAbs(shift_centered_max[i], 1e-6));
            }
        }
    }
}

TEMPLATE_TEST_CASE("signal::correlation_map/peak batched", "", (Vec<f32, 2>), (Vec<f64, 2>), (Vec<f32, 3>), (Vec<f64, 3>)) {
    using value_t = TestType::value_type;
    constexpr usize R = TestType::SIZE;
    constexpr usize B = 2;
    constexpr usize N = B + R;

    auto devices = std::vector<Device>{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    auto shape = test::random_shape_batched<isize, N>(R, {.size_range = {64, 100}, .batch_range = {1, 3}});
    if (R == 3)
        shape[N - 3] += 80;
    shape[N - 2] += 80;
    shape[N - 1] += 80;
    // shape = nf::next_fast_shape(shape, R); // much faster, but we want to test odd sizes too so leave commented out
    INFO(shape);

    const auto [shape_b, shape_r] = shape.template split<B>();
    const auto data_batched = Array<TestData<R>, B>(shape_b);
    const auto lhs_inverse_affine_matrices = Array<Mat<f32, R + 1, R + 1>, B>(shape_b);
    const auto rhs_inverse_affine_matrices = Array<Mat<f32, R + 1, R + 1>, B>(shape_b);
    for (auto&& [data, lhs, rhs]: noa::zip(
        data_batched.span_1d(),
        lhs_inverse_affine_matrices.span_1d(),
        rhs_inverse_affine_matrices.span_1d()
    )) {
        data = generate_data(shape_r);
        lhs = ng::translate(-data.lhs_center).template as<f32>();
        rhs = ng::translate(-data.rhs_center).template as<f32>();
    }

    for (auto correlation_mode: cross_correlation_modes) {
        const auto xmap_options = ns::CrossCorrelationMapOptions{
            .mode=correlation_mode,
            .rank = R,
            .ifft_norm=nf::NORM_DEFAULT,
        };
        const auto xpeak_options = ns::CrossCorrelationPeakOptions<R>{};

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, Stream::SYNC);
            const auto options = ArrayOption(device, Allocator::MANAGED);

            auto [lhs, lhs_rfft] = nf::empty<value_t>(shape, options);
            auto [rhs, rhs_rfft] = nf::empty<value_t>(shape, options);
            const auto xmap = noa::empty<value_t>(shape, options);
            const auto buffer = noa::empty_like(lhs_rfft);

            const auto radius = data_batched.first().radius;
            const auto smoothness = data_batched.first().smoothness;
            const auto rectangle = ng::Rectangle{.radius=radius, .smoothness=smoothness}.template get<f32>();
            ng::draw<R>({}, lhs, rectangle, lhs_inverse_affine_matrices.to({device}));
            ng::draw<R>({}, rhs, rectangle, rhs_inverse_affine_matrices.to({device}));

            nf::r2c(lhs, lhs_rfft, {.rank = R});
            nf::r2c(rhs, rhs_rfft, {.rank = R});

            auto run = [&]<nf::Layout REMAP>(){
                INFO(REMAP);
                ns::cross_correlation_map<REMAP>(lhs_rfft, rhs_rfft, xmap, xmap_options, buffer);

                const auto shifts = Array<Vec<f32, R>, B>(shape_b, options);
                const auto values = Array<value_t, B>(shape_b, options);
                constexpr auto REMAP_ = REMAP.flip().erase_output();
                ns::cross_correlation_peaks<REMAP_>(xmap, shifts, values, xpeak_options);

                auto xmap_br = xmap.reshape(Shape{shape_b.n_elements(), shape_r.n_elements()});
                for (usize i{}; auto&& [data, shift, value]: noa::zip(
                    data_batched.span_1d(),
                    shifts.span_1d(),
                    values.span_1d()
                )) {
                    INFO(i);
                    auto computed_shift = -(shift.template as<f64>() - data.lhs_center);
                    if (correlation_mode == ns::Correlation::DOUBLE_PHASE)
                        computed_shift /= 2;

                    // fmt::println("Computed shift={::.6f}, expected={::.6f}, diff={::.6f}",
                    //              computed_shift, data[i].expected_shift, abs(computed_shift - data[i].expected_shift));

                    for (size_t r: noa::irange(R))
                        REQUIRE_THAT(computed_shift[r], Catch::Matchers::WithinAbs(data.expected_shift[r], 5e-2));

                    const auto max = noa::max(xmap_br.subregion(i++));
                    REQUIRE(max <= value);
                }
            };
            run.template operator()<"H2FC">();
            run.template operator()<"H2F">();
        }
    }
}

TEST_CASE("signal:: autocorrelate") {
    auto devices = std::vector<Device>{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const auto shape = test::random_shape<isize, 3>(3);
    const auto center = (shape / 2).vec.as<f64>();

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        const auto lhs = noa::random(noa::Uniform{-50.f, 50.f}, shape, options);
        const auto lhs_rfft = nf::rfft3(lhs);
        const auto rhs_rfft = lhs_rfft.copy();
        const auto xmap = noa::empty_like<f32>(lhs);
        ns::cross_correlation_map_3d<"H2F">(lhs_rfft, rhs_rfft, xmap);
        const auto [shift, _] = ns::cross_correlation_peak_3d<"f">(xmap);
        REQUIRE_THAT(shift[0], Catch::Matchers::WithinAbs(center[0], 5e-2));
        REQUIRE_THAT(shift[1], Catch::Matchers::WithinAbs(center[1], 5e-2));
        REQUIRE_THAT(shift[2], Catch::Matchers::WithinAbs(center[2], 5e-2));

        // turn off the reduction, take the zero-lag
        const auto shift2 = ns::cross_correlation_peak_3d<"f">(xmap, {.maximum_lag = {}}).first;
        REQUIRE(shift == shift2);
    }
}

TEST_CASE("signal::cross_correlation_score") {
    auto devices = std::vector<Device>{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const auto shape = test::random_shape(3);
    for (auto& device: devices) {
        const auto stream = noa::StreamGuard(device);
        const auto options = noa::ArrayOption(device, Allocator::MANAGED);

        // Mean=2.
        const auto lhs = noa::random(noa::Normal(0., 1.), shape, options);
        noa::normalize(lhs, lhs, {.mode = noa::Norm::MEAN_STD});
        noa::ewise(noa::wrap(lhs, 2.), lhs, noa::Plus{});

        const auto rhs = noa::random(noa::Normal(0., 1.), shape, options);
        noa::normalize(rhs, rhs, {.mode = noa::Norm::MEAN_STD});
        noa::ewise(noa::wrap(rhs, 2.), rhs, noa::Plus{});

        const auto lhs_l2_normalized = noa::empty_like(lhs);
        const auto lhs_zero_normalized = noa::empty_like(lhs);
        const auto lhs_zero_l2_normalized = noa::empty_like(lhs);
        noa::normalize(lhs, lhs_l2_normalized, {.mode = noa::Norm::L2});
        noa::normalize(lhs, lhs_zero_normalized, {.mode = noa::Norm::MEAN_STD});
        noa::normalize(lhs_zero_normalized, lhs_zero_l2_normalized, {.mode = noa::Norm::L2});

        const auto rhs_l2_normalized = noa::empty_like(rhs);
        const auto rhs_zero_normalized = noa::empty_like(rhs);
        const auto rhs_zero_l2_normalized = noa::empty_like(rhs);
        noa::normalize(rhs, rhs_l2_normalized, {.mode = noa::Norm::L2});
        noa::normalize(rhs, rhs_zero_normalized, {.mode = noa::Norm::MEAN_STD});
        noa::normalize(rhs_zero_normalized, rhs_zero_l2_normalized, {.mode = noa::Norm::L2});

        const auto dot = noa::dot(lhs.flat(), rhs.flat());
        const auto cc = ns::cross_correlation_score(lhs, rhs, {.center = false, .normalize = false});
        REQUIRE_THAT(dot, Catch::Matchers::WithinAbs(cc, 1e-5));

        const auto zcc0 = ns::cross_correlation_score(lhs, rhs, {.center = true, .normalize = false});
        const auto zcc1 = ns::cross_correlation_score(lhs_zero_normalized, rhs_zero_normalized, {.center = false, .normalize = false});
        REQUIRE_THAT(zcc0, Catch::Matchers::WithinAbs(zcc1, 1e-5));

        const auto ncc0 = ns::cross_correlation_score(lhs, rhs, {.center = false, .normalize = true});
        const auto ncc1 = ns::cross_correlation_score(lhs_l2_normalized, rhs_l2_normalized, {.center = false, .normalize = false});
        REQUIRE_THAT(ncc0, Catch::Matchers::WithinAbs(ncc1, 1e-5));

        const auto zncc0 = ns::cross_correlation_score(lhs, rhs, {.center = true, .normalize = true});
        const auto zncc1 = ns::cross_correlation_score(lhs_zero_l2_normalized, rhs_zero_l2_normalized, {.center = false, .normalize = false});
        REQUIRE_THAT(zncc0, Catch::Matchers::WithinAbs(zncc1, 1e-5));
    }
}

TEST_CASE("signal::cross_correlation_peak, no registration") {
    auto devices = std::vector<Device>{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const auto shape = Shape<isize, 2>{64, 64};
    for (auto& device: devices) {
        const auto stream = noa::StreamGuard(device);
        const auto options = noa::ArrayOption(device, Allocator::MANAGED);

        const auto lhs = Array<f64, 2>(shape, options);
        ng::draw_2d({}, lhs, ng::Sphere{.center=Vec{32., 32.}, .radius=6., .smoothness = 4.}.get());
        noa::normalize(lhs, lhs, {.mode = noa::Norm::MEAN_STD});
        noa::normalize(lhs, lhs, {.mode = noa::Norm::L2});
        const auto lhs_rfft = nf::rfft2(lhs, {.norm = nf::Norm::BACKWARD});

        const auto rhs = Array<f64, 2>(shape, options);
        ng::draw_2d({}, rhs, ng::Sphere{.center=Vec{35., 34.}, .radius=6., .smoothness = 4.}.get());
        noa::normalize(rhs, rhs, {.mode = noa::Norm::MEAN_STD});
        noa::normalize(rhs, rhs, {.mode = noa::Norm::L2});
        const auto rhs_rfft = nf::rfft2(rhs, {.norm = nf::Norm::BACKWARD});

        // Using a cross-correlation map.
        const auto xmap = noa::empty_like(rhs);
        ns::cross_correlation_map_2d<"h2fc">(lhs_rfft, rhs_rfft, xmap, {.ifft_norm = nf::Norm::BACKWARD});
        auto [peak_coord, peak_value] = ns::cross_correlation_peak_2d<"fc2fc">(xmap, {.registration_radius = Vec<i32, 2>{}});

        // Using argmax.
        auto [argmax_value, argmax_offset] = noa::argmax(xmap);
        auto indices = noa::offset2index(argmax_offset, xmap);

        REQUIRE(noa::allclose(argmax_value, peak_value));
        REQUIRE(noa::allclose(indices.as<f64>(), peak_coord));
    }
}
