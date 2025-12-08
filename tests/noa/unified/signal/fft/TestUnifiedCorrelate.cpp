#include <noa/unified/fft/Transform.hpp>
#include <noa/unified/fft/Factory.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/geometry/Draw.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/Reduce.hpp>
#include <noa/unified/signal/Correlate.hpp>
#include <noa/unified/Blas.hpp>
#include <noa/unified/IO.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
namespace nf = ::noa::fft;
namespace ng = ::noa::geometry;
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
    auto generate_data(const auto& shape) {
        auto data = TestData<N>{};
        data.radius = Vec<f64, N>::from_value(N == 2 ? 25.: 20.);
        data.smoothness = 7.;
        data.lhs_center = (shape.vec.template pop_front<4 - N>() / 2).template as<f64>();

        constexpr auto max_shift = N == 2 ? 30.: 10.;
        test::Randomizer<f64> randomizer{-max_shift, max_shift};
        test::randomize(data.expected_shift.data(), N, randomizer);
        data.rhs_center = data.lhs_center + data.expected_shift;
        return data;
    }
}

TEMPLATE_TEST_CASE("unified::signal, correlation peak", "", Vec2<f32>, Vec2<f64>, Vec3<f32>, Vec3<f64>) {
    using value_t = TestType::value_type;
    constexpr size_t N = TestType::SIZE;

    auto devices = std::vector<Device>{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    auto shape = test::random_shape(N) + Shape<i64, N>::from_value(N == 2 ? 200 : 50).template push_front<4 - N>(0);
    auto data = generate_data<N>(shape);

    for (auto correlation_mode: cross_correlation_modes) {
        const auto xmap_options = ns::CrossCorrelationMapOptions{
            .mode=correlation_mode,
            .ifft_norm=nf::NORM_DEFAULT,
        };

        for (auto device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);

            auto [lhs, lhs_rfft] = nf::empty<value_t>(shape, options);
            auto [rhs, rhs_rfft] = nf::empty<value_t>(shape, options);
            const auto xmap = Array<value_t>(shape, options);
            const auto buffer = noa::like(lhs_rfft);

            ng::draw({}, lhs, ng::Rectangle{data.lhs_center, data.radius, data.smoothness}.draw());
            ng::draw({}, rhs, ng::Rectangle{data.rhs_center, data.radius, data.smoothness}.draw());
            nf::r2c(lhs, lhs_rfft);
            nf::r2c(rhs, rhs_rfft);

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

TEMPLATE_TEST_CASE("unified::signal, correlation peak batched", "", Vec2<f32>, Vec2<f64>, Vec3<f32>,  Vec3<f64>) {
    using value_t = TestType::value_type;
    constexpr size_t N = TestType::SIZE;

    auto devices = std::vector<Device>{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    auto shape = test::random_shape_batched(N, {.batch_range = {2, 3}});
    if (N == 3)
        shape[1] += 80;
    shape[2] += 80;
    shape[3] += 80;
    INFO(shape);

    std::vector<TestData<N>> data;
    const auto lhs_inverse_affine_matrices = Array<Mat<f32, N + 1, N + 1>>(shape[0]);
    const auto rhs_inverse_affine_matrices = Array<Mat<f32, N + 1, N + 1>>(shape[0]);
    for (auto i: noa::irange(shape[0])) {
        auto tmp = generate_data<N>(shape);
        data.emplace_back(tmp);
        lhs_inverse_affine_matrices(0, 0, 0, i) = ng::translate(-tmp.lhs_center).template as<f32>();
        rhs_inverse_affine_matrices(0, 0, 0, i) = ng::translate(-tmp.rhs_center).template as<f32>();
    }

    for (auto correlation_mode: cross_correlation_modes) {
        const auto xmap_options = ns::CrossCorrelationMapOptions{.mode=correlation_mode, .ifft_norm=nf::NORM_DEFAULT};
        const auto xpeak_options = ns::CrossCorrelationPeakOptions<N>{};

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, Stream::DEFAULT);
            const auto options = ArrayOption(device, Allocator::MANAGED);

            auto [lhs, lhs_rfft] = nf::empty<value_t>(shape, options);
            auto [rhs, rhs_rfft] = nf::empty<value_t>(shape, options);
            const auto xmap = Array<value_t>(shape, options);
            const auto buffer = noa::like(lhs_rfft);

            ng::draw({}, lhs, ng::Rectangle{
                .radius=data[0].radius, .smoothness=data[0].smoothness}.template draw<f32>(),
                 lhs_inverse_affine_matrices.to({device}));
            ng::draw({}, rhs, ng::Rectangle{
                .radius=data[0].radius, .smoothness=data[0].smoothness}.template draw<f32>(),
                 rhs_inverse_affine_matrices.to({device}));

            nf::r2c(lhs, lhs_rfft);
            nf::r2c(rhs, rhs_rfft);

            auto run = [&]<nf::Layout REMAP>(){
                INFO(REMAP);
                ns::cross_correlation_map<REMAP>(lhs_rfft, rhs_rfft, xmap, xmap_options, buffer);

                const auto shifts = Array<Vec<f32, N>>(shape[0], options);
                const auto values = Array<value_t>(shape[0], options);
                constexpr auto REMAP_ = REMAP.flip().erase_output();
                ns::cross_correlation_peak<REMAP_>(xmap, shifts, values, xpeak_options);

                for (size_t i{}; i < shifts.eval().size(); ++i) {
                    INFO(i);
                    auto computed_shift = -(shifts(0, 0, 0, i).template as<f64>() - data[i].lhs_center);
                    if (correlation_mode == ns::Correlation::DOUBLE_PHASE)
                        computed_shift /= 2;

                    // fmt::println("Computed shift={::.6f}, expected={::.6f}, diff={::.6f}",
                    //              computed_shift, data[i].expected_shift, abs(computed_shift - data[i].expected_shift));

                    for (size_t j: noa::irange(N))
                        REQUIRE_THAT(computed_shift[j], Catch::Matchers::WithinAbs(data[i].expected_shift[j], 5e-2));

                    const auto max = noa::max(xmap.subregion(i));
                    REQUIRE(max <= values(0, 0, 0, i));
                }
            };
            run.template operator()<"H2FC">();
            run.template operator()<"H2F">();
        }
    }
}

TEST_CASE("unified::signal, autocorrelate") {
    auto devices = std::vector<Device>{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const auto shape = test::random_shape(3);
    const auto center = (shape.pop_front() / 2).vec.as<f64>();

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        const auto lhs = noa::random(noa::Uniform{-50.f, 50.f}, shape, options);
        const auto lhs_rfft = nf::r2c(lhs);
        const auto rhs_rfft = lhs_rfft.copy();
        const auto xmap = noa::like<f32>(lhs);
        ns::cross_correlation_map<"H2F">(lhs_rfft, rhs_rfft, xmap);
        const auto [shift, _] = ns::cross_correlation_peak_3d<"f">(xmap);
        REQUIRE_THAT(shift[0], Catch::Matchers::WithinAbs(center[0], 5e-2));
        REQUIRE_THAT(shift[1], Catch::Matchers::WithinAbs(center[1], 5e-2));
        REQUIRE_THAT(shift[2], Catch::Matchers::WithinAbs(center[2], 5e-2));

        // turn off the reduction, take the zero-lag
        const auto shift2 = ns::cross_correlation_peak_3d<"f">(xmap, {.maximum_lag = {}}).first;
        REQUIRE(all(shift == shift2));
    }
}

TEST_CASE("unified::signal::cross_correlation_score") {
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

        const auto lhs_l2_normalized = noa::like(lhs);
        const auto lhs_zero_normalized = noa::like(lhs);
        const auto lhs_zero_l2_normalized = noa::like(lhs);
        noa::normalize(lhs, lhs_l2_normalized, {.mode = noa::Norm::L2});
        noa::normalize(lhs, lhs_zero_normalized, {.mode = noa::Norm::MEAN_STD});
        noa::normalize(lhs_zero_normalized, lhs_zero_l2_normalized, {.mode = noa::Norm::L2});

        const auto rhs_l2_normalized = noa::like(rhs);
        const auto rhs_zero_normalized = noa::like(rhs);
        const auto rhs_zero_l2_normalized = noa::like(rhs);
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

TEST_CASE("unified::signal::cross_correlation_peak, no registration") {
    auto devices = std::vector<Device>{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const auto shape = Shape<i64, 4>{1, 1, 64, 64};
    for (auto& device: devices) {
        const auto stream = noa::StreamGuard(device);
        const auto options = noa::ArrayOption(device, Allocator::MANAGED);

        const auto lhs = Array<f64>(shape, options);
        ng::draw({}, lhs, ng::Sphere{.center=Vec{32., 32.}, .radius=6., .smoothness = 4.}.draw());
        noa::normalize(lhs, lhs, {.mode = noa::Norm::MEAN_STD});
        noa::normalize(lhs, lhs, {.mode = noa::Norm::L2});
        const auto lhs_rfft = nf::r2c(lhs, {.norm = nf::Norm::BACKWARD});

        const auto rhs = Array<f64>(shape, options);
        ng::draw({}, rhs, ng::Sphere{.center=Vec{35., 34.}, .radius=6., .smoothness = 4.}.draw());
        noa::normalize(rhs, rhs, {.mode = noa::Norm::MEAN_STD});
        noa::normalize(rhs, rhs, {.mode = noa::Norm::L2});
        const auto rhs_rfft = nf::r2c(rhs, {.norm = nf::Norm::BACKWARD});

        // Using a cross-correlation map.
        const auto xmap = noa::like(rhs);
        ns::cross_correlation_map<"h2fc">(lhs_rfft, rhs_rfft, xmap, {.ifft_norm = nf::Norm::BACKWARD});
        auto [peak_coord, peak_value] = ns::cross_correlation_peak_2d<"fc2fc">(xmap, {.registration_radius = Vec<i64, 2>{}});

        // Using argmax.
        auto [argmax_value, argmax_offset] = noa::argmax(xmap);
        auto indices = noa::indexing::offset2index(argmax_offset, xmap);

        REQUIRE(noa::allclose(argmax_value, peak_value));
        REQUIRE((indices[0] == 0 and indices[1] == 0));
        REQUIRE(noa::all(noa::allclose(indices.filter(2, 3).as<f64>(), peak_coord)));
    }
}
