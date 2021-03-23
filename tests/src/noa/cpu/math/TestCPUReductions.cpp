#include <noa/cpu/math/Reductions.h>

#include <noa/Types.h>
#include <noa/cpu/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

template<typename T>
std::pair<T, T> getMinMax(T* data, size_t elements) {
    T min{data[0]}, max{data[0]};

    for (size_t idx = 0; idx < elements; ++idx) {
        if (data[idx] < min)
            min = data[idx];
        if (max < data[idx])
            max = data[idx];
    }
    return {min, max};
}

template<typename T>
std::pair<T, T> getSumMean(T* data, size_t elements) {
    T sum{};
    for (size_t idx = 0; idx < elements; ++idx)
        sum += data[idx];
    return {sum, sum / static_cast<Noa::Traits::value_type_t<T>>(elements)};
}

template<typename T>
T getWeightedSum(T* data, T* weights, size_t elements) {
    T sum{};
    for (size_t idx = 0; idx < elements; ++idx)
        sum += data[idx] * weights[idx];
    return sum;
}

template<typename T>
T getVariance(T* data, size_t elements, T mean) {
    T var{0};
    for (size_t idx = 0; idx < elements; ++idx)
        var += (data[idx] - mean) * (data[idx] - mean);
    var /= static_cast<Noa::Traits::value_type_t<T>>(elements);
    return var;
}

template<typename T>
Noa::Stats<T> getStats(T* data, size_t elements) {
    auto[min, max] = getMinMax(data, elements);
    auto[sum, mean] = getSumMean(data, elements);
    T variance = getVariance(data, elements, mean);
    return {min, max, sum, mean, variance, Math::sqrt(variance)};
}

TEMPLATE_TEST_CASE("CPU: Reductions: real", "[noa][cpu][math]", float, double) {
    Test::Randomizer<TestType> randomizer(0., 1.);
    size_t elements = Test::IntRandomizer<size_t>(100, 1000).get();
    Noa::PtrHost<TestType> data(elements);
    Test::initDataRandom(data.get(), elements, randomizer);

    Noa::Stats<TestType> expected_stats = getStats(data.get(), elements);
    Noa::Stats<TestType> result_stats;

    AND_THEN("Min & Max") {
        REQUIRE(expected_stats.min == Math::min(data.get(), elements));
        REQUIRE(expected_stats.max == Math::max(data.get(), elements));
        auto[min, max] = Math::minMax(data.get(), elements);
        REQUIRE(expected_stats.min == min);
        REQUIRE(expected_stats.max == max);
    }

    AND_THEN("Sum & Min & Max") {
        result_stats = Math::sumMinMax(data.get(), elements);
        REQUIRE(result_stats.min == expected_stats.min);
        REQUIRE(result_stats.max == expected_stats.max);
        REQUIRE_THAT(result_stats.sum, Test::isWithinAbs(expected_stats.sum, 1e-8));
    }

    AND_THEN("Mean & Sum weighted") {
        TestType mean = Math::mean(data.get(), elements);
        REQUIRE(mean == expected_stats.mean);

        Noa::PtrHost<TestType> weights(elements);
        Test::initDataRandom(weights.get(), elements, randomizer);
        TestType weighted_sum = Math::sumWeighted(data.get(), weights.get(), elements);
        TestType weighted_sum_expected = getWeightedSum(data.get(), weights.get(), elements);

        REQUIRE_THAT(weighted_sum, Test::isWithinAbs(weighted_sum_expected, 1e-8));
    }

    AND_THEN("Variance") {
        auto[mean, variance] = Math::variance(data.get(), elements);
        REQUIRE_THAT(mean, Test::isWithinAbs(expected_stats.mean, 1e-8));
        REQUIRE_THAT(variance, Test::isWithinAbs(expected_stats.variance, 1e-6));
    }

    AND_THEN("Statistics") {
        result_stats = Math::statistics(data.get(), elements);
        REQUIRE(result_stats.min == expected_stats.min);
        REQUIRE(result_stats.max == expected_stats.max);
        REQUIRE_THAT(result_stats.sum, Test::isWithinAbs(expected_stats.sum, 1e-8));
        REQUIRE_THAT(result_stats.mean, Test::isWithinAbs(expected_stats.mean, 1e-8));
        REQUIRE_THAT(result_stats.variance, Test::isWithinAbs(expected_stats.variance, 1e-6));
        REQUIRE_THAT(result_stats.stdev, Test::isWithinAbs(expected_stats.stdev, 1e-6));
    }
}

void my_copy(cfloat_t* complex, float* real) {
    for (int i = 0; i < 1000; ++i) {
        real[i] = complex[i].imag();
    }
}

TEMPLATE_TEST_CASE("CPU: Reductions: complex", "[noa][cpu][math]", cfloat_t) {
    Test::Randomizer<TestType> randomizer(0., 1.);
    size_t elements = 1000;
    Noa::PtrHost<TestType> data(elements);
    Test::initDataRandom(data.get(), elements, randomizer);

    Noa::Stats<TestType> expected_stats;
    std::tie(expected_stats.sum, expected_stats.mean) = getSumMean(data.get(), elements);

    expected_stats.variance = getVariance(data.get(), elements, expected_stats.mean);
    Noa::Stats<TestType> result_stats;

    AND_THEN("Sum & Mean & Sum weighted") {
        result_stats.sum = Math::sum(data.get(), elements);
        result_stats.mean = Math::mean(data.get(), elements);
        REQUIRE_THAT(result_stats.sum, Test::isWithinAbs(expected_stats.sum, 1e-3)); // about 500.000
        REQUIRE_THAT(result_stats.mean, Test::isWithinAbs(expected_stats.mean, 1e-6));
    }

    AND_THEN("Variance") {
        auto[mean, variance] = Math::variance(data.get(), elements);
        REQUIRE_THAT(mean, Test::isWithinAbs(expected_stats.mean, 1e-6));
        REQUIRE_THAT(variance, Test::isWithinAbs(expected_stats.variance, 1e-6));
    }
}
