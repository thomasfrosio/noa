#include <noa/common/io/ImageFile.h>
#include <noa/common/io/TextFile.h>
#include <noa/common/string/Convert.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Reductions.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cpu::math:: basic statistics", "[assets][noa][cpu][math]") {
    path_t path = test::PATH_TEST_DATA / "math";
    YAML::Node tests = YAML::LoadFile(path / "param.yaml")["stats"];

    const YAML::Node& input = tests["input"];
    auto batches = input["batches"].as<uint>();
    auto shape = input["shape"].as<size2_t>();
    auto input_filename = path / input["path"].as<path_t>();
    auto output_filename = path / tests["output"].as<path_t>();

    io::ImageFile file(input_filename, io::READ);
    size_t elements = noa::elements(shape);
    cpu::memory::PtrHost<float> data(elements * batches);
    file.readAll(data.get());

    // Collect results:
    tests = YAML::LoadFile(output_filename);
    std::vector<float> expected_max, expected_min, expected_mean, expected_std, expected_sum, expected_var;
    for (size_t i = 0; i < tests.size(); i++) {
        expected_max.emplace_back(tests[i]["max"].as<float>());
        expected_min.emplace_back(tests[i]["min"].as<float>());
        expected_mean.emplace_back(tests[i]["mean"].as<float>());
        expected_std.emplace_back(tests[i]["std"].as<float>());
        expected_sum.emplace_back(tests[i]["sum"].as<float>());
        expected_var.emplace_back(tests[i]["var"].as<float>());
    }

    cpu::memory::PtrHost<float> results(batches * 6);
    float* mins = results.get();
    float* maxs = results.get() + batches * 1;
    float* sums = results.get() + batches * 2;
    float* means = results.get() + batches * 3;
    float* variances = results.get() + batches * 4;
    float* stddevs = results.get() + batches * 5;

    WHEN("min, max, sum, mean, variance, stddev") {
        cpu::math::min(data.get(), mins, elements, batches);
        cpu::math::max(data.get(), maxs, elements, batches);
        cpu::math::sum(data.get(), sums, elements, batches);
        cpu::math::mean(data.get(), means, elements, batches);
        cpu::math::variance(data.get(), means, variances, elements, batches);
        cpu::math::stddev(data.get(), means, stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], test::isWithinAbs(expected_min[batch], 1e-6));
            REQUIRE_THAT(maxs[batch], test::isWithinAbs(expected_max[batch], 1e-6));
            REQUIRE_THAT(sums[batch], test::isWithinRel(expected_sum[batch]));
            REQUIRE_THAT(means[batch], test::isWithinRel(expected_mean[batch]));
            REQUIRE_THAT(variances[batch], test::isWithinRel(expected_var[batch]));
            REQUIRE_THAT(stddevs[batch], test::isWithinRel(expected_std[batch]));
        }
        cpu::math::variance(data.get(), variances, elements, batches);
        cpu::math::stddev(data.get(), stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(variances[batch], test::isWithinRel(expected_var[batch]));
            REQUIRE_THAT(stddevs[batch], test::isWithinRel(expected_std[batch]));
        }
    }

    WHEN("minMax, sum, mean, variance, stddev") {
        cpu::math::minMax(data.get(), mins, maxs, elements, batches);
        cpu::math::sum(data.get(), sums, elements, batches);
        cpu::math::mean(data.get(), means, elements, batches);
        cpu::math::varianceStddev(data.get(), means, variances, stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], test::isWithinAbs(expected_min[batch], 1e-6));
            REQUIRE_THAT(maxs[batch], test::isWithinAbs(expected_max[batch], 1e-6));
            REQUIRE_THAT(sums[batch], test::isWithinRel(expected_sum[batch]));
            REQUIRE_THAT(means[batch], test::isWithinRel(expected_mean[batch]));
            REQUIRE_THAT(variances[batch], test::isWithinRel(expected_var[batch]));
            REQUIRE_THAT(stddevs[batch], test::isWithinRel(expected_std[batch]));
        }
    }

    WHEN("minMaxSumMean, variance, stddev") {
        cpu::math::minMaxSumMean(data.get(), mins, maxs, sums, means, elements, batches);
        cpu::math::varianceStddev(data.get(), means, variances, stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], test::isWithinAbs(expected_min[batch], 1e-6));
            REQUIRE_THAT(maxs[batch], test::isWithinAbs(expected_max[batch], 1e-6));
            REQUIRE_THAT(sums[batch], test::isWithinRel(expected_sum[batch]));
            REQUIRE_THAT(means[batch], test::isWithinRel(expected_mean[batch]));
            REQUIRE_THAT(variances[batch], test::isWithinRel(expected_var[batch]));
            REQUIRE_THAT(stddevs[batch], test::isWithinRel(expected_std[batch]));
        }
    }

    WHEN("minMaxSumMean, variance") {
        cpu::math::minMax(data.get(), mins, maxs, elements, batches);
        cpu::math::sumMeanVarianceStddev(data.get(), sums, means, variances, stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], test::isWithinAbs(expected_min[batch], 1e-6));
            REQUIRE_THAT(maxs[batch], test::isWithinAbs(expected_max[batch], 1e-6));
            REQUIRE_THAT(sums[batch], test::isWithinRel(expected_sum[batch]));
            REQUIRE_THAT(means[batch], test::isWithinRel(expected_mean[batch]));
            REQUIRE_THAT(variances[batch], test::isWithinRel(expected_var[batch]));
            REQUIRE_THAT(stddevs[batch], test::isWithinRel(expected_std[batch]));
        }
    }

    WHEN("statistics") {
        cpu::math::statistics(data.get(), mins, maxs, sums, means, variances, stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], test::isWithinAbs(expected_min[batch], 1e-6));
            REQUIRE_THAT(maxs[batch], test::isWithinAbs(expected_max[batch], 1e-6));
            REQUIRE_THAT(sums[batch], test::isWithinRel(expected_sum[batch]));
            REQUIRE_THAT(means[batch], test::isWithinRel(expected_mean[batch]));
            REQUIRE_THAT(variances[batch], test::isWithinRel(expected_var[batch]));
            REQUIRE_THAT(stddevs[batch], test::isWithinRel(expected_std[batch]));
        }
    }
}

TEST_CASE("cpu::math:: reductions", "[assets][noa][cpu][math]") {
    path_t path = test::PATH_TEST_DATA / "math";
    YAML::Node tests = YAML::LoadFile(path / "param.yaml")["reductions"];

    const YAML::Node& input = tests["input"];
    auto batches = input["batches"].as<uint>();
    auto nb_vectors = input["vectors"].as<uint>();
    auto elements = input["elements"].as<size_t>();
    path_t path_vectors = path / input["path"].as<path_t>();
    path_t path_weights = path / input["path_weights"].as<path_t>();

    path_t path_reduce_add = path / tests["sum"].as<path_t>();
    path_t path_reduce_mean = path / tests["mean"].as<path_t>();
    path_t path_reduce_weighted_mean = path / tests["weighted_mean"].as<path_t>();

    io::ImageFile file(path_vectors, io::READ);
    cpu::memory::PtrHost<float> vectors(elements * nb_vectors * batches);
    file.readAll(vectors.get());

    cpu::memory::PtrHost<float> expected_reduce(elements * batches);
    cpu::memory::PtrHost<float> result_reduce(elements * batches);

    AND_THEN("reduceAdd") {
        file.open(path_reduce_add, io::READ);
        file.readAll(expected_reduce.get());
        cpu::math::reduceAdd(vectors.get(), result_reduce.get(), elements, nb_vectors, batches);
        float diff = test::getAverageDifference(expected_reduce.get(), result_reduce.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-5));
    }

    AND_THEN("reduceMean") {
        file.open(path_reduce_mean, io::READ);
        file.readAll(expected_reduce.get());
        cpu::math::reduceMean(vectors.get(), result_reduce.get(), elements, nb_vectors, batches);
        float diff = test::getAverageDifference(expected_reduce.get(), result_reduce.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-5));
    }

    AND_THEN("reduceMeanWeighted") {
        file.open(path_reduce_weighted_mean, io::READ);
        file.readAll(expected_reduce.get());

        cpu::memory::PtrHost<float> weights(elements * nb_vectors);
        file.open(path_weights, io::READ);
        file.readAll(weights.get());

        cpu::math::reduceMeanWeighted(vectors.get(), weights.get(), result_reduce.get(), elements, nb_vectors, batches);
        float diff = test::getAverageDifference(expected_reduce.get(), result_reduce.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-5));
    }
}
