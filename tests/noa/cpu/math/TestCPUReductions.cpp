#include <noa/common/files/MRCFile.h>
#include <noa/common/files/TextFile.h>
#include <noa/common/string/Convert.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Reductions.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cpu::math:: basic statistics", "[noa][cpu][math]") {
    path_t directory = test::PATH_TEST_DATA / "math";
    path_t path_data = directory / "tmp_stats_random_array.mrc";
    path_t path_stats = directory / "tmp_stats_random_array.txt";
    MRCFile file_data(path_data, io::READ);

    size3_t shape = file_data.getShape();
    size_t elements = getElementsSlice(shape);
    uint batches = static_cast<uint>(shape.z);

    cpu::memory::PtrHost<float> data(elements * batches);
    file_data.readAll(data.get());

    TextFile<std::ifstream> file_stats(path_stats, io::READ);
    std::string line;
    cpu::memory::PtrHost<float> expected_stats(batches * 6);
    for (uint idx = 0; idx < batches * 6; ++idx) {
        file_stats.getLine(line);
        expected_stats[idx] = string::toFloat(line);
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
            REQUIRE_THAT(mins[batch], test::isWithinAbs(expected_stats[batch], 1e-6));
            REQUIRE_THAT(maxs[batch], test::isWithinAbs(expected_stats[batch + batches * 1], 1e-6));
            REQUIRE_THAT(sums[batch], test::isWithinRel(expected_stats[batch + batches * 2]));
            REQUIRE_THAT(means[batch], test::isWithinRel(expected_stats[batch + batches * 3]));
            REQUIRE_THAT(variances[batch], test::isWithinRel(expected_stats[batch + batches * 4]));
            REQUIRE_THAT(stddevs[batch], test::isWithinRel(expected_stats[batch + batches * 5]));
        }
        cpu::math::variance(data.get(), variances, elements, batches);
        cpu::math::stddev(data.get(), stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(variances[batch], test::isWithinRel(expected_stats[batch + batches * 4]));
            REQUIRE_THAT(stddevs[batch], test::isWithinRel(expected_stats[batch + batches * 5]));
        }
    }

    WHEN("minMax, sum, mean, variance, stddev") {
        cpu::math::minMax(data.get(), mins, maxs, elements, batches);
        cpu::math::sum(data.get(), sums, elements, batches);
        cpu::math::mean(data.get(), means, elements, batches);
        cpu::math::varianceStddev(data.get(), means, variances, stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], test::isWithinAbs(expected_stats[batch], 1e-6));
            REQUIRE_THAT(maxs[batch], test::isWithinAbs(expected_stats[batch + batches * 1], 1e-6));
            REQUIRE_THAT(sums[batch], test::isWithinRel(expected_stats[batch + batches * 2]));
            REQUIRE_THAT(means[batch], test::isWithinRel(expected_stats[batch + batches * 3]));
            REQUIRE_THAT(variances[batch], test::isWithinRel(expected_stats[batch + batches * 4]));
            REQUIRE_THAT(stddevs[batch], test::isWithinRel(expected_stats[batch + batches * 5]));
        }
    }

    WHEN("minMaxSumMean, variance, stddev") {
        cpu::math::minMaxSumMean(data.get(), mins, maxs, sums, means, elements, batches);
        cpu::math::varianceStddev(data.get(), means, variances, stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], test::isWithinAbs(expected_stats[batch], 1e-6));
            REQUIRE_THAT(maxs[batch], test::isWithinAbs(expected_stats[batch + batches * 1], 1e-6));
            REQUIRE_THAT(sums[batch], test::isWithinRel(expected_stats[batch + batches * 2]));
            REQUIRE_THAT(means[batch], test::isWithinRel(expected_stats[batch + batches * 3]));
            REQUIRE_THAT(variances[batch], test::isWithinRel(expected_stats[batch + batches * 4]));
            REQUIRE_THAT(stddevs[batch], test::isWithinRel(expected_stats[batch + batches * 5]));
        }
    }

    WHEN("minMaxSumMean, variance") {
        cpu::math::minMax(data.get(), mins, maxs, elements, batches);
        cpu::math::sumMeanVarianceStddev(data.get(), sums, means, variances, stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], test::isWithinAbs(expected_stats[batch], 1e-6));
            REQUIRE_THAT(maxs[batch], test::isWithinAbs(expected_stats[batch + batches * 1], 1e-6));
            REQUIRE_THAT(sums[batch], test::isWithinRel(expected_stats[batch + batches * 2]));
            REQUIRE_THAT(means[batch], test::isWithinRel(expected_stats[batch + batches * 3]));
            REQUIRE_THAT(variances[batch], test::isWithinRel(expected_stats[batch + batches * 4]));
            REQUIRE_THAT(stddevs[batch], test::isWithinRel(expected_stats[batch + batches * 5]));
        }
    }

    WHEN("statistics") {
        cpu::math::statistics(data.get(), mins, maxs, sums, means, variances, stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], test::isWithinAbs(expected_stats[batch], 1e-6));
            REQUIRE_THAT(maxs[batch], test::isWithinAbs(expected_stats[batch + batches * 1], 1e-6));
            REQUIRE_THAT(sums[batch], test::isWithinRel(expected_stats[batch + batches * 2]));
            REQUIRE_THAT(means[batch], test::isWithinRel(expected_stats[batch + batches * 3]));
            REQUIRE_THAT(variances[batch], test::isWithinRel(expected_stats[batch + batches * 4]));
            REQUIRE_THAT(stddevs[batch], test::isWithinRel(expected_stats[batch + batches * 5]));
        }
    }
}

TEST_CASE("cpu::math:: reductions", "[noa][cpu][math]") {
    path_t directory = test::PATH_TEST_DATA / "math";
    path_t path_vectors = directory / "tmp_reduction_random_vectors.mrc";
    path_t path_weights = directory / "tmp_reduction_random_weights.mrc";
    path_t path_reduce_add = directory / "tmp_reduction_reduce_add.mrc";
    path_t path_reduce_mean = directory / "tmp_reduction_reduce_mean.mrc";
    path_t path_reduce_weighted_mean = directory / "tmp_reduction_reduce_weighted_mean.mrc";

    MRCFile mrc_file(path_vectors, io::READ);

    size3_t shape = mrc_file.getShape();
    uint batches = static_cast<uint>(shape.z);
    uint nb_vectors = static_cast<uint>(shape.y);
    size_t elements = shape.x;

    cpu::memory::PtrHost<float> vectors(elements * nb_vectors * batches);
    mrc_file.readAll(vectors.get());

    cpu::memory::PtrHost<float> expected_reduce(elements * batches);
    cpu::memory::PtrHost<float> result_reduce(elements * batches);

    AND_THEN("reduceAdd") {
        mrc_file.open(path_reduce_add, io::READ);
        mrc_file.readAll(expected_reduce.get());
        cpu::math::reduceAdd(vectors.get(), result_reduce.get(), elements, nb_vectors, batches);
        float diff = test::getAverageDifference(expected_reduce.get(), result_reduce.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-5));
    }

    AND_THEN("reduceMean") {
        mrc_file.open(path_reduce_mean, io::READ);
        mrc_file.readAll(expected_reduce.get());
        cpu::math::reduceMean(vectors.get(), result_reduce.get(), elements, nb_vectors, batches);
        float diff = test::getAverageDifference(expected_reduce.get(), result_reduce.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-5));
    }

    AND_THEN("reduceMeanWeighted") {
        mrc_file.open(path_reduce_weighted_mean, io::READ);
        mrc_file.readAll(expected_reduce.get());

        cpu::memory::PtrHost<float> weights(elements * nb_vectors);
        mrc_file.open(path_weights, io::READ);
        mrc_file.readAll(weights.get());

        cpu::math::reduceMeanWeighted(vectors.get(), weights.get(), result_reduce.get(), elements, nb_vectors, batches);
        float diff = test::getAverageDifference(expected_reduce.get(), result_reduce.get(), elements * batches);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-5));
    }
}
