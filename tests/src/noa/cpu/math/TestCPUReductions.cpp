#include <noa/cpu/math/Reductions.h>

#include <noa/cpu/PtrHost.h>
#include "noa/io/files/MRCFile.h"
#include "noa/io/files/TextFile.h"
#include "noa/util/string/Convert.h"

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEST_CASE("CPU::Math: Reductions: Stats", "[noa][cpu][math]") {
    path_t directory = path_t(NOA_TESTS_DATA) / "src" / "math";
    path_t path_data = directory / "stats_random_array.mrc";
    path_t path_stats = directory / "stats_random_array.txt";
    MRCFile file_data(path_data, IO::READ);

    size3_t shape = file_data.getShape();
    size_t elements = getElementsSlice(shape);
    uint batches = static_cast<uint>(shape.z);

    PtrHost<float> data(elements * batches);
    file_data.readAll(data.get());

    TextFile<std::ifstream> file_stats(path_stats, IO::READ);
    std::string line;
    PtrHost<float> expected_stats(batches * 6);
    for (uint idx = 0; idx < batches * 6; ++idx) {
        file_stats.getLine(line);
        expected_stats[idx] = String::toFloat(line);
    }

    PtrHost<float> results(batches * 6);
    float* mins = results.get();
    float* maxs = results.get() + batches * 1;
    float* sums = results.get() + batches * 2;
    float* means = results.get() + batches * 3;
    float* variances = results.get() + batches * 4;
    float* stddevs = results.get() + batches * 5;

    WHEN("min, max, sum, mean, variance, stddev") {
        Math::min(data.get(), mins, elements, batches);
        Math::max(data.get(), maxs, elements, batches);
        Math::sum(data.get(), sums, elements, batches);
        Math::mean(data.get(), means, elements, batches);
        Math::variance(data.get(), means, variances, elements, batches);
        Math::stddev(data.get(), means, stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], Test::isWithinAbs(expected_stats[batch], 1e-6));
            REQUIRE_THAT(maxs[batch], Test::isWithinAbs(expected_stats[batch + batches * 1], 1e-6));
            REQUIRE_THAT(sums[batch], Test::isWithinRel(expected_stats[batch + batches * 2]));
            REQUIRE_THAT(means[batch], Test::isWithinRel(expected_stats[batch + batches * 3]));
            REQUIRE_THAT(variances[batch], Test::isWithinRel(expected_stats[batch + batches * 4]));
            REQUIRE_THAT(stddevs[batch], Test::isWithinRel(expected_stats[batch + batches * 5]));
        }
        Math::variance(data.get(), variances, elements, batches);
        Math::stddev(data.get(), stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(variances[batch], Test::isWithinRel(expected_stats[batch + batches * 4]));
            REQUIRE_THAT(stddevs[batch], Test::isWithinRel(expected_stats[batch + batches * 5]));
        }
    }

    WHEN("minMax, sum, mean, variance, stddev") {
        Math::minMax(data.get(), mins, maxs, elements, batches);
        Math::sum(data.get(), sums, elements, batches);
        Math::mean(data.get(), means, elements, batches);
        Math::varianceStddev(data.get(), means, variances, stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], Test::isWithinAbs(expected_stats[batch], 1e-6));
            REQUIRE_THAT(maxs[batch], Test::isWithinAbs(expected_stats[batch + batches * 1], 1e-6));
            REQUIRE_THAT(sums[batch], Test::isWithinRel(expected_stats[batch + batches * 2]));
            REQUIRE_THAT(means[batch], Test::isWithinRel(expected_stats[batch + batches * 3]));
            REQUIRE_THAT(variances[batch], Test::isWithinRel(expected_stats[batch + batches * 4]));
            REQUIRE_THAT(stddevs[batch], Test::isWithinRel(expected_stats[batch + batches * 5]));
        }
    }

    WHEN("minMaxSumMean, variance, stddev") {
        Math::minMaxSumMean(data.get(), mins, maxs, sums, means, elements, batches);
        Math::varianceStddev(data.get(), means, variances, stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], Test::isWithinAbs(expected_stats[batch], 1e-6));
            REQUIRE_THAT(maxs[batch], Test::isWithinAbs(expected_stats[batch + batches * 1], 1e-6));
            REQUIRE_THAT(sums[batch], Test::isWithinRel(expected_stats[batch + batches * 2]));
            REQUIRE_THAT(means[batch], Test::isWithinRel(expected_stats[batch + batches * 3]));
            REQUIRE_THAT(variances[batch], Test::isWithinRel(expected_stats[batch + batches * 4]));
            REQUIRE_THAT(stddevs[batch], Test::isWithinRel(expected_stats[batch + batches * 5]));
        }
    }

    WHEN("minMaxSumMean, variance") {
        Math::minMax(data.get(), mins, maxs, elements, batches);
        Math::sumMeanVarianceStddev(data.get(), sums, means, variances, stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], Test::isWithinAbs(expected_stats[batch], 1e-6));
            REQUIRE_THAT(maxs[batch], Test::isWithinAbs(expected_stats[batch + batches * 1], 1e-6));
            REQUIRE_THAT(sums[batch], Test::isWithinRel(expected_stats[batch + batches * 2]));
            REQUIRE_THAT(means[batch], Test::isWithinRel(expected_stats[batch + batches * 3]));
            REQUIRE_THAT(variances[batch], Test::isWithinRel(expected_stats[batch + batches * 4]));
            REQUIRE_THAT(stddevs[batch], Test::isWithinRel(expected_stats[batch + batches * 5]));
        }
    }

    WHEN("statistics") {
        Math::statistics(data.get(), mins, maxs, sums, means, variances, stddevs, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], Test::isWithinAbs(expected_stats[batch], 1e-6));
            REQUIRE_THAT(maxs[batch], Test::isWithinAbs(expected_stats[batch + batches * 1], 1e-6));
            REQUIRE_THAT(sums[batch], Test::isWithinRel(expected_stats[batch + batches * 2]));
            REQUIRE_THAT(means[batch], Test::isWithinRel(expected_stats[batch + batches * 3]));
            REQUIRE_THAT(variances[batch], Test::isWithinRel(expected_stats[batch + batches * 4]));
            REQUIRE_THAT(stddevs[batch], Test::isWithinRel(expected_stats[batch + batches * 5]));
        }
    }
}

TEST_CASE("CPU::Math: Reductions: reduce", "[noa][cpu][math]") {
    // See data layout in data/src/math/DataReductions.py

    path_t directory = path_t(NOA_TESTS_DATA) / "src" / "math";
    path_t path_vectors = directory / "reduction_random_vectors.mrc";
    path_t path_weights = directory / "reduction_random_weights.mrc";
    path_t path_reduce_add = directory / "reduction_reduce_add.mrc";
    path_t path_reduce_mean = directory / "reduction_reduce_mean.mrc";
    path_t path_reduce_weighted_mean = directory / "reduction_reduce_weighted_mean.mrc";

    MRCFile mrc_file(path_vectors, IO::READ);

    size3_t shape = mrc_file.getShape();
    uint batches = static_cast<uint>(shape.z);
    uint nb_vectors = static_cast<uint>(shape.y);
    size_t elements = shape.x;

    PtrHost<float> vectors(elements * nb_vectors * batches);
    mrc_file.readAll(vectors.get());

    PtrHost<float> expected_reduce(elements * batches);
    PtrHost<float> result_reduce(elements * batches);

    AND_THEN("reduceAdd") {
        mrc_file.open(path_reduce_add, IO::READ);
        mrc_file.readAll(expected_reduce.get());
        Math::reduceAdd(vectors.get(), result_reduce.get(), elements, nb_vectors, batches);
        float diff = Test::getAverageDifference(expected_reduce.get(), result_reduce.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(0.f, 1e-5));
    }

    AND_THEN("reduceMean") {
        mrc_file.open(path_reduce_mean, IO::READ);
        mrc_file.readAll(expected_reduce.get());
        Math::reduceMean(vectors.get(), result_reduce.get(), elements, nb_vectors, batches);
        float diff = Test::getAverageDifference(expected_reduce.get(), result_reduce.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(0.f, 1e-5));
    }

    AND_THEN("reduceMeanWeighted") {
        mrc_file.open(path_reduce_weighted_mean, IO::READ);
        mrc_file.readAll(expected_reduce.get());

        PtrHost<float> weights(elements * nb_vectors);
        mrc_file.open(path_weights, IO::READ);
        mrc_file.readAll(weights.get());

        Math::reduceMeanWeighted(vectors.get(), weights.get(), result_reduce.get(), elements, nb_vectors, batches);
        float diff = Test::getAverageDifference(expected_reduce.get(), result_reduce.get(), elements * batches);
        REQUIRE_THAT(diff, Test::isWithinAbs(0.f, 1e-5));
    }
}
