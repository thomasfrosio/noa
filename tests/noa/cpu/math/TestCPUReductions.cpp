#include <noa/common/io/ImageFile.h>
#include <noa/common/io/TextFile.h>
#include <noa/common/string/Convert.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Reductions.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cpu::math::statistics()", "[assets][noa][cpu][math]") {
    path_t path = test::PATH_NOA_DATA / "math";
    YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["stats"];

    const YAML::Node& input = tests["input"];
    const auto batches = input["batches"].as<uint>();
    const auto shape_2d = input["shape"].as<size2_t>();
    const auto input_filename = path / input["path"].as<path_t>();
    const auto output_filename = path / tests["output"].as<path_t>();
    const size3_t shape{shape_2d.x, shape_2d.y, 1};
    const size_t elements = noa::elements(shape);

    io::ImageFile file(input_filename, io::READ);
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

    cpu::Stream stream;
    cpu::memory::PtrHost<float> results(batches * 6);
    float* mins = results.get();
    float* maxs = results.get() + batches * 1;
    float* sums = results.get() + batches * 2;
    float* means = results.get() + batches * 3;
    float* vars = results.get() + batches * 4;
    float* stds = results.get() + batches * 5;

    WHEN("min, max, sum, mean, var, std") {
        cpu::math::min(data.get(), shape, shape, mins, batches, stream);
        cpu::math::max(data.get(), shape, shape, maxs, batches, stream);
        cpu::math::sum(data.get(), shape, shape, sums, batches, stream);
        cpu::math::mean(data.get(), shape, shape, means, batches, stream);
        cpu::math::var(data.get(), shape, shape, vars, batches, stream);
        cpu::math::std(data.get(), shape, shape, stds, batches, stream);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], Catch::WithinAbs(static_cast<double>(expected_min[batch]), 1e-6));
            REQUIRE_THAT(maxs[batch], Catch::WithinAbs(static_cast<double>(expected_max[batch]), 1e-6));
            REQUIRE_THAT(sums[batch], Catch::WithinRel(expected_sum[batch]));
            REQUIRE_THAT(means[batch], Catch::WithinRel(expected_mean[batch]));
            REQUIRE_THAT(vars[batch], Catch::WithinRel(expected_var[batch]));
            REQUIRE_THAT(stds[batch], Catch::WithinRel(expected_std[batch]));
        }
    }

    WHEN("statistics") {
        cpu::math::statistics(data.get(), shape, shape, mins, maxs, sums, means, vars, stds, batches, stream);
        for (uint batch = 0; batch < batches; ++batch) {
            REQUIRE_THAT(mins[batch], Catch::WithinAbs(static_cast<double>(expected_min[batch]), 1e-6));
            REQUIRE_THAT(maxs[batch], Catch::WithinAbs(static_cast<double>(expected_max[batch]), 1e-6));
            REQUIRE_THAT(sums[batch], Catch::WithinRel(expected_sum[batch]));
            REQUIRE_THAT(means[batch], Catch::WithinRel(expected_mean[batch]));
            REQUIRE_THAT(vars[batch], Catch::WithinRel(expected_var[batch]));
            REQUIRE_THAT(stds[batch], Catch::WithinRel(expected_std[batch]));
        }
    }
}

TEST_CASE("cpu::math:: reductions", "[assets][noa][cpu][math]") {
    const path_t path = test::PATH_NOA_DATA / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reductions"];

    const YAML::Node& input = tests["input"];
    const auto batches = input["batches"].as<uint>();
    const auto nb_vectors = input["vectors"].as<uint>();
    const auto elements = input["elements"].as<size_t>();
    const path_t path_vectors = path / input["path"].as<path_t>();
    const path_t path_weights = path / input["path_weights"].as<path_t>();
    const size3_t shape{elements, 1, 1};

    const path_t path_reduce_add = path / tests["sum"].as<path_t>();
    const path_t path_reduce_mean = path / tests["mean"].as<path_t>();
    const path_t path_reduce_weighted_mean = path / tests["weighted_mean"].as<path_t>();

    io::ImageFile file(path_vectors, io::READ);
    cpu::memory::PtrHost<float> vectors(elements * nb_vectors * batches);
    file.readAll(vectors.get());

    cpu::Stream stream;
    cpu::memory::PtrHost<float> expected_reduce(elements * batches);
    cpu::memory::PtrHost<float> result_reduce(elements * batches);

    AND_THEN("reduceAdd") {
        file.open(path_reduce_add, io::READ);
        file.readAll(expected_reduce.get());
        cpu::math::reduceAdd(vectors.get(), shape, result_reduce.get(), shape, shape, nb_vectors, batches, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, expected_reduce.get(), result_reduce.get(), elements * batches, 1e-5));
    }

    AND_THEN("reduceMean") {
        file.open(path_reduce_mean, io::READ);
        file.readAll(expected_reduce.get());
        cpu::math::reduceMean(vectors.get(), shape, result_reduce.get(), shape, shape, nb_vectors, batches, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, expected_reduce.get(), result_reduce.get(), elements * batches, 1e-5));
    }

    AND_THEN("reduceMeanWeighted") {
        file.open(path_reduce_weighted_mean, io::READ);
        file.readAll(expected_reduce.get());

        cpu::memory::PtrHost<float> weights(elements * nb_vectors);
        file.open(path_weights, io::READ);
        file.readAll(weights.get());

        cpu::math::reduceMeanWeighted(vectors.get(), shape, weights.get(), shape, result_reduce.get(), shape,
                                      shape, nb_vectors, batches, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, expected_reduce.get(), result_reduce.get(), elements * batches, 1e-5));
    }
}
