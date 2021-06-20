/// \file noa/util/Stats.h
/// \brief The Stats type, which gather the basic statistics of an array.
/// \author Thomas - ffyr2w
/// \date 10/12/2020

#pragma once

#include "noa/Definitions.h"

namespace noa {
    /// Statistics of a vector. Careful as some fields will not be computed depending on the function.
    template<typename T>
    struct Stats {
        T min{}, max{}, sum{}, mean{}, variance{}, stddev{};

        /// Joins a "struct of arrays" to an "array of structs".
        /// \param[in] input_stats      A struct of arrays containing the stats.
        ///                             Should contain the arrays for min, max, sum, mean, variance and then stdev.
        ///                             Each array should have one value per batch.
        /// \param[out] output_stats    Output stats. One per batch.
        /// \param batches              Number of batches.
        NOA_HOST static void join(T* input_stats, Stats<T>* output_stats, uint batches) {
            for (uint batch = 0; batch < batches; ++batch) {
                output_stats[batch].min = input_stats[batch];
                output_stats[batch].max = input_stats[batch + batches * 1];
                output_stats[batch].mean = input_stats[batch + batches * 2];
                output_stats[batch].variance = input_stats[batch + batches * 3];
                output_stats[batch].stddev = input_stats[batch + batches * 4];
            }
        }
    };
}
