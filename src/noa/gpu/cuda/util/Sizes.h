#pragma once

#include "noa/Definitions.h"
#include "noa/util/IntX.h"
#include "noa/util/Sizes.h"

/*
 * This file contains overloads for the NOA::CUDA namespace of some functions from noa/util/Sizes.h.
 * These are here to take into account cuFFT stronger requirements, as opposed to FFTW, about the dimension sizes.
 */

namespace Noa::CUDA::Details {
    /// Even values satisfying (2^a) * (3^b) * (5^c) * (7^d).
    static constexpr uint sizes_even_cufft[315] = {
            2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 42, 48, 50, 54, 56, 60, 64, 70, 72, 80, 84,
            90, 96, 98, 100, 108, 112, 120, 126, 128, 140, 144, 150, 160, 162, 168, 180, 192, 196, 200, 210, 216,
            224, 240, 250, 252, 256, 270, 280, 288, 294, 300, 320, 324, 336, 350, 360, 378, 384, 392, 400, 420, 432,
            448, 450, 480, 486, 490, 500, 504, 512, 540, 560, 576, 588, 600, 630, 640, 648, 672, 686, 700, 720, 750,
            756, 768, 784, 800, 810, 840, 864, 882, 896, 900, 960, 972, 980, 1000, 1008, 1024, 1050, 1080, 1120,
            1134, 1152, 1176, 1200, 1250, 1260, 1280, 1296, 1344, 1350, 1372, 1400, 1440, 1458, 1470, 1500, 1512,
            1536, 1568, 1600, 1620, 1680, 1728, 1750, 1764, 1792, 1800, 1890, 1920, 1944, 1960, 2000, 2016, 2048,
            2058, 2100, 2160, 2240, 2250, 2268, 2304, 2352, 2400, 2430, 2450, 2500, 2520, 2560, 2592, 2646, 2688,
            2700, 2744, 2800, 2880, 2916, 2940, 3000, 3024, 3072, 3136, 3150, 3200, 3240, 3360, 3402, 3430, 3456,
            3500, 3528, 3584, 3600, 3750, 3780, 3840, 3888, 3920, 4000, 4032, 4050, 4096, 4116, 4200, 4320, 4374,
            4410, 4480, 4500, 4536, 4608, 4704, 4800, 4860, 4900, 5000, 5040, 5120, 5184, 5250, 5292, 5376, 5400,
            5488, 5600, 5670, 5760, 5832, 5880, 6000, 6048, 6144, 6174, 6250, 6272, 6300, 6400, 6480, 6720, 6750,
            6804, 6860, 6912, 7000, 7056, 7168, 7200, 7290, 7350, 7500, 7560, 7680, 7776, 7840, 7938, 8000, 8064,
            8100, 8192, 8232, 8400, 8640, 8748, 8750, 8820, 8960, 9000, 9072, 9216, 9408, 9450, 9600, 9720, 9800,
            10000, 10080, 10206, 10240, 10290, 10368, 10500, 10584, 10752, 10800, 10976, 11200, 11250, 11340, 11520,
            11664, 11760, 12000, 12096, 12150, 12250, 12288, 12348, 12500, 12544, 12600, 12800, 12960, 13230, 13440,
            13500, 13608, 13720, 13824, 14000, 14112, 14336, 14400, 14580, 14700, 15000, 15120, 15360, 15552, 15680,
            15750, 15876, 16000, 16128, 16200, 16384, 16464, 16800
    };
}

namespace Noa::CUDA {
    /**
     * Returns a "nice" even size, greater or equal than @a size.
     * @note A "nice" size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d).
     * @warning If @a size is >16800, this function will simply return the next even number and will not necessarily
     *          satisfy the aforementioned requirements.
     */
    NOA_IH size_t getNiceSize(size_t size) {
        auto tmp = static_cast<uint>(size);
        for (uint nice_size : Details::sizes_even_cufft)
            if (tmp < nice_size)
                return static_cast<size_t>(nice_size);
        return (size % 2 == 0) ? size : (size + 1); // fall back to next even number
    }

    /// Returns a "nice" shape. @note Dimensions of size 0 or 1 are ignored, e.g. {51,51,1} is rounded up to {52,52,1}.
    NOA_IH size3_t getNiceShape(size3_t shape) {
        return size3_t(shape.x > 1 ? getNiceSize(shape.x) : shape.x,
                       shape.y > 1 ? getNiceSize(shape.y) : shape.y,
                       shape.z > 1 ? getNiceSize(shape.z) : shape.z);
    }
}
