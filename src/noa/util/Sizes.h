/// \file noa/util/IntX.h
/// \brief "Size" related utilities.
/// \author Thomas - ffyr2w
/// \date 10/12/2020

#pragma once

#include "noa/Definitions.h"
#include "noa/util/IntX.h"

// Size of dimensions
// ==================
//
// The code base often refers to "shapes" when dealing with sizes, especially with sizes of multidimensional arrays.
//  -   Shapes are always organized as such (unless specified otherwise): {x=fast, y=medium, z=slow}.
//      This directly refers to the memory layout of the data and is therefore less ambiguous than
//      {row|width, column|height, page|depth}.
//
//  -   Shapes should not have any zeros. An "empty" dimension is specified with 1.
//      The API follows this convention (unless specified otherwise), where x, y and z are > 1:
//          A 1D array is specified as {x}, {x, 1} or {x, 1, 1}.
//          A 2D array is specified as {x, y} or {x, y, 1}.
//          A 3D array is specified as {x, y, z}.

namespace noa::details {
    /// Even values satisfying (2^a) * (3^b) * (5^c) * (7^d) * (11^e) * (13^f), with e + f = 0 or 1.
    static constexpr uint sizes_even_fftw[] = {
            2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 42, 44, 48, 50, 52, 54, 56, 60, 64,
            66, 70, 72, 78, 80, 84, 88, 90, 96, 98, 100, 104, 108, 110, 112, 120, 126, 128, 130, 132, 140, 144, 150,
            154, 156, 160, 162, 168, 176, 180, 182, 192, 196, 198, 200, 208, 210, 216, 220, 224, 234, 240, 250, 252,
            256, 260, 264, 270, 280, 288, 294, 300, 308, 312, 320, 324, 330, 336, 350, 352, 360, 364, 378, 384, 390,
            392, 396, 400, 416, 420, 432, 440, 448, 450, 462, 468, 480, 486, 490, 500, 504, 512, 520, 528, 540, 546,
            550, 560, 576, 588, 594, 600, 616, 624, 630, 640, 648, 650, 660, 672, 686, 700, 702, 704, 720, 728, 750,
            756, 768, 770, 780, 784, 792, 800, 810, 832, 840, 864, 880, 882, 896, 900, 910, 924, 936, 960, 972, 980,
            990, 1000, 1008, 1024, 1040, 1050, 1056, 1078, 1080, 1092, 1100, 1120, 1134, 1152, 1170, 1176, 1188,
            1200, 1232, 1248, 1250, 1260, 1274, 1280, 1296, 1300, 1320, 1344, 1350, 1372, 1386, 1400, 1404, 1408,
            1440, 1456, 1458, 1470, 1500, 1512, 1536, 1540, 1560, 1568, 1584, 1600, 1620, 1638, 1650, 1664, 1680,
            1728, 1750, 1760, 1764, 1782, 1792, 1800, 1820, 1848, 1872, 1890, 1920, 1944, 1950, 1960, 1980, 2000,
            2016, 2048, 2058, 2080, 2100, 2106, 2112, 2156, 2160, 2184, 2200, 2240, 2250, 2268, 2304, 2310, 2340,
            2352, 2376, 2400, 2430, 2450, 2464, 2496, 2500, 2520, 2548, 2560, 2592, 2600, 2640, 2646, 2688, 2700,
            2730, 2744, 2750, 2772, 2800, 2808, 2816, 2880, 2912, 2916, 2940, 2970, 3000, 3024, 3072, 3080, 3120,
            3136, 3150, 3168, 3200, 3234, 3240, 3250, 3276, 3300, 3328, 3360, 3402, 3430, 3456, 3500, 3510, 3520,
            3528, 3564, 3584, 3600, 3640, 3696, 3744, 3750, 3780, 3822, 3840, 3850, 3888, 3900, 3920, 3960, 4000,
            4032, 4050, 4096, 4116, 4158, 4160, 4200, 4212, 4224, 4312, 4320, 4368, 4374, 4400, 4410, 4480, 4500,
            4536, 4550, 4608, 4620, 4680, 4704, 4752, 4800, 4860, 4900, 4914, 4928, 4950, 4992, 5000, 5040, 5096,
            5120, 5184, 5200, 5250, 5280, 5292, 5346, 5376, 5390, 5400, 5460, 5488, 5500, 5544, 5600, 5616, 5632,
            5670, 5760, 5824, 5832, 5850, 5880, 5940, 6000, 6048, 6144, 6160, 6174, 6240, 6250, 6272, 6300, 6318,
            6336, 6370, 6400, 6468, 6480, 6500, 6552, 6600, 6656, 6720, 6750, 6804, 6860, 6912, 6930, 7000, 7020,
            7040, 7056, 7128, 7168, 7200, 7280, 7290, 7350, 7392, 7488, 7500, 7546, 7560, 7644, 7680, 7700, 7776,
            7800, 7840, 7920, 7938, 8000, 8064, 8100, 8190, 8192, 8232, 8250, 8316, 8320, 8400, 8424, 8448, 8624,
            8640, 8736, 8748, 8750, 8800, 8820, 8910, 8918, 8960, 9000, 9072, 9100, 9216, 9240, 9360, 9408, 9450,
            9504, 9600, 9702, 9720, 9750, 9800, 9828, 9856, 9900, 9984, 10000, 10080, 10192, 10206, 10240, 10290,
            10368, 10400, 10500, 10530, 10560, 10584, 10692, 10752, 10780, 10800, 10920, 10976, 11000, 11088, 11200,
            11232, 11250, 11264, 11340, 11466, 11520, 11550, 11648, 11664, 11700, 11760, 11880, 12000, 12096, 12150,
            12250, 12288, 12320, 12348, 12474, 12480, 12500, 12544, 12600, 12636, 12672, 12740, 12800, 12936, 12960,
            13000, 13104, 13200, 13230, 13312, 13440, 13500, 13608, 13650, 13720, 13750, 13824, 13860, 14000, 14040,
            14080, 14112, 14256, 14336, 14400, 14560, 14580, 14700, 14742, 14784, 14850, 14976, 15000, 15092, 15120,
            15288, 15360, 15400, 15552, 15600, 15680, 15750, 15840, 15876, 16000, 16038, 16128, 16170, 16200, 16250,
            16380, 16384, 16464, 16500, 16632, 16640, 16800, 16848, 16896
    };
}

namespace noa {
    using size2_t = Int2<size_t>;
    using size3_t = Int3<size_t>;
    using size4_t = Int4<size_t>;

    /// Returns a "nice" even size, greater or equal than \a size.
    /// \note A "nice" size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d)*(11^e)*(13^f), with e + f = 0 or 1.
    /// \note If \a size is >16896, this function will simply return the next even number and will not necessarily
    ///       satisfy the aforementioned requirements.
    NOA_IH size_t getNiceSize(size_t size) {
        auto tmp = static_cast<uint>(size);
        for (uint nice_size : details::sizes_even_fftw)
            if (tmp < nice_size)
                return static_cast<size_t>(nice_size);
        return (size % 2 == 0) ? size : (size + 1); // fall back to next even number
    }

    /// Returns a "nice" shape. \note Dimensions of size 0 or 1 are ignored, e.g. {51,51,1} is rounded up to {52,52,1}.
    NOA_IH size3_t getNiceShape(size3_t shape) {
        return size3_t(shape.x > 1 ? getNiceSize(shape.x) : shape.x,
                       shape.y > 1 ? getNiceSize(shape.y) : shape.y,
                       shape.z > 1 ? getNiceSize(shape.z) : shape.z);
    }

    /// Returns the number of elements within an array with a given \a shape.
    NOA_FHD size_t getElements(size3_t shape) { return shape.x * shape.y * shape.z; }

    /// Returns the number of elements in one slice within an array with a given \a shape.
    NOA_FHD size_t getElementsSlice(size3_t shape) { return shape.x * shape.y; }

    /// Returns the number of complex elements in the non-redundant Fourier transform of an array with a given \a shape.
    NOA_FHD size_t getElementsFFT(size3_t shape) { return (shape.x / 2 + 1) * shape.y * shape.z; }

    /// Returns the shape of the slice of an array with a given \a shape.
    NOA_FHD size3_t getShapeSlice(size3_t shape) { return size3_t{shape.x, shape.y, 1}; }

    /// Returns the physical shape (i.e. non-redundant) given the logical \a shape.
    NOA_FHD size3_t getShapeFFT(size3_t shape) { return size3_t{shape.x / 2 + 1, shape.y, shape.z}; }

    /// Returns the number of rows in a array with a given \a shape.
    NOA_FHD size_t getRows(size3_t shape) { return shape.y * shape.z; }
    NOA_FHD uint getRows(Int3<uint> shape) { return shape.y * shape.z; }

    /// Returns the number of dimensions of an array with a given \a shape. Can be either 1, 2 or 3.
    NOA_FHD uint getNDim(size3_t shape) { return shape.z > 1 ? 3 : shape.y > 1 ? 2 : 1; }
    NOA_FHD uint getRank(size3_t shape) { return getNDim(shape); }

    /// Returns the {x, y} coordinates corresponding to \a idx.
    NOA_IHD size2_t getCoords(size_t idx, size_t shape_x) {
        size_t coord_y = idx / shape_x;
        size_t coord_x = idx - coord_y * shape_x;
        return {coord_x, coord_y};
    }

    /// Returns the {x, y, z} coordinates corresponding to \a idx.
    NOA_IHD size3_t getCoords(size_t idx, size_t shape_y, size_t shape_x) {
        size_t coord_z = idx / (shape_y * shape_x);
        size_t tmp = idx - coord_z * shape_y * shape_x;
        size_t coord_y = tmp / shape_x;
        size_t coord_x = tmp - coord_y * shape_x;
        return {coord_x, coord_y, coord_z};
    }

    /// Returns the index corresponding to the {x, y} coordinates \a coord_x, \a coord_y and \a coord_z.
    NOA_FHD size_t getIdx(size_t coord_x, size_t coord_y, size_t coord_z, size_t shape_y, size_t shape_x) {
        return (coord_z * shape_y + coord_y) * shape_x + coord_x;
    }

    /// Returns the index corresponding to the {x, y} coordinates \a coord_x, \a coord_y and \a coord_z.
    NOA_FHD size_t getIdx(size_t coord_x, size_t coord_y, size_t shape_x) {
        return coord_y * shape_x + coord_x;
    }
}
