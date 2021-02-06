/**
 * @file noa/Types.h
 * @brief Define some typedefs...
 * @author Thomas - ffyr2w
 * @date 11/01/2021
 */
#pragma once

#include <complex>
#include <cstdint>
#include <filesystem>
#include <ios>

#include "noa/util/IntX.h"
#include "noa/util/FloatX.h"
#include "noa/util/Complex.h"

namespace Noa {
#ifdef NOA_SINGLE_PRECISION
    using real_t = float;
#else
    using real_t = double;
#endif

    // Real2
    using real2_t = Float2<real_t>;
    using real3_t = Float3<real_t>;
    using real4_t = Float4<real_t>;

    using float2_t = Float2<float>;
    using float3_t = Float3<float>;
    using float4_t = Float4<float>;

    using double2_t = Float2<double>;
    using double3_t = Float3<double>;
    using double4_t = Float4<double>;

    // Complex type. OpenCL could use Complex<> as well.
#ifdef NOA_BUILD_CUDA
    using cfloat_t = Complex<float>;
    using cdouble_t = Complex<double>;
    using complex_t = Complex<real_t>;
#else
    using cfloat_t = std::complex<float>;
    using cdouble_t = std::complex<double>;
    using complex_t = std::complex<real_t>;
#endif

    namespace Traits {
        template<> struct proclaim_is_complex<cfloat_t> : std::true_type {};
        template<> struct proclaim_is_complex<cdouble_t> : std::true_type {};
    }

    using size2_t = Int2<size_t>;
    using size3_t = Int3<size_t>;
    using size4_t = Int4<size_t>;

    using int_t = int;
    using int2_t = Int2<int_t>;
    using int3_t = Int3<int_t>;
    using int4_t = Int4<int_t>;

    using uint_t = unsigned int;
    using uint2_t = Int2<uint_t>;
    using uint3_t = Int3<uint_t>;
    using uint4_t = Int4<uint_t>;

    using long_t = long long;
    using long2_t = Int2<long_t>;
    using long3_t = Int3<long_t>;
    using long4_t = Int4<long_t>;

    using ulong_t = unsigned long long;
    using ulong2_t = Int2<ulong_t>;
    using ulong3_t = Int3<ulong_t>;
    using ulong4_t = Int4<ulong_t>;

    namespace fs = std::filesystem;
    using path_t = fs::path;
    using openmode_t = std::ios_base::openmode;

    struct ImageStats {
        float mean, min, max, stddev, var;
    };
}
