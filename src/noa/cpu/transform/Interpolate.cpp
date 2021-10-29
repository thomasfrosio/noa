#include "noa/common/Profiler.h"
#include "noa/cpu/transform/Interpolate.h"

// This is adapted from https://github.com/DannyRuijters/CubicInterpolationCUDA
// See licences/CubicInterpolationCUDA.txt

// Compared to original implementation:
//  - steps were switched to number of elements.
//  - const was added when necessary.
//  - Out-of-place filtering was added.
//  - Support for double precision and complex types.
namespace {
    using namespace ::noa;

    // math::sqrt(3.0f)-2.0f; pole for cubic b-spline
    #define POLE_ (-0.2679491924311228)

    template<typename T> // float/double or cfloat_t/cdouble_t
    T initialCausalCoefficient_(const T* c, uint step_increment, uint steps) {
        using real_t = noa::traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        const uint horizon = math::min(12U, steps);

        // this initialization corresponds to clamping boundaries accelerated loop
        real_t zn = POLE;
        T sum = *c;
        for (uint n = 0; n < horizon; n++) {
            sum += zn * *c;
            zn *= POLE;
            c += step_increment;
        }
        return sum;
    }

    template<typename T> // float/double or cfloat_t/cdouble_t
    inline T initialAntiCausalCoefficient_(const T* c) {
        // this initialization corresponds to clamping boundaries
        using real_t = noa::traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        return ((POLE / (POLE - 1)) * *c);
    }

    template<typename T> // float/double or cfloat_t/cdouble_t
    void toCoeffs_(T* output, uint step_increment, uint steps) {
        using real_t = noa::traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        constexpr auto LAMBDA = (1 - POLE) * (1 - 1 / POLE); // compute the overall gain

        // causal initialization and recursion
        T* c = output;
        T previous_c;  //cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = LAMBDA * initialCausalCoefficient_(c, step_increment, steps);
        for (uint n = 1; n < steps; n++) {
            c += step_increment;
            *c = previous_c = LAMBDA * *c + POLE * previous_c;
        }

        // anticausal initialization and recursion
        *c = previous_c = initialAntiCausalCoefficient_(c);
        for (int n = static_cast<int>(steps) - 2; 0 <= n; n--) {
            c -= step_increment;
            *c = previous_c = POLE * (previous_c - *c);
        }
    }

    template<typename T> // float/double or cfloat_t/cdouble_t
    void toCoeffs_(const T* input, T* output, uint step_increment, uint steps) {
        using real_t = noa::traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        constexpr auto LAMBDA = (1 - POLE) * (1 - 1 / POLE); // compute the overall gain

        // causal initialization and recursion
        T* c = output;
        T previous_c;  // cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = LAMBDA * initialCausalCoefficient_(input, step_increment, steps);
        for (uint n = 1; n < steps; n++) {
            input += step_increment;
            c += step_increment;
            *c = previous_c = LAMBDA * *input + POLE * previous_c;
        }

        // anticausal initialization and recursion
        *c = previous_c = initialAntiCausalCoefficient_(c);
        for (int n = static_cast<int>(steps) - 2; 0 <= n; n--) {
            c -= step_increment;
            *c = previous_c = POLE * (previous_c - *c);
        }
    }
}

namespace noa::cpu::transform::bspline {
    template<typename T>
    void prefilter1D(const T* inputs, T* outputs, size_t size, uint batches) {
        NOA_PROFILE_FUNCTION();
        auto tmp = static_cast<uint>(size);

        if (inputs == outputs) {
            for (uint batch = 0; batch < batches; ++batch)
                toCoeffs_(outputs + tmp * batch, 1, tmp);
        } else {
            for (uint batch = 0; batch < batches; ++batch) {
                const T* input = inputs + tmp * batch;
                T* output = outputs + tmp * batch;
                toCoeffs_(input, output, 1, tmp);
            }
        }
    }

    template<typename T>
    void prefilter2D(const T* inputs, T* outputs, size2_t shape, uint batches) {
        NOA_PROFILE_FUNCTION();
        uint2_t dim(shape);
        size_t elements = noa::elements(shape);

        if (inputs == outputs) {
            for (uint batch = 0; batch < batches; ++batch) {
                T* output = outputs + elements * batch;
                for (uint y = 0; y < dim.y; ++y) // process every row
                    toCoeffs_(output + y * dim.x, 1, dim.x);
                for (uint x = 0; x < dim.x; ++x) // process every column
                    toCoeffs_(output + x, dim.x, dim.y);
            }
        } else {
            for (uint batch = 0; batch < batches; ++batch) {
                const T* input = inputs + elements * batch;
                T* output = outputs + elements * batch;
                for (uint y = 0; y < dim.y; ++y) // process every row
                    toCoeffs_(input + y * dim.x, output + y * dim.x, 1, dim.x);
                for (uint x = 0; x < dim.x; ++x) // process every column
                    toCoeffs_(output + x, dim.x, dim.y);
            }
        }
    }

    template<typename T>
    void prefilter3D(const T* inputs, T* outputs, size3_t shape, uint batches) {
        NOA_PROFILE_FUNCTION();
        uint3_t dim(shape);
        size_t elements = noa::elements(shape);

        if (inputs == outputs) {
            for (uint batch = 0; batch < batches; ++batch) {
                T* output = outputs + elements * batch;
                for (uint z = 0; z < dim.z; ++z)
                    for (uint y = 0; y < dim.y; ++y)
                        toCoeffs_(output + (z * dim.y + y) * dim.x, 1, dim.x); // process every row
                for (uint z = 0; z < dim.z; ++z)
                    for (uint x = 0; x < dim.x; ++x)
                        toCoeffs_(output + z * dim.y * dim.x + x, dim.x, dim.y); // process every column
                for (uint y = 0; y < dim.y; ++y)
                    for (uint x = 0; x < dim.x; ++x)
                        toCoeffs_(output + y * dim.x + x, dim.y * dim.x, dim.z); // process every page
            }
        } else {
            for (uint batch = 0; batch < batches; ++batch) {
                const T* input = inputs + elements * batch;
                T* output = outputs + elements * batch;
                for (uint z = 0; z < dim.z; ++z) {
                    uint tmp = z * dim.y;
                    for (uint y = 0; y < dim.y; ++y) {
                        uint offset = (tmp + y) * dim.x;
                        toCoeffs_(input + offset, output + offset, 1, dim.x); // process every row
                    }
                }
                for (uint z = 0; z < dim.z; ++z)
                    for (uint x = 0; x < dim.x; ++x)
                        toCoeffs_(output + z * dim.y * dim.x + x, dim.x, dim.y); // process every column
                for (uint y = 0; y < dim.y; ++y)
                    for (uint x = 0; x < dim.x; ++x)
                        toCoeffs_(output + y * dim.x + x, dim.y * dim.x, dim.z); // process every page
            }
        }
    }

    #define NOA_INSTANTIATE_PREFILTER_(T)                       \
    template void prefilter1D<T>(const T*, T*, size_t, uint);   \
    template void prefilter2D<T>(const T*, T*, size2_t, uint);  \
    template void prefilter3D<T>(const T*, T*, size3_t, uint)

    NOA_INSTANTIATE_PREFILTER_(float);
    NOA_INSTANTIATE_PREFILTER_(double);
    NOA_INSTANTIATE_PREFILTER_(cfloat_t);
    NOA_INSTANTIATE_PREFILTER_(cdouble_t);
}
