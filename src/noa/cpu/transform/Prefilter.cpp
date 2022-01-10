#include "noa/common/Profiler.h"
#include "noa/cpu/transform/Prefilter.h"

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

    template<typename T>
    // float/double or cfloat_t/cdouble_t
    T initialCausalCoefficient_(const T* c, size_t step_increment, size_t steps) {
        using real_t = noa::traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        const size_t horizon = math::min(size_t{12}, steps);

        // this initialization corresponds to clamping boundaries accelerated loop
        real_t zn = POLE;
        T sum = *c;
        for (size_t n = 0; n < horizon; n++) {
            sum += zn * *c;
            zn *= POLE;
            c += step_increment;
        }
        return sum;
    }

    template<typename T>
    // float/double or cfloat_t/cdouble_t
    inline T initialAntiCausalCoefficient_(const T* c) {
        // this initialization corresponds to clamping boundaries
        using real_t = noa::traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        return ((POLE / (POLE - 1)) * *c);
    }

    template<typename T>
    // float/double or cfloat_t/cdouble_t
    void toCoeffs_(T* output, size_t step_increment, size_t steps) {
        using real_t = noa::traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        constexpr auto LAMBDA = (1 - POLE) * (1 - 1 / POLE); // compute the overall gain

        // causal initialization and recursion
        T* c = output;
        T previous_c;  //cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = LAMBDA * initialCausalCoefficient_(c, step_increment, steps);
        for (size_t n = 1; n < steps; n++) {
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

    template<typename T>
    // float/double or cfloat_t/cdouble_t
    void toCoeffs_(const T* input, T* output, size_t step_increment, size_t steps) {
        using real_t = noa::traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        constexpr auto LAMBDA = (1 - POLE) * (1 - 1 / POLE); // compute the overall gain

        // causal initialization and recursion
        T* c = output;
        T previous_c;  // cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = LAMBDA * initialCausalCoefficient_(input, step_increment, steps);
        for (size_t n = 1; n < steps; n++) {
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

namespace noa::cpu::transform::bspline::details {
    template<typename T>
    void prefilter1D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                     size_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        if (inputs == outputs) {
            for (size_t batch = 0; batch < batches; ++batch)
                toCoeffs_(outputs + output_pitch * batch, 1, shape);
        } else {
            for (size_t batch = 0; batch < batches; ++batch) {
                const T* input = inputs + input_pitch * batch;
                T* output = outputs + output_pitch * batch;
                toCoeffs_(input, output, 1, shape);
            }
        }
    }

    template<typename T>
    void prefilter2D(const T* inputs, size2_t input_pitch, T* outputs, size2_t output_pitch,
                     size2_t shape, size_t batches, size_t threads) {
        NOA_PROFILE_FUNCTION();
        if (inputs == outputs) {
            const size_t offset = elements(output_pitch);
            #pragma omp parallel num_threads(threads) default(none) shared(outputs, output_pitch, shape, batches, offset)
            {
                #pragma omp for collapse(2)
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t y = 0; y < shape.y; ++y) // every row
                        toCoeffs_(outputs + offset * batch + y * output_pitch.x, 1, shape.x);
                #pragma omp for collapse(2)
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t x = 0; x < shape.x; ++x) // every column
                        toCoeffs_(outputs + offset * batch + x, output_pitch.x, shape.y);
            }
        } else {
            const size_t iffset = elements(input_pitch);
            const size_t offset = elements(output_pitch);

            #pragma omp parallel num_threads(threads) default(none) \
            shared(inputs, input_pitch, outputs, output_pitch, shape, batches, iffset, offset)
            {
                #pragma omp for collapse(2)
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t y = 0; y < shape.y; ++y) // every row
                        toCoeffs_(inputs + iffset * batch + y * input_pitch.x,
                                  outputs + offset * batch + y * output_pitch.x,
                                  1, shape.x);
                #pragma omp for collapse(2)
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t x = 0; x < shape.x; ++x) // every column
                        toCoeffs_(outputs + offset * batch + x, output_pitch.x, shape.y);
            }
        }
    }

    template<typename T>
    void prefilter3D(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                     size3_t shape, size_t batches, size_t threads) {
        NOA_PROFILE_FUNCTION();
        if (inputs == outputs) {
            const size_t offset = elements(output_pitch);

            #pragma omp parallel num_threads(threads) default(none) shared(outputs, output_pitch, shape, batches, offset)
            {
                #pragma omp for collapse(3)
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t z = 0; z < shape.z; ++z)
                        for (size_t y = 0; y < shape.y; ++y)
                            toCoeffs_(outputs + offset * batch + index(y, z, output_pitch),
                                      1, shape.x); // every row
                #pragma omp for collapse(3)
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t z = 0; z < shape.z; ++z)
                        for (size_t x = 0; x < shape.x; ++x)
                            toCoeffs_(outputs + offset * batch + z * output_pitch.y * output_pitch.x + x,
                                      output_pitch.x, shape.y); // every column
                #pragma omp for collapse(3)
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t y = 0; y < shape.y; ++y)
                        for (size_t x = 0; x < shape.x; ++x)
                            toCoeffs_(outputs + offset * batch + y * output_pitch.x + x,
                                      output_pitch.y * output_pitch.x, shape.z); // every page
            }
        } else {
            const size_t iffset = elements(input_pitch);
            const size_t offset = elements(output_pitch);

            #pragma omp parallel num_threads(threads) default(none) \
            shared(inputs, input_pitch, outputs, output_pitch, shape, batches, iffset, offset)
            {
                #pragma omp for collapse(3)
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t z = 0; z < shape.z; ++z)
                        for (size_t y = 0; y < shape.y; ++y)
                            toCoeffs_(inputs + iffset * batch + index(y, z, input_pitch),
                                      outputs + offset * batch + index(y, z, output_pitch),
                                      1, shape.x); // every row
                #pragma omp for collapse(3)
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t z = 0; z < shape.z; ++z)
                        for (size_t x = 0; x < shape.x; ++x)
                            toCoeffs_(outputs + offset * batch + z * output_pitch.y * output_pitch.x + x,
                                      output_pitch.x, shape.y); // every column
                #pragma omp for collapse(3)
                for (size_t batch = 0; batch < batches; ++batch)
                    for (size_t y = 0; y < shape.y; ++y)
                        for (size_t x = 0; x < shape.x; ++x)
                            toCoeffs_(outputs + offset * batch + y * output_pitch.x + x,
                                      output_pitch.y * output_pitch.x, shape.z); // every page
            }
        }
    }

    #define NOA_INSTANTIATE_PREFILTER_(T)                                                   \
    template void prefilter1D<T>(const T*, size_t, T*, size_t, size_t, size_t);             \
    template void prefilter2D<T>(const T*, size2_t, T*, size2_t, size2_t, size_t, size_t);  \
    template void prefilter3D<T>(const T*, size3_t, T*, size3_t, size3_t, size_t, size_t)

    NOA_INSTANTIATE_PREFILTER_(float);
    NOA_INSTANTIATE_PREFILTER_(double);
    NOA_INSTANTIATE_PREFILTER_(cfloat_t);
    NOA_INSTANTIATE_PREFILTER_(cdouble_t);
}
