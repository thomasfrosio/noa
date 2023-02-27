#include "noa/cpu/geometry/Prefilter.hpp"

// This is adapted from https://github.com/DannyRuijters/CubicInterpolationCUDA
// Copyright (c) 2008-2013, Danny Ruijters. All rights reserved.
// http://www.dannyruijters.nl/cubicinterpolation/

// Compared to Danny's implementation:
//  - steps were switched to number of elements and strides were added.
//  - const was added when appropriate.
//  - In-place filtering was added.
//  - Support for double precision and complex types.
namespace {
    using namespace ::noa;

    // math::sqrt(3.0f)-2.0f; pole for cubic b-spline
    #define POLE_ (-0.2679491924311228)

    template<typename T>
    T initial_causal_coefficient_(const T* c, i64 strides, i64 shape) {
        using real_t = noa::traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        const i64 horizon = noa::math::min(i64{12}, shape);

        // this initialization corresponds to clamping boundaries accelerated loop
        real_t zn = POLE;
        T sum = *c;
        for (i64 n = 0; n < horizon; n++) {
            sum += zn * *c;
            zn *= POLE;
            c += strides;
        }
        return sum;
    }

    template<typename T>
    T initial_anticausal_coefficient_(const T* c) {
        // this initialization corresponds to clamping boundaries
        using real_t = noa::traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        return ((POLE / (POLE - 1)) * *c);
    }

    // float/double or cfloat_t/cdouble_t
    template<typename T>
    void to_coeffs_(T* output, i64 strides, i64 shape) {
        using real_t = noa::traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        constexpr auto LAMBDA = (1 - POLE) * (1 - 1 / POLE); // compute the overall gain

        // causal initialization and recursion
        T* c = output;
        T previous_c;  //cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = LAMBDA * initial_causal_coefficient_(c, strides, shape);
        for (i64 n = 1; n < shape; n++) {
            c += strides;
            *c = previous_c = LAMBDA * *c + POLE * previous_c;
        }

        // anticausal initialization and recursion
        *c = previous_c = initial_anticausal_coefficient_(c);
        for (i64 n = shape - 2; 0 <= n; n--) {
            c -= strides;
            *c = previous_c = POLE * (previous_c - *c);
        }
    }

    template<typename T>
    // float/double or cfloat_t/cdouble_t
    void to_coeffs_(const T* input, i64 input_strides, T* output, i64 output_strides, i64 shape) {
        using real_t = noa::traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        constexpr auto LAMBDA = (1 - POLE) * (1 - 1 / POLE); // compute the overall gain

        // causal initialization and recursion
        T* c = output;
        T previous_c;  // cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = LAMBDA * initial_causal_coefficient_(input, input_strides, shape);
        for (i64 n = 1; n < shape; n++) {
            input += input_strides;
            c += output_strides;
            *c = previous_c = LAMBDA * *input + POLE * previous_c;
        }

        // anticausal initialization and recursion
        *c = previous_c = initial_anticausal_coefficient_(c);
        for (i64 n = shape - 2; 0 <= n; n--) {
            c -= output_strides;
            *c = previous_c = POLE * (previous_c - *c);
        }
    }

    template<typename T>
    void prefilter_1d_(const T* input, const Strides2<i64>& input_strides,
                       T* output, const Strides2<i64>& output_strides,
                       const Shape2<i64>& shape) {
        if (input == output) {
            for (i64 i = 0; i < shape[0]; ++i)
                to_coeffs_(output + output_strides[0] * i, output_strides[1], shape[1]);
        } else {
            for (i64 i = 0; i < shape[0]; ++i) {
                to_coeffs_(input + input_strides[0] * i, input_strides[1],
                           output + output_strides[0] * i, output_strides[1], shape[1]);
            }
        }
    }

    template<typename T>
    void prefilter_2d_(const T* input, const Strides3<i64>& input_strides,
                       T* output, const Strides3<i64>& output_strides,
                       const Shape3<i64>& shape) {
        if (input == output) {
            for (i64 i = 0; i < shape[0]; ++i)
                for (i64 y = 0; y < shape[1]; ++y) // every row
                    to_coeffs_(output + output_strides[0] * i + y * output_strides[1], output_strides[2], shape[2]);
            for (i64 i = 0; i < shape[0]; ++i)
                for (i64 x = 0; x < shape[2]; ++x) // every column
                    to_coeffs_(output + output_strides[0] * i + x * output_strides[2], output_strides[1], shape[1]);
        } else {
            for (i64 i = 0; i < shape[0]; ++i)
                for (i64 y = 0; y < shape[1]; ++y) // every row
                    to_coeffs_(input + i * input_strides[0] + y * input_strides[1], input_strides[2],
                               output + i * output_strides[0] + y * output_strides[1], output_strides[2], shape[2]);
            for (i64 i = 0; i < shape[0]; ++i)
                for (i64 x = 0; x < shape[2]; ++x) // every column
                    to_coeffs_(output + i * output_strides[0] + x * output_strides[2], output_strides[1], shape[1]);
        }
    }

    template<typename T>
    void prefilter_3d_(const T* input, const Strides4<i64>& input_strides,
                       T* output, const Strides4<i64>& output_strides,
                       const Shape4<i64>& shape, i64 threads) {
        constexpr i64 OMP_THRESHOLD = 1048576; // 1024*1024
        const i64 iterations = shape.pop_back().elements();

        if (input == output) {
            #pragma omp parallel num_threads(threads) default(none) if(iterations > OMP_THRESHOLD) \
            shared(output, output_strides, shape)
            {
                #pragma omp for collapse(3)
                for (i64 i = 0; i < shape[0]; ++i)
                    for (i64 z = 0; z < shape[1]; ++z)
                        for (i64 y = 0; y < shape[2]; ++y)
                            to_coeffs_(output + indexing::at(i, z, y, output_strides),
                                       output_strides[3], shape[3]); // every row
                #pragma omp for collapse(3)
                for (i64 i = 0; i < shape[0]; ++i)
                    for (i64 z = 0; z < shape[1]; ++z)
                        for (i64 x = 0; x < shape[3]; ++x)
                            to_coeffs_(output + i * output_strides[0] + z * output_strides[1] + x * output_strides[3],
                                       output_strides[2], shape[2]); // every column
                #pragma omp for collapse(3)
                for (i64 i = 0; i < shape[0]; ++i)
                    for (i64 y = 0; y < shape[2]; ++y)
                        for (i64 x = 0; x < shape[3]; ++x)
                            to_coeffs_(output + i * output_strides[0] + y * output_strides[2] + x * output_strides[3],
                                       output_strides[1], shape[1]); // every page
            }
        } else {
            #pragma omp parallel num_threads(threads) default(none) if(iterations > OMP_THRESHOLD) \
            shared(input, input_strides, output, output_strides, shape)
            {
                #pragma omp for collapse(3)
                for (i64 i = 0; i < shape[0]; ++i)
                    for (i64 z = 0; z < shape[1]; ++z)
                        for (i64 y = 0; y < shape[2]; ++y)
                            to_coeffs_(input + indexing::at(i, z, y, input_strides), input_strides[3],
                                       output + indexing::at(i, z, y, output_strides), output_strides[3],
                                       shape[3]); // every row
                #pragma omp for collapse(3)
                for (i64 i = 0; i < shape[0]; ++i)
                    for (i64 z = 0; z < shape[1]; ++z)
                        for (i64 x = 0; x < shape[3]; ++x)
                            to_coeffs_(output + i * output_strides[0] + z * output_strides[1] + x * output_strides[3],
                                       output_strides[2], shape[2]); // every column
                #pragma omp for collapse(3)
                for (i64 i = 0; i < shape[0]; ++i)
                    for (i64 y = 0; y < shape[2]; ++y)
                        for (i64 x = 0; x < shape[3]; ++x)
                            to_coeffs_(output + i * output_strides[0] + y * output_strides[2] + x * output_strides[3],
                                       output_strides[1], shape[1]); // every page
            }
        }
    }
}

namespace noa::cpu::geometry {
    template<typename Value, typename>
    void cubic_bspline_prefilter(const Value* input, Strides4<i64> input_strides,
                                 Value* output, Strides4<i64> output_strides,
                                 Shape4<i64> shape, i64 threads) {
        NOA_ASSERT(input && output && noa::all(shape > 0));

        // Reorder to rightmost.
        const auto order = noa::indexing::order(output_strides.pop_front(), shape.pop_front());
        if (noa::any(order != Vec3<i64>{0, 1, 2})) {
            const auto order_4d = (order + 1).push_front(0);
            input_strides = input_strides.reorder(order_4d);
            output_strides = output_strides.reorder(order_4d);
            shape = shape.reorder(order_4d);
        }

        const auto ndim = shape.ndim();
        if (ndim == 3) {
            prefilter_3d_(input, input_strides, output, output_strides, shape, threads);
        } else if (ndim == 2) {
            prefilter_2d_(input, input_strides.filter(0, 2, 3),
                          output, output_strides.filter(0, 2, 3),
                          shape.filter(0, 2, 3));
        } else {
            prefilter_1d_(input, input_strides.filter(0, 3),
                          output, output_strides.filter(0, 3),
                          shape.filter(0, 3));
        }
    }

    #define NOA_INSTANTIATE_PREFILTER_(T) \
    template void cubic_bspline_prefilter<T,void>(const T*, Strides4<i64>, T*, Strides4<i64>, Shape4<i64>, i64)

    NOA_INSTANTIATE_PREFILTER_(f32);
    NOA_INSTANTIATE_PREFILTER_(f64);
    NOA_INSTANTIATE_PREFILTER_(c32);
    NOA_INSTANTIATE_PREFILTER_(c64);
}
