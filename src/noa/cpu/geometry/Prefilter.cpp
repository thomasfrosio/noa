#include "noa/cpu/geometry/Prefilter.h"

// This is adapted from https://github.com/DannyRuijters/CubicInterpolationCUDA
// See licences/CubicInterpolationCUDA.txt

// Compared to original implementation:
//  - steps were switched to number of elements and strides were added.
//  - const was added when necessary.
//  - Out-of-place filtering was added.
//  - Support for double precision and complex types.
namespace {
    using namespace ::noa;

    // math::sqrt(3.0f)-2.0f; pole for cubic b-spline
    #define POLE_ (-0.2679491924311228)

    template<typename T>
    // float/double or cfloat_t/cdouble_t
    T initialCausalCoefficient_(const T* c, dim_t strides, dim_t shape) {
        using real_t = traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        const dim_t horizon = math::min(dim_t{12}, shape);

        // this initialization corresponds to clamping boundaries accelerated loop
        real_t zn = POLE;
        T sum = *c;
        for (dim_t n = 0; n < horizon; n++) {
            sum += zn * *c;
            zn *= POLE;
            c += strides;
        }
        return sum;
    }

    // float/double or cfloat_t/cdouble_t
    template<typename T>
    inline T initialAntiCausalCoefficient_(const T* c) {
        // this initialization corresponds to clamping boundaries
        using real_t = traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        return ((POLE / (POLE - 1)) * *c);
    }

    // float/double or cfloat_t/cdouble_t
    template<typename T>
    void toCoeffs_(T* output, dim_t strides, dim_t shape) {
        using real_t = traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        constexpr auto LAMBDA = (1 - POLE) * (1 - 1 / POLE); // compute the overall gain

        // causal initialization and recursion
        T* c = output;
        T previous_c;  //cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = LAMBDA * initialCausalCoefficient_(c, strides, shape);
        for (dim_t n = 1; n < shape; n++) {
            c += strides;
            *c = previous_c = LAMBDA * *c + POLE * previous_c;
        }

        // anticausal initialization and recursion
        *c = previous_c = initialAntiCausalCoefficient_(c);
        for (int n = static_cast<int>(shape) - 2; 0 <= n; n--) {
            c -= strides;
            *c = previous_c = POLE * (previous_c - *c);
        }
    }

    template<typename T>
    // float/double or cfloat_t/cdouble_t
    void toCoeffs_(const T* input, dim_t input_strides, T* output, dim_t output_strides, dim_t shape) {
        using real_t = traits::value_type_t<T>;
        constexpr auto POLE = static_cast<real_t>(POLE_);
        constexpr auto LAMBDA = (1 - POLE) * (1 - 1 / POLE); // compute the overall gain

        // causal initialization and recursion
        T* c = output;
        T previous_c;  // cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = LAMBDA * initialCausalCoefficient_(input, input_strides, shape);
        for (dim_t n = 1; n < shape; n++) {
            input += input_strides;
            c += output_strides;
            *c = previous_c = LAMBDA * *input + POLE * previous_c;
        }

        // anticausal initialization and recursion
        *c = previous_c = initialAntiCausalCoefficient_(c);
        for (int n = static_cast<int>(shape) - 2; 0 <= n; n--) {
            c -= output_strides;
            *c = previous_c = POLE * (previous_c - *c);
        }
    }

    template<typename T>
    void prefilter1D_(const T* input, dim2_t input_strides, T* output, dim2_t output_strides, dim2_t shape) {
        if (input == output) {
            for (dim_t i = 0; i < shape[0]; ++i)
                toCoeffs_(output + output_strides[0] * i, output_strides[1], shape[1]);
        } else {
            for (dim_t i = 0; i < shape[0]; ++i) {
                toCoeffs_(input + input_strides[0] * i, input_strides[1],
                          output + output_strides[0] * i, output_strides[1],
                          shape[1]);
            }
        }
    }

    template<typename T>
    void prefilter2D_(const T* input, dim3_t input_strides,
                      T* output, dim3_t output_strides,
                      dim3_t shape, dim_t threads) {
        if (input == output) {
            #pragma omp parallel num_threads(threads) default(none) shared(output, output_strides, shape)
            {
                #pragma omp for collapse(2)
                for (dim_t i = 0; i < shape[0]; ++i)
                    for (dim_t y = 0; y < shape[1]; ++y) // every row
                        toCoeffs_(output + output_strides[0] * i + y * output_strides[1], output_strides[2], shape[2]);
                #pragma omp for collapse(2)
                for (dim_t i = 0; i < shape[0]; ++i)
                    for (dim_t x = 0; x < shape[2]; ++x) // every column
                        toCoeffs_(output + output_strides[0] * i + x * output_strides[2], output_strides[1], shape[1]);
            }
        } else {
            #pragma omp parallel num_threads(threads) default(none) \
            shared(input, input_strides, output, output_strides, shape)
            {
                #pragma omp for collapse(2)
                for (dim_t i = 0; i < shape[0]; ++i)
                    for (dim_t y = 0; y < shape[1]; ++y) // every row
                        toCoeffs_(input + i * input_strides[0] + y * input_strides[1], input_strides[2],
                                  output + i * output_strides[0] + y * output_strides[1], output_strides[2], shape[2]);
                #pragma omp for collapse(2)
                for (dim_t i = 0; i < shape[0]; ++i)
                    for (dim_t x = 0; x < shape[2]; ++x) // every column
                        toCoeffs_(output + i * output_strides[0] + x * output_strides[2], output_strides[1], shape[1]);
            }
        }
    }

    template<typename T>
    void prefilter3D_(const T* input, dim4_t input_strides, T* output, dim4_t output_strides,
                      dim4_t shape, dim_t threads) {
        if (input == output) {
            #pragma omp parallel num_threads(threads) default(none) shared(output, output_strides, shape)
            {
                #pragma omp for collapse(3)
                for (dim_t i = 0; i < shape[0]; ++i)
                    for (dim_t z = 0; z < shape[1]; ++z)
                        for (dim_t y = 0; y < shape[2]; ++y)
                            toCoeffs_(output + indexing::at(i, z, y, output_strides),
                                      output_strides[3], shape[3]); // every row
                #pragma omp for collapse(3)
                for (dim_t i = 0; i < shape[0]; ++i)
                    for (dim_t z = 0; z < shape[1]; ++z)
                        for (dim_t x = 0; x < shape[3]; ++x)
                            toCoeffs_(output + i * output_strides[0] + z * output_strides[1] + x * output_strides[3],
                                      output_strides[2], shape[2]); // every column
                #pragma omp for collapse(3)
                for (dim_t i = 0; i < shape[0]; ++i)
                    for (dim_t y = 0; y < shape[2]; ++y)
                        for (dim_t x = 0; x < shape[3]; ++x)
                            toCoeffs_(output + i * output_strides[0] + y * output_strides[2] + x * output_strides[3],
                                      output_strides[1], shape[1]); // every page
            }
        } else {
            #pragma omp parallel num_threads(threads) default(none) \
            shared(input, input_strides, output, output_strides, shape)
            {
                #pragma omp for collapse(3)
                for (dim_t i = 0; i < shape[0]; ++i)
                    for (dim_t z = 0; z < shape[1]; ++z)
                        for (dim_t y = 0; y < shape[2]; ++y)
                            toCoeffs_(input + indexing::at(i, z, y, input_strides), input_strides[3],
                                      output + indexing::at(i, z, y, output_strides), output_strides[3],
                                      shape[3]); // every row
                #pragma omp for collapse(3)
                for (dim_t i = 0; i < shape[0]; ++i)
                    for (dim_t z = 0; z < shape[1]; ++z)
                        for (dim_t x = 0; x < shape[3]; ++x)
                            toCoeffs_(output + i * output_strides[0] + z * output_strides[1] + x * output_strides[3],
                                      output_strides[2], shape[2]); // every column
                #pragma omp for collapse(3)
                for (dim_t i = 0; i < shape[0]; ++i)
                    for (dim_t y = 0; y < shape[2]; ++y)
                        for (dim_t x = 0; x < shape[3]; ++x)
                            toCoeffs_(output + i * output_strides[0] + y * output_strides[2] + x * output_strides[3],
                                      output_strides[1], shape[1]); // every page
            }
        }
    }
}

namespace noa::cpu::geometry::bspline {
    template<typename Value, typename>
    void prefilter(const Value* input, dim4_t input_strides,
                   Value* output, dim4_t output_strides,
                   dim4_t shape, dim_t threads) {
        NOA_ASSERT(input && output && all(shape > 0));

        const dim_t ndim = dim3_t(shape.get(1)).ndim();
        if (ndim == 3) {
            prefilter3D_(input, input_strides, output, output_strides, shape, threads);
        } else if (ndim == 2) {
            prefilter2D_(input, dim3_t{input_strides[0], input_strides[2], input_strides[3]},
                         output, dim3_t{output_strides[0], output_strides[2], output_strides[3]},
                         dim3_t{shape[0], shape[2], shape[3]}, threads);
        } else {
            const bool is_column = shape[3] == 1;
            prefilter1D_(input, dim2_t{input_strides[0], input_strides[3 - is_column]},
                         output, dim2_t{output_strides[0], output_strides[3 - is_column]},
                         dim2_t{shape[0], shape[3 - is_column]});
        }
    }

    template<typename Value, typename>
    void prefilter(const shared_t<Value[]>& input, dim4_t input_strides,
                   const shared_t<Value[]>& output, dim4_t output_strides,
                   dim4_t shape, Stream& stream) {
        const dim_t threads = stream.threads();
        stream.enqueue([=]() {
            prefilter(input.get(), input_strides, output.get(), output_strides, shape, threads);
        });
    }

    #define NOA_INSTANTIATE_PREFILTER_(T)                                           \
    template void prefilter<T,void>(const T*, dim4_t, T*, dim4_t, dim4_t, dim_t);   \
    template void prefilter<T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&)

    NOA_INSTANTIATE_PREFILTER_(float);
    NOA_INSTANTIATE_PREFILTER_(double);
    NOA_INSTANTIATE_PREFILTER_(cfloat_t);
    NOA_INSTANTIATE_PREFILTER_(cdouble_t);
}
