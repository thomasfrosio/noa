#pragma once

#include "noa/core/Interpolation.hpp"

namespace noa::cpu::details {
    template<typename T>
    void cubic_bspline_prefilter_1d(
        const T* input, const Strides2& input_strides,
        T* output, const Strides2& output_strides,
        const Shape2& shape
    ) {
        using bspline = nd::BSplinePrefilter1d<T, isize>;

        if (input == output) {
            for (isize i = 0; i < shape[0]; ++i)
                bspline::filter_inplace(output + output_strides[0] * i, output_strides[1], shape[1]);
        } else {
            for (isize i = 0; i < shape[0]; ++i) {
                bspline::filter(input + input_strides[0] * i, input_strides[1],
                                output + output_strides[0] * i, output_strides[1], shape[1]);
            }
        }
    }

    template<typename T>
    void cubic_bspline_prefilter_2d(
        const T* input, const Strides3& input_strides,
        T* output, const Strides3& output_strides,
        const Shape3& shape
    ) {
        using bspline = nd::BSplinePrefilter1d<T, isize>;
        if (input == output) {
            for (isize i = 0; i < shape[0]; ++i)
                for (isize y = 0; y < shape[1]; ++y) // every row
                    bspline::filter_inplace(
                        output + output_strides[0] * i + y * output_strides[1],
                        output_strides[2], shape[2]);

            for (isize i = 0; i < shape[0]; ++i)
                for (isize x = 0; x < shape[2]; ++x) // every column
                    bspline::filter_inplace(
                        output + output_strides[0] * i + x * output_strides[2],
                        output_strides[1], shape[1]);
        } else {
            for (isize i = 0; i < shape[0]; ++i)
                for (isize y = 0; y < shape[1]; ++y) // every row
                    bspline::filter(
                        input + i * input_strides[0] + y * input_strides[1], input_strides[2],
                        output + i * output_strides[0] + y * output_strides[1], output_strides[2], shape[2]);

            for (isize i = 0; i < shape[0]; ++i)
                for (isize x = 0; x < shape[2]; ++x) // every column
                    bspline::filter_inplace(
                        output + i * output_strides[0] + x * output_strides[2],
                        output_strides[1], shape[1]);
        }
    }

    template<typename T>
    void cubic_bspline_prefilter_3d(
        const T* input, const Strides4& input_strides,
        T* output, const Strides4& output_strides,
        const Shape4& shape, isize threads
    ) {
        using bspline = nd::BSplinePrefilter1d<T, isize>;
        [[maybe_unused]] constexpr isize OMP_THRESHOLD = 1048576; // 1024*1024
        const isize n_iterations = shape.pop_back().n_elements();

        if (input == output) {
            #pragma omp parallel num_threads(threads) default(none) if(n_iterations > OMP_THRESHOLD) \
            shared(output, output_strides, shape)
            {
                #pragma omp for collapse(3)
                for (isize i = 0; i < shape[0]; ++i)
                    for (isize z = 0; z < shape[1]; ++z)
                        for (isize y = 0; y < shape[2]; ++y)
                            bspline::filter_inplace(
                                output + ni::offset_at(output_strides, i, z, y),
                                output_strides[3], shape[3]); // every row
                #pragma omp for collapse(3)
                for (isize i = 0; i < shape[0]; ++i)
                    for (isize z = 0; z < shape[1]; ++z)
                        for (isize x = 0; x < shape[3]; ++x)
                            bspline::filter_inplace(
                                output + i * output_strides[0] + z * output_strides[1] + x * output_strides[3],
                                output_strides[2], shape[2]); // every column
                #pragma omp for collapse(3)
                for (isize i = 0; i < shape[0]; ++i)
                    for (isize y = 0; y < shape[2]; ++y)
                        for (isize x = 0; x < shape[3]; ++x)
                            bspline::filter_inplace(
                                output + i * output_strides[0] + y * output_strides[2] + x * output_strides[3],
                                output_strides[1], shape[1]); // every page
            }
        } else {
            #pragma omp parallel num_threads(threads) default(none) if(n_iterations > OMP_THRESHOLD) \
            shared(input, input_strides, output, output_strides, shape)
            {
                #pragma omp for collapse(3)
                for (isize i = 0; i < shape[0]; ++i)
                    for (isize z = 0; z < shape[1]; ++z)
                        for (isize y = 0; y < shape[2]; ++y)
                            bspline::filter(
                                input + ni::offset_at(input_strides, i, z, y), input_strides[3],
                                output + ni::offset_at(output_strides, i, z, y), output_strides[3],
                                shape[3]); // every row
                #pragma omp for collapse(3)
                for (isize i = 0; i < shape[0]; ++i)
                    for (isize z = 0; z < shape[1]; ++z)
                        for (isize x = 0; x < shape[3]; ++x)
                            bspline::filter_inplace(
                                output + i * output_strides[0] + z * output_strides[1] + x * output_strides[3],
                                output_strides[2], shape[2]); // every column
                #pragma omp for collapse(3)
                for (isize i = 0; i < shape[0]; ++i)
                    for (isize y = 0; y < shape[2]; ++y)
                        for (isize x = 0; x < shape[3]; ++x)
                            bspline::filter_inplace(
                                output + i * output_strides[0] + y * output_strides[2] + x * output_strides[3],
                                output_strides[1], shape[1]); // every page
            }
        }
    }
}

namespace noa::cpu {
    template<typename Value>
    void cubic_bspline_prefilter(
        const Value* input, Strides4 input_strides,
        Value* output, Strides4 output_strides,
        Shape4 shape, isize n_threads
    ) {
        // Reorder to rightmost.
        const auto order = ni::order(output_strides.pop_front(), shape.pop_front());
        if (order != Vec<isize, 3>{0, 1, 2}) {
            const auto order_4d = (order + 1).push_front(0);
            input_strides = input_strides.reorder(order_4d);
            output_strides = output_strides.reorder(order_4d);
            shape = shape.reorder(order_4d);
        }

        const auto ndim = shape.ndim();
        if (ndim == 3) {
            details::cubic_bspline_prefilter_3d(input, input_strides, output, output_strides, shape, n_threads);
        } else if (ndim == 2) {
            details::cubic_bspline_prefilter_2d(
                input, input_strides.filter(0, 2, 3),
                output, output_strides.filter(0, 2, 3),
                shape.filter(0, 2, 3));
        } else {
            details::cubic_bspline_prefilter_1d(
                input, input_strides.filter(0, 3),
                output, output_strides.filter(0, 3),
                shape.filter(0, 3));
        }
    }
}
