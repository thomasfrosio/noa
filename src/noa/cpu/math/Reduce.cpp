#include "noa/common/Math.h"

#include "noa/cpu/math/Reduce.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Set.h"
#include "noa/cpu/memory/PtrHost.h"

// Reduce a 4D array to one element:
namespace {
    using namespace noa;

    template<typename T>
    void reduceMin_(const T* input, size4_t stride, size4_t shape, T* output, size_t threads) {
        const size_t elements = shape.elements();
        using value_t = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        value_t min_value = math::Limits<value_t>::max();

        #pragma omp parallel for if (elements > 100000) \
        collapse(4) num_threads(threads) reduction(min:min_value) default(none) \
        shared(input, stride, shape)

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        const size_t offset = indexing::at(i, j, k, l, stride);
                        const auto val = static_cast<value_t>(input[offset]);
                        min_value = val < min_value ? val : min_value;
                    }
                }
            }
        }
        *output = static_cast<T>(min_value);
    }

    template<typename T>
    void reduceMax_(const T* input, size4_t stride, size4_t shape, T* output, size_t threads) {
        const size_t elements = shape.elements();
        using value_t = std::conditional_t<std::is_same_v<half_t, T>, float, T>;
        value_t max_value = math::Limits<value_t>::lowest();

        #pragma omp parallel for if (elements > 100000) \
        collapse(4) num_threads(threads) reduction(max:max_value) default(none) \
        shared(input, stride, shape)

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        const size_t offset = indexing::at(i, j, k, l, stride);
                        const auto val = static_cast<value_t>(input[offset]);
                        max_value = max_value < val  ? val : max_value;
                    }
                }
            }
        }
        *output = static_cast<T>(max_value);
    }

    template<typename T>
    void reduceSum_(const T* input, size4_t stride, size4_t shape, T* output, size_t threads) {
        const size_t elements = shape.elements();
        T sum = 0;

        #pragma omp parallel for if (elements > 100000) \
        collapse(4) num_threads(threads) reduction(+:sum) default(none) \
        shared(input, stride, shape)

        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                for (size_t k = 0; k < shape[2]; ++k)
                    for (size_t l = 0; l < shape[3]; ++l)
                        sum += input[indexing::at(i, j, k, l, stride)];
        *output = sum;
    }

    template<typename T>
    void reduceAccurateSum_(const T* input, size4_t stride, size4_t shape, T* output, size_t threads) {
        const size_t elements = shape.elements();
        double sum = 0, err = 0;

        #pragma omp parallel for if (elements > 100000) \
        collapse(4) num_threads(threads) reduction(+:sum, err) default(none) \
        shared(input, stride, shape)

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        auto tmp = static_cast<double>(input[indexing::at(i, j, k, l, stride)]);
                        auto t = sum + tmp;
                        if (noa::math::abs(sum) >= noa::math::abs(tmp)) // Neumaier variation
                            err += (sum - t) + tmp;
                        else
                            err += (tmp - t) + sum;
                        sum = t;
                    }
                }
            }
        }
        *output = static_cast<T>(sum + err);
    }

    template<typename T>
    void reduceAccurateSumComplex_(const T* input, size4_t stride, size4_t shape, T* output, size_t threads) {
        const size_t elements = shape.elements();
        double sum_real = 0, sum_imag = 0, err_real = 0, err_imag = 0;

        #pragma omp parallel for if (elements > 100000) \
        collapse(4) num_threads(threads) reduction(+:sum_real, sum_imag, err_real, err_imag) \
        default(none) shared(input, stride, shape)

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        auto tmp = static_cast<cdouble_t>(input[indexing::at(i, j, k, l, stride)]);

                        auto t_real = sum_real + tmp.real;
                        if (noa::math::abs(sum_real) >= noa::math::abs(tmp.real)) // Neumaier variation
                            err_real += (sum_real - t_real) + tmp.real;
                        else
                            err_real += (tmp.real - t_real) + sum_real;
                        sum_real = t_real;

                        auto t_imag = sum_imag + tmp.imag;
                        if (noa::math::abs(sum_imag) >= noa::math::abs(tmp.imag)) // Neumaier variation
                            err_imag += (sum_imag - t_imag) + tmp.imag;
                        else
                            err_imag += (tmp.imag - t_imag) + sum_imag;
                        sum_imag = t_imag;
                    }
                }
            }
        }
        using real_t = traits::value_type_t<T>;
        output->real = static_cast<real_t>(sum_real + err_real);
        output->imag = static_cast<real_t>(sum_imag + err_imag);
    }

    template<int DDOF, typename T>
    void reduceAccurateVariance_(const T* input, size4_t stride, size4_t shape, T mean, T* variance, size_t threads) {
        const auto count = static_cast<double>(shape.elements() - DDOF);
        const auto tmp_mean = static_cast<double>(mean);
        double tmp_variance = 0;

        #pragma omp parallel for if (count > 100000) \
        collapse(4) num_threads(threads) reduction(+:tmp_variance) default(none) \
        shared(input, stride, shape, tmp_mean)

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        const auto tmp = static_cast<double>(input[indexing::at(i, j, k, l, stride)]);
                        const auto distance = tmp - tmp_mean;
                        tmp_variance += distance * distance;
                    }
                }
            }
        }
        *variance = static_cast<T>(tmp_variance / count);
    }

    template<int DDOF, typename T>
    void reduceAccurateVarianceComplex_(const Complex<T>* input, size4_t stride, size4_t shape,
                                        Complex<T> mean, T* variance, size_t threads) {
        const auto count = static_cast<double>(shape.elements() - DDOF);
        const auto tmp_mean = static_cast<cdouble_t>(mean);
        double tmp_variance = 0;

        #pragma omp parallel for if (count > 100000) \
        collapse(4) num_threads(threads) reduction(+:tmp_variance) default(none) \
        shared(input, stride, shape, tmp_mean)

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        const auto tmp = static_cast<cdouble_t>(input[indexing::at(i, j, k, l, stride)]);
                        double distance = math::abs(tmp - tmp_mean);
                        distance *= distance;
                        tmp_variance += distance;
                    }
                }
            }
        }
        *variance = static_cast<T>(tmp_variance / count);
    }
}

// Reduce one axis to one value:
namespace {
    using namespace noa;

    template<typename T>
    double reduceAxisAccurateSum_(const T* input, size_t stride, size_t elements) {
        double sum = 0, err = 0;
        for (size_t i = 0; i < elements; ++i) {
            const auto tmp = static_cast<double>(input[i * stride]);
            auto t = sum + tmp;
            if (noa::math::abs(sum) >= noa::math::abs(tmp)) // Neumaier variation
                err += (sum - t) + tmp;
            else
                err += (tmp - t) + sum;
            sum = t;
        }
        return sum + err;
    }

    template<typename T>
    cdouble_t reduceAxisAccurateSum_(const Complex<T>* input, size_t stride, size_t elements) {
        double sum_real = 0, sum_imag = 0, err_real = 0, err_imag = 0;
        for (size_t i = 0; i < elements; ++i) {
            const auto tmp = static_cast<cdouble_t>(input[i * stride]);
            auto t_real = sum_real + tmp.real;
            if (noa::math::abs(sum_real) >= noa::math::abs(tmp.real))
                err_real += (sum_real - t_real) + tmp.real;
            else
                err_real += (tmp.real - t_real) + sum_real;
            sum_real = t_real;

            auto t_imag = sum_imag + tmp.imag;
            if (noa::math::abs(sum_imag) >= noa::math::abs(tmp.imag))
                err_imag += (sum_imag - t_imag) + tmp.imag;
            else
                err_imag += (tmp.imag - t_imag) + sum_imag;
            sum_imag = t_imag;
        }
        return {sum_real + err_real, sum_imag + err_imag};
    }

    template<int DDOF, typename T>
    auto reduceAxisAccurateVariance_(const T* input, size_t stride, size_t elements) {
        const auto count = static_cast<double>(elements - DDOF);
        const auto mean = reduceAxisAccurateSum_(input, stride, elements) / count;
        double variance = 0;
        for (size_t i = 0; i < elements; ++i) {
            const auto tmp = static_cast<double>(input[i * stride]);
            const double distance = tmp - mean;
            variance += distance * distance;
        }
        return variance / count;
    }

    template<int DDOF, typename T>
    auto reduceAxisAccurateVarianceComplex_(const T* input, size_t stride, size_t elements) {
        const auto count = static_cast<double>(elements - DDOF);
        const auto mean = reduceAxisAccurateSum_(input, stride, elements) / count;
        double variance = 0;
        for (size_t i = 0; i < elements; ++i) {
            const auto tmp = static_cast<cdouble_t>(input[i * stride]);
            const double distance = math::abs(tmp - mean);
            variance += distance * distance;
        }
        return variance / count;
    }

    template<int AXIS, typename T, typename U, typename ReduceOp>
    inline void reduceAxis_(const T* input, size4_t input_stride, size4_t input_shape,
                            U* output, size4_t output_stride, ReduceOp reduce) {
        if constexpr (AXIS == 0) {
            for (size_t j = 0; j < input_shape[1]; ++j)
                for (size_t k = 0; k < input_shape[2]; ++k)
                    for (size_t l = 0; l < input_shape[3]; ++l)
                        output[indexing::at(0, j, k, l, output_stride)] =
                                static_cast<U>(reduce(input + indexing::at(0, j, k, l, input_stride),
                                                      input_stride[0], input_shape[0]));
        } else if constexpr (AXIS == 1) {
            for (size_t i = 0; i < input_shape[0]; ++i)
                for (size_t k = 0; k < input_shape[2]; ++k)
                    for (size_t l = 0; l < input_shape[3]; ++l)
                        output[indexing::at(i, 0, k, l, output_stride)] =
                                static_cast<U>(reduce(input + indexing::at(i, 0, k, l, input_stride),
                                                      input_stride[1], input_shape[1]));
        } else if constexpr (AXIS == 2) {
            for (size_t i = 0; i < input_shape[0]; ++i)
                for (size_t j = 0; j < input_shape[1]; ++j)
                    for (size_t l = 0; l < input_shape[3]; ++l)
                        output[indexing::at(i, j, 0, l, output_stride)] =
                                static_cast<U>(reduce(input + indexing::at(i, j, 0, l, input_stride),
                                                      input_stride[2], input_shape[2]));
        } else if constexpr (AXIS == 3) {
            for (size_t i = 0; i < input_shape[0]; ++i)
                for (size_t j = 0; j < input_shape[1]; ++j)
                    for (size_t k = 0; k < input_shape[2]; ++k)
                        output[indexing::at(i, j, k, output_stride)] =
                                static_cast<U>(reduce(input + indexing::at(i, j, k, input_stride),
                                                      input_stride[3], input_shape[3]));
        }
    }

    template<typename T, typename U, typename ReduceOp>
    inline void reduceAxis_(const char* func, cpu::Stream& stream,
                            const shared_t<T[]> input, size4_t input_stride, size4_t input_shape,
                            const shared_t<U[]> output, size4_t output_stride, size4_t output_shape,
                            bool4_t mask, ReduceOp reduce) {
        if (noa::math::sum(int4_t{mask}) > 1) {
            NOA_THROW_FUNC(func, "Reducing more than one axis at a time is only supported if the reduction results in "
                                 "one value per batch, i.e. the 3 innermost dimensions are shape=1 after reduction. "
                                 "Got input:{}, output:{}, reduce:{}", input_shape, output_shape, mask);
        }

        if (mask[3]) {
            stream.enqueue([=]() {
                reduceAxis_<3>(input.get(), input_stride, input_shape, output.get(), output_stride, reduce);
            });
        } else if (mask[2]) {
            stream.enqueue([=]() {
                reduceAxis_<2>(input.get(), input_stride, input_shape, output.get(), output_stride, reduce);
            });
        } else if (mask[1]) {
            stream.enqueue([=]() {
                reduceAxis_<1>(input.get(), input_stride, input_shape, output.get(), output_stride, reduce);
            });
        } else if (mask[0]) {
            stream.enqueue([=]() {
                reduceAxis_<0>(input.get(), input_stride, input_shape, output.get(), output_stride, reduce);
            });
        }
    }

    bool4_t getMask_(const char* func, size4_t input_shape, size4_t output_shape) {
        const bool4_t mask{input_shape != output_shape};
        if (any(mask && (output_shape != 1))) {
            NOA_THROW_FUNC(func, "Dimensions should match the input shape, or be 1, indicating the dimension should be "
                                 "reduced to one element. Got input:{}, output:{}", input_shape, output_shape);
        }
        return mask;
    }
}

namespace noa::cpu::math {
    template<typename T, typename>
    T min(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        T output;
        const size_t threads = stream.threads();
        stream.enqueue([=, &output](){ reduceMin_<T>(input.get(), stride, shape, &output, threads); });
        stream.synchronize();
        return output;
    }

    template<typename T, typename>
    T max(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        T output;
        const size_t threads = stream.threads();
        stream.enqueue([=, &output](){ reduceMax_<T>(input.get(), stride, shape, &output, threads); });
        stream.synchronize();
        return output;
    }

    template<typename T, typename>
    T sum(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        T output;
        const size_t threads = stream.threads();
        stream.enqueue([=, &output](){
            if constexpr (noa::traits::is_float_v<T>) {
                reduceAccurateSum_<T>(input.get(), stride, shape, &output, threads);
            } else if constexpr (noa::traits::is_complex_v<T>) {
                reduceAccurateSumComplex_<T>(input.get(), stride, shape, &output, threads);
            } else {
                reduceSum_<T>(input.get(), stride, shape, &output, threads);
            }
        });
        stream.synchronize();
        return output;
    }

    template<int DDOF, typename T, typename U, typename>
    U var(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        U output;
        stream.enqueue([=, &output, &stream]() {
            T mean = sum(input, stride, shape, stream);
            using value_t = noa::traits::value_type_t<T>;
            const auto count = static_cast<value_t>(shape.elements() - DDOF);
            mean /= count;

            if constexpr (noa::traits::is_float_v<T>) {
                reduceAccurateVariance_<DDOF>(input.get(), stride, shape, mean, &output, stream.threads());
            } else if constexpr (noa::traits::is_complex_v<T>) {
                reduceAccurateVarianceComplex_<DDOF>(input.get(), stride, shape, mean, &output, stream.threads());
            } else {
                static_assert(traits::always_false_v<T>);
            }
        });
        stream.synchronize();
        return output;
    }
}

namespace noa::cpu::math {
    template<typename T, typename>
    void min(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
             const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        const bool4_t mask = getMask_("min", input_shape, output_shape);
        const bool4_t is_or_should_reduce{output_shape == 1 || mask};

        if (!any(mask)) {
            cpu::memory::copy(input, input_stride, output, output_stride, output_shape, stream);

        } else if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            // Reduce the input to one value or one value per batch.
            stream.enqueue([=](){
                const T* iptr = input.get();
                T* optr = output.get();
                const size_t threads = stream.threads();
                const size4_t shape_to_reduce{is_or_should_reduce[0] ? input_shape[0] : 1,
                                              input_shape[1], input_shape[2], input_shape[3]};
                for (size_t i = 0; i < output_shape[0]; ++i) {
                    reduceMin_<T>(iptr + i * input_stride[0], input_stride, shape_to_reduce,
                                  optr + i * output_stride[0], threads);
                }
            });
        } else {
            reduceAxis_("min", stream, input, input_stride, input_shape,
                        output, output_stride, output_shape, mask,
                        [](const T* axis, size_t stride, size_t elements) {
                            T min = *axis;
                            for (size_t i = 0; i < elements; ++i)
                                min = noa::math::min(min, axis[i * stride]);
                            return min;
                        });
        }
    }

    template<typename T, typename>
    void max(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
             const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        const bool4_t mask = getMask_("max", input_shape, output_shape);
        const bool4_t is_or_should_reduce{output_shape == 1 || mask};

        if (!any(mask)) {
            cpu::memory::copy(input, input_stride, output, output_stride, output_shape, stream);

        } else if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            // Reduce the input to one value or one value per batch.
            stream.enqueue([=](){
                const T* iptr = input.get();
                T* optr = output.get();
                const size_t threads = stream.threads();
                const size4_t shape_to_reduce{is_or_should_reduce[0] ? input_shape[0] : 1,
                                              input_shape[1], input_shape[2], input_shape[3]};
                for (size_t i = 0; i < output_shape[0]; ++i) {
                    reduceMax_<T>(iptr + i * input_stride[0], input_stride, shape_to_reduce,
                                  optr + i * output_stride[0], threads);
                }
            });
        } else {
            reduceAxis_("max", stream, input, input_stride, input_shape,
                        output, output_stride, output_shape, mask,
                        [](const T* axis, size_t stride, size_t elements) {
                            T max = *axis;
                            for (size_t i = 0; i < elements; ++i)
                                max = noa::math::max(max, axis[i * stride]);
                            return max;
                        });
        }
    }

    #define NOA_INSTANTIATE_MIN_MAX_(T)                                       \
    template T min<T, void>(const shared_t<T[]>&, size4_t, size4_t, Stream&); \
    template T max<T, void>(const shared_t<T[]>&, size4_t, size4_t, Stream&); \
    template void min<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&); \
    template void max<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_MIN_MAX_(int16_t);
    NOA_INSTANTIATE_MIN_MAX_(int32_t);
    NOA_INSTANTIATE_MIN_MAX_(int64_t);
    NOA_INSTANTIATE_MIN_MAX_(uint16_t);
    NOA_INSTANTIATE_MIN_MAX_(uint32_t);
    NOA_INSTANTIATE_MIN_MAX_(uint64_t);
    NOA_INSTANTIATE_MIN_MAX_(half_t);
    NOA_INSTANTIATE_MIN_MAX_(float);
    NOA_INSTANTIATE_MIN_MAX_(double);


    template<typename T, typename>
    void sum(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
             const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        const bool4_t mask = getMask_("sum", input_shape, output_shape);
        const bool4_t is_or_should_reduce{output_shape == 1 || mask};

        if (!any(mask)) {
            cpu::memory::copy(input, input_stride, output, output_stride, output_shape, stream);

        } else if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            // Reduce the input to one value or one value per batch.
            stream.enqueue([=]() mutable {
                T* optr = output.get();
                const size4_t shape_to_reduce{is_or_should_reduce[0] ? input_shape[0] : 1,
                                              input_shape[1], input_shape[2], input_shape[3]};
                for (size_t i = 0; i < output_shape[0]; ++i) {
                    const shared_t<T[]> tmp{input, input.get() + i * input_stride[0]};
                    optr[i * output_stride[0]] = sum(tmp, input_stride, shape_to_reduce, stream);
                }
            });
        } else {
            reduceAxis_("sum", stream, input, input_stride, input_shape, output, output_stride, output_shape, mask,
                        [](const T* axis, size_t stride, size_t elements) {
                            if constexpr (traits::is_complex_v<T> || traits::is_float_v<T>) {
                                return reduceAxisAccurateSum_(axis, stride, elements);
                            } else if constexpr (traits::is_int_v<T>) {
                                T sum = 0;
                                for (size_t i = 0; i < elements; ++i)
                                    sum += axis[i * stride];
                                return sum;
                            } else {
                                static_assert(traits::always_false_v<T>);
                            }
                        });
        }
    }

    template<typename T, typename>
    void mean(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
              const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        const bool4_t mask = getMask_("mean", input_shape, output_shape);
        const bool4_t is_or_should_reduce{output_shape == 1 || mask};

        if (!any(mask)) {
            cpu::memory::copy(input, input_stride, output, output_stride, output_shape, stream);

        } else if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            // Reduce the input to one value or one value per batch.
            stream.enqueue([=]() mutable {
                T* optr = output.get();
                const size4_t shape_to_reduce{is_or_should_reduce[0] ? input_shape[0] : 1,
                                              input_shape[1], input_shape[2], input_shape[3]};
                for (size_t i = 0; i < output_shape[0]; ++i) {
                    const shared_t<T[]> tmp{input, input.get() + i * input_stride[0]};
                    optr[i * output_stride[0]] = mean(tmp, input_stride, shape_to_reduce, stream);
                }
            });
        } else {
            reduceAxis_("mean", stream, input, input_stride, input_shape, output, output_stride, output_shape, mask,
                        [](const T* axis, size_t stride, size_t elements) {
                            if constexpr (traits::is_complex_v<T> || traits::is_float_v<T>) {
                                const auto count = static_cast<double>(elements);
                                return reduceAxisAccurateSum_(axis, stride, elements) / count;
                            } else if constexpr (traits::is_int_v<T>) {
                                T sum = 0;
                                for (size_t i = 0; i < elements; ++i)
                                    sum += axis[i * stride];
                                return sum / static_cast<T>(elements);
                            } else {
                                static_assert(traits::always_false_v<T>);
                            }
                        });
        }
    }

    #define NOA_INSTANTIATE_SUM_MEAN_(T)                                        \
    template T sum<T, void>(const shared_t<T[]>&, size4_t, size4_t, Stream&);   \
    template T mean<T, void>(const shared_t<T[]>&, size4_t, size4_t, Stream&);  \
    template void sum<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&); \
    template void mean<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_SUM_MEAN_(int32_t);
    NOA_INSTANTIATE_SUM_MEAN_(int64_t);
    NOA_INSTANTIATE_SUM_MEAN_(uint32_t);
    NOA_INSTANTIATE_SUM_MEAN_(uint64_t);
    NOA_INSTANTIATE_SUM_MEAN_(float);
    NOA_INSTANTIATE_SUM_MEAN_(double);
    NOA_INSTANTIATE_SUM_MEAN_(cfloat_t);
    NOA_INSTANTIATE_SUM_MEAN_(cdouble_t);


    template<int DDOF, typename T, typename U, typename>
    void var(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
             const shared_t<U[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        const bool4_t mask = getMask_("var", input_shape, output_shape);
        const bool4_t is_or_should_reduce{output_shape == 1 || mask};

        if (!any(mask)) {
            if constexpr (noa::traits::is_complex_v<T>)
                math::ewise(input, input_stride, output, output_stride, output_shape, noa::math::abs_t{}, stream);
            else
                memory::copy(input, input_stride, output, output_stride, output_shape, stream);

        } else if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            // Reduce the input to one value or one value per batch.
            // Reduce the input to one value or one value per batch.
            stream.enqueue([=]() mutable {
                U* optr = output.get();
                const size4_t shape_to_reduce{is_or_should_reduce[0] ? input_shape[0] : 1,
                                              input_shape[1], input_shape[2], input_shape[3]};
                for (size_t i = 0; i < output_shape[0]; ++i) {
                    const shared_t<T[]> tmp{input, input.get() + i * input_stride[0]};
                    optr[i * output_stride[0]] = var<DDOF>(tmp, input_stride, shape_to_reduce, stream);
                }
            });
        } else {
            reduceAxis_("var", stream, input, input_stride, input_shape, output, output_stride, output_shape, mask,
                        [](const T* axis, size_t stride, size_t elements) {
                        if constexpr (traits::is_complex_v<T>) {
                            return reduceAxisAccurateVarianceComplex_<DDOF>(axis, stride, elements);
                        } else if constexpr (traits::is_float_v<T>) {
                            return reduceAxisAccurateVariance_<DDOF>(axis, stride, elements);
                        } else {
                            static_assert(traits::always_false_v<T>);
                        }
                    });
        }
    }

    #define NOA_INSTANTIATE_VAR_(T,U,DDOF)                                              \
    template U var<DDOF,T,U,void>(const shared_t<T[]>& , size4_t, size4_t, Stream&);    \
    template void var<DDOF,T,U,void>(const shared_t<T[]>& , size4_t, size4_t, const shared_t<U[]>&, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_VAR_(float, float, 0);
    NOA_INSTANTIATE_VAR_(double, double, 0);
    NOA_INSTANTIATE_VAR_(float, float, 1);
    NOA_INSTANTIATE_VAR_(double, double, 1);
    NOA_INSTANTIATE_VAR_(cfloat_t, float, 0);
    NOA_INSTANTIATE_VAR_(cdouble_t, double, 0);
    NOA_INSTANTIATE_VAR_(cfloat_t, float, 1);
    NOA_INSTANTIATE_VAR_(cdouble_t, double, 1);
}
