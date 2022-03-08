#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/fft/Filters.h"

namespace {
    using namespace noa;

    enum class Type {
        LOWPASS,
        HIGHPASS
    };

    template<bool IS_CENTERED>
    inline int64_t getFrequency_(int64_t idx, int64_t dim) {
        if constexpr(IS_CENTERED)
            return idx - dim / 2;
        else
            return idx < (dim + 1) / 2 ? idx : idx - dim;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED>
    inline int64_t getOutputIndex_(int64_t i_idx, [[maybe_unused]] int64_t dim) {
        if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED)
            return i_idx;
        else if constexpr (IS_SRC_CENTERED)
            return noa::math::iFFTShift(i_idx, dim);
        else
            return noa::math::FFTShift(i_idx, dim);
    }

    template<Type PASS, typename T = void>
    inline float getSoftWindow_(float freq_cutoff, float freq_width, float freq) {
        constexpr float PI = math::Constants<float>::PI;
        float filter;
        if constexpr (PASS == Type::LOWPASS) {
            if (freq <= freq_cutoff)
                filter = 1;
            else if (freq_cutoff + freq_width <= freq)
                filter = 0;
            else
                filter = (1.f + math::cos(PI * (freq_cutoff - freq) / freq_width)) * 0.5f;
        } else if constexpr (PASS == Type::HIGHPASS) {
            if (freq_cutoff <= freq)
                filter = 1;
            else if (freq <= freq_cutoff - freq_width)
                filter = 0;
            else
                filter = (1.f + math::cos(PI * (freq - freq_cutoff) / freq_width)) * 0.5f;
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return filter;
    }

    template<Type PASS, typename T = void>
    inline float getHardWindow_(float freq_cutoff_sqd, float freq_sqd) {
        float filter;
        if constexpr (PASS == Type::LOWPASS) {
            if (freq_cutoff_sqd < freq_sqd)
                filter = 0;
            else
                filter = 1;
        } else if constexpr (PASS == Type::HIGHPASS) {
            if (freq_sqd < freq_cutoff_sqd)
                filter = 0;
            else
                filter = 1;
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return filter;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T, typename U>
    void applyPass_(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                    size4_t shape, size_t threads, U&& getPass) {
        using real_t = noa::traits::value_type_t<T>;
        const size_t batches = shape[0];
        const long3_t l_shape(shape.get() + 1);
        const float3_t f_shape(l_shape / 2 * 2 + long3_t{l_shape == 1}); // if odd, subtract 1 to keep Nyquist at 0.5

        #pragma omp parallel for collapse(4) num_threads(threads) default(none) \
                    shared(input, input_stride, output, output_stride, getPass, batches, l_shape, f_shape)
        for (size_t ii = 0; ii < batches; ++ii) {
            for (int64_t ij = 0; ij < l_shape[0]; ++ij) {
                for (int64_t ik = 0; ik < l_shape[1]; ++ik) {
                    for (int64_t il = 0; il < l_shape[2] / 2 + 1; ++il) {

                        // Compute the filter value for the current frequency:
                        float3_t frequency(getFrequency_<IS_SRC_CENTERED>(ij, l_shape[0]),
                                           getFrequency_<IS_SRC_CENTERED>(ik, l_shape[1]),
                                           il);
                        frequency /= f_shape;
                        const float frequency_sqd = math::dot(frequency, frequency);
                        const auto filter = static_cast<real_t>(getPass(frequency_sqd));

                        // Compute the index of the current frequency in the output:
                        const int64_t oj = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(ij, l_shape[0]);
                        const int64_t ol = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(ik, l_shape[1]);
                        output[at(ii, oj, ol, il, output_stride)] =
                                input ? input[at(ii, ij, ik, il, input_stride)] * filter : filter;
                    }
                }
            }
        }
    }

    // The omp collapse clause is preventing some optimizations. If threads > 1, it's worth it. Since I don't
    // want to penalize the single threaded version, just duplicate the entire kernel...
    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T, typename U>
    void applyPass_(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                    size4_t shape, U&& getPass) {
        using real_t = noa::traits::value_type_t<T>;
        const long3_t l_shape(shape.get() + 1);
        const float3_t f_shape(l_shape / 2 * 2 + long3_t{l_shape == 1}); // if odd, subtract 1 to keep Nyquist at 0.5

        for (size_t ii = 0; ii < shape[0]; ++ii) {
            for (int64_t ij = 0; ij < l_shape[0]; ++ij) {
                for (int64_t ik = 0; ik < l_shape[1]; ++ik) {
                    for (int64_t il = 0; il < l_shape[2] / 2 + 1; ++il) {

                        // Compute the filter value for the current frequency:
                        float3_t frequency(getFrequency_<IS_SRC_CENTERED>(ij, l_shape[0]),
                                           getFrequency_<IS_SRC_CENTERED>(ik, l_shape[1]),
                                           il);
                        frequency /= f_shape;
                        const float frequency_sqd = math::dot(frequency, frequency);
                        const auto filter = static_cast<real_t>(getPass(frequency_sqd));

                        // Compute the index of the current frequency in the output:
                        const int64_t oz = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(ij, l_shape[0]);
                        const int64_t oy = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(ik, l_shape[1]);
                        output[at(ii, oz, oy, il, output_stride)] =
                                input ? input[at(ii, ij, ik, il, input_stride)] * filter : filter;
                    }
                }
            }
        }
    }
}

namespace {
    template<Type PASS, bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    inline void singlePassSoft_(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                                size4_t shape, size_t threads, float cutoff, float width) {
        if (threads > 1) {
            applyPass_<IS_SRC_CENTERED, IS_DST_CENTERED>(
                    input, input_stride, output, output_stride, shape, threads,
                    [cutoff, width](float frequency_sqd) -> float {
                        return getSoftWindow_<PASS>(cutoff, width, math::sqrt(frequency_sqd));
                    });
        } else {
            applyPass_<IS_SRC_CENTERED, IS_DST_CENTERED>(
                    input, input_stride, output, output_stride, shape,
                    [cutoff, width](float frequency_sqd) -> float {
                        return getSoftWindow_<PASS>(cutoff, width, math::sqrt(frequency_sqd));
                    });
        }
    }

    template<Type PASS, bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    inline void singlePassHard_(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                                size4_t shape, size_t threads, float cutoff, [[maybe_unused]] float width) {
        const float freq_cutoff_sqd = cutoff * cutoff;
        if (threads > 1) {
            applyPass_<IS_SRC_CENTERED, IS_DST_CENTERED>(
                    input, input_stride, output, output_stride, shape, threads,
                    [freq_cutoff_sqd](float frequency_sqd) -> float {
                        return getHardWindow_<PASS>(freq_cutoff_sqd, frequency_sqd);
                    });
        } else {
            applyPass_<IS_SRC_CENTERED, IS_DST_CENTERED>(
                    input, input_stride, output, output_stride, shape,
                    [freq_cutoff_sqd](float frequency_sqd) -> float {
                        return getHardWindow_<PASS>(freq_cutoff_sqd, frequency_sqd);
                    });
        }
    }

    template<Type PASS, fft::Remap REMAP, typename T>
    inline void singlePass(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                           size4_t shape, float cutoff, float width, cpu::Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & fft::Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & fft::Layout::DST_CENTERED;
        if constexpr (REMAP_ & fft::Layout::SRC_FULL || REMAP_ & fft::Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        NOA_ASSERT(input != output || IS_SRC_CENTERED == IS_DST_CENTERED);

        if (input) {
            stream.enqueue(width > 1e-6f ?
                           singlePassSoft_<PASS, IS_SRC_CENTERED, IS_DST_CENTERED, T> :
                           singlePassHard_<PASS, IS_SRC_CENTERED, IS_DST_CENTERED, T>,
                           input, input_stride, output, output_stride, shape,
                           stream.threads(), cutoff, width);
        } else {
            if constexpr (!traits::is_complex_v<T>) {
                stream.enqueue(width > 1e-6f ?
                               singlePassSoft_<PASS, IS_SRC_CENTERED, IS_DST_CENTERED, T> :
                               singlePassHard_<PASS, IS_SRC_CENTERED, IS_DST_CENTERED, T>,
                               nullptr, size4_t{}, output, output_stride, shape,
                               stream.threads(), cutoff, width);
            } else {
                NOA_THROW_FUNC("(low|high)pass", "Cannot compute a filter of complex type");
            }
        }
    }
}

namespace noa::cpu::fft {
    template<Remap REMAP, typename T>
    void lowpass(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape,
                 float cutoff, float width, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        singlePass<Type::LOWPASS, REMAP>(input, input_stride, output, output_stride, shape, cutoff, width, stream);
    }

    template<Remap REMAP, typename T>
    void highpass(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape,
                  float cutoff, float width, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        singlePass<Type::HIGHPASS, REMAP>(input, input_stride, output, output_stride, shape, cutoff, width, stream);
    }
}

namespace {
    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    inline void bandPassSoft_(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                              size4_t shape, size_t threads, float cutoff_1, float cutoff_2,
                              float width_1, float width_2) {
        if (threads > 1) {
            applyPass_<IS_SRC_CENTERED, IS_DST_CENTERED>(
                    input, input_stride, output, output_stride, shape, threads,
                    [cutoff_1, cutoff_2, width_1, width_2](float frequency_sqd) -> float {
                        frequency_sqd = math::sqrt(frequency_sqd);
                        return getSoftWindow_<Type::HIGHPASS>(cutoff_1, width_1, frequency_sqd) *
                               getSoftWindow_<Type::LOWPASS>(cutoff_2, width_2, frequency_sqd);
                    });
        } else {
            applyPass_<IS_SRC_CENTERED, IS_DST_CENTERED>(
                    input, input_stride, output, output_stride, shape,
                    [cutoff_1, cutoff_2, width_1, width_2](float frequency_sqd) -> float {
                        frequency_sqd = math::sqrt(frequency_sqd);
                        return getSoftWindow_<Type::HIGHPASS>(cutoff_1, width_1, frequency_sqd) *
                               getSoftWindow_<Type::LOWPASS>(cutoff_2, width_2, frequency_sqd);
                    });
        }
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    inline void bandPassHard_(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                              size4_t shape, size_t threads, float cutoff_1, float cutoff_2,
                              [[maybe_unused]] float width_1, [[maybe_unused]] float width_2) {
        const float cutoff_sqd_1 = cutoff_1 * cutoff_1;
        const float cutoff_sqd_2 = cutoff_2 * cutoff_2;
        if (threads > 1) {
            applyPass_<IS_SRC_CENTERED, IS_DST_CENTERED>(
                    input, input_stride, output, output_stride, shape, threads,
                    [cutoff_sqd_1, cutoff_sqd_2](float frequency_sqd) -> float {
                        return getHardWindow_<Type::HIGHPASS>(cutoff_sqd_1, frequency_sqd) *
                               getHardWindow_<Type::LOWPASS>(cutoff_sqd_2, frequency_sqd);
                    });
        } else {
            applyPass_<IS_SRC_CENTERED, IS_DST_CENTERED>(
                    input, input_stride, output, output_stride, shape,
                    [cutoff_sqd_1, cutoff_sqd_2](float frequency_sqd) -> float {
                        return getHardWindow_<Type::HIGHPASS>(cutoff_sqd_1, frequency_sqd) *
                               getHardWindow_<Type::LOWPASS>(cutoff_sqd_2, frequency_sqd);
                    });
        }
    }
}

namespace noa::cpu::fft {
    template<Remap REMAP, typename T>
    void bandpass(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape,
                  float cutoff_1, float cutoff_2, float width_1, float width_2, Stream& stream) {
        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        NOA_ASSERT(input != output || IS_SRC_CENTERED == IS_DST_CENTERED);
        NOA_PROFILE_FUNCTION();
        if (input) {
            stream.enqueue(width_1 > 1e-6f || width_2 > 1e-6f ?
                           bandPassSoft_<IS_SRC_CENTERED, IS_DST_CENTERED, T> :
                           bandPassHard_<IS_SRC_CENTERED, IS_DST_CENTERED, T>,
                           input, input_stride, output, output_stride, shape, stream.threads(),
                           cutoff_1, cutoff_2, width_1, width_2);
        } else {
            if constexpr (!traits::is_complex_v<T>) {
                stream.enqueue(width_1 > 1e-6f || width_2 > 1e-6f ?
                               bandPassSoft_<IS_SRC_CENTERED, IS_DST_CENTERED, T> :
                               bandPassHard_<IS_SRC_CENTERED, IS_DST_CENTERED, T>,
                               nullptr, size4_t{}, output, output_stride, shape, stream.threads(),
                               cutoff_1, cutoff_2, width_1, width_2);
            } else {
                NOA_THROW("Cannot compute a filter of complex type");
            }
        }
    }

    #define NOA_INSTANTIATE_FILTERS_(T)                                                                                 \
    template void lowpass<Remap::H2H, T>(const T*, size4_t, T*, size4_t, size4_t, float, float, Stream&);               \
    template void highpass<Remap::H2H,T>(const T*, size4_t, T*, size4_t, size4_t, float, float, Stream&);               \
    template void bandpass<Remap::H2H,T>(const T*, size4_t, T*, size4_t, size4_t, float, float, float, float, Stream&); \
    template void lowpass<Remap::H2HC, T>(const T*, size4_t, T*, size4_t, size4_t, float, float, Stream&);              \
    template void highpass<Remap::H2HC,T>(const T*, size4_t, T*, size4_t, size4_t, float, float, Stream&);              \
    template void bandpass<Remap::H2HC,T>(const T*, size4_t, T*, size4_t, size4_t, float, float, float, float, Stream&);\
    template void lowpass<Remap::HC2H, T>(const T*, size4_t, T*, size4_t, size4_t, float, float, Stream&);              \
    template void highpass<Remap::HC2H,T>(const T*, size4_t, T*, size4_t, size4_t, float, float, Stream&);              \
    template void bandpass<Remap::HC2H,T>(const T*, size4_t, T*, size4_t, size4_t, float, float, float, float, Stream&);\
    template void lowpass<Remap::HC2HC, T>(const T*, size4_t, T*, size4_t, size4_t, float, float, Stream&);             \
    template void highpass<Remap::HC2HC,T>(const T*, size4_t, T*, size4_t, size4_t, float, float, Stream&);             \
    template void bandpass<Remap::HC2HC,T>(const T*, size4_t, T*, size4_t, size4_t, float, float, float, float, Stream&)

    NOA_INSTANTIATE_FILTERS_(half_t);
    NOA_INSTANTIATE_FILTERS_(float);
    NOA_INSTANTIATE_FILTERS_(double);
    NOA_INSTANTIATE_FILTERS_(chalf_t);
    NOA_INSTANTIATE_FILTERS_(cfloat_t);
    NOA_INSTANTIATE_FILTERS_(cdouble_t);
}
