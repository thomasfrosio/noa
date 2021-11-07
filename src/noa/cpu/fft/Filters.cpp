#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/fft/Filters.h"
#include "noa/cpu/memory/Copy.h"

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

    template<Type PASS>
    float getSoftWindow_(float freq_cutoff, float freq_width, float freq) {
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
        }
        return filter;
    }

    template<Type PASS>
    float getHardWindow_(float freq_cutoff_sqd, float freq_sqd) {
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
        }
        return filter;
    }

    template<typename T, typename U>
    void applyPass_(const T* inputs, T* outputs, size3_t shape, U&& getPass) {
        using real_t = noa::traits::value_type_t<T>;
        const long3_t l_shape(shape);
        const int64_t shape_x = l_shape.x / 2 + 1;
        const float3_t f_shape(shape.x / 2 * 2,
                               shape.y > 1 ? shape.y / 2 * 2 : 1,
                               shape.z > 1 ? shape.z / 2 * 2 : 1); // if odd, subtract 1 to keep Nyquist at 0.5
        float3_t distance_sqd;
        for (int64_t z = 0; z < l_shape.z; ++z) {
            const int64_t w = getFrequency_<false>(z, l_shape.z); // false: non-centered
            const int64_t offset_z = z * shape_x * l_shape.y;
            distance_sqd.z = static_cast<float>(w) / f_shape.z;
            distance_sqd.z *= distance_sqd.z;
            for (int64_t y = 0; y < l_shape.y; ++y) {
                const int64_t v = getFrequency_<false>(y, l_shape.y);
                const int64_t offset_yz = offset_z + y * shape_x;
                distance_sqd.y = static_cast<float>(v) / f_shape.y;
                distance_sqd.y *= distance_sqd.y;
                for (int64_t x = 0; x < shape_x; ++x) {
                    // x = u
                    distance_sqd.x = static_cast<float>(x) / f_shape.x;
                    distance_sqd.x *= distance_sqd.x;
                    const float frequency_sqd = math::sum(distance_sqd); // from 0 to 0.5
                    const float filter = getPass(frequency_sqd); // compiler should see through that

                    if (inputs) // this should be fully predicted
                        outputs[offset_yz + x] = inputs[offset_yz + x] * static_cast<real_t>(filter);
                    else
                        outputs[offset_yz + x] = static_cast<real_t>(filter);
                }
            }
        }
    }

    template<Type PASS, typename T>
    inline void singlePassSoft_(const T* inputs, T* outputs, size3_t shape,
                                float freq_cutoff, float freq_width) {
        applyPass_(inputs, outputs, shape,
                   [freq_cutoff, freq_width](float frequency_sqd) -> float {
                       return getSoftWindow_<PASS>(freq_cutoff, freq_width, math::sqrt(frequency_sqd));
                   });
    }

    template<Type PASS, typename T>
    inline void singlePassHard_(const T* inputs, T* outputs, size3_t shape, float freq_cutoff) {
        const float freq_cutoff_sqd = freq_cutoff * freq_cutoff;
        applyPass_(inputs, outputs, shape,
                   [freq_cutoff_sqd](float frequency_sqd) -> float {
                       return getHardWindow_<PASS>(freq_cutoff_sqd, frequency_sqd);
                   });
    }

    template<typename T>
    inline void bandPassSoft_(const T* inputs, T* outputs, size3_t shape,
                              float freq_cutoff_1, float freq_cutoff_2,
                              float freq_width_1, float freq_width_2) {
        applyPass_(inputs, outputs, shape,
                   [freq_cutoff_1, freq_cutoff_2, freq_width_1, freq_width_2](float frequency_sqd) -> float {
                       frequency_sqd = math::sqrt(frequency_sqd);
                       return getSoftWindow_<Type::HIGHPASS>(freq_cutoff_1, freq_width_1, frequency_sqd) *
                              getSoftWindow_<Type::LOWPASS>(freq_cutoff_2, freq_width_2, frequency_sqd);
                   });
    }

    template<typename T>
    inline void bandPassHard_(const T* inputs, T* outputs, size3_t shape,
                              float freq_cutoff_1, float freq_cutoff_2) {
        const float freq_cutoff_sqd_1 = freq_cutoff_1 * freq_cutoff_1;
        const float freq_cutoff_sqd_2 = freq_cutoff_2 * freq_cutoff_2;
        applyPass_(inputs, outputs, shape,
                   [freq_cutoff_sqd_1, freq_cutoff_sqd_2](float frequency_sqd) -> float {
                       return getHardWindow_<Type::HIGHPASS>(freq_cutoff_sqd_1, frequency_sqd) *
                              getHardWindow_<Type::LOWPASS>(freq_cutoff_sqd_2, frequency_sqd);
                   });
    }

    template<Type PASS, typename T>
    void singlePass(const T* inputs, T* outputs, size3_t shape, size_t batches, float freq_cutoff, float freq_width) {
        size_t elements = elementsFFT(shape);
        if (inputs) {
            for (size_t batch = 0; batch < batches; ++batch) {
                const T* input = inputs + elements * batch;
                T* output = outputs + elements * batch;

                if (freq_width > 1e-6f)
                    singlePassSoft_<PASS>(input, output, shape, freq_cutoff, freq_width);
                else
                    singlePassHard_<PASS>(input, output, shape, freq_cutoff);
            }
        } else {
            if constexpr (!traits::is_complex_v<T>) {
                if (freq_width > 1e-6f)
                    singlePassSoft_<PASS, T>(nullptr, outputs, shape, freq_cutoff, freq_width);
                else
                    singlePassHard_<PASS, T>(nullptr, outputs, shape, freq_cutoff);
                for (size_t batch = 1; batch < batches; ++batch)
                    cpu::memory::copy(outputs, outputs + elements * batch, elements);
            } else {
                NOA_THROW_FUNC("(low|high)pass", "Cannot compute a filter of complex type");
            }
        }
    }
}

namespace noa::cpu::fft {
    template<typename T>
    void lowpass(const T* inputs, T* outputs, size3_t shape, size_t batches, float freq_cutoff, float freq_width) {
        NOA_PROFILE_FUNCTION();
        singlePass<Type::LOWPASS>(inputs, outputs, shape, batches, freq_cutoff, freq_width);
    }

    template<typename T>
    void highpass(const T* inputs, T* outputs, size3_t shape, size_t batches, float freq_cutoff, float freq_width) {
        NOA_PROFILE_FUNCTION();
        singlePass<Type::HIGHPASS>(inputs, outputs, shape, batches, freq_cutoff, freq_width);
    }

    template<typename T>
    void bandpass(const T* inputs, T* outputs, size3_t shape, size_t batches,
                  float freq_cutoff_1, float freq_cutoff_2, float freq_width_1, float freq_width_2) {
        NOA_PROFILE_FUNCTION();
        size_t elements = elementsFFT(shape);
        if (inputs) {
            for (size_t batch = 0; batch < batches; ++batch) {
                const T* input = inputs + elements * batch;
                T* output = outputs + elements * batch;

                if (freq_width_1 > 1e-6f || freq_width_2 > 1e-6f)
                    bandPassSoft_(input, output, shape,
                                  freq_cutoff_1, freq_cutoff_2, freq_width_1, freq_width_2);
                else
                    bandPassHard_(input, output, shape,
                                  freq_cutoff_1, freq_cutoff_2);
            }
        } else {
            if constexpr (!traits::is_complex_v<T>) {
                if (freq_width_1 > 1e-6f || freq_width_2 > 1e-6f)
                    bandPassSoft_<T>(nullptr, outputs, shape,
                                     freq_cutoff_1, freq_cutoff_2, freq_width_1, freq_width_2);
                else
                    bandPassHard_<T>(nullptr, outputs, shape,
                                     freq_cutoff_1, freq_cutoff_2);
                for (size_t batch = 1; batch < batches; ++batch)
                    cpu::memory::copy(outputs, outputs + elements * batch, elements);
            } else {
                NOA_THROW("Cannot compute a filter of complex type");
            }
        }
    }

    #define NOA_INSTANTIATE_FILTERS_(T)                                         \
    template void lowpass<T>(const T*, T*, size3_t, size_t, float, float);      \
    template void highpass<T>(const T*, T*, size3_t, size_t, float, float);     \
    template void bandpass<T>(const T*, T*, size3_t, size_t, float, float, float, float)

    NOA_INSTANTIATE_FILTERS_(float);
    NOA_INSTANTIATE_FILTERS_(double);
    NOA_INSTANTIATE_FILTERS_(cfloat_t);
    NOA_INSTANTIATE_FILTERS_(cdouble_t);
}
