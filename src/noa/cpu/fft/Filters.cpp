#include "noa/common/Profiler.h"
#include "noa/cpu/fft/Filters.h"

// Commons:
namespace {
    enum class Type {
        LOWPASS,
        HIGHPASS
    };

    float getDistanceSquared_(size_t dimension, size_t half, size_t idx) {
        float dist = idx >= half ? static_cast<float>(idx) - static_cast<float>(dimension) : static_cast<float>(idx);
        dist /= static_cast<float>(dimension);
        dist *= dist;
        return dist;
    }
}

// Soft edges (Hann window):
namespace {
    using namespace noa;

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

    template<Type PASS, typename T>
    void singlePassSoft_(const T* inputs, T* outputs, size3_t shape, float freq_cutoff, float freq_width) {
        using real_t = noa::traits::value_type_t<T>;
        size3_t half(shape / size_t{2} + size_t{1});

        float3_t distance_sqd;
        float frequency, filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    frequency = math::sqrt(math::sum(distance_sqd)); // from 0 to 0.5
                    filter = getSoftWindow_<PASS>(freq_cutoff, freq_width, frequency);
                    outputs[offset + x] = inputs[offset + x] * static_cast<real_t>(filter);
                }
            }
        }
    }

    template<typename T>
    void bandPassSoft_(const T* inputs, T* outputs, size3_t shape, float freq_cutoff_1, float freq_cutoff_2,
                       float freq_width_1, float freq_width_2) {
        using real_t = noa::traits::value_type_t<T>;
        size3_t half(shape / size_t{2} + size_t{1});

        float3_t distance_sqd;
        float frequency, filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    frequency = math::sqrt(math::sum(distance_sqd)); // from 0 to 0.5
                    filter = getSoftWindow_<Type::HIGHPASS>(freq_cutoff_1, freq_width_1, frequency);
                    filter *= getSoftWindow_<Type::LOWPASS>(freq_cutoff_2, freq_width_2, frequency);
                    outputs[offset + x] = inputs[offset + x] * static_cast<real_t>(filter);
                }
            }
        }
    }

    template<Type PASS, typename T>
    void singlePassSoft_(T* output_filter, size3_t shape, float freq_cutoff, float freq_width) {
        using real_t = noa::traits::value_type_t<T>;
        size3_t half(shape / size_t{2} + size_t{1});

        float3_t distance_sqd;
        float frequency, filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    frequency = math::sqrt(math::sum(distance_sqd));
                    filter = getSoftWindow_<PASS>(freq_cutoff, freq_width, frequency);
                    output_filter[offset + x] = static_cast<real_t>(filter);
                }
            }
        }
    }

    template<typename T>
    void bandPassSoft_(T* output_filter, size3_t shape, float freq_cutoff_1, float freq_cutoff_2,
                       float freq_width_1, float freq_width_2) {
        using real_t = noa::traits::value_type_t<T>;
        size3_t half(shape / size_t{2} + size_t{1});

        float3_t distance_sqd;
        float frequency, filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    frequency = math::sqrt(math::sum(distance_sqd));
                    filter = getSoftWindow_<Type::HIGHPASS>(freq_cutoff_1, freq_width_1, frequency);
                    filter *= getSoftWindow_<Type::LOWPASS>(freq_cutoff_2, freq_width_2, frequency);
                    output_filter[offset + x] = static_cast<real_t>(filter);
                }
            }
        }
    }
}

// Hard edges:
namespace {
    using namespace noa;

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

    template<Type PASS, typename T>
    void singlePassHard_(const T* inputs, T* outputs, size3_t shape, float freq_cutoff) {
        using real_t = noa::traits::value_type_t<T>;
        size3_t half(shape / size_t{2} + size_t{1});

        float3_t distance_sqd;
        float freq_cutoff_sqd = freq_cutoff * freq_cutoff;
        float filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    filter = getHardWindow_<PASS>(freq_cutoff_sqd, math::sum(distance_sqd));
                    outputs[offset + x] = inputs[offset + x] * static_cast<real_t>(filter);
                }
            }
        }
    }

    template<typename T>
    void bandPassHard_(const T* inputs, T* outputs, size3_t shape,
                       float freq_cutoff_1, float freq_cutoff_2) {
        using real_t = noa::traits::value_type_t<T>;
        size3_t half(shape / size_t{2} + size_t{1});

        float3_t distance_sqd;
        float freq_cutoff_sqd_1 = freq_cutoff_1 * freq_cutoff_1;
        float freq_cutoff_sqd_2 = freq_cutoff_2 * freq_cutoff_2;
        float freq_sqd, filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    freq_sqd = math::sum(distance_sqd);
                    filter = getHardWindow_<Type::HIGHPASS>(freq_cutoff_sqd_1, freq_sqd);
                    filter *= getHardWindow_<Type::LOWPASS>(freq_cutoff_sqd_2, freq_sqd);
                    outputs[offset + x] = inputs[offset + x] * static_cast<real_t>(filter);
                }
            }
        }
    }

    template<Type PASS, typename T>
    void singlePassHard_(T* output_filter, size3_t shape, float freq_cutoff) {
        using real_t = noa::traits::value_type_t<T>;
        size3_t half(shape / size_t{2} + size_t{1});

        float3_t distance_sqd;
        float freq_cutoff_sqd = freq_cutoff * freq_cutoff;
        float filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    filter = getHardWindow_<PASS>(freq_cutoff_sqd, math::sum(distance_sqd));
                    output_filter[offset + x] = static_cast<real_t>(filter);
                }
            }
        }
    }

    template<typename T>
    void bandPassHard_(T* output_filter, size3_t shape, float freq_cutoff_1, float freq_cutoff_2) {
        using real_t = noa::traits::value_type_t<T>;
        size3_t half(shape / size_t{2} + size_t{1});

        float3_t distance_sqd;
        float freq_cutoff_sqd_1 = freq_cutoff_1 * freq_cutoff_1;
        float freq_cutoff_sqd_2 = freq_cutoff_2 * freq_cutoff_2;
        float freq_sqd, filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    freq_sqd = math::sum(distance_sqd);
                    filter = getHardWindow_<Type::HIGHPASS>(freq_cutoff_sqd_1, freq_sqd);
                    filter *= getHardWindow_<Type::LOWPASS>(freq_cutoff_sqd_2, freq_sqd);
                    output_filter[offset + x] = static_cast<real_t>(filter);
                }
            }
        }
    }
}

namespace noa::cpu::fft {
    template<typename T>
    void lowpass(const T* inputs, T* outputs, size3_t shape, size_t batches, float freq_cutoff, float freq_width) {
        NOA_PROFILE_FUNCTION();
        size_t elements = elementsFFT(shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            T* output = outputs + elements * batch;

            if (freq_width > 1e-6f)
                singlePassSoft_<Type::LOWPASS>(input, output, shape, freq_cutoff, freq_width);
            else
                singlePassHard_<Type::LOWPASS>(input, output, shape, freq_cutoff);
        }
    }

    template<typename T>
    void lowpass(T* output_lowpass, size3_t shape, float freq_cutoff, float freq_width) {
        NOA_PROFILE_FUNCTION();
        if (freq_width > 1e-6f)
            singlePassSoft_<Type::LOWPASS>(output_lowpass, shape, freq_cutoff, freq_width);
        else
            singlePassHard_<Type::LOWPASS>(output_lowpass, shape, freq_cutoff);
    }

    template<typename T>
    void highpass(const T* inputs, T* outputs, size3_t shape,  size_t batches, float freq_cutoff, float freq_width) {
        NOA_PROFILE_FUNCTION();
        size_t elements = elementsFFT(shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            T* output = outputs + elements * batch;

            if (freq_width > 1e-6f)
                singlePassSoft_<Type::HIGHPASS>(input, output, shape, freq_cutoff, freq_width);
            else
                singlePassHard_<Type::HIGHPASS>(input, output, shape, freq_cutoff);
        }
    }

    template<typename T>
    void highpass(T* output_highpass, size3_t shape, float freq_cutoff, float freq_width) {
        NOA_PROFILE_FUNCTION();
        if (freq_width > 1e-6f)
            singlePassSoft_<Type::HIGHPASS>(output_highpass, shape, freq_cutoff, freq_width);
        else
            singlePassHard_<Type::HIGHPASS>(output_highpass, shape, freq_cutoff);
    }

    template<typename T>
    void bandpass(const T* inputs, T* outputs, size3_t shape, size_t batches,
                  float freq_cutoff_1, float freq_cutoff_2, float freq_width_1, float freq_width_2) {
        NOA_PROFILE_FUNCTION();
        size_t elements = elementsFFT(shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            T* output = outputs + elements * batch;

            if (freq_width_1 > 1e-6f || freq_width_2 > 1e-6f)
                bandPassSoft_(input, output, shape, freq_cutoff_1, freq_cutoff_2, freq_width_1, freq_width_2);
            else
                bandPassHard_(input, output, shape, freq_cutoff_1, freq_cutoff_2);
        }
    }

    template<typename T>
    void bandpass(T* output_bandpass, size3_t shape, float freq_cutoff_1, float freq_cutoff_2,
                  float freq_width_1, float freq_width_2) {
        NOA_PROFILE_FUNCTION();
        if (freq_width_1 > 1e-6f || freq_width_2 > 1e-6f)
            bandPassSoft_(output_bandpass, shape, freq_cutoff_1, freq_cutoff_2, freq_width_1, freq_width_2);
        else
            bandPassHard_(output_bandpass, shape, freq_cutoff_1, freq_cutoff_2);
    }

    #define NOA_INSTANTIATE_FILTERS_(REAL, COMPLEX)                                                         \
    template void lowpass<COMPLEX>(const COMPLEX*, COMPLEX*, size3_t, size_t, float, float);                \
    template void lowpass<REAL>(const REAL*, REAL*, size3_t, size_t, float, float);                         \
    template void lowpass<REAL>(REAL*, size3_t, float, float);                                              \
    template void highpass<COMPLEX>(const COMPLEX*, COMPLEX*, size3_t, size_t, float, float);               \
    template void highpass<REAL>(const REAL*, REAL*, size3_t, size_t, float, float);                        \
    template void highpass<REAL>(REAL*, size3_t, float, float);                                             \
    template void bandpass<COMPLEX>(const COMPLEX*, COMPLEX*, size3_t, size_t, float, float, float, float); \
    template void bandpass<REAL>(const REAL*, REAL*, size3_t, size_t, float, float, float, float);          \
    template void bandpass<REAL>(REAL*, size3_t, float, float, float, float)

    NOA_INSTANTIATE_FILTERS_(float, cfloat_t);
    NOA_INSTANTIATE_FILTERS_(double, cdouble_t);
}
