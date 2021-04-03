#include "noa/cpu/math/Reductions.h"

namespace Noa::Math::Details {
    template<class T>
    void accurateMeanDP(T* input, size_t elements, double* out_sum, double* out_mean) {
        double sum = 0.0;
        double c = 0.0;
        T* end = input + elements;
        while (input < end) {
            auto tmp = static_cast<double>(*input++);
            double t = sum + tmp;
            if (Math::abs(sum) >= Math::abs(tmp)) // Neumaier variation
                c += (sum - t) + tmp;
            else
                c += (tmp - t) + sum;
            sum = t;
        }
        sum += c;
        *out_sum = sum;
        *out_mean = sum / static_cast<double>(elements);
    }
    template void accurateMeanDP<float>(float*, size_t, double*, double*);
    template void accurateMeanDP<double>(double*, size_t, double*, double*);

    template<class T>
    void accurateMeanDP(T* input, size_t elements, cdouble_t* out_sum, cdouble_t* out_mean) {
        cdouble_t sum = 0.0;
        cdouble_t c = 0.0;
        T* end = input + elements;
        while (input < end) {
            auto tmp = static_cast<cdouble_t>(*input++);
            cdouble_t t = sum + tmp;
            if (Math::abs(sum) >= Math::abs(tmp)) // Neumaier variation
                c += (sum - t) + tmp;
            else
                c += (tmp - t) + sum;
            sum = t;
        }
        sum += c;
        *out_sum = sum;
        *out_mean = sum / static_cast<double>(elements);
    }
    template void accurateMeanDP<cfloat_t>(cfloat_t*, size_t, cdouble_t*, cdouble_t*);
    template void accurateMeanDP<cdouble_t>(cdouble_t*, size_t, cdouble_t*, cdouble_t*);

    template<class T>
    void accurateMeanDPAndMinMax(T* input, size_t elements, double* out_sum, double* out_mean, T* out_min, T* out_max) {
        double sum = 0.0;
        double c = 0.0;
        T* end = input + elements;
        *out_min = *input;
        *out_max = *input;
        while (input < end) {
            *out_min = Math::min(*input, *out_min);
            *out_max = Math::max(*input, *out_max);
            auto tmp = static_cast<double>(*input++);
            double t = sum + tmp;
            if (Math::abs(sum) >= Math::abs(tmp)) // Neumaier variation
                c += (sum - t) + tmp;
            else
                c += (tmp - t) + sum;
            sum = t;
        }
        sum += c;
        *out_sum = sum;
        *out_mean = sum / static_cast<double>(elements);
    }
    template void accurateMeanDPAndMinMax<float>(float*, size_t, double*, double*, float*, float*);
    template void accurateMeanDPAndMinMax<double>(double*, size_t, double*, double*, double*, double*);
}
