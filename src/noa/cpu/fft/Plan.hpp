#pragma once

#include <fftw3.h>

#include "noa/core/Types.hpp"

namespace noa::cpu::fft {
    // Returns the optimum even size, greater or equal than "size".
    // An optimum size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d)*(11^e)*(13^f), with e + f = 0 or 1.
    // If "size" is >16896, this function will simply return the next even number and will not necessarily
    // satisfy the aforementioned requirements.
    i64 fast_size(i64 size);

    // Returns the optimum BDHW logical shape.
    template<typename T, size_t N>
    Shape<T, N> fast_shape(Shape<T, N> shape) noexcept {
        for (size_t i = 1; i < N; ++i) // ignore batch dimension
            if (shape[i] > 1)
                shape[i] = static_cast<T>(fast_size(static_cast<i64>(shape[i])));
        return shape;
    }

    // Wrapper for FFTW flags.
    enum Flag : u32 {
        // -- Planning-rigor field -- //

        // Instead of actual measurements of different algorithms, a simple heuristic is used to pick a
        // (probably suboptimal) plan quickly. With this flag, the input/output arrays are not overwritten
        // during planning. This is often enough.
        ESTIMATE = FFTW_ESTIMATE,

        // Find and optimized plan by actually computing several FFTs and measuring their execution time.
        // Depending on your machine, this can take some time (often a few seconds).
        MEASURE = FFTW_MEASURE,

        // Same as \a MEASURE, but considers a wider range of algorithms and often produces a "more optimal"
        // plan (especially for large transforms), but at the expense of several times longer planning time
        // (especially for large transforms).
        PATIENT = FFTW_PATIENT,

        // Same as \a PATIENT, but considers an even wider range of algorithms, including many that we think
        // are unlikely to be fast, to produce the most optimal plan but with a substantially increased planning time.
        EXHAUSTIVE = FFTW_EXHAUSTIVE,

        // -- Algorithm-restriction field -- //

        // Specifies that an out-of-place transform is allowed to overwrite its input array with arbitrary data;
        // this can sometimes allow more efficient algorithms to be employed.
        DESTROY_INPUT = FFTW_DESTROY_INPUT,

        // Specifies that an out-of-place transform must not change its input array. This is ordinarily the default,
        // except for C2R transforms for which DESTROY_INPUT is the default. In the latter cases, passing this
        // flag will attempt to use algorithms that do not destroy the input, at the expense of worse performance;
        // for multidimensional C2R transforms, however, no input-preserving algorithms are implemented and the
        // fft::Plan will throw an exception.
        PRESERVE_INPUT = FFTW_PRESERVE_INPUT
    };

    // Wrapper managing FFTW plans.
    // NOTE: This object does not keep track of the associated data.
    //       It is the user's responsibility to create, delete and keep track of the input/output arrays.
    // NOTE: For R2C/C2R transforms, the 2D/3D arrays should be in the rightmost order for best performance since the
    //       library currently always assumes the non-redundant dimension is the rows, i.e. the rightmost dimension.
    //       The current exception is for column vectors, which are explicitly detected and supported.
    // NOTE: In-place R2C/C2R transforms are allowed (input == output). In this case, the input requires extra
    //       padding: the rows should have an extra float if the dimension is odd, or two extra floats if it is even.
    //       If strides are provided, this padding should be reflected in the strides.
    // NOTE: The FFTW planner is intended to be called from a single thread. Thus, even if the constructors of this
    //       class are thread safe, understand that you may be holding for a long time and other threads might have
    //       to wait, which might be undesirable.
    // NOTE: For C2C, column-major is also supported.
    //       If strides are not provided, arrays should be C-contiguous.
    //       Any of the FFT flags is accepted.
    template<typename T>
    class Plan {
    public:
        static_assert(noa::traits::is_any_v<T, f32, f64>);
        static constexpr bool IS_SINGLE_PRECISION = std::is_same_v<T, f32>;
        using fftw_plan_type = std::conditional_t<IS_SINGLE_PRECISION, fftwf_plan, fftw_plan>;
        using fftw_complex_type = std::conditional_t<IS_SINGLE_PRECISION, fftwf_complex, fftw_complex>;

    public: // R2C
        Plan(T* input, Complex<T>* output, const Shape4<i64>& shape, u32 flag, i64 max_threads);

        Plan(T* input, const Strides4<i64>& input_strides,
             Complex<T>* output, const Strides4<i64>& output_strides,
             const Shape4<i64>& shape, u32 flag, i64 max_threads);

    public: // C2R
        Plan(Complex<T>* input, T* output, const Shape4<i64>& shape, u32 flag, i64 max_threads);

        Plan(Complex<T>* input, const Strides4<i64>& input_strides,
             T* output, const Strides4<i64>& output_strides,
             const Shape4<i64>& shape, u32 flag, i64 max_threads);

    public: // C2C
        Plan(Complex<T>* input, Complex<T>* output, const Shape4<i64>& shape,
             noa::fft::Sign sign, u32 flag, i64 max_threads);

        Plan(Complex<T>* input, const Strides4<i64>& input_strides,
             Complex<T>* output, const Strides4<i64>& output_strides,
             const Shape4<i64>& shape, noa::fft::Sign sign, u32 flag, i64 max_threads);

    public:
        [[nodiscard]] fftw_plan_type get() const noexcept { return m_plan; }

        Plan(const Plan<T>&) = delete;
        Plan<T>& operator=(const Plan<T>&) = delete;
        Plan(Plan<T>&&) noexcept = default;
        Plan<T>& operator=(Plan<T>&&) noexcept = default;
        ~Plan();

        // The plans are cached and FFTW caches accumulated wisdom and a list of algorithms available in the current
        // configuration. If you want to deallocate all of that and reset to the pristine state it was in when you
        // started your program, then call this function.
        // This functions should only be call when all plans are destroyed. All existing plans become
        // undefined, and one should not attempt to execute them nor to destroy them. You can however
        // create and execute/destroy new plans.
        static void cleanup();

    private:
        fftw_plan_type m_plan{};
    };
}
