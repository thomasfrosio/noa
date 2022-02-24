/// \file noa/cpu/fft/Plan.h
/// \brief The single and double precision FFT plans.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <fftw3.h>

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::fft {
    using namespace noa::fft;

    /// Returns the optimum even size, greater or equal than \p size.
    /// \note A optimum size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d)*(11^e)*(13^f), with e + f = 0 or 1.
    /// \note If \p size is >16896, this function will simply return the next even number and will not necessarily
    ///       satisfy the aforementioned requirements.
    NOA_HOST size_t fastSize(size_t size);

    /// Returns the optimum rightmost shape.
    /// \note Dimensions of size 0 or 1 are ignored, e.g. {1,51,51} is rounded up to {1,52,52}.
    template<typename T>
    NOA_IH Int3<T> fastShape(Int3<T> shape) {
        return {shape[0] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[0]))) : shape[0],
                shape[1] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[1]))) : shape[1],
                shape[2] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[2]))) : shape[2]};
    }

    /// Returns the optimum rightmost shape.
    /// \note Dimensions of size 0 or 1 are ignored as well as the leftmost dimension, e.g. {1,1,51,51}
    ///       is rounded up to {1,1,52,52}.
    template<typename T>
    NOA_IH Int4<T> fastShape(Int4<T> shape) {
        return {shape[0],
                shape[1] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[1]))) : shape[1],
                shape[2] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[2]))) : shape[2],
                shape[3] > 1 ? static_cast<T>(fastSize(static_cast<size_t>(shape[3]))) : shape[3]};
    }

    /// Wrapper for FFTW flags.
    enum Flag : uint {
        // -- Planning-rigor field -- //

        /// Instead of actual measurements of different algorithms, a simple heuristic is used to pick a
        /// (probably sub-optimal) plan quickly. With this flag, the input/output arrays are not overwritten
        /// during planning. This is often enough.
        ESTIMATE = FFTW_ESTIMATE,

        /// Find and optimized plan by actually computing several FFTs and measuring their execution time.
        /// Depending on your machine, this can take some time (often a few seconds).
        MEASURE = FFTW_MEASURE,

        /// Same as \a MEASURE, but considers a wider range of algorithms and often produces a "more optimal"
        /// plan (especially for large transforms), but at the expense of several times longer planning time
        /// (especially for large transforms).
        PATIENT = FFTW_PATIENT,

        /// Same as \a PATIENT, but considers an even wider range of algorithms, including many that we think
        /// are unlikely to be fast, to produce the most optimal plan but with a substantially increased planning time.
        EXHAUSTIVE = FFTW_EXHAUSTIVE,

        // -- Algorithm-restriction field -- //

        /// Specifies that an out-of-place transform is allowed to overwrite its input array with arbitrary data;
        /// this can sometimes allow more efficient algorithms to be employed.
        DESTROY_INPUT = FFTW_DESTROY_INPUT,

        /// Specifies that an out-of-place transform must not change its input array. This is ordinarily the default,
        /// except for C2R transforms for which DESTROY_INPUT is the default. In the latter cases, passing this
        /// flag will attempt to use algorithms that do not destroy the input, at the expense of worse performance;
        /// for multi-dimensional C2R transforms, however, no input-preserving algorithms are implemented and the
        /// fft::Plan will throw an exception.
        PRESERVE_INPUT = FFTW_PRESERVE_INPUT
    };

    /// Wrapper managing FFTW plans.
    /// \note This object does not keep track of the associated data.
    ///       It is the user's responsibility to create, delete and keep track of the input/output arrays.
    template<typename T>
    class Plan {
    public: // typedefs
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
        static constexpr bool IS_SINGLE_PRECISION = std::is_same_v<T, float>;
        using fftw_plan_t = std::conditional_t<IS_SINGLE_PRECISION, fftwf_plan, fftw_plan>;
        using fftw_complex_t = std::conditional_t<IS_SINGLE_PRECISION, fftwf_complex, fftw_complex>;

    public: // Constructors and member functions
        /// Creates the plan for a forward R2C transform.
        /// \param[in,out] input    On the \b host. Input real data. Must be allocated.
        /// \param[out] output      On the \b host. Output non-redundant non-centered FFT. Must be allocated.
        /// \param shape            Rightmost shape.
        /// \param flag             Any of the FFT flags. Flag::ESTIMATE is the only flag that guarantees
        ///                         to not overwrite the input during planning.
        /// \param[in,out] stream   Stream on which to enqueue this function.
        ///
        /// \note Depending on the stream, this function may be asynchronous and may return before completion.
        /// \note The FFTW planner is intended to be called from a single thread. Even if this constructor
        ///       is thread safe, understand that you may be holding for that plan for a long time and other
        ///       streams (i.e. threads) might have to wait, which might be undesirable.
        /// \note In-place transforms are allowed (\p input == \p output). In this case, \p input requires extra
        ///       padding: the innermost dimension should have an extra float if the dimension is odd, or
        ///       two extra floats if it is even.
        NOA_HOST Plan(T* input, noa::Complex<T>* output, size4_t shape, uint flag, Stream& stream) {
            size_t max_threads = stream.threads();
            stream.enqueue([=]() {
                this->m_plan = Plan<T>::getR2C_(input, output, shape, flag, max_threads);
            });
        }

        /// Creates the plan for a forward R2C transform.
        /// \param[in,out] input    On the \b host. Input real data. Must be allocated.
        /// \param input_stride     Rightmost strides of \p input.
        /// \param[out] output      On the \b host. Output non-redundant non-centered FFT. Must be allocated.
        /// \param output_stride    Rightmost strides of \p output.
        /// \param shape            Rightmost shape.
        /// \param flag             Any of the FFT flags. Flag::ESTIMATE is the only flag that guarantees
        ///                         to not overwrite the input during planning.
        /// \param[in,out] stream   Stream on which to enqueue this function.
        ///
        /// \note Depending on the stream, this function may be asynchronous and may return before completion.
        /// \note The FFTW planner is intended to be called from a single thread. Even if this constructor
        ///       is thread safe, understand that you may be holding for that plan for a long time and other
        ///       streams (i.e. threads) might have to wait, which might be undesirable.
        /// \note In-place transforms are allowed (\p input == \p output). In this case, \p input requires extra
        ///       padding, which should be encoded in \p input_stride. Each row should have at least an extra float
        ///       if the dimension is odd, or at least two extra floats if it is even.
        NOA_HOST Plan(T* input, size4_t input_stride,
                      noa::Complex<T>* output, size4_t output_stride,
                      size4_t shape, uint flag, Stream& stream) {
            size_t max_threads = stream.threads();
            stream.enqueue([=]() {
                this->m_plan = Plan<T>::getR2C_(input, input_stride, output, output_stride,
                                                shape, flag, max_threads);
            });
        }

        /// Creates the plan for an inverse C2R transform.
        /// \param[in,out] input    On the \b host. Input non-redundant non-centered FFT. Must be allocated.
        /// \param[out] output      On the \b host. Output real data. Must be allocated.
        /// \param shape            Rightmost shape.
        /// \param flag             Any of the FFT flags. Flag::ESTIMATE is the only flag that guarantees to not
        ///                         overwrite the input during planning. Flag::PRESERVE_INPUT cannot be used with
        ///                         multi-dimensional out-of-place C2R plans.
        /// \param[in,out] stream   Stream on which to enqueue this function.
        ///
        /// \note Depending on the stream, this function may be asynchronous and may return before completion.
        /// \note The FFTW planner is intended to be called from a single thread. Even if this constructor
        ///       is thread safe, understand that you may be holding for that plan for a long time and other
        ///       streams (i.e. threads) might have to wait, which might be undesirable.
        /// \note In-place transforms are allowed (\p input == \p output). In this case, \p output requires extra
        ///       padding: the innermost dimension should have an extra float if the dimension is odd, or
        ///       two extra float if it is even. See FFTW documentation for more details.
        NOA_HOST Plan(noa::Complex<T>* input, T* output, size4_t shape, uint flag, Stream& stream) {
            size_t max_threads = stream.threads();
            stream.enqueue([=]() {
                this->m_plan = Plan<T>::getC2R_(input, output, shape, flag, max_threads);
            });
        }

        /// Creates the plan for an inverse C2R transform.
        /// \param[in,out] input    On the \b host. Input non-redundant non-centered FFT. Must be allocated.
        /// \param input_stride     Rightmost strides of \p input.
        /// \param[out] output      On the \b host. Output real data. Must be allocated.
        /// \param output_stride    Rightmost strides of \p output.
        /// \param shape            Rightmost shape.
        /// \param flag             Any of the FFT flags. Flag::ESTIMATE is the only flag that guarantees to not
        ///                         overwrite the input during planning. Flag::PRESERVE_INPUT cannot be used with
        ///                         multi-dimensional out-of-place C2R plans.
        /// \param[in,out] stream   Stream on which to enqueue this function.
        ///
        /// \note Depending on the stream, this function may be asynchronous and may return before completion.
        /// \note The FFTW planner is intended to be called from a single thread. Even if this constructor
        ///       is thread safe, understand that you may be holding for that plan for a long time and other
        ///       streams (i.e. threads) might have to wait, which might be undesirable.
        /// \note In-place transforms are allowed (\p input == \p output). In this case, \p input requires extra
        ///       padding, which should be encoded in \p input_stride. Each row should have at least an extra float
        ///       if the dimension is odd, or at least two extra floats if it is even.
        NOA_HOST Plan(noa::Complex<T>* input, size4_t input_stride,
                      T* output, size4_t output_stride,
                      size4_t shape, uint flag, Stream& stream) {
            size_t max_threads = stream.threads();
            stream.enqueue([=]() {
                this->m_plan = Plan<T>::getC2R_(input, input_stride, output, output_stride,
                                                shape, flag, max_threads);
            });
        }

        /// Creates the plan for a C2C transform (i.e. forward/backward complex-to-complex transform).
        /// \param[in,out] input    On the \b host. Input complex data. Must be allocated.
        /// \param[out] output      On the \b host. Output non-redundant non-centered FFT. Must be allocated.
        /// \param shape            Rightmost shape.
        /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
        ///                         It can be −1 (FORWARD) or +1 (BACKWARD).
        /// \param flag             Any of the planning-rigor and/or algorithm-restriction flags. \c fft::ESTIMATE and
        ///                         \c fft::WISDOM_ONLY are the only flag that guarantees to not overwrite the input
        ///                         during planning.
        /// \param[in,out] stream   Stream on which to enqueue this function.
        ///
        /// \note Depending on the stream, this function may be asynchronous and may return before completion.
        /// \note The FFTW planner is intended to be called from a single thread. Even if this constructor
        ///       is thread safe, understand that you may be holding for that plan for a long time and other
        ///       streams (i.e. threads) might have to wait, which might be undesirable.
        /// \note In-place transforms are allowed (\p input == \p output).
        NOA_HOST Plan(noa::Complex<T>* input, noa::Complex<T>* output, size4_t shape,
                      Sign sign, uint flag, Stream& stream) {
            size_t max_threads = stream.threads();
            stream.enqueue([=]() {
                this->m_plan = Plan<T>::getC2C_(input, output, shape, sign, flag, max_threads);
            });
        }

        /// Creates the plan for a C2C transform (i.e. forward/backward complex-to-complex transform).
        /// \param[in,out] input    On the \b host. Input complex data. Must be allocated.
        /// \param input_stride     Rightmost strides of \p input.
        /// \param[out] output      On the \b host. Output non-redundant non-centered FFT. Must be allocated.
        /// \param output_stride    Rightmost strides of \p output.
        /// \param shape            Rightmost shape.
        /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
        ///                         It can be −1 (FORWARD) or +1 (BACKWARD).
        /// \param flag             Any of the planning-rigor and/or algorithm-restriction flags. \c fft::ESTIMATE and
        ///                         \c fft::WISDOM_ONLY are the only flag that guarantees to not overwrite the input
        ///                         during planning.
        /// \param[in,out] stream   Stream on which to enqueue this function.
        ///
        /// \note Depending on the stream, this function may be asynchronous and may return before completion.
        /// \note The FFTW planner is intended to be called from a single thread. Even if this constructor
        ///       is thread safe, understand that you may be holding for that plan for a long time and other
        ///       streams (i.e. threads) might have to wait, which might be undesirable.
        /// \note In-place transforms are allowed (\p input == \p output).
        NOA_HOST Plan(noa::Complex<T>* input, size4_t input_stride,
                      noa::Complex<T>* output, size4_t output_stride,
                      size4_t shape, Sign sign, uint flag, Stream& stream) {
            size_t max_threads = stream.threads();
            stream.enqueue([=]() {
                this->m_plan = Plan<T>::getC2C_(input, input_stride, output, output_stride,
                                                shape, sign, flag, max_threads);
            });
        }

        /// Destroys the underlying plan.
        NOA_HOST ~Plan();

        /// Gets the underlying FFTW3 plan.
        NOA_HOST [[nodiscard]] fftw_plan_t get() const noexcept { return m_plan; }

    public: // Static functions
        ///  FFTW’s planner saves some other persistent data, such as the accumulated wisdom and a list of algorithms
        ///  available in the current configuration. If you want to deallocate all of that and reset FFTW to the
        ///  pristine state it was in when you started your program, then call this function.
        ///  \note This functions should only be call when all plan are destroyed. All existing plans become
        ///        undefined, and one should not attempt to execute them nor to destroy them. You can however
        ///        create and execute/destroy new plans, in which case FFTW starts accumulating wisdom
        ///        information again.
        NOA_HOST static void cleanup();

    private:
        fftw_plan_t m_plan{};

    private:
        NOA_HOST static fftw_plan_t getR2C_(T* input, size4_t input_stride,
                                            noa::Complex<T>* output, size4_t output_stride,
                                            size4_t shape, uint flag, size_t threads);
        NOA_HOST static fftw_plan_t getR2C_(T* input, noa::Complex<T>* output,
                                            size4_t shape, uint flag, size_t threads);
        NOA_HOST static fftw_plan_t getC2R_(noa::Complex<T>* input, size4_t input_stride,
                                            T* output, size4_t output_stride,
                                            size4_t shape, uint flag, size_t threads);
        NOA_HOST static fftw_plan_t getC2R_(noa::Complex<T>* input, T* output,
                                            size4_t shape, uint flag, size_t threads);
        NOA_HOST static fftw_plan_t getC2C_(noa::Complex<T>* input, size4_t input_stride,
                                            noa::Complex<T>* output, size4_t output_stride,
                                            size4_t shape, Sign sign, uint flag, size_t threads);
        NOA_HOST static fftw_plan_t getC2C_(noa::Complex<T>* input, noa::Complex<T>* output,
                                            size4_t shape, Sign sign, uint flag, size_t threads);
    };
}
