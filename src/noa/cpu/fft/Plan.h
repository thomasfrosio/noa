/// \file noa/cpu/fft/Plan.h
/// \brief The single and double precision FFT plans.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <fftw3.h>

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::cpu::fft {
    using namespace noa::fft;

    /// Returns the optimum even size, greater or equal than \p size.
    /// \note A optimum size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d)*(11^e)*(13^f), with e + f = 0 or 1.
    /// \note If \p size is >16896, this function will simply return the next even number and will not necessarily
    ///       satisfy the aforementioned requirements.
    NOA_HOST size_t fastSize(size_t size);

    /// Returns the optimum shape.
    /// \note Dimensions of size 0 or 1 are ignored, e.g. {51,51,1} is rounded up to {52,52,1}.
    NOA_IH size3_t fastShape(size3_t shape) {
        return {shape.x > 1 ? fastSize(shape.x) : shape.x,
                shape.y > 1 ? fastSize(shape.y) : shape.y,
                shape.z > 1 ? fastSize(shape.z) : shape.z};
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
        /// \param[in,out] inputs   On the \b host. Input real data. Must be allocated.
        /// \param[out] outputs     On the \b host. Output non-redundant non-centered FFT. Must be allocated.
        /// \param shape            Logical {fast, medium, slow} shape, in elements, ignoring the batches.
        /// \param batches          The number of contiguous transforms to compute.
        /// \param flag             Any of the FFT flags. Flag::ESTIMATE is the only flag that guarantees
        ///                         to not overwrite the inputs during planning.
        ///
        /// \note The FFTW planner is intended to be called from a single thread. Even if this constructor
        ///       is thread safe, understand that you may be holding for that plan for a long time, which
        ///       might be undesirable.
        /// \note In-place transforms are allowed (\p inputs == \p outputs). In this case, \p inputs requires extra
        ///       padding: each row (the innermost dimension) should have an extra float if the dimension is odd, or
        ///       two extra floats if it is even. See FFTW documentation for more details.
        NOA_HOST Plan(T* inputs, Complex<T>* outputs, size3_t shape, size_t batches, uint flag);

        /// Creates the plan for an inverse C2R transform.
        /// \param[in,out] inputs   On the \b host. Input non-redundant non-centered FFT. Must be allocated.
        /// \param[out] outputs     On the \b host. Output real data. Must be allocated.
        /// \param shape            Logical {fast, medium, slow} shape, in elements, ignoring the batches.
        /// \param batches          The number of contiguous transforms to compute.
        /// \param flag             Any of the FFT flags.
        ///                 Flag::ESTIMATE is the only flag that guarantees to not overwrite the inputs during planning.
        ///                 Flag::PRESERVE_INPUT cannot be used with multi-dimensional out-of-place C2R plans.
        ///
        /// \note The FFTW planner is intended to be called from a single thread. Even if this constructor
        ///       is thread safe, understand that you may be waiting for that plan for a long time, which
        ///       might be undesirable.
        /// \note In-place transforms are allowed (\p inputs == \p outputs). In this case, \p outputs requires extra
        ///       padding: each row (the fastest dimension) should have an extra float if the dimension is odd, or
        ///       two extra float if it is even. See FFTW documentation for more details.
        NOA_HOST Plan(Complex<T>* inputs, T* outputs, size3_t shape, size_t batches, uint flag);

        /// Creates the plan for a C2C transform (i.e. forward/backward complex-to-complex transform).
        /// \param[in,out] inputs   On the \b host. Input complex data. Must be allocated.
        /// \param[out] outputs     On the \b host. Output non-redundant non-centered FFT. Must be allocated.
        /// \param shape            Logical {fast, medium, slow} shape, in elements, ignoring the batches.
        /// \param batches          The number of transforms to compute. Batches should be contiguous to each other.
        /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
        ///                         It can be −1 (FORWARD) or +1 (BACKWARD).
        /// \param flag             Any of the planning-rigor and/or algorithm-restriction flags. \c fft::ESTIMATE and
        ///                         \c fft::WISDOM_ONLY are the only flag that guarantees to not overwrite the inputs
        ///                         during planning.
        ///
        /// \note The FFTW planner is intended to be called from a single thread. Even if this constructor
        ///       is thread safe, understand that you may be waiting for that plan for a long time, which
        ///       might be undesirable.
        /// \note In-place transforms are allowed (\p inputs == \p outputs).
        NOA_HOST Plan(Complex<T>* inputs, Complex<T>* outputs, size3_t shape, size_t batches, Sign sign, uint flag);

        /// Destroys the underlying plan.
        NOA_HOST ~Plan();

        /// Gets the underlying FFTW3 plan.
        NOA_HOST [[nodiscard]] fftw_plan_t get() const noexcept { return m_plan; }

    public: // Static functions
        /// Sets the maximum number of threads for the next plans. By default, it is limited to noa::maxThreads().
        NOA_HOST static void setMaxThreads(uint threads);

        ///  FFTW’s planner saves some other persistent data, such as the accumulated wisdom and a list of algorithms
        ///  available in the current configuration. If you want to deallocate all of that and reset FFTW to the
        ///  pristine state it was in when you started your program, then call this function.
        ///  \note This functions should only be call when all plan are destroyed. All existing plans become
        ///        undefined, and one should not attempt to execute them nor to destroy them. You can however
        ///        create and execute/destroy new plans, in which case FFTW starts accumulating wisdom
        ///        information again.
        NOA_HOST static void cleanup();

    private:
        static inline bool s_is_initialized;
        static inline int s_max_threads;
        fftw_plan_t m_plan{};

    private:
        NOA_HOST static void initialize_();
        NOA_HOST static void setThreads_(size3_t shape, size_t batches, int rank);
    };
}
