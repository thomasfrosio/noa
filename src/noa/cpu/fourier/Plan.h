#pragma once

#include <fftw3.h>

#include <mutex>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Types.h"

namespace Noa::Fourier::Details {
    /// The only thread-safe routine in FFTW is fftw_execute (and the new-array variants). All other routines
    /// (e.g. the planner) should only be called from one thread at a time. Thus, calls to FFTW should be
    /// protected by this mutex.
    class Mutex {
        static std::mutex mutex;
    public:
        NOA_IH static std::mutex& get() {
            return mutex;
        }
    };

    /// From FFTW: "Creating a new plan is quick once one exists for a given size".
    /// Just in case it helps FFTW, hold on to plans in a ring buffer (this is similar to what IMOD does).
    /// With the wisdom mechanism, I would be surprised if it actually helps... Wisdom is cumulative, and
    /// is stored in a global, private data structure managed internally by FFTW.
    class PlansBuffer {
        constexpr static uint MAX_PLAN = 10;
        static fftwf_plan m_plans_float[MAX_PLAN];
        static fftw_plan m_plans_double[MAX_PLAN];
        static uint m_index_float;
        static uint m_index_double;

    public:
        static void push(fftwf_plan plan); /// Hold on to this pointer for a while. Hopefully it helps FFTW.
        static void push(fftw_plan plan); /// Hold on to this pointer for a while. Hopefully it helps FFTW.
        static void clear_plans_double(); /// Destroy the plans in the buffer. This is usually not called.
        static void clear_plans_float(); /// Destroy the plans in the buffer. This is usually not called.
    };
}

namespace Noa::Fourier {
    /// Wrapper for FFTW flags.
    struct Flag {
        // -- Planning-rigor flags -- //

        /// Instead of actual measurements of different algorithms, a simple heuristic is used to pick a
        /// (probably sub-optimal) plan quickly. With this flag, the input/output arrays are not overwritten
        /// during planning. This is often enough.
        static constexpr uint estimate{FFTW_ESTIMATE};

        /// Find and optimized plan by actually computing several FFTs and measuring their execution time.
        /// Depending on your machine, this can take some time (often a few seconds).
        static constexpr uint measure{FFTW_MEASURE};

        /// Same as Flag::measure, but considers a wider range of algorithms and often produces a "more optimal"
        /// plan (especially for large transforms), but at the expense of several times longer planning time
        /// (especially for large transforms).
        static constexpr uint patient{FFTW_PATIENT};

        // -- Algorithm-restriction flags -- //

        /// Same as Flag::patient, but considers an even wider range of algorithms, including many that we think
        /// are unlikely to be fast, to produce the most optimal plan but with a substantially increased planning time.
        static constexpr uint exhaustive{FFTW_EXHAUSTIVE};

        /// Specifies that an out-of-place transform is allowed to overwrite its input array with arbitrary data;
        /// this can sometimes allow more efficient algorithms to be employed.
        static constexpr uint destroy_input{FFTW_DESTROY_INPUT};

        /// Specifies that an out-of-place transform must not change its input array. This is ordinarily the default,
        /// except for c2r transforms for which Flag::destroy_input is the default. In the latter cases, passing this
        /// flag will attempt to use algorithms that do not destroy the input, at the expense of worse performance;
        /// for multi-dimensional c2r transforms, however, no input-preserving algorithms are implemented and the
        /// Fourier::Plan will throw an exception.
        static constexpr uint preserve_input{FFTW_PRESERVE_INPUT};
    };

    template<typename T>
    class Plan;

    /**
     * Wrapper managing FFTW plans.
     * @note This object does not keep track of the associated data.
     *       It is the user's responsibility to create, delete and keep track of the input/output arrays.
     */
    template<>
    class Plan<float> {
        fftwf_plan m_plan{nullptr};

    public:
        Plan() = default;

        /**
         * Creates the plan for a c2r transform (i.e. inverse transform).
         * @param[out] output   Output data of shape @a shape. Must be allocated.
         * @param[out] input    Input data. Must be allocated.
         * @param shape         {fast, medium, slow} shape of the real data, i.e. the shape of @a output.
         *                      The dimensionality (i.e. rank) of the transform is equal to @c ndim(shape).
         * @param flag          Any of the available flags from Fourier::Flag. @c Flag::estimate is the only flag
         *                      that guarantees to not overwrite the @a input and @a output data during planning.
         *                      If @c Flag::preserve_input is used with multi-dimensional out-of-place plans.
         *
         * @note The FFTW planner is intended to be called from a single thread. Even if this constructor
         *       is thread safe, understand that you may be waiting for that plan for a long time, which
         *       is undesirable.
         */
        NOA_HOST Plan(float* output, cfloat_t* input, shape_t shape, uint flag) {
            int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
            int rank = static_cast<int>(ndim(shape));
            {
                std::unique_lock<std::mutex> lock(Details::Mutex::get());
                m_plan = fftwf_plan_dft_c2r(rank, n + 3 - rank, reinterpret_cast<fftwf_complex*>(input), output, flag);
            }
            // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
            // configuration supporting a restricted set of transforms or with the Flag::preserve_input
            // flag with a multi-dimensional out-of-place c2r transform.
            if (!m_plan)
                NOA_THROW("Failed to create the c2r plan, with shape {}", shape);
        }

        /**
         * Creates the plan for a r2c transform (i.e. forward transform).
         * @param[out] output   Output data of shape @a shape. Must be allocated.
         * @param[out] input    Input data. Must be allocated.
         * @param shape         {fast, medium, slow} shape of the real data, i.e. the shape of @a output.
         *                      The dimensionality (i.e. rank) of the transform is equal to @c ndim(shape).
         * @param flag          Any of the available flags from Fourier::Flag. @c Flag::estimate is the only flag
         *                      that guarantees to not overwrite the @a input and @a output data during planning.
         *
         * @note The FFTW planner is intended to be called from a single thread. Even if this constructor
         *       is thread safe, understand that you may be holding for that plan for a long time, which
         *       is undesirable.
         */
        NOA_HOST Plan(cfloat_t* output, float* input, size3_t shape, uint flag) {
            int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
            int rank = static_cast<int>(ndim(shape));
            {
                std::unique_lock<std::mutex> lock(Details::Mutex::get());
                m_plan = fftwf_plan_dft_r2c(rank, n + 3 - rank, input, reinterpret_cast<fftwf_complex*>(output), flag);
            }
            // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
            // configuration supporting a restricted set of transforms.
            if (!m_plan)
                NOA_THROW("Failed to create the r2c plan, with shape {}", shape);
        }

        /**
         * Send the plan to the buffer. It will destroy the plans eventually.
         * @warning Even if the plan is not immediately destroyed, it should not be used since the PlansBuffer
         *          is allowed to destroy whatever plan it holds.
         */
        NOA_HOST ~Plan() {
            if (m_plan)
                Details::PlansBuffer::push(m_plan);
        }

        /**
         * FFTW’s planner saves some other persistent data, such as the accumulated wisdom and a list of algorithms
         * available in the current configuration. If you want to deallocate all of that and reset FFTW to the
         * pristine state it was in when you started your program, then call this function.
         *
         * @warning This functions should only be call when all plan are destroyed. All existing plans become
         *          undefined, and one should not attempt to execute them nor to destroy them. You can however
         *          create and execute/destroy new plans, in which case FFTW starts accumulating wisdom
         *          information again.
         */
        NOA_HOST static void cleanup() {
            std::unique_lock<std::mutex> lock(Details::Mutex::get());
            fftwf_cleanup();
        }

        /**
         * Clears the underlying buffer holding on old plans. Calling this function should not be necessary.
         * @note This function will not destroy the current (i.e. active plans).
         * @see PlanBuffer.
         */
        NOA_HOST static void clearBuffer() { Details::PlansBuffer::clear_plans_float(); }

        NOA_HOST fftwf_plan get() const noexcept { return m_plan; }
        NOA_HOST fftwf_plan get() noexcept { return m_plan; }
    };

    template<>
    class Plan<double> {
        fftw_plan m_plan;
    public:
        Plan() = default;

        NOA_HOST Plan(double* output, cdouble_t* input, shape_t shape, uint flag) {
            int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
            int rank = static_cast<int>(ndim(shape));
            {
                std::unique_lock<std::mutex> lock(Details::Mutex::get());
                m_plan = fftw_plan_dft_c2r(rank, n + 3 - rank, reinterpret_cast<fftw_complex*>(input), output, flag);
            }
            if (!m_plan)
                NOA_THROW("Failed to create the c2r plan, with shape {}", shape);
        }

        NOA_HOST Plan(cdouble_t* output, double* input, size3_t shape, uint flag) {
            int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
            int rank = static_cast<int>(ndim(shape));
            {
                std::unique_lock<std::mutex> lock(Details::Mutex::get());
                m_plan = fftw_plan_dft_r2c(rank, n, input, reinterpret_cast<fftw_complex*>(output), flag);
            }
            if (!m_plan)
                NOA_THROW("Failed to create the r2c plan, with shape {}", shape);
        }

        NOA_HOST ~Plan() {
            if (m_plan)
                Details::PlansBuffer::push(m_plan);
        }

        NOA_HOST static void cleanup() {
            std::unique_lock<std::mutex> lock(Details::Mutex::get());
            fftw_cleanup();
        }

        NOA_HOST static void clearBuffer() { Details::PlansBuffer::clear_plans_double(); }

        NOA_HOST fftw_plan get() const noexcept { return m_plan; }
        NOA_HOST fftw_plan get() noexcept { return m_plan; }
    };
}
