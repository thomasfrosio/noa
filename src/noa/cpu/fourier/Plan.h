#pragma once

#include <fftw3.h>

#include <mutex>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Types.h"
#include "noa/Environment.h"

namespace Noa::Fourier::Details {
    /// The only thread-safe routine in FFTW is fftw_execute (and the new-array variants). All other routines
    /// (e.g. the planners) should only be called from one thread at a time. Thus, to make our API thread-safe,
    /// calls to FFTW should be protected by this mutex.
    class Mutex {
        static std::mutex mutex;
    public:
        NOA_IH static std::mutex& get() { return mutex; }
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
        NOA_HOST static void push(fftwf_plan plan); /// Hold on to this pointer for a while. Hopefully it helps FFTW.
        NOA_HOST static void push(fftw_plan plan); /// Hold on to this pointer for a while. Hopefully it helps FFTW.
        NOA_HOST static void clearDouble(); /// Destroy the plans in the buffer. This is usually not called.
        NOA_HOST static void clearFloat(); /// Destroy the plans in the buffer. This is usually not called.
    };
}

namespace Noa::Fourier {
    /// Wrapper for FFTW flags.
    enum : uint {
        // -- Planning-rigor flags -- //

        /// Instead of actual measurements of different algorithms, a simple heuristic is used to pick a
        /// (probably sub-optimal) plan quickly. With this flag, the input/output arrays are not overwritten
        /// during planning. This is often enough.
        ESTIMATE = FFTW_ESTIMATE,

        /// Find and optimized plan by actually computing several FFTs and measuring their execution time.
        /// Depending on your machine, this can take some time (often a few seconds).
        MEASURE = FFTW_MEASURE,

        /// Same as Flag::measure, but considers a wider range of algorithms and often produces a "more optimal"
        /// plan (especially for large transforms), but at the expense of several times longer planning time
        /// (especially for large transforms).
        PATIENT = FFTW_PATIENT,

        /// Same as Flag::patient, but considers an even wider range of algorithms, including many that we think
        /// are unlikely to be fast, to produce the most optimal plan but with a substantially increased planning time.
        EXHAUSTIVE = FFTW_EXHAUSTIVE,

        // -- Algorithm-restriction flags -- //

        /// Specifies that an out-of-place transform is allowed to overwrite its input array with arbitrary data;
        /// this can sometimes allow more efficient algorithms to be employed.
        DESTROY_INPUT = FFTW_DESTROY_INPUT,

        /// Specifies that an out-of-place transform must not change its input array. This is ordinarily the default,
        /// except for C2R transforms for which Flag::destroy_input is the default. In the latter cases, passing this
        /// flag will attempt to use algorithms that do not destroy the input, at the expense of worse performance;
        /// for multi-dimensional C2R transforms, however, no input-preserving algorithms are implemented and the
        /// Fourier::Plan will throw an exception.
        PRESERVE_INPUT = FFTW_PRESERVE_INPUT
    };

    /// Sign flags
    enum : int {
        FORWARD = FFTW_FORWARD,
        BACKWARD = FFTW_BACKWARD
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
        static bool is_initialized;
        static int max_threads;
        fftwf_plan m_plan{nullptr};

    private:
        /// Initializes FFTW threads. @warning NOT thread-safe.
        NOA_HOST static void initialize_() {
            if (!fftwf_init_threads())
                NOA_THROW("Failed to initialize the single precision FFTW-threads");
            if (!max_threads) // in case setMaxThreads() was called before initialization, do not override.
                setMaxThreads(Noa::maxThreads());
            is_initialized = true;
        }

        /// Sets the number of threads for the next plans. From IMOD/libfft/fftw_wrap.c. @warning NOT thread-safe
        NOA_HOST static void setThreads_(size3_t shape, int rank) {
            double geom_size = Math::pow(static_cast<double>(getElements(shape)), 1. / rank);
            int threads = static_cast<int>((Math::log(geom_size) / Math::log(2.) - 5.95) * 2.);
            fftwf_plan_with_nthreads(Math::clamp(threads, 1, max_threads));
        }

    public:
        /// Sets the number of threads for the next plans. By default, everything is limited to Noa::maxThreads().
        NOA_HOST static void setMaxThreads(uint threads) {
            if (!threads)
                NOA_THROW("Thread count should be a non-zero positive number, got 0");
            max_threads = static_cast<int>(threads);
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
        NOA_HOST static void clearBuffer() { Details::PlansBuffer::clearFloat(); }

    public:
        /**
         * Creates the plan for a R2C transform (i.e. forward transform).
         * @param[out] input    Input data. Must be allocated.
         * @param[out] output   Output data. Must be allocated.
         * @param shape         Logical {fast, medium, slow} shape of the real data, i.e. the shape of @a input,
         *                      in floats. The dimensionality (i.e. rank) of the transform is equal to @c ndim(shape).
         * @param batch         The number of transforms to compute. Data should be contiguous.
         * @param flag          Any of the Fourier flags. @c Fourier::ESTIMATE is the only flag that guarantees to not
         *                      overwrite the inputs during planning.
         *
         * @note The FFTW planner is intended to be called from a single thread. Even if this constructor
         *       is thread safe, understand that you may be holding for that plan for a long time, which
         *       is undesirable.
         * @warning In-place transforms are allowed (@a input == @a output). In this case, the array requires extra
         *          padding: each row (the fastest dimension) should have an extra float if the dimension is odd, or
         *          two extra floats if it is even. See FFTW documentation.
         */
        NOA_HOST Plan(float* input, cfloat_t* output, size3_t shape, uint batch, uint flag) {
            int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
            int rank = static_cast<int>(getRank(shape));
            {
                std::unique_lock<std::mutex> lock(Details::Mutex::get());
                if (!is_initialized)
                    initialize_();
                if (max_threads > 1)
                    setThreads_(shape, rank);
                if (batch == 1) {
                    m_plan = fftwf_plan_dft_r2c(rank, n + 3 - rank,
                                                input, reinterpret_cast<fftwf_complex*>(output), flag);
                } else {
                    m_plan = fftwf_plan_many_dft_r2c(rank, n + 3 - rank, static_cast<int>(batch),
                                                     input, nullptr, 1,
                                                     static_cast<int>(getElements(shape)),
                                                     reinterpret_cast<fftwf_complex*>(output), nullptr, 1,
                                                     static_cast<int>(getElementsFFT(shape)),
                                                     flag);
                }
            }
            // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
            // configuration supporting a restricted set of transforms.
            if (!m_plan)
                NOA_THROW("Failed to create the R2C plan, with shape {}", shape);
        }

        /**
         * Creates the plan for a C2R transform (i.e. inverse transform).
         * @param[out] input    Input data. Must be allocated.
         * @param[out] output   Output data. Must be allocated.
         * @param shape         Logical {fast, medium, slow} shape of the real data, i.e. the shape of @a output,
         *                      in floats. The dimensionality (i.e. rank) of the transform is equal to @c ndim(shape).
         * @param batch         The number of transforms to compute. Data should be contiguous.
         * @param flag  Any of the Fourier flags.
         *              @c Fourier::ESTIMATE is the only flag that guarantees to not overwrite the inputs during planning.
         *              @c Fourier::PRESERVE_INPUT cannot be used with multi-dimensional out-of-place C2R plans.
         *
         * @note The FFTW planner is intended to be called from a single thread. Even if this constructor
         *       is thread safe, understand that you may be waiting for that plan for a long time, which
         *       is undesirable.
         * @warning In-place transforms are allowed (@a input == @a output). In this case, the array requires extra
         *          padding: each row (the fastest dimension) should have an extra float if the dimension is odd, or
         *          two extra float if it is even. See FFTW documentation.
         */
        NOA_HOST Plan(cfloat_t* input, float* output, size3_t shape, uint batch, uint flag) {
            int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
            int rank = static_cast<int>(getRank(shape));
            {
                std::unique_lock<std::mutex> lock(Details::Mutex::get());
                if (!is_initialized)
                    initialize_();
                if (max_threads > 1)
                    setThreads_(shape, rank);
                if (batch == 1) {
                    m_plan = fftwf_plan_dft_c2r(rank, n + 3 - rank,
                                                reinterpret_cast<fftwf_complex*>(input), output, flag);
                } else {
                    m_plan = fftwf_plan_many_dft_c2r(rank, n + 3 - rank, static_cast<int>(batch),
                                                     reinterpret_cast<fftwf_complex*>(input), nullptr, 1,
                                                     static_cast<int>(getElementsFFT(shape)),
                                                     output, nullptr, 1, static_cast<int>(getElements(shape)),
                                                     flag);
                }
            }
            // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
            // configuration supporting a restricted set of transforms or with the PRESERVE_INPUT flag
            // with a multi-dimensional out-of-place c2r transform.
            if (!m_plan)
                NOA_THROW("Failed to create the C2R plan, with shape {}", shape);
        }

        /**
         * Creates the plan for a C2C transform (i.e. forward/backward complex-to-complex transform).
         * @param[out] input    Input data. Must be allocated.
         * @param[out] output   Output data. Must be allocated.
         * @param shape         Logical {fast, medium, slow} shape of the arrays in cfloat_t.
         *                      The dimensionality (i.e. rank) of the transform is equal to @c ndim(shape).
         * @param sign          Sign of the exponent in the formula that defines the Fourier transform.
         *                      It can be −1 (@c FORWARD) or +1 (@c BACKWARD).
         * @param flag          Any of the planning-rigor and/or algorithm-restriction flags. @c ESTIMATE and
         *                      @c WISDOM_ONLY are the only flag that guarantees to not overwrite the inputs
         *                      during planning.
         *
         * @note The FFTW planner is intended to be called from a single thread. Even if this constructor
         *       is thread safe, understand that you may be waiting for that plan for a long time, which
         *       is undesirable.
         * @note In-place transforms are allowed (@a input == @a output).
         */
        NOA_HOST Plan(cfloat_t* input, cfloat_t* output, size3_t shape, uint batch, int sign, uint flag) {
            int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
            int rank = static_cast<int>(getRank(shape));
            {
                std::unique_lock<std::mutex> lock(Details::Mutex::get());
                if (!is_initialized)
                    initialize_();
                if (max_threads > 1)
                    setThreads_(shape, rank);
                if (batch == 1) {
                    m_plan = fftwf_plan_dft(rank, n + 3 - rank,
                                            reinterpret_cast<fftwf_complex*>(input),
                                            reinterpret_cast<fftwf_complex*>(output),
                                            sign, flag);
                } else {
                    int dist = static_cast<int>(getElements(shape));
                    m_plan = fftwf_plan_many_dft(rank, n + 3 - rank, static_cast<int>(batch),
                                                 reinterpret_cast<fftwf_complex*>(input), nullptr, 1, dist,
                                                 reinterpret_cast<fftwf_complex*>(output), nullptr, 1, dist,
                                                 sign, flag);
                }
            }
            // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
            // configuration supporting a restricted set of transforms.
            if (!m_plan)
                NOA_THROW("Failed to create the C2C plan, with shape {}", shape);
        }

        /**
         * Send the plan to the underlying buffer. It will destroy the plans eventually.
         * @warning Even if the plan is not immediately destroyed, it should not be used since the underlying buffer
         *          is allowed to destroy whatever plan it holds.
         */
        NOA_HOST ~Plan() {
            if (m_plan)
                Details::PlansBuffer::push(m_plan);
        }

        /// Gets the underlying plan.
        NOA_HOST fftwf_plan get() const noexcept { return m_plan; }
        NOA_HOST fftwf_plan get() noexcept { return m_plan; }
    };

    /// See documentation for Plan<float>.
    template<>
    class Plan<double> {
        static bool is_initialized;
        static int max_threads;
        fftw_plan m_plan{nullptr};

    private:
        NOA_HOST static void initialize_() {
            if (!fftw_init_threads())
                NOA_THROW("Failed to initialize the double precision FFTW-threads");
            if (!max_threads)
                setMaxThreads(Noa::maxThreads());
            is_initialized = true;
        }

        NOA_HOST static void setThreads_(size3_t shape, int rank) {
            double geom_size = Math::pow(static_cast<double>(getElements(shape)), 1. / rank);
            int threads = static_cast<int>((Math::log(geom_size) / Math::log(2.) - 5.95) * 2.);
            fftw_plan_with_nthreads(Math::clamp(threads, 1, max_threads));
        }

    public:
        NOA_HOST static void setMaxThreads(uint threads) {
            if (!threads)
                NOA_THROW("Thread count should be a non-zero positive number, got 0");
            max_threads = static_cast<int>(threads);
        }

        NOA_HOST static void cleanup() {
            std::unique_lock<std::mutex> lock(Details::Mutex::get());
            fftw_cleanup();
        }

        NOA_HOST static void clearBuffer() { Details::PlansBuffer::clearFloat(); }

    public:
        NOA_HOST Plan(double* input, cdouble_t* output, size3_t shape, uint batch, uint flag) {
            int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
            int rank = static_cast<int>(getRank(shape));
            {
                std::unique_lock<std::mutex> lock(Details::Mutex::get());
                if (!is_initialized)
                    initialize_();
                if (max_threads > 1)
                    setThreads_(shape, rank);
                if (batch == 1) {
                    m_plan = fftw_plan_dft_r2c(rank, n + 3 - rank,
                                               input, reinterpret_cast<fftw_complex*>(output), flag);
                } else {
                    m_plan = fftw_plan_many_dft_r2c(rank, n + 3 - rank, static_cast<int>(batch),
                                                    input, nullptr, 1,
                                                    static_cast<int>(getElements(shape)),
                                                    reinterpret_cast<fftw_complex*>(output), nullptr, 1,
                                                    static_cast<int>(getElementsFFT(shape)),
                                                    flag);
                }
            }
            if (!m_plan)
                NOA_THROW("Failed to create the R2C plan, with shape {}", shape);
        }

        NOA_HOST Plan(cdouble_t* input, double* output, size3_t shape, uint batch, uint flag) {
            int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
            int rank = static_cast<int>(getRank(shape));
            {
                std::unique_lock<std::mutex> lock(Details::Mutex::get());
                if (!is_initialized)
                    initialize_();
                if (max_threads > 1)
                    setThreads_(shape, rank);
                if (batch == 1) {
                    m_plan = fftw_plan_dft_c2r(rank, n + 3 - rank,
                                                reinterpret_cast<fftw_complex*>(input), output, flag);
                } else {
                    m_plan = fftw_plan_many_dft_c2r(rank, n + 3 - rank, static_cast<int>(batch),
                                                     reinterpret_cast<fftw_complex*>(input), nullptr, 1,
                                                     static_cast<int>(getElementsFFT(shape)),
                                                     output, nullptr, 1, static_cast<int>(getElements(shape)),
                                                     flag);
                }
            }
            if (!m_plan)
                NOA_THROW("Failed to create the C2R plan, with shape {}", shape);
        }

        NOA_HOST Plan(cdouble_t* input, cdouble_t* output, size3_t shape, uint batch, int sign, uint flag) {
            int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
            int rank = static_cast<int>(getRank(shape));
            {
                std::unique_lock<std::mutex> lock(Details::Mutex::get());
                if (!is_initialized)
                    initialize_();
                if (max_threads > 1)
                    setThreads_(shape, rank);
                if (batch == 1) {
                    m_plan = fftw_plan_dft(rank, n + 3 - rank,
                                            reinterpret_cast<fftw_complex*>(input),
                                            reinterpret_cast<fftw_complex*>(output),
                                            sign, flag);
                } else {
                    int dist = static_cast<int>(getElements(shape));
                    m_plan = fftw_plan_many_dft(rank, n + 3 - rank, static_cast<int>(batch),
                                                 reinterpret_cast<fftw_complex*>(input), nullptr, 1, dist,
                                                 reinterpret_cast<fftw_complex*>(output), nullptr, 1, dist,
                                                 sign, flag);
                }
            }
            if (!m_plan)
                NOA_THROW("Failed to create the C2C plan, with shape {}", shape);
        }

        NOA_HOST ~Plan() {
            if (m_plan)
                Details::PlansBuffer::push(m_plan);
        }

        NOA_HOST fftw_plan get() const noexcept { return m_plan; }
        NOA_HOST fftw_plan get() noexcept { return m_plan; }
    };
}
