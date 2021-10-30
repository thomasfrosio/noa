#include <mutex>

#include "noa/common/Environment.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/fft/Plan.h"

namespace {
    using namespace ::noa;

    // The only thread-safe routine in FFTW is fftw_execute (and the new-array variants). All other routines
    // (e.g. the planners) should only be called from one thread at a time. Thus, to make our API thread-safe,
    // calls to FFTW should be protected by this mutex.
    std::mutex s_mutex;

    // Gets the number of threads given a shape, batches and rank. From IMOD/libfft/fftw_wrap.c.
    int getThreads(size3_t shape, size_t batches, int rank) {
        double geom_size;
        if (rank == 1)
            geom_size = (math::sqrt(static_cast<double>(shape.x * batches)) + static_cast<double>(batches)) / 2.;
        else
            geom_size = math::pow(static_cast<double>(elements(shape)), 1. / rank);
        int threads = static_cast<int>((math::log(geom_size) / math::log(2.) - 5.95) * 2.);
        return math::clamp(threads, 1, all(getNiceShape(shape) == shape) ? 8 : 4);
    }
}

namespace noa::cpu::fft {
    bool Plan<float>::m_is_initialized{false};
    int Plan<float>::m_max_threads{};

    Plan<float>::Plan(float* inputs, cfloat_t* outputs, size3_t shape, size_t batches, uint flag) {
        NOA_PROFILE_FUNCTION();
        int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
        int rank = static_cast<int>(ndim(shape));
        {
            std::unique_lock<std::mutex> lock(s_mutex);
            if (!m_is_initialized)
                initialize_();
            if (m_max_threads > 1)
                setThreads_(shape, batches, rank);
            if (batches == 1) {
                m_plan = fftwf_plan_dft_r2c(rank, n + 3 - rank,
                                            inputs, reinterpret_cast<fftwf_complex*>(outputs), flag);
            } else {
                m_plan = fftwf_plan_many_dft_r2c(rank, n + 3 - rank, static_cast<int>(batches),
                                                 inputs, nullptr, 1,
                                                 static_cast<int>(elements(shape)),
                                                 reinterpret_cast<fftwf_complex*>(outputs), nullptr, 1,
                                                 static_cast<int>(elementsFFT(shape)),
                                                 flag);
            }
        }
        // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
        // configuration supporting a restricted set of transforms.
        if (!m_plan)
            NOA_THROW("Failed to create the R2C plan, with shape {}", shape);
    }

    Plan<float>::Plan(cfloat_t* inputs, float* outputs, size3_t shape, size_t batches, uint flag) {
        NOA_PROFILE_FUNCTION();
        int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
        int rank = static_cast<int>(ndim(shape));
        {
            std::unique_lock<std::mutex> lock(s_mutex);
            if (!m_is_initialized)
                initialize_();
            if (m_max_threads > 1)
                setThreads_(shape, batches, rank);
            if (batches == 1) {
                m_plan = fftwf_plan_dft_c2r(rank, n + 3 - rank,
                                            reinterpret_cast<fftwf_complex*>(inputs), outputs, flag);
            } else {
                m_plan = fftwf_plan_many_dft_c2r(rank, n + 3 - rank, static_cast<int>(batches),
                                                 reinterpret_cast<fftwf_complex*>(inputs), nullptr, 1,
                                                 static_cast<int>(elementsFFT(shape)),
                                                 outputs, nullptr, 1, static_cast<int>(elements(shape)),
                                                 flag);
            }
        }
        // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
        // configuration supporting a restricted set of transforms or with the PRESERVE_INPUT flag
        // with a multi-dimensional out-of-place c2r transform.
        if (!m_plan)
            NOA_THROW("Failed to create the C2R plan, with shape {}", shape);
    }

    Plan<float>::Plan(cfloat_t* inputs, cfloat_t* outputs, size3_t shape, size_t batches, Sign sign, uint flag) {
        NOA_PROFILE_FUNCTION();
        int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
        int rank = static_cast<int>(ndim(shape));
        {
            std::unique_lock<std::mutex> lock(s_mutex);
            if (!m_is_initialized)
                initialize_();
            if (m_max_threads > 1)
                setThreads_(shape, batches, rank);
            if (batches == 1) {
                m_plan = fftwf_plan_dft(rank, n + 3 - rank,
                                        reinterpret_cast<fftwf_complex*>(inputs),
                                        reinterpret_cast<fftwf_complex*>(outputs),
                                        sign, flag);
            } else {
                int dist = static_cast<int>(elements(shape));
                m_plan = fftwf_plan_many_dft(rank, n + 3 - rank, static_cast<int>(batches),
                                             reinterpret_cast<fftwf_complex*>(inputs), nullptr, 1, dist,
                                             reinterpret_cast<fftwf_complex*>(outputs), nullptr, 1, dist,
                                             sign, flag);
            }
        }
        // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
        // configuration supporting a restricted set of transforms.
        if (!m_plan)
            NOA_THROW("Failed to create the C2C plan, with shape {}", shape);
    }

    Plan<float>::~Plan() {
        NOA_PROFILE_FUNCTION();
        if (m_plan) {
            std::unique_lock<std::mutex> lock(s_mutex);
            fftwf_destroy_plan(m_plan);
        }
    }

    void Plan<float>::setMaxThreads(uint threads) {
        if (!threads)
            NOA_THROW("Thread count should be a non-zero positive number, got 0");
        m_max_threads = static_cast<int>(threads);
    }

    void Plan<float>::cleanup() {
        NOA_PROFILE_FUNCTION();
        std::unique_lock<std::mutex> lock(s_mutex);
        fftwf_cleanup();
    }

    void Plan<float>::initialize_() {
        NOA_PROFILE_FUNCTION();
        if (!fftwf_init_threads())
            NOA_THROW("Failed to initialize the single precision FFTW-threads");
        if (!m_max_threads) // in case setMaxThreads() was called before initialization, do not override.
            setMaxThreads(noa::maxThreads());
        m_is_initialized = true;
    }

    void Plan<float>::setThreads_(size3_t shape, size_t batches, int rank) {
        NOA_PROFILE_FUNCTION();
        fftwf_plan_with_nthreads(noa::math::min(getThreads(shape, batches, rank), m_max_threads));
    }
}

namespace noa::cpu::fft {
    bool Plan<double>::m_is_initialized{false};
    int Plan<double>::m_max_threads{};

    Plan<double>::Plan(double* inputs, cdouble_t* outputs, size3_t shape, size_t batches, uint flag) {
        NOA_PROFILE_FUNCTION();
        int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
        int rank = static_cast<int>(ndim(shape));
        {
            std::unique_lock<std::mutex> lock(s_mutex);
            if (!m_is_initialized)
                initialize_();
            if (m_max_threads > 1)
                setThreads_(shape, batches, rank);
            if (batches == 1) {
                m_plan = fftw_plan_dft_r2c(rank, n + 3 - rank,
                                           inputs, reinterpret_cast<fftw_complex*>(outputs), flag);
            } else {
                m_plan = fftw_plan_many_dft_r2c(rank, n + 3 - rank, static_cast<int>(batches),
                                                inputs, nullptr, 1,
                                                static_cast<int>(elements(shape)),
                                                reinterpret_cast<fftw_complex*>(outputs), nullptr, 1,
                                                static_cast<int>(elementsFFT(shape)),
                                                flag);
            }
        }
        if (!m_plan)
            NOA_THROW("Failed to create the R2C plan, with shape {}", shape);
    }

    Plan<double>::Plan(cdouble_t* inputs, double* outputs, size3_t shape, size_t batches, uint flag) {
        NOA_PROFILE_FUNCTION();
        int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
        int rank = static_cast<int>(ndim(shape));
        {
            std::unique_lock<std::mutex> lock(s_mutex);
            if (!m_is_initialized)
                initialize_();
            if (m_max_threads > 1)
                setThreads_(shape, batches, rank);
            if (batches == 1) {
                m_plan = fftw_plan_dft_c2r(rank, n + 3 - rank,
                                           reinterpret_cast<fftw_complex*>(inputs), outputs, flag);
            } else {
                m_plan = fftw_plan_many_dft_c2r(rank, n + 3 - rank, static_cast<int>(batches),
                                                reinterpret_cast<fftw_complex*>(inputs), nullptr, 1,
                                                static_cast<int>(elementsFFT(shape)),
                                                outputs, nullptr, 1, static_cast<int>(elements(shape)),
                                                flag);
            }
        }
        if (!m_plan)
            NOA_THROW("Failed to create the C2R plan, with shape {}", shape);
    }

    Plan<double>::Plan(cdouble_t* inputs, cdouble_t* outputs, size3_t shape, size_t batches, Sign sign, uint flag) {
        NOA_PROFILE_FUNCTION();
        int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
        int rank = static_cast<int>(ndim(shape));
        {
            std::unique_lock<std::mutex> lock(s_mutex);
            if (!m_is_initialized)
                initialize_();
            if (m_max_threads > 1)
                setThreads_(shape, batches, rank);
            if (batches == 1) {
                m_plan = fftw_plan_dft(rank, n + 3 - rank,
                                       reinterpret_cast<fftw_complex*>(inputs),
                                       reinterpret_cast<fftw_complex*>(outputs),
                                       sign, flag);
            } else {
                int dist = static_cast<int>(elements(shape));
                m_plan = fftw_plan_many_dft(rank, n + 3 - rank, static_cast<int>(batches),
                                            reinterpret_cast<fftw_complex*>(inputs), nullptr, 1, dist,
                                            reinterpret_cast<fftw_complex*>(outputs), nullptr, 1, dist,
                                            sign, flag);
            }
        }
        if (!m_plan)
            NOA_THROW("Failed to create the C2C plan, with shape {}", shape);
    }

    Plan<double>::~Plan() {
        NOA_PROFILE_FUNCTION();
        if (m_plan) {
            std::unique_lock<std::mutex> lock(s_mutex);
            fftw_destroy_plan(m_plan);
        }
    }

    void Plan<double>::initialize_() {
        NOA_PROFILE_FUNCTION();
        if (!fftw_init_threads())
            NOA_THROW("Failed to initialize the double precision FFTW-threads");
        if (!m_max_threads)
            setMaxThreads(noa::maxThreads());
        m_is_initialized = true;
    }

    void Plan<double>::setThreads_(size3_t shape, size_t batches, int rank) {
        NOA_PROFILE_FUNCTION();
        fftw_plan_with_nthreads(noa::math::min(getThreads(shape, batches, rank), m_max_threads));
    }

    void Plan<double>::setMaxThreads(uint threads) {
        if (!threads)
            NOA_THROW("Thread count should be a non-zero positive number, got 0");
        m_max_threads = static_cast<int>(threads);
    }

    void Plan<double>::cleanup() {
        NOA_PROFILE_FUNCTION();
        std::unique_lock<std::mutex> lock(s_mutex);
        fftw_cleanup();
    }
}
