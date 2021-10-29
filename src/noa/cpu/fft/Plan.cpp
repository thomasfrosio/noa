#include "noa/cpu/fft/Plan.h"

namespace noa::cpu::fft::details {
    std::mutex Mutex::mutex;

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

    Plan<float>::Plan(float* input, cfloat_t* output, size3_t shape, size_t batches, uint flag) {
        NOA_PROFILE_FUNCTION();
        int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
        int rank = static_cast<int>(ndim(shape));
        {
            std::unique_lock<std::mutex> lock(details::Mutex::get());
            if (!m_is_initialized)
                initialize_();
            if (m_max_threads > 1)
                setThreads_(shape, batches, rank);
            if (batches == 1) {
                m_plan = fftwf_plan_dft_r2c(rank, n + 3 - rank,
                                            input, reinterpret_cast<fftwf_complex*>(output), flag);
            } else {
                m_plan = fftwf_plan_many_dft_r2c(rank, n + 3 - rank, static_cast<int>(batches),
                                                 input, nullptr, 1,
                                                 static_cast<int>(elements(shape)),
                                                 reinterpret_cast<fftwf_complex*>(output), nullptr, 1,
                                                 static_cast<int>(elementsFFT(shape)),
                                                 flag);
            }
        }
        // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
        // configuration supporting a restricted set of transforms.
        if (!m_plan)
            NOA_THROW("Failed to create the R2C plan, with shape {}", shape);
    }

    Plan<float>::Plan(cfloat_t* input, float* output, size3_t shape, size_t batches, uint flag) {
        NOA_PROFILE_FUNCTION();
        int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
        int rank = static_cast<int>(ndim(shape));
        {
            std::unique_lock<std::mutex> lock(details::Mutex::get());
            if (!m_is_initialized)
                initialize_();
            if (m_max_threads > 1)
                setThreads_(shape, batches, rank);
            if (batches == 1) {
                m_plan = fftwf_plan_dft_c2r(rank, n + 3 - rank,
                                            reinterpret_cast<fftwf_complex*>(input), output, flag);
            } else {
                m_plan = fftwf_plan_many_dft_c2r(rank, n + 3 - rank, static_cast<int>(batches),
                                                 reinterpret_cast<fftwf_complex*>(input), nullptr, 1,
                                                 static_cast<int>(elementsFFT(shape)),
                                                 output, nullptr, 1, static_cast<int>(elements(shape)),
                                                 flag);
            }
        }
        // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
        // configuration supporting a restricted set of transforms or with the PRESERVE_INPUT flag
        // with a multi-dimensional out-of-place c2r transform.
        if (!m_plan)
            NOA_THROW("Failed to create the C2R plan, with shape {}", shape);
    }

    Plan<float>::Plan(cfloat_t* input, cfloat_t* output, size3_t shape, size_t batches, Sign sign, uint flag) {
        NOA_PROFILE_FUNCTION();
        int n[3] = {static_cast<int>(shape.z), static_cast<int>(shape.y), static_cast<int>(shape.x)};
        int rank = static_cast<int>(ndim(shape));
        {
            std::unique_lock<std::mutex> lock(details::Mutex::get());
            if (!m_is_initialized)
                initialize_();
            if (m_max_threads > 1)
                setThreads_(shape, batches, rank);
            if (batches == 1) {
                m_plan = fftwf_plan_dft(rank, n + 3 - rank,
                                        reinterpret_cast<fftwf_complex*>(input),
                                        reinterpret_cast<fftwf_complex*>(output),
                                        sign, flag);
            } else {
                int dist = static_cast<int>(elements(shape));
                m_plan = fftwf_plan_many_dft(rank, n + 3 - rank, static_cast<int>(batches),
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

    Plan<float>::~Plan() {
        NOA_PROFILE_FUNCTION();
        if (m_plan) {
            std::unique_lock<std::mutex> lock(details::Mutex::get());
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
        std::unique_lock<std::mutex> lock(details::Mutex::get());
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
        fftwf_plan_with_nthreads(noa::math::min(details::getThreads(shape, batches, rank), m_max_threads));
    }

    bool Plan<double>::m_is_initialized{false};
    int Plan<double>::m_max_threads{};
}
