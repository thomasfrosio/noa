#include <mutex>

#include "noa/common/Environment.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/fft/Plan.h"

namespace {
    using namespace ::noa;

    // Even values satisfying (2^a) * (3^b) * (5^c) * (7^d) * (11^e) * (13^f), with e + f = 0 or 1.
    constexpr uint sizes_even_fftw_[] = {
            2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 42, 44, 48, 50, 52, 54, 56, 60, 64,
            66, 70, 72, 78, 80, 84, 88, 90, 96, 98, 100, 104, 108, 110, 112, 120, 126, 128, 130, 132, 140, 144, 150,
            154, 156, 160, 162, 168, 176, 180, 182, 192, 196, 198, 200, 208, 210, 216, 220, 224, 234, 240, 250, 252,
            256, 260, 264, 270, 280, 288, 294, 300, 308, 312, 320, 324, 330, 336, 350, 352, 360, 364, 378, 384, 390,
            392, 396, 400, 416, 420, 432, 440, 448, 450, 462, 468, 480, 486, 490, 500, 504, 512, 520, 528, 540, 546,
            550, 560, 576, 588, 594, 600, 616, 624, 630, 640, 648, 650, 660, 672, 686, 700, 702, 704, 720, 728, 750,
            756, 768, 770, 780, 784, 792, 800, 810, 832, 840, 864, 880, 882, 896, 900, 910, 924, 936, 960, 972, 980,
            990, 1000, 1008, 1024, 1040, 1050, 1056, 1078, 1080, 1092, 1100, 1120, 1134, 1152, 1170, 1176, 1188,
            1200, 1232, 1248, 1250, 1260, 1274, 1280, 1296, 1300, 1320, 1344, 1350, 1372, 1386, 1400, 1404, 1408,
            1440, 1456, 1458, 1470, 1500, 1512, 1536, 1540, 1560, 1568, 1584, 1600, 1620, 1638, 1650, 1664, 1680,
            1728, 1750, 1760, 1764, 1782, 1792, 1800, 1820, 1848, 1872, 1890, 1920, 1944, 1950, 1960, 1980, 2000,
            2016, 2048, 2058, 2080, 2100, 2106, 2112, 2156, 2160, 2184, 2200, 2240, 2250, 2268, 2304, 2310, 2340,
            2352, 2376, 2400, 2430, 2450, 2464, 2496, 2500, 2520, 2548, 2560, 2592, 2600, 2640, 2646, 2688, 2700,
            2730, 2744, 2750, 2772, 2800, 2808, 2816, 2880, 2912, 2916, 2940, 2970, 3000, 3024, 3072, 3080, 3120,
            3136, 3150, 3168, 3200, 3234, 3240, 3250, 3276, 3300, 3328, 3360, 3402, 3430, 3456, 3500, 3510, 3520,
            3528, 3564, 3584, 3600, 3640, 3696, 3744, 3750, 3780, 3822, 3840, 3850, 3888, 3900, 3920, 3960, 4000,
            4032, 4050, 4096, 4116, 4158, 4160, 4200, 4212, 4224, 4312, 4320, 4368, 4374, 4400, 4410, 4480, 4500,
            4536, 4550, 4608, 4620, 4680, 4704, 4752, 4800, 4860, 4900, 4914, 4928, 4950, 4992, 5000, 5040, 5096,
            5120, 5184, 5200, 5250, 5280, 5292, 5346, 5376, 5390, 5400, 5460, 5488, 5500, 5544, 5600, 5616, 5632,
            5670, 5760, 5824, 5832, 5850, 5880, 5940, 6000, 6048, 6144, 6160, 6174, 6240, 6250, 6272, 6300, 6318,
            6336, 6370, 6400, 6468, 6480, 6500, 6552, 6600, 6656, 6720, 6750, 6804, 6860, 6912, 6930, 7000, 7020,
            7040, 7056, 7128, 7168, 7200, 7280, 7290, 7350, 7392, 7488, 7500, 7546, 7560, 7644, 7680, 7700, 7776,
            7800, 7840, 7920, 7938, 8000, 8064, 8100, 8190, 8192, 8232, 8250, 8316, 8320, 8400, 8424, 8448, 8624,
            8640, 8736, 8748, 8750, 8800, 8820, 8910, 8918, 8960, 9000, 9072, 9100, 9216, 9240, 9360, 9408, 9450,
            9504, 9600, 9702, 9720, 9750, 9800, 9828, 9856, 9900, 9984, 10000, 10080, 10192, 10206, 10240, 10290,
            10368, 10400, 10500, 10530, 10560, 10584, 10692, 10752, 10780, 10800, 10920, 10976, 11000, 11088, 11200,
            11232, 11250, 11264, 11340, 11466, 11520, 11550, 11648, 11664, 11700, 11760, 11880, 12000, 12096, 12150,
            12250, 12288, 12320, 12348, 12474, 12480, 12500, 12544, 12600, 12636, 12672, 12740, 12800, 12936, 12960,
            13000, 13104, 13200, 13230, 13312, 13440, 13500, 13608, 13650, 13720, 13750, 13824, 13860, 14000, 14040,
            14080, 14112, 14256, 14336, 14400, 14560, 14580, 14700, 14742, 14784, 14850, 14976, 15000, 15092, 15120,
            15288, 15360, 15400, 15552, 15600, 15680, 15750, 15840, 15876, 16000, 16038, 16128, 16170, 16200, 16250,
            16380, 16384, 16464, 16500, 16632, 16640, 16800, 16848, 16896
    };

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
        return math::clamp(threads, 1, all(cpu::fft::fastShape(shape) == shape) ? 8 : 4);
    }
}

namespace noa::cpu::fft {
    size_t fastSize(size_t size) {
        auto tmp = static_cast<uint>(size);
        for (uint nice_size: sizes_even_fftw_)
            if (tmp < nice_size)
                return static_cast<size_t>(nice_size);
        return size % 2 ? size : (size + 1); // fall back to next even number
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