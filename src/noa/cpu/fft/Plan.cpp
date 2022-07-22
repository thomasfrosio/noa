#include <mutex>

#include "noa/common/Exception.h"
#include "noa/cpu/fft/Plan.h"

namespace {
    using namespace ::noa;

    // Even values satisfying (2^a) * (3^b) * (5^c) * (7^d) * (11^e) * (13^f), with e + f = 0 or 1.
    constexpr uint16_t sizes_even_fftw_[] = {
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
    std::mutex g_noa_fftw3_mutex_;

#ifdef NOA_FFTW_THREADS
    // To call once before anything
    template<bool IS_SINGLE_PRECISION>
    void initialize_() {
        static bool is_initialized = false;
        if (is_initialized)
            return;
        if constexpr (IS_SINGLE_PRECISION) {
            if (!fftwf_init_threads())
                NOA_THROW("Failed to initialize the single precision FFTW-threads");
        } else {
            if (!fftw_init_threads())
                NOA_THROW("Failed to initialize the double precision FFTW-threads");
        }
    }

    // Gets the number of threads given a shape, batches and rank. From IMOD/libfft/fftw_wrap.c.
    // FFTW3 seems to be quite sensitive. If too many threads, the plan creation is just too slow...
    int getThreads_(int3_t shape, int batches, int rank) {
        double geom_size;
        if (rank == 1)
            geom_size = (math::sqrt(static_cast<double>(shape[2] * batches)) + static_cast<double>(batches)) / 2.;
        else
            geom_size = math::pow(static_cast<double>(shape.elements()), 1. / static_cast<double>(rank));
        int threads = static_cast<int>((math::log(geom_size) / math::log(2.) - 5.95) * 2.);
        return math::clamp(threads, 1, all(cpu::fft::fastShape(shape) == shape) ? 8 : 4);
    }

    // All subsequent plans will use this number of threads.
    template<bool IS_SINGLE_PRECISION>
    void setThreads_(int3_t shape, int batches, size_t max_threads) {
        initialize_<IS_SINGLE_PRECISION>();
        if (max_threads > 1) {
            const int threads = math::min(getThreads_(shape, batches, shape.ndim()), static_cast<int>(max_threads));
            if constexpr (IS_SINGLE_PRECISION)
                fftwf_plan_with_nthreads(threads);
            else
                fftw_plan_with_nthreads(threads);
        } else {
            if constexpr (IS_SINGLE_PRECISION)
                fftwf_plan_with_nthreads(1);
            else
                fftw_plan_with_nthreads(1);
        }
    }
#endif
}

namespace noa::cpu::fft {
    size_t fastSize(size_t size) {
        auto tmp = static_cast<uint>(size);
        for (uint16_t nice_size: sizes_even_fftw_)
            if (tmp < nice_size)
                return static_cast<size_t>(nice_size);
        return size % 2 ? size : (size + 1); // fall back to next even number
    }
}

namespace noa::cpu::fft {
    template<typename T>
    Plan<T>::~Plan() {
        cache_(std::exchange(m_plan, nullptr), false);
    }

    template<typename T>
    void Plan<T>::cleanup() {
        cache_(nullptr, true);
        std::lock_guard<std::mutex> lock(g_noa_fftw3_mutex_);
        if constexpr (IS_SINGLE_PRECISION)
            fftwf_cleanup();
        else
            fftw_cleanup();
        #ifdef NOA_FFTW_THREADS
        if constexpr (IS_SINGLE_PRECISION)
            fftwf_cleanup_threads();
        else
            fftw_cleanup_threads();
        #endif
    }

    template<typename T>
    void Plan<T>::cache_(fftw_plan_t plan, bool clear) {
        // FFTW accumulates a "wisdom" automatically. This circular buffer is here
        // in case fftw_destroy_plan destructs that wisdom.
        static constexpr size_t MAX_SIZE = 6;
        static std::array<fftw_plan_t, MAX_SIZE> s_bin{nullptr};
        static size_t s_index{0};
        auto destruct_plan = [](fftw_plan_t plan_) {
            if (plan_) {
                if constexpr (IS_SINGLE_PRECISION)
                    fftwf_destroy_plan(plan_);
                else
                    fftw_destroy_plan(plan_);
            }
        };

        if (clear) {
            std::lock_guard<std::mutex> lock(g_noa_fftw3_mutex_);
            for (auto& i_plan: s_bin)
                destruct_plan(std::exchange(i_plan, nullptr));
        }
        if (plan) {
            std::lock_guard<std::mutex> lock(g_noa_fftw3_mutex_);
            destruct_plan(std::exchange(s_bin[s_index], plan));
            s_index = (s_index + 1) % MAX_SIZE;
        }
    }

    template<typename T>
    typename Plan<T>::fftw_plan_t Plan<T>::getR2C_(T* input, Complex<T>* output, size4_t shape,
                                                   uint flag, size_t threads) {
        const int3_t s_shape(shape.get(1));
        const int rank = s_shape.ndim();
        const int how_many = static_cast<int>(shape[0]);
        const int odist = s_shape.fft().elements();
        const int idist = input == reinterpret_cast<T*>(output) ? odist * 2 : s_shape.elements();
        fftw_plan_t plan;
        {
            std::lock_guard<std::mutex> lock(g_noa_fftw3_mutex_);
            #ifdef NOA_FFTW_THREADS
            setThreads_<IS_SINGLE_PRECISION>(s_shape, how_many, threads);
            #endif
            if constexpr (IS_SINGLE_PRECISION) {
                plan = fftwf_plan_many_dft_r2c(
                        rank, s_shape.get(3 - rank), how_many, input, nullptr, 1, idist,
                        reinterpret_cast<fftwf_complex*>(output), nullptr, 1, odist, flag);
            } else {
                plan = fftw_plan_many_dft_r2c(
                        rank, s_shape.get(3 - rank), how_many, input, nullptr, 1, idist,
                        reinterpret_cast<fftw_complex*>(output), nullptr, 1, odist, flag);
            }
        }

        // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
        // configuration supporting a restricted set of transforms.
        if (!plan)
            NOA_THROW("Failed to create the R2C plan, with shape:{}", s_shape);
        return plan;
    }

    template<typename T>
    typename Plan<T>::fftw_plan_t Plan<T>::getR2C_(T* input, size4_t input_stride,
                                                   Complex<T>* output, size4_t output_stride,
                                                   size4_t shape, uint flag, size_t threads) {
        const int3_t s_shape(shape.get(1));
        const int4_t i_stride(input_stride);
        const int4_t o_stride(output_stride);
        const int3_t inembed(i_stride.pitches());
        const int3_t onembed(o_stride.pitches());
        const int rank = s_shape.ndim();
        const int how_many = static_cast<int>(shape[0]);
        const int off = 3 - rank;
        fftw_plan_t plan;
        {
            std::lock_guard<std::mutex> lock(g_noa_fftw3_mutex_);
            #ifdef NOA_FFTW_THREADS
            setThreads_<IS_SINGLE_PRECISION>(s_shape, how_many, threads);
            #endif

            if constexpr (IS_SINGLE_PRECISION) {
                plan = fftwf_plan_many_dft_r2c(
                        rank, s_shape.get(off), how_many, input, inembed.get(off), i_stride[3], i_stride[0],
                        reinterpret_cast<fftwf_complex*>(output), onembed.get(off), o_stride[3], o_stride[0], flag);
            } else {
                plan = fftw_plan_many_dft_r2c(
                        rank, s_shape.get(off), how_many, input, inembed.get(off), i_stride[3], i_stride[0],
                        reinterpret_cast<fftw_complex*>(output), onembed.get(off), o_stride[3], o_stride[0], flag);
            }
        }
        if (!plan)
            NOA_THROW("Failed to create the R2C plan, with shape:{}, istride:{}, ostride:{}",
                      s_shape, input_stride, output_stride);
        return plan;
    }

    template<typename T>
    typename Plan<T>::fftw_plan_t Plan<T>::getC2R_(Complex<T>* input, T* output,
                                                   size4_t shape, uint flag, size_t threads) {
        const int3_t s_shape(shape.get() + 1);
        const int rank = s_shape.ndim();
        const int idist = s_shape.fft().elements();
        const int odist = reinterpret_cast<T*>(input) == output ? idist * 2 : s_shape.elements();
        const int how_many = static_cast<int>(shape[0]);
        fftw_plan_t plan;
        {
            std::lock_guard<std::mutex> lock(g_noa_fftw3_mutex_);
            #ifdef NOA_FFTW_THREADS
            setThreads_<IS_SINGLE_PRECISION>(s_shape, how_many, threads);
            #endif
            if constexpr (IS_SINGLE_PRECISION) {
                plan = fftwf_plan_many_dft_c2r(
                        rank, s_shape.get() + 3 - rank, how_many,
                        reinterpret_cast<fftwf_complex*>(input), nullptr, 1, idist,
                        output, nullptr, 1, odist, flag);
            } else {
                plan = fftw_plan_many_dft_c2r(
                        rank, s_shape.get() + 3 - rank, how_many,
                        reinterpret_cast<fftw_complex*>(input), nullptr, 1, idist,
                        output, nullptr, 1, odist, flag);
            }
        }
        if (!plan)
            NOA_THROW("Failed to create the C2R plan, with shape:{}", s_shape);
        return plan;
    }

    template<typename T>
    typename Plan<T>::fftw_plan_t Plan<T>::getC2R_(Complex<T>* input, size4_t input_stride,
                                                   T* output, size4_t output_stride,
                                                   size4_t shape, uint flag, size_t threads) {
        const int3_t s_shape(shape.get() + 1);
        const int4_t i_stride(input_stride);
        const int4_t o_stride(output_stride);
        const int3_t inembed(i_stride.pitches());
        const int3_t onembed(o_stride.pitches());
        const int rank = s_shape.ndim();
        const int off = 3 - rank;
        const int how_many = static_cast<int>(shape[0]);
        fftw_plan_t plan;
        {
            std::lock_guard<std::mutex> lock(g_noa_fftw3_mutex_);
            #ifdef NOA_FFTW_THREADS
            setThreads_<IS_SINGLE_PRECISION>(s_shape, how_many, threads);
            #endif
            if constexpr (IS_SINGLE_PRECISION) {
                plan = fftwf_plan_many_dft_c2r(
                        rank, s_shape.get() + off, how_many,
                        reinterpret_cast<fftwf_complex*>(input), inembed.get() + off, i_stride[3], i_stride[0],
                        output, onembed.get() + off, o_stride[3], o_stride[0], flag);
            } else {
                plan = fftw_plan_many_dft_c2r(
                        rank, s_shape.get() + off, how_many,
                        reinterpret_cast<fftw_complex*>(input), inembed.get() + off, i_stride[3], i_stride[0],
                        output, onembed.get() + off, o_stride[3], o_stride[0], flag);
            }
        }
        // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
        // configuration supporting a restricted set of transforms or with the PRESERVE_INPUT flag
        // with a multi-dimensional out-of-place c2r transform.
        if (!plan)
            NOA_THROW("Failed to create the R2C plan, with shape:{}, istride:{}, ostride:{}",
                      s_shape, input_stride, output_stride);
        return plan;
    }

    template<typename T>
    typename Plan<T>::fftw_plan_t Plan<T>::getC2C_(Complex<T>* input, Complex<T>* output, size4_t shape,
                                                   Sign sign, uint flag, size_t threads) {
        static_assert(Sign::FORWARD == FFTW_FORWARD);
        static_assert(Sign::BACKWARD == FFTW_BACKWARD);

        int3_t s_shape(shape.get() + 1);
        int rank = s_shape.ndim();
        int how_many = static_cast<int>(shape[0]);
        int dist = s_shape.elements();
        fftw_plan_t plan;
        {
            std::lock_guard<std::mutex> lock(g_noa_fftw3_mutex_);
            #ifdef NOA_FFTW_THREADS
            setThreads_<IS_SINGLE_PRECISION>(s_shape, how_many, threads);
            #endif

            if constexpr (IS_SINGLE_PRECISION) {
                plan = fftwf_plan_many_dft(rank, s_shape.get() + 3 - rank, how_many,
                                           reinterpret_cast<fftwf_complex*>(input), nullptr, 1, dist,
                                           reinterpret_cast<fftwf_complex*>(output), nullptr, 1, dist,
                                           sign, flag);
            } else {
                plan = fftw_plan_many_dft(rank, s_shape.get() + 3 - rank, how_many,
                                          reinterpret_cast<fftw_complex*>(input), nullptr, 1, dist,
                                          reinterpret_cast<fftw_complex*>(output), nullptr, 1, dist,
                                          sign, flag);
            }
        }
        // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
        // configuration supporting a restricted set of transforms.
        if (!plan)
            NOA_THROW("Failed to create the C2C plan, with shape:{}", s_shape);
        return plan;
    }

    template<typename T>
    typename Plan<T>::fftw_plan_t Plan<T>::getC2C_(Complex<T>* input, size4_t input_stride,
                                                   Complex<T>* output, size4_t output_stride,
                                                   size4_t shape, Sign sign, uint flag,
                                                   size_t threads) {
        static_assert(Sign::FORWARD == FFTW_FORWARD);
        static_assert(Sign::BACKWARD == FFTW_BACKWARD);

        const int3_t s_shape(shape.get() + 1);
        const int4_t i_stride(input_stride);
        const int4_t o_stride(output_stride);
        const int3_t inembed(i_stride.pitches());
        const int3_t onembed(o_stride.pitches());
        const int rank = s_shape.ndim();
        const int how_many = static_cast<int>(shape[0]);
        const int off = 3 - rank;
        fftw_plan_t plan;
        {
            std::lock_guard<std::mutex> lock(g_noa_fftw3_mutex_);
            #ifdef NOA_FFTW_THREADS
            setThreads_<IS_SINGLE_PRECISION>(s_shape, how_many, threads);
            #endif

            if constexpr (IS_SINGLE_PRECISION) {
                plan = fftwf_plan_many_dft(
                        rank, s_shape.get() + off, how_many,
                        reinterpret_cast<fftwf_complex*>(input), inembed.get() + off, i_stride[3], i_stride[0],
                        reinterpret_cast<fftwf_complex*>(output), onembed.get() + off, o_stride[3], o_stride[0],
                        sign, flag);
            } else {
                plan = fftw_plan_many_dft(
                        rank, s_shape.get() + off, how_many,
                        reinterpret_cast<fftw_complex*>(input), inembed.get() + off, i_stride[3], i_stride[0],
                        reinterpret_cast<fftw_complex*>(output), onembed.get() + off, o_stride[3], o_stride[0],
                        sign, flag);
            }
        }
        // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
        // configuration supporting a restricted set of transforms.
        if (!plan)
            NOA_THROW("Failed to create the R2C plan, with shape:{}, istride:{}, ostride:{}",
                      s_shape, input_stride, output_stride);
        return plan;
    }

    template
    class Plan<float>;

    template
    class Plan<double>;
}
