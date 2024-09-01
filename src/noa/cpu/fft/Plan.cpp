#include <mutex>
#include <fftw3/fftw3.h>

#include "noa/core/Exception.hpp"
#include "noa/core/indexing/Layout.hpp"
#include "noa/core/utils/Irange.hpp"
#include "noa/core/utils/Misc.hpp"
#include "noa/cpu/fft/Plan.hpp"

namespace {
    using namespace ::noa;

    static_assert(noa::to_underlying(noa::fft::Sign::FORWARD) == FFTW_FORWARD);
    static_assert(noa::to_underlying(noa::fft::Sign::BACKWARD) == FFTW_BACKWARD);

    // Even values satisfying (2^a) * (3^b) * (5^c) * (7^d) * (11^e) * (13^f), with e + f = 0 or 1.
    constexpr u16 sizes_even_fftw_[] = {
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

    // Just makes sure OMP launches the exact number of necessary threads, as opposed to:
    // https://github.com/FFTW/fftw3/blob/master/threads/openmp.c#L77
    // See https://www.fftw.org/fftw3_doc/Usage-of-Multi_002dthreaded-FFTW.html
    void fftw_callback_(void *(*work)(char *), char *jobdata, size_t elsize, i32 njobs, void*) {
        #pragma omp parallel for num_threads(njobs)
        for (size_t i = 0; i < static_cast<size_t>(njobs); ++i)
            work(jobdata + elsize * i);
    }

    template<typename T>
    class fftw {
    public:
        static constexpr bool is_single_precision = std::is_same_v<T, f32>;
        using plan_t = std::conditional_t<is_single_precision, fftwf_plan, fftw_plan>; // pointers to struct
        using real_t = T;
        using complex_t = Complex<T>;
        using fftw_complex_t = std::conditional_t<is_single_precision, fftwf_complex, fftw_complex>;

        static plan_t create_r2c(
                real_t* input, i32 input_batch_stride,
                complex_t* output, i32 output_batch_stride,
                i32 batch, const Shape3<i32>& shape_3d, i32 rank, i64 max_n_threads, u32 flags
        ) {
            const std::scoped_lock lock(mutex);
            set_planner_(batch, shape_3d, max_n_threads);
            plan_t plan;
            auto optr = reinterpret_cast<fftw_complex_t*>(output);
            if constexpr (is_single_precision) {
                plan = fftwf_plan_many_dft_r2c(
                        rank, shape_3d.data() + 3 - rank, batch,
                        input, nullptr, 1, input_batch_stride,
                        optr, nullptr, 1, output_batch_stride, flags);
            } else {
                plan = fftw_plan_many_dft_r2c(
                        rank, shape_3d.data() + 3 - rank, batch,
                        input, nullptr, 1, input_batch_stride,
                        optr, nullptr, 1, output_batch_stride, flags);
            }
            // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
            // configuration supporting a restricted set of transforms.
            noa::check(plan != nullptr, "Failed to create the r2c plan with shape={}", shape_3d.push_back(batch));
            return plan;
        }

        static plan_t create_r2c(
                real_t* input, const Strides4<i32>& input_strides,
                complex_t* output, const Strides4<i32>& output_strides,
                i32 batch, const Shape3<i32>& shape_3d, i32 rank, i64 max_n_threads, u32 flags
        ) {
            const std::scoped_lock lock(mutex);
            set_planner_(batch, shape_3d, max_n_threads);
            plan_t plan;

            const auto inembed = input_strides.physical_shape();
            const auto onembed = output_strides.physical_shape();
            const i32 off = 3 - rank;

            auto optr = reinterpret_cast<fftw_complex_t*>(output);
            if constexpr (is_single_precision) {
                plan = fftwf_plan_many_dft_r2c(
                        rank, shape_3d.data() + off, batch,
                        input, inembed.data() + off, input_strides[3], input_strides[0],
                        optr, onembed.data() + off, output_strides[3], output_strides[0], flags);
            } else {
                plan = fftw_plan_many_dft_r2c(
                        rank, shape_3d.data() + off, batch,
                        input, inembed.data() + off, input_strides[3], input_strides[0],
                        optr, onembed.data() + off, output_strides[3], output_strides[0], flags);
            }

            noa::check(plan != nullptr,
                       "Failed to create the r2c plan with shape={}, input_strides={}, output_strides={}",
                       shape_3d, input_strides, output_strides);
            return plan;
        }

        static plan_t create_c2r(
                complex_t* input, i32 input_batch_stride,
                real_t* output, i32 output_batch_stride,
                i32 batch, const Shape3<i32>& shape_3d, i32 rank, i64 max_n_threads, u32 flags
        ) {
            const std::scoped_lock lock(mutex);
            set_planner_(batch, shape_3d, max_n_threads);
            plan_t plan;
            auto iptr = reinterpret_cast<fftw_complex_t*>(input);
            if constexpr (is_single_precision) {
                plan = fftwf_plan_many_dft_c2r(
                        rank, shape_3d.data() + 3 - rank, batch,
                        iptr, nullptr, 1, input_batch_stride,
                        output, nullptr, 1, output_batch_stride, flags);
            } else {
                plan = fftw_plan_many_dft_c2r(
                        rank, shape_3d.data() + 3 - rank, batch,
                        iptr, nullptr, 1, input_batch_stride,
                        output, nullptr, 1, output_batch_stride, flags);
            }
            noa::check(plan != nullptr, "Failed to create the c2r plan with shape={}", shape_3d.push_back(batch));
            return plan;
        }

        static plan_t create_c2r(
                complex_t* input, const Strides4<i32>& input_strides,
                real_t* output, const Strides4<i32>& output_strides,
                i32 batch, const Shape3<i32>& shape_3d, i32 rank, i64 max_n_threads, u32 flags
        ) {
            const std::scoped_lock lock(mutex);
            set_planner_(batch, shape_3d, max_n_threads);
            plan_t plan;

            const auto inembed = input_strides.physical_shape();
            const auto onembed = output_strides.physical_shape();
            const i32 off = 3 - rank;

            auto iptr = reinterpret_cast<fftw_complex_t*>(input);
            if constexpr (is_single_precision) {
                plan = fftwf_plan_many_dft_c2r(
                        rank, shape_3d.data() + off, batch,
                        iptr, inembed.data() + off, input_strides[3], input_strides[0],
                        output, onembed.data() + off, output_strides[3], output_strides[0], flags);
            } else {
                plan = fftw_plan_many_dft_c2r(
                        rank, shape_3d.data() + off, batch,
                        iptr, inembed.data() + off, input_strides[3], input_strides[0],
                        output, onembed.data() + off, output_strides[3], output_strides[0], flags);
            }

            // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
            // configuration supporting a restricted set of transforms or with the PRESERVE_INPUT flag
            // with a multidimensional out-of-place c2r transform.
            noa::check(plan != nullptr,
                       "Failed to create the c2r plan with shape={}, input_strides={}, output_strides={}",
                       shape_3d, input_strides, output_strides);
            return plan;
        }

        static plan_t create_c2c(
                complex_t* input, i32 input_batch_stride,
                complex_t* output, i32 output_batch_stride,
                noa::fft::Sign sign,
                i32 batch, const Shape3<i32>& shape_3d, i32 rank, i64 max_n_threads, u32 flags
        ) {
            const std::scoped_lock lock(mutex);
            set_planner_(batch, shape_3d, max_n_threads);
            plan_t plan;
            auto iptr = reinterpret_cast<fftw_complex_t*>(input);
            auto optr = reinterpret_cast<fftw_complex_t*>(output);
            if constexpr (is_single_precision) {
                plan = fftwf_plan_many_dft(
                        rank, shape_3d.data() + 3 - rank, batch,
                        iptr, nullptr, 1, input_batch_stride,
                        optr, nullptr, 1, output_batch_stride,
                        noa::to_underlying(sign), flags);
            } else {
                plan = fftw_plan_many_dft(
                        rank, shape_3d.data() + 3 - rank, batch,
                        iptr, nullptr, 1, input_batch_stride,
                        optr, nullptr, 1, output_batch_stride,
                        noa::to_underlying(sign), flags);
            }

            // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
            // configuration supporting a restricted set of transforms.
            noa::check(plan != nullptr, "Failed to create the c2c plan with shape={}", shape_3d.push_back(batch));
            return plan;
        }

        static plan_t create_c2c(
                complex_t* input, const Strides4<i32>& input_strides,
                complex_t* output, const Strides4<i32>& output_strides,
                noa::fft::Sign sign,
                i32 batch, const Shape3<i32>& shape_3d, i32 rank, i64 max_n_threads, u32 flags
        ) {
            const std::scoped_lock lock(mutex);
            set_planner_(batch, shape_3d, max_n_threads);
            plan_t plan;

            const auto inembed = input_strides.physical_shape();
            const auto onembed = output_strides.physical_shape();
            const i32 off = 3 - rank;

            auto iptr = reinterpret_cast<fftw_complex_t*>(input);
            auto optr = reinterpret_cast<fftw_complex_t*>(output);
            if constexpr (is_single_precision) {
                plan = fftwf_plan_many_dft(
                        rank, shape_3d.data() + off, batch,
                        iptr, inembed.data() + off, input_strides[3], input_strides[0],
                        optr, onembed.data() + off, output_strides[3], output_strides[0],
                        noa::to_underlying(sign), flags);
            } else {
                plan = fftw_plan_many_dft(
                        rank, shape_3d.data() + off, batch,
                        iptr, inembed.data() + off, input_strides[3], input_strides[0],
                        optr, onembed.data() + off, output_strides[3], output_strides[0],
                        noa::to_underlying(sign), flags);
            }

            // A non-NULL plan is always returned by the basic interface unless using a customized FFTW
            // configuration supporting a restricted set of transforms.
            noa::check(plan != nullptr,
                       "Failed to create the c2c plan with shape={}, input_strides={}, output_strides={}",
                       shape_3d, input_strides, output_strides);
            return plan;
        }

        static i32 destroy(void* plan, bool clear_cache_first = false) noexcept {
            // FFTW accumulates a "wisdom" automatically. This circular buffer is here
            // in case fftw_destroy_plan destructs that wisdom.
            // TODO Benchmark to make sure this does something.
            constexpr size_t CACHE_SIZE = 6;
            struct SimpleCache {
                std::array<void*, CACHE_SIZE> cache{nullptr};
                size_t index{0};

                auto push(void* new_plan) -> i32 {
                    void* old_plan = cache[index]; // std::exchange
                    cache[index] = new_plan;
                    index = (index + 1) % CACHE_SIZE; // increment to next plan, for next call
                    // destruct plan
                    if (old_plan) {
                        if constexpr (is_single_precision)
                            fftwf_destroy_plan(static_cast<plan_t>(old_plan));
                        else
                            fftw_destroy_plan(static_cast<plan_t>(old_plan));
                        return 1;
                    }
                    return 0;
                };
            };
            static SimpleCache cache;

            i32 n_plans_destructed{0};
            if (clear_cache_first) {
                const std::scoped_lock lock(mutex);
                for ([[maybe_unused]] auto _: noa::irange(CACHE_SIZE))
                    n_plans_destructed += cache.push(nullptr);
            }
            if (plan) {
                const std::scoped_lock lock(mutex);
                n_plans_destructed += cache.push(plan);
            }
            return n_plans_destructed;
        }

        static i32 cleanup() noexcept {
            i32 n_plans_destructed = destroy(nullptr, true);
            const std::scoped_lock lock(mutex);
            #if defined(NOA_FFTW_THREADS)
            if constexpr (is_single_precision)
                fftwf_cleanup_threads();
            else
                fftw_cleanup_threads();
            #else
            if constexpr (is_single_precision)
                fftwf_cleanup();
            else
                fftw_cleanup();
            #endif
            return n_plans_destructed;
        }

        // Executes the plan.
        // It is safe to execute the same plan in parallel by multiple threads. However, since a given plan operates
        // by default on a fixed array, one needs to use one of the new-array functions so that different threads
        // compute the transform on different data.
        static void execute(void* plan) noexcept {
            auto p = static_cast<plan_t>(plan);
            if constexpr (is_single_precision)
                fftwf_execute(p);
            else
                fftw_execute(p);
        }

    private:
        // The only thread-safe routine in FFTW is fftw_execute (and the new-array variants). All other routines
        // (e.g. the planners) should only be called from one thread at a time. Thus, to make our API thread-safe,
        // calls to FFTW should be protected by this mutex.
        static std::mutex mutex;

        // Gets the number of threads given a shape, batches and rank. From IMOD/libfft/fftw_wrap.c.
        // FFTW3 seems to be quite sensitive. If too many threads, the plan creation is just too slow...
        static i32 suggest_n_threads_(i32 batch, const Shape3<i32>& shape, i32 rank) noexcept {
            f64 geom_size;
            if (rank == 1) {
                const auto n_elements = static_cast<f64>(shape[2] * batch);
                geom_size = (std::sqrt(n_elements) + static_cast<f64>(batch)) / 2.;
            } else {
                const auto n_elements = static_cast<f64>(shape.n_elements());
                geom_size = std::pow(n_elements, 1. / static_cast<f64>(rank));
            }

            const auto n_threads = static_cast<i32>((std::log(geom_size) / std::log(2.) - 5.95) * 2.);
            const bool is_fast_shape = noa::all(noa::cpu::fft::fast_shape(shape) == shape) ;
            return std::clamp(n_threads, 1, is_fast_shape ? 8 : 4);
        }

        // All subsequent plans will use this number of threads.
        // This function is not thread-safe; it should be called in a thread-safe environment.
        static void set_planner_(i32 batch, const Shape3<i32>& shape, i64 max_threads) {
            #ifdef NOA_MULTITHREADED_FFTW3
            // Initialize (once)...
            static bool is_initialized = false;
            if (not is_initialized) {
                if constexpr (is_single_precision)
                    check(fftwf_init_threads(), "Failed to initialize the single precision FFTW-threads");
                else
                    check(fftw_init_threads(), "Failed to initialize the single precision FFTW-threads");

                fftw_threads_set_callback(fftw_callback_, nullptr);
                fftwf_threads_set_callback(fftw_callback_, nullptr);

                is_initialized = true;
            }

            if (max_threads > 1) {
                const i32 suggested_n_threads = suggest_n_threads_(batch, shape, shape.ndim());
                const i32 actual_n_threads = std::min(suggested_n_threads, static_cast<i32>(max_threads));

                if constexpr (is_single_precision)
                    fftwf_plan_with_nthreads(actual_n_threads);
                else
                    fftw_plan_with_nthreads(actual_n_threads);
            } else {
                if constexpr (is_single_precision)
                    fftwf_plan_with_nthreads(1);
                else
                    fftw_plan_with_nthreads(1);
            }
            #endif
        }
    };

    template<typename T>
    std::mutex fftw<T>::mutex{};

    bool is_inplace_(const void* input, const void* output) {
        return input == output;
    }
}

namespace noa::cpu::fft {
    i64 fast_size(i64 size) {
        for (u16 nice_size: sizes_even_fftw_) {
            const auto tmp = static_cast<i64>(nice_size);
            if (size <= tmp)
                return tmp;
        }
        return size % 2 ? size : (size + 1); // fall back to next even number
    }
}

namespace noa::cpu::fft {
    template<typename T>
    Plan<T>::Plan(T* input, Complex<T>* output, const Shape4<i64>& shape, u32 flag, i64 max_n_threads) {
        auto [batch, shape_3d] = shape.as_safe<i32>().split_batch();
        const i32 rank = shape_3d.ndim();
        const i32 odist = shape_3d.rfft().n_elements();
        const i32 idist = is_inplace_(input, output) ? odist * 2 : shape_3d.n_elements();

        NOA_ASSERT(rank == 1 || !ni::is_vector(shape_3d));
        if (rank == 1 and shape_3d[2] == 1) // column vector -> row vector
            std::swap(shape_3d[1], shape_3d[2]);

        m_plan = fftw<T>::create_r2c(input, idist, output, odist, batch, shape_3d, rank, max_n_threads, flag);
    }

    template<typename T>
    Plan<T>::Plan(
            T* input, const Strides4<i64>& input_strides,
            Complex<T>* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, u32 flag, i64 max_n_threads
    ) {
        auto [batch, shape_3d] = shape.as_safe<i32>().split_batch();
        const i32 rank = shape_3d.ndim();
        auto istrides = input_strides.as_safe<i32>();
        auto ostrides = output_strides.as_safe<i32>();

        NOA_ASSERT(rank == 1 or not ni::is_vector(shape_3d));
        if (rank == 1 and shape_3d[2] == 1) { // column vector -> row vector
            std::swap(shape_3d[1], shape_3d[2]);
            std::swap(istrides[2], istrides[3]);
            std::swap(ostrides[2], ostrides[3]);
        }
        m_plan = fftw<T>::create_r2c(input, istrides, output, ostrides, batch, shape_3d, rank, max_n_threads, flag);
    }

    template<typename T>
    Plan<T>::Plan(Complex<T>* input, T* output, const Shape4<i64>& shape, u32 flag, i64 max_threads) {
        auto [batch, shape_3d] = shape.as_safe<i32>().split_batch();
        const i32 rank = shape_3d.ndim();
        const i32 idist = shape_3d.rfft().n_elements();
        const i32 odist = is_inplace_(input, output) ? idist * 2 : shape_3d.n_elements();

        NOA_ASSERT(rank == 1 or not ni::is_vector(shape_3d));
        if (rank == 1 and shape_3d[2] == 1) // column vector -> row vector
            std::swap(shape_3d[1], shape_3d[2]);

        m_plan = fftw<T>::create_c2r(input, idist, output, odist, batch, shape_3d, rank, max_threads, flag);
    }

    template<typename T>
    Plan<T>::Plan(
            Complex<T>* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, u32 flag, i64 max_n_threads
    ) {
        auto [batch, shape_3d] = shape.as_safe<i32>().split_batch();
        const i32 rank = shape_3d.ndim();
        auto istrides = input_strides.as_safe<i32>();
        auto ostrides = output_strides.as_safe<i32>();

        NOA_ASSERT(rank == 1 or not ni::is_vector(shape_3d));
        if (rank == 1 and shape_3d[2] == 1) { // column vector -> row vector
            std::swap(shape_3d[1], shape_3d[2]);
            std::swap(istrides[2], istrides[3]);
            std::swap(ostrides[2], ostrides[3]);
        }
        m_plan = fftw<T>::create_c2r(input, istrides, output, ostrides, batch, shape_3d, rank, max_n_threads, flag);
    }

    template<typename T>
    Plan<T>::Plan(
            Complex<T>* input, Complex<T>* output, const Shape4<i64>& shape,
            noa::fft::Sign sign, u32 flag, i64 max_threads
    ) {
        auto [batch, shape_3d] = shape.as_safe<i32>().split_batch();
        const i32 rank = shape_3d.ndim();
        const i32 dist = shape_3d.n_elements();

        NOA_ASSERT(rank == 1 or not ni::is_vector(shape_3d));
        if (rank == 1 and shape_3d[2] == 1) // column vector -> row vector
            std::swap(shape_3d[1], shape_3d[2]);

        m_plan = fftw<T>::create_c2c(input, dist, output, dist, sign, batch, shape_3d, rank, max_threads, flag);
    }

    template<typename T>
    Plan<T>::Plan(
            Complex<T>* input, const Strides4<i64>& input_strides,
            Complex<T>* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, noa::fft::Sign sign, u32 flag, i64 max_n_threads
    ) {
        auto [batch, shape_3d] = shape.as_safe<i32>().split_batch();
        const i32 rank = shape_3d.ndim();
        auto istrides = input_strides.as_safe<i32>();
        auto ostrides = output_strides.as_safe<i32>();

        NOA_ASSERT(rank == 1 or not ni::is_vector(shape_3d));
        if (ni::is_column_major(istrides) and ni::is_column_major(ostrides)) {
            // column major -> row major
            std::swap(shape_3d[1], shape_3d[2]);
            std::swap(istrides[2], istrides[3]);
            std::swap(ostrides[2], ostrides[3]);
        }
        m_plan = fftw<T>::create_c2c(input, istrides, output, ostrides, sign, batch, shape_3d, rank, max_n_threads, flag);
    }

    template<typename T>
    Plan<T>::~Plan() noexcept {
        fftw<T>::destroy(std::exchange(m_plan, nullptr), false);
    }

    template<typename T>
    void Plan<T>::execute() noexcept {
        fftw<T>::execute(m_plan);
    }

    template<typename T>
    i32 Plan<T>::cleanup() noexcept {
        return fftw<T>::cleanup();
    }

    template class Plan<f32>;
    template class Plan<f64>;
}
