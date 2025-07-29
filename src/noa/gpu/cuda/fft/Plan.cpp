#include <cufft.h>
#include <deque>

#include "noa/core/utils/Strings.hpp"
#include "noa/gpu/cuda/Error.hpp"
#include "noa/gpu/cuda/fft/Plan.hpp"

namespace noa::cuda {
    std::string error2string(cufftResult_t result) {
        switch (result) {
            case CUFFT_SUCCESS:
                return "CUFFT_SUCCESS";
            case CUFFT_INVALID_PLAN:
                return "CUFFT_INVALID_PLAN";
            case CUFFT_ALLOC_FAILED:
                return "CUFFT_ALLOC_FAILED";
            case CUFFT_INVALID_TYPE:
                return "CUFFT_INVALID_TYPE";
            case CUFFT_INVALID_VALUE:
                return "CUFFT_INVALID_VALUE";
            case CUFFT_INTERNAL_ERROR:
                return "CUFFT_INTERNAL_ERROR";
            case CUFFT_EXEC_FAILED:
                return "CUFFT_EXEC_FAILED";
            case CUFFT_SETUP_FAILED:
                return "CUFFT_SETUP_FAILED";
            case CUFFT_INVALID_SIZE:
                return "CUFFT_INVALID_SIZE";
            case CUFFT_UNALIGNED_DATA:
                return "CUFFT_UNALIGNED_DATA";
            case CUFFT_INCOMPLETE_PARAMETER_LIST:
                return "CUFFT_INCOMPLETE_PARAMETER_LIST";
            case CUFFT_INVALID_DEVICE:
                return "CUFFT_INVALID_DEVICE";
            case CUFFT_PARSE_ERROR:
                return "CUFFT_PARSE_ERROR";
            case CUFFT_NO_WORKSPACE:
                return "CUFFT_NO_WORKSPACE";
            case CUFFT_NOT_IMPLEMENTED:
                return "CUFFT_NOT_IMPLEMENTED";
            case CUFFT_LICENSE_ERROR:
                return "CUFFT_LICENSE_ERROR";
            case CUFFT_NOT_SUPPORTED:
                return "CUFFT_NOT_SUPPORTED";
        }
        return {};
    }

    constexpr void check(cufftResult_t result, const std::source_location& location = std::source_location::current()) {
        if (result == CUFFT_SUCCESS) {
            /*do nothing*/
        } else {
            panic_at_location(location, "cuFFT failed with error: {}", error2string(result));
        }
    }
}

namespace {
    using namespace ::noa::types;

    // Even values satisfying (2^a) * (3^b) * (5^c) * (7^d).
    constexpr u32 sizes_even_cufft_[315] = {
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 42, 48, 50, 54, 56, 60, 64, 70, 72, 80, 84,
        90, 96, 98, 100, 108, 112, 120, 126, 128, 140, 144, 150, 160, 162, 168, 180, 192, 196, 200, 210, 216,
        224, 240, 250, 252, 256, 270, 280, 288, 294, 300, 320, 324, 336, 350, 360, 378, 384, 392, 400, 420, 432,
        448, 450, 480, 486, 490, 500, 504, 512, 540, 560, 576, 588, 600, 630, 640, 648, 672, 686, 700, 720, 750,
        756, 768, 784, 800, 810, 840, 864, 882, 896, 900, 960, 972, 980, 1000, 1008, 1024, 1050, 1080, 1120,
        1134, 1152, 1176, 1200, 1250, 1260, 1280, 1296, 1344, 1350, 1372, 1400, 1440, 1458, 1470, 1500, 1512,
        1536, 1568, 1600, 1620, 1680, 1728, 1750, 1764, 1792, 1800, 1890, 1920, 1944, 1960, 2000, 2016, 2048,
        2058, 2100, 2160, 2240, 2250, 2268, 2304, 2352, 2400, 2430, 2450, 2500, 2520, 2560, 2592, 2646, 2688,
        2700, 2744, 2800, 2880, 2916, 2940, 3000, 3024, 3072, 3136, 3150, 3200, 3240, 3360, 3402, 3430, 3456,
        3500, 3528, 3584, 3600, 3750, 3780, 3840, 3888, 3920, 4000, 4032, 4050, 4096, 4116, 4200, 4320, 4374,
        4410, 4480, 4500, 4536, 4608, 4704, 4800, 4860, 4900, 5000, 5040, 5120, 5184, 5250, 5292, 5376, 5400,
        5488, 5600, 5670, 5760, 5832, 5880, 6000, 6048, 6144, 6174, 6250, 6272, 6300, 6400, 6480, 6720, 6750,
        6804, 6860, 6912, 7000, 7056, 7168, 7200, 7290, 7350, 7500, 7560, 7680, 7776, 7840, 7938, 8000, 8064,
        8100, 8192, 8232, 8400, 8640, 8748, 8750, 8820, 8960, 9000, 9072, 9216, 9408, 9450, 9600, 9720, 9800,
        10000, 10080, 10206, 10240, 10290, 10368, 10500, 10584, 10752, 10800, 10976, 11200, 11250, 11340, 11520,
        11664, 11760, 12000, 12096, 12150, 12250, 12288, 12348, 12500, 12544, 12600, 12800, 12960, 13230, 13440,
        13500, 13608, 13720, 13824, 14000, 14112, 14336, 14400, 14580, 14700, 15000, 15120, 15360, 15552, 15680,
        15750, 15876, 16000, 16128, 16200, 16384, 16464, 16800
    };

    struct CufftPlan {
        cufftHandle handle{};
        ~CufftPlan() {
            const cufftResult result = cufftDestroy(handle);
            NOA_ASSERT(result == cufftResult::CUFFT_SUCCESS);
            (void) result;
        }
    };

    class CufftManager {
    public:
        CufftManager() = default;

        auto set_cache_limit(i32 size) noexcept {
            m_max_size = noa::safe_cast<size_t>(size);
            i32 n_plans_destructed{};
            while (m_queue.size() > m_max_size) {
                m_queue.pop_back();
                ++n_plans_destructed;
            }
            return n_plans_destructed;
        }

        [[nodiscard]] auto cache_limit() const noexcept {
            return static_cast<i32>(m_max_size);
        }

        [[nodiscard]] auto cache_size() const noexcept {
            return static_cast<i32>(m_queue.size());
        }

        auto clear_cache() noexcept {
            i32 n_plans_destructed{};
            while (not m_queue.empty()) {
                m_queue.pop_back();
                ++n_plans_destructed;
            }
            return n_plans_destructed;
        }

        void reserve_one() noexcept {
            if (not m_queue.empty() and m_queue.size() == m_max_size)
                m_queue.pop_back();
        }

        [[nodiscard]] auto find_in_cache(const std::string& key) const noexcept -> std::shared_ptr<CufftPlan> {
            std::shared_ptr<CufftPlan> out;
            for (const auto& [hash, handle]: m_queue) {
                if (key == hash) {
                    out = handle;
                    break;
                }
            }
            return out;
        }

        auto share_and_push_to_cache(std::string&& key, cufftHandle handle) -> std::shared_ptr<CufftPlan> {
            if (m_max_size == 0) // the cache is turned off
                return std::make_shared<CufftPlan>(handle);

            if (m_queue.size() >= m_max_size)
                m_queue.pop_back();
            m_queue.emplace_front(std::move(key), std::make_shared<CufftPlan>(handle));
            return m_queue.front().second;
        }

        static auto share(cufftHandle handle) -> std::shared_ptr<CufftPlan> {
            return std::make_shared<CufftPlan>(handle);
        }

    private:
        using plan_pair_type = std::pair<std::string, std::shared_ptr<CufftPlan>>;
        std::deque<plan_pair_type> m_queue;
        size_t m_max_size{4};
    };

    // Since a cufft plan can only be used by one thread at a time, for simplicity,
    // have a per-host-thread cache. Each GPU has its own cache, of course.
    CufftManager& get_cache_(noa::cuda::Device device) {
        constexpr i32 MAX_DEVICES = 64;
        noa::cuda::check(device.id() < MAX_DEVICES,
                         "Internal buffer for caching cufft plans is limited to 64 visible devices");

        thread_local std::unique_ptr<CufftManager> g_cache[MAX_DEVICES];
        std::unique_ptr<CufftManager>& cache = g_cache[device.id()];
        if (not cache)
            cache = std::make_unique<CufftManager>();
        return *cache;
    }

    // Offset the type if double precision.
    cufftType_t to_cufft_type_(noa::cuda::fft::Type type, bool is_single_precision) noexcept {
        static_assert(
                CUFFT_Z2Z - CUFFT_C2C == 64 and
                CUFFT_Z2D - CUFFT_C2R == 64 and
                CUFFT_D2Z - CUFFT_R2C == 64);
        switch (type) {
            case noa::cuda::fft::Type::R2C:
                return static_cast<cufftType_t>(CUFFT_R2C + (is_single_precision ? 0 : 64));
            case noa::cuda::fft::Type::C2R:
                return static_cast<cufftType_t>(CUFFT_C2R + (is_single_precision ? 0 : 64));
            case noa::cuda::fft::Type::C2C:
                return static_cast<cufftType_t>(CUFFT_C2C + (is_single_precision ? 0 : 64));
            default:
                noa::panic("Missing type");
        }
    }

    template<typename Real>
    bool is_aligned_to_complex_(Real* ptr) {
        // This is apparently not guaranteed to work by the standard.
        // But this should work in all modern and mainstream platforms.
        constexpr size_t ALIGNMENT = alignof(Complex<Real>);
        return not(reinterpret_cast<std::uintptr_t>(ptr) % ALIGNMENT);
    }
}

namespace noa::cuda::fft::guts {
    auto get_plan(
        Type type,
        bool is_single_precision,
        const Shape4<i64>& shape,
        Device device,
        bool save_in_cache
    ) -> std::shared_ptr<void> {
        auto [batch, shape_3d] = shape.as_safe<i32>().split_batch();
        const i32 rank = shape_3d.ndim();
        NOA_ASSERT(rank == 1 or not ni::is_vector(shape_3d));
        if (rank == 1 and shape_3d[2] == 1) // column vector -> row vector
            std::swap(shape_3d[1], shape_3d[2]);

        const auto i_type = to_underlying(type);
        std::string hash;
        if (rank == 2)
            hash = fmt::format("{}:{},{}:{}:{}", rank, shape_3d[1], shape_3d[2], i_type, batch);
        else if (rank == 3)
            hash = fmt::format("{}:{},{},{}:{}:{}", rank, shape_3d[0], shape_3d[1], shape_3d[2], i_type, batch);
        else // 1
            hash = fmt::format("{}:{}:{}:{}", rank, shape_3d[2], i_type, batch);

        // Look for a cached plan:
        CufftManager& cache = get_cache_(device);
        if (std::shared_ptr handle_ptr = cache.find_in_cache(hash))
            return handle_ptr;

        // We are about to create a new plan and save it in the cache,
        // so make sure there's at least one space left in the cache.
        if (save_in_cache)
            cache.reserve_one();

        // Create and cache the plan:
        cufftHandle handle{};
        for (i32 i = 0; i < 2; ++i) {
            const auto err = ::cufftPlanMany(
                &handle, rank, shape_3d.data() + 3 - rank,
                nullptr, 1, 0, nullptr, 1, 0,
                to_cufft_type_(type, is_single_precision), batch);
            if (err == CUFFT_SUCCESS)
                break;
            // It may have failed because of not enough memory,
            // so clear cache and try again.
            if (i)
                panic("Failed to create a cuFFT plan. {}", error2string(err));
            else
                cache.clear_cache();
        }
        if (save_in_cache)
            return cache.share_and_push_to_cache(std::move(hash), handle);
        return CufftManager::share(handle);
    }

    auto get_plan(
        Type type, bool is_single_precision,
        Strides4<i64> input_strides, Strides4<i64> output_strides,
        const Shape4<i64>& shape, Device device, bool save_in_cache
    ) -> std::shared_ptr<void> {
        auto [batch, shape_3d] = shape.as_safe<i32>().split_batch();
        const i32 rank = shape_3d.ndim();

        NOA_ASSERT(rank == 1 or not ni::is_vector(shape_3d));
        if (rank == 1 and shape_3d[2] == 1) { // column vector -> row vector
            std::swap(shape_3d[1], shape_3d[2]);
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
        }

        const auto i_strides = input_strides.as<i32>();
        const auto o_strides = output_strides.as<i32>();
        Shape3<i32> i_pitch = i_strides.physical_shape();
        Shape3<i32> o_pitch = o_strides.physical_shape();
        const i32 offset = 3 - rank;

        const auto i_type = noa::to_underlying(type);
        std::string hash;
        {
            std::ostringstream tmp;
            tmp << rank << ':';
            for (i32 i = 0; i < rank; ++i)
                tmp << shape_3d[offset + i] << ',';

            for (i32 i = 0; i < rank; ++i)
                tmp << i_pitch[offset + i] << ',';
            tmp << i_strides[3] << ':';
            tmp << i_strides[0] << ':';

            for (i32 i = 0; i < rank; ++i)
                tmp << o_pitch[offset + i] << ',';
            tmp << o_strides[3] << ':';
            tmp << o_strides[0] << ':';

            tmp << i_type << ':';
            tmp << batch;

            hash = tmp.str();
        }

        // Look for a cached plan:
        CufftManager& cache = get_cache_(device);
        if (std::shared_ptr handle_ptr = cache.find_in_cache(hash))
            return handle_ptr;

        // We are about to create a new plan and save it in the cache,
        // so make sure there's at least one space left in the cache.
        if (save_in_cache)
            cache.reserve_one();

        // Create and cache the plan:
        cufftHandle handle{};
        for (i32 i = 0; i < 2; ++i) {
            const auto err = ::cufftPlanMany(
                &handle, rank, shape_3d.data() + offset,
                i_pitch.data() + offset, i_strides[3], i_strides[0],
                o_pitch.data() + offset, o_strides[3], o_strides[0],
                to_cufft_type_(type, is_single_precision), batch);
            if (err == CUFFT_SUCCESS)
                break;
            if (i == 1)
                panic("Failed to create a cuFFT plan. {}", error2string(err));
            else
                cache.clear_cache();
        }
        if (save_in_cache)
            return cache.share_and_push_to_cache(std::move(hash), handle);
        return CufftManager::share(handle);
    }
}

namespace noa::cuda::fft {
    // cuFFT has stronger requirements compared to FFTW.
    auto fast_size(i64 size) -> i64 {
        for (auto nice_size: sizes_even_cufft_) {
            const auto tmp = static_cast<i64>(nice_size);
            if (size <= tmp)
                return tmp;
        }
        return is_even(size) ? size : (size + 1); // fall back to next even number
    }

    auto clear_cache(Device device) noexcept -> i32 {
        return get_cache_(device).clear_cache();
    }

    auto cache_limit(Device device) noexcept -> i32 {
        return get_cache_(device).cache_limit();
    }

    auto cache_size(Device device) noexcept -> i32 {
        return get_cache_(device).cache_size();
    }

    auto set_cache_limit(Device device, i32 count) noexcept -> i32 {
        return get_cache_(device).set_cache_limit(count);
    }

    template<typename T>
    void Plan<T>::execute(T* input, Complex<T>* output, Stream& stream) {
        check(is_aligned_to_complex_(input),
              "cufft requires both the input and output to be aligned to the complex type. This might not "
              "be the case for the real input when operating on a subregion starting at an odd offset. "
              "Hint: copy the real array to a new array or add adequate padding.");

        auto plan = static_cast<CufftPlan*>(m_plan.get())->handle; // int
        check(cufftSetStream(plan, stream.get()));
        if constexpr (is_single_precision)
            check(cufftExecR2C(plan, input, reinterpret_cast<cufftComplex*>(output)));
        else
            check(cufftExecD2Z(plan, input, reinterpret_cast<cufftDoubleComplex*>(output)));
        stream.enqueue_attach(m_plan);
    }

    template<typename T>
    void Plan<T>::execute(Complex<T>* input, T* output, Stream& stream) {
        check(is_aligned_to_complex_(output),
              "cufft requires both the input and output to be aligned to the complex type. This might not "
              "be the case for the real output when operating on a subregion starting at an odd offset. "
              "Hint: copy the real array to a new array or add adequate padding.");

        auto plan = static_cast<CufftPlan*>(m_plan.get())->handle; // int
        check(cufftSetStream(plan, stream.get()));
        if constexpr (is_single_precision)
            check(cufftExecC2R(plan, reinterpret_cast<cufftComplex*>(input), output));
        else
            check(cufftExecZ2D(plan, reinterpret_cast<cufftDoubleComplex*>(input), output));
        stream.enqueue_attach(m_plan);
    }

    template<typename T>
    void Plan<T>::execute(Complex<T>* input, Complex<T>* output, noa::fft::Sign sign, Stream& stream) {
        auto plan = static_cast<CufftPlan*>(m_plan.get())->handle; // int
        check(cufftSetStream(plan, stream.get()));
        if constexpr (is_single_precision) {
            check(cufftExecC2C(plan,
                               reinterpret_cast<cufftComplex*>(input),
                               reinterpret_cast<cufftComplex*>(output),
                               to_underlying(sign)));
        } else {
            check(cufftExecZ2Z(plan,
                               reinterpret_cast<cufftDoubleComplex*>(input),
                               reinterpret_cast<cufftDoubleComplex*>(output),
                               to_underlying(sign)));
        }
        stream.enqueue_attach(m_plan);
    }

    template class Plan<f32>;
    template class Plan<f64>;
}
