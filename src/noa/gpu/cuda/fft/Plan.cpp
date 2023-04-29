#include <deque>

#include "noa/gpu/cuda/fft/Plan.hpp"
#include "noa/core/string/Format.hpp"

namespace {
    using namespace ::noa;

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

    class CufftManager {
    public:
        CufftManager() = default;

        void set_cache_limit(i32 size) noexcept {
            NOA_ASSERT(size >= 0);
            m_max_size = clamp_cast<size_t>(size);
            while (m_queue.size() > m_max_size)
                m_queue.pop_back();
        }

        void clear_cache() noexcept {
            while (!m_queue.empty())
                m_queue.pop_back();
        }

        [[nodiscard]] i32 cache_limit() const noexcept { return static_cast<i32>(m_max_size); }

        [[nodiscard]] Shared<cufftHandle> find_in_cache(const std::string& key) const noexcept {
            Shared<cufftHandle> res;
            for (const auto& i: m_queue) {
                if (key == i.first) {
                    res = i.second;
                    break;
                }
            }
            return res;
        }

        Shared<cufftHandle> share_and_push_to_cache(std::string&& key, cufftHandle plan) {
            auto deleter = [](const cufftHandle* ptr) {
                cufftDestroy(*ptr);
                delete ptr;
            };

            // The cache is turned off.
            if (m_max_size == 0)
                return Shared<cufftHandle>(new cufftHandle{plan}, deleter);

            if (m_queue.size() >= m_max_size)
                m_queue.pop_back();
            m_queue.emplace_front(std::move(key), Shared<cufftHandle>(new cufftHandle{plan}, deleter));
            return m_queue.front().second;
        }

        static Shared<cufftHandle> share(cufftHandle plan) {
            auto deleter = [](const cufftHandle* ptr) {
                cufftDestroy(*ptr);
                delete ptr;
            };
            return Shared<cufftHandle>(new cufftHandle{plan}, deleter);
        }

    private:
        using plan_pair_type = typename std::pair<std::string, Shared<cufftHandle>>;
        std::deque<plan_pair_type> m_queue;
        size_t m_max_size{4};
    };

    // Since a cufft plan can only be used by one thread at a time, for simplicity, have a per-host-thread cache.
    // Each GPU has its own cache of course.
    CufftManager& get_cache_(i32 device) {
        constexpr i32 MAX_DEVICES = 16;
        thread_local Unique<CufftManager> g_cache[MAX_DEVICES];

        Unique<CufftManager>& cache = g_cache[device];
        if (!cache)
            cache = std::make_unique<CufftManager>();
        return *cache;
    }
}

namespace noa::cuda::fft::details {
    auto get_plan(cufftType_t type, const Shape4<i64>& shape, i32 device, bool save_in_cache) -> Shared<cufftHandle> {
        const auto s_shape = shape.as_safe<i32>();
        const auto batch = s_shape[0];
        auto shape_3d = s_shape.pop_front();
        const i32 rank = shape_3d.ndim();
        NOA_ASSERT(rank == 1 || !noa::indexing::is_vector(shape_3d));
        if (rank == 1 && shape_3d[2] == 1) // column vector -> row vector
            std::swap(shape_3d[1], shape_3d[2]);

        const auto i_type = noa::traits::to_underlying(type);
        std::string hash;
        if (rank == 2)
            hash = noa::string::format("{}:{},{}:{}:{}", rank, shape_3d[1], shape_3d[2], i_type, batch);
        else if (rank == 3)
            hash = noa::string::format("{}:{},{},{}:{}:{}", rank, shape_3d[0], shape_3d[1], shape_3d[2], i_type, batch);
        else // 1
            hash = noa::string::format("{}:{}:{}:{}", rank, shape_3d[2], i_type, batch);

        // Look for a cached plan:
        CufftManager& cache = get_cache_(device);
        Shared<cufftHandle> handle_ptr = cache.find_in_cache(hash);
        if (handle_ptr)
            return handle_ptr;

        // Create and cache the plan:
        cufftHandle plan{};
        for (i32 i = 0; i < 2; ++i) {
            const auto err = ::cufftPlanMany(
                    &plan, rank, shape_3d.data() + 3 - rank,
                    nullptr, 1, 0, nullptr, 1, 0,
                    type, batch);
            if (err == CUFFT_SUCCESS)
                break;
            // It may have failed because of not enough memory,
            // so clear cache and try again.
            if (i)
                NOA_THROW("Failed to create a cuFFT plan. {}", to_string(err));
            else
                cache.clear_cache();
        }
        if (save_in_cache)
            return cache.share_and_push_to_cache(std::move(hash), plan);
        else
            return CufftManager::share(plan);
    }

    auto get_plan(
            cufftType_t type,
            Strides4<i64> input_strides, Strides4<i64> output_strides,
            const Shape4<i64>& shape, i32 device, bool save_in_cache
    ) -> Shared<cufftHandle> {
        auto s_shape = shape.as_safe<i32>();
        const auto batch = s_shape[0];
        auto shape_3d = s_shape.pop_front();
        const i32 rank = shape_3d.ndim();

        NOA_ASSERT(rank == 1 || !noa::indexing::is_vector(shape_3d));
        if (rank == 1 && shape_3d[2] == 1) { // column vector -> row vector
            std::swap(shape_3d[1], shape_3d[2]);
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
        }

        const auto i_strides = input_strides.as<i32>();
        const auto o_strides = output_strides.as<i32>();
        Shape3<i32> i_pitch = i_strides.physical_shape();
        Shape3<i32> o_pitch = o_strides.physical_shape();
        const i32 offset = 3 - rank;

        const auto i_type = noa::traits::to_underlying(type);
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
        Shared<cufftHandle> handle_ptr = cache.find_in_cache(hash);
        if (handle_ptr)
            return handle_ptr;

        // Create and cache the plan:
        cufftHandle plan{};
        for (i32 i = 0; i < 2; ++i) {
            const auto err = ::cufftPlanMany(
                    &plan, rank, shape_3d.data() + offset,
                    i_pitch.data() + offset, i_strides[3], i_strides[0],
                    o_pitch.data() + offset, o_strides[3], o_strides[0],
                    type, batch);
            if (err == CUFFT_SUCCESS)
                break;
            if (i == 1)
                NOA_THROW("Failed to create a cuFFT plan. {}", to_string(err));
            else
                cache.clear_cache();
        }
        if (save_in_cache)
            return cache.share_and_push_to_cache(std::move(hash), plan);
        else
            return CufftManager::share(plan);
    }

    void cache_clear(i32 device) noexcept {
        get_cache_(device).clear_cache();
    }

    void cache_set_limit(i32 device, i32 count) noexcept {
        get_cache_(device).set_cache_limit(count);
    }
}

namespace noa::cuda::fft {
    // cuFFT has stronger requirements compared FFTW.
    i64 fast_size(i64 size) {
        for (auto nice_size: sizes_even_cufft_) {
            const auto tmp = static_cast<i64>(nice_size);
            if (size < tmp)
                return tmp;
        }
        return (size % 2 == 0) ? size : (size + 1); // fall back to next even number
    }
}
