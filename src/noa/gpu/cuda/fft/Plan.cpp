#include <deque>
#include "noa/gpu/cuda/fft/Plan.h"
#include "noa/common/string/Format.h"

namespace {
    // Even values satisfying (2^a) * (3^b) * (5^c) * (7^d).
    constexpr uint sizes_even_cufft_[315] = {
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

    class CufftCache {
    public:
        CufftCache() : m_max_size(4) {}

        void limit(size_t size) noexcept {
            m_max_size = size;
            while (m_queue.size() > m_max_size)
                m_queue.pop_back();
        }

        void clear() noexcept {
            while (!m_queue.empty())
                m_queue.pop_back();
        }

        [[nodiscard]] size_t limit() const noexcept { return m_max_size; }

        [[nodiscard]] std::shared_ptr<cufftHandle> find(const std::string& key) const noexcept {
            std::shared_ptr<cufftHandle> res;
            for (const auto& i: m_queue) {
                if (key == i.first) {
                    res = i.second;
                    break;
                }
            }
            return res;
        }

        std::shared_ptr<cufftHandle>& push(std::string&& key, cufftHandle plan) {
            auto deleter = [](const cufftHandle* ptr) {
                cufftDestroy(*ptr);
                delete ptr;
            };

            if (m_queue.size() >= m_max_size)
                m_queue.pop_back();
            m_queue.emplace_front(std::move(key),
                                  std::shared_ptr<cufftHandle>{new cufftHandle{plan}, deleter});
            return m_queue.front().second;
        }

    private:
        using plan_pair_t = typename std::pair<std::string, std::shared_ptr<cufftHandle>>;
        std::deque<plan_pair_t> m_queue;
        size_t m_max_size;
    };

    // Since a cufft plan can only be used by one thread at a time, for simplicity, have a per-thread cache.
    // Each GPU has its own cache of course.
    CufftCache& getCache_(int device) {
        constexpr size_t MAX_DEVICES = 16;
        thread_local std::unique_ptr<CufftCache> g_cache[MAX_DEVICES];

        std::unique_ptr<CufftCache>& cache = g_cache[device];
        if (!cache)
            cache = std::make_unique<CufftCache>();
        return *cache;
    }
}

namespace noa::cuda::fft::details {
    std::shared_ptr<cufftHandle> getPlan(cufftType_t type, size4_t shape, int device) {
        const auto batch = static_cast<int>(shape[0]);
        int3_t s_shape{shape.get(1)};
        const int rank = s_shape.ndim();

        using enum_type = std::underlying_type_t<cufftType_t>;
        const auto type_ = static_cast<enum_type>(type);
        std::string hash;
        if (rank == 2)
            hash = string::format("{}:{},{}:{}:{}", rank, s_shape[1], s_shape[2], type_, batch);
        else if (rank == 3)
            hash = string::format("{}:{},{},{}:{}:{}", rank, s_shape[0], s_shape[1], s_shape[2], type_, batch);
        else
            hash = string::format("{}:{}:{}:{}", rank, s_shape[2], type_, batch);

        // Look for a cached plan:
        CufftCache& cache = getCache_(device);
        std::shared_ptr<cufftHandle> handle_ptr = cache.find(hash);
        if (handle_ptr)
            return handle_ptr;

        // Create and cache the plan:
        cufftHandle plan{};
        for (size_t i = 0; i < 2; ++i) {
            const auto err = ::cufftPlanMany(&plan, rank, s_shape.get(3 - rank),
                                             nullptr, 1, 0, nullptr, 1, 0,
                                             type, batch);
            if (err == CUFFT_SUCCESS)
                break;
            // It may have failed because of not enough memory,
            // so clear cache and try again.
            if (i)
                NOA_THROW("Failed to create a cuFFT plan. {}", err);
            else
                cache.clear();
        }
        return cache.push(std::move(hash), plan);
    }

    std::shared_ptr<cufftHandle> getPlan(cufftType_t type, size4_t input_stride, size4_t output_stride,
                                         size4_t shape, int device) {
        int3_t s_shape(shape.get(1));
        const int4_t i_stride(input_stride);
        const int4_t o_stride(output_stride);
        int3_t i_pitch(i_stride.pitches());
        int3_t o_pitch(o_stride.pitches());
        const int rank = s_shape.ndim();
        const auto batch = static_cast<int>(shape[0]);
        const int offset = 3 - rank;

        using enum_type = std::underlying_type_t<cufftType_t>;
        const auto type_ = static_cast<enum_type>(type);
        std::string hash;
        {
            static std::ostringstream tmp;
            tmp.seekp(0); // clear

            tmp << rank << ':';
            for (int i = 0; i < rank; ++i)
                tmp << s_shape[offset + i] << ',';

            for (int i = 0; i < rank; ++i)
                tmp << i_pitch[offset + i] << ',';
            tmp << i_stride[3] << ':';
            tmp << i_stride[0] << ':';

            for (int i = 0; i < rank; ++i)
                tmp << o_pitch[offset + i] << ',';
            tmp << o_stride[3] << ':';
            tmp << o_stride[0] << ':';

            tmp << type_ << ':';
            tmp << batch;
            hash = tmp.str();
        }

        // Look for a cached plan:
        CufftCache& cache = getCache_(device);
        std::shared_ptr<cufftHandle> handle_ptr = cache.find(hash);
        if (handle_ptr)
            return handle_ptr;

        // Create and cache the plan:
        cufftHandle plan;
        for (size_t i = 0; i < 2; ++i) {
            const auto err = ::cufftPlanMany(&plan, rank, s_shape.get(offset),
                                             i_pitch.get(offset), i_stride[3], i_stride[0],
                                             o_pitch.get(offset), o_stride[3], o_stride[0],
                                             type, batch);
            if (err == CUFFT_SUCCESS)
                break;
            if (i)
                NOA_THROW("Failed to create a cuFFT plan. {}", err);
            else
                cache.clear();
        }
        return cache.push(std::move(hash), plan);
    }

    void cacheClear(int device) noexcept {
        getCache_(device).clear();
    }

    void cacheLimit(int device, size_t count) noexcept {
        getCache_(device).limit(count);
    }
}

namespace noa::cuda::fft {
    // cuFFT has stronger requirements compared FFTW.
    size_t fastSize(size_t size) {
        auto tmp = static_cast<uint>(size);
        for (uint nice_size: sizes_even_cufft_)
            if (tmp < nice_size)
                return static_cast<size_t>(nice_size);
        return (size % 2 == 0) ? size : (size + 1); // fall back to next even number
    }
}
