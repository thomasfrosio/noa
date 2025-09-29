#include <cufft.h>
#include <deque>
#include <ranges>

#include "noa/core/utils/Strings.hpp"
#include "noa/gpu/cuda/Allocators.hpp"
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

    struct CufftPlan {
        cufftHandle handle{};

        // Whether the workspace is ready.
        // If false, it is waiting for a workspace and is not ready for execution.
        bool has_workspace{};

        // The workspace used by the plan.
        // If managed by cuFFT (default), it is null.
        // It managed by us, it can still be null if no workspace is needed or if has_workspace=false.
        std::shared_ptr<unsigned char[]> workspace{};

        ~CufftPlan() {
            const cufftResult result = cufftDestroy(handle);
            NOA_ASSERT(result == cufftResult::CUFFT_SUCCESS);
            (void) result;
        }
    };

    class CufftCache {
    public:
        CufftCache() = default;

        auto set_limit(i32 size) noexcept {
            m_max_size = noa::safe_cast<size_t>(size);
            i32 n_plans_destructed{};
            while (m_queue.size() > m_max_size) {
                m_queue.pop_back();
                ++n_plans_destructed;
            }
            return n_plans_destructed;
        }

        [[nodiscard]] auto limit() const noexcept {
            return static_cast<i32>(m_max_size);
        }

        [[nodiscard]] auto size() const noexcept {
            return static_cast<i32>(m_queue.size());
        }

        [[nodiscard]] auto workspace_size() const noexcept {
            return m_workspace_size;
        }

        auto clear() noexcept {
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

        void pop() noexcept {
            if (not m_queue.empty())
                m_queue.pop_back();
        }

        [[nodiscard]] auto has(const std::string& key) const noexcept -> bool {
            for (const auto& hash: m_queue | std::views::keys)
                if (key == hash)
                    return true;
            return false;
        }

        [[nodiscard]] auto find(const std::string& key) const noexcept -> std::shared_ptr<CufftPlan> {
            std::shared_ptr<CufftPlan> out;
            for (const auto& [hash, handle]: m_queue) {
                if (key == hash) {
                    out = handle;
                    break;
                }
            }
            return out;
        }

        void push(std::string&& key, cufftHandle handle, size_t workspace_size) {
            check(m_max_size > 0);
            if (m_queue.size() >= m_max_size)
                m_queue.pop_back();

            m_queue.emplace_front(std::move(key), std::make_shared<CufftPlan>(handle, workspace_size == 0));
            m_workspace_size = std::max(m_workspace_size, workspace_size); // record workspace requirement
        }

        void allocate_and_share_workspace(noa::cuda::Device device) {
            if (m_workspace_size <= 0)
                return; // no need to allocate; plans are ready for execution

            using namespace noa::cuda;
            using workspace_deleter_t = AllocatorDevice::Deleter;
            using workspace_shared_ptr_t = std::shared_ptr<unsigned char[]>;
            workspace_shared_ptr_t workspace{nullptr};

            // Allocate memory on the device, safely.
            // AllocatorDevice::allocate(...) throws if it can't allocate, which isn't what we want here.
            auto allocate = [&]() -> bool {
                const auto guard = DeviceGuard(device);
                void* tmp{nullptr};
                cudaError_t err = cudaMalloc(&tmp, m_workspace_size);
                if (err == cudaSuccess) {
                    workspace = workspace_shared_ptr_t(
                        static_cast<unsigned char*>(tmp),
                        workspace_deleter_t{.size = static_cast<i64>(m_workspace_size)}
                    );
                    return true;
                }
                return false;
            };

            // If the allocation fails, release some cached plans
            // (with the hope to release some memory) and try again.
            const i32 n_trials = size() + 1;
            bool success{false};
            for (i32 i{}; i < n_trials; ++i) {
                if (i > 0)
                    pop();
                success = allocate();
                if (success)
                    break;
            }
            check(success, "Could not allocate {} bytes for the cuFFT workspace", m_workspace_size);

            // Set the newly allocated buffer as the workspace for the plans that were waiting for one.
            for (auto& plan: m_queue | std::views::values) {
                if (not plan->has_workspace) {
                    plan->has_workspace = true;
                    plan->workspace = workspace;
                    check(::cufftSetWorkArea(plan->handle, workspace.get()));
                }
            }
            m_workspace_size = 0; // reset for future calls with record_workspace=true
        }

        static auto make_shared(cufftHandle handle, bool has_workspace) {
            return std::make_shared<CufftPlan>(handle, has_workspace);
        }

    public:
        using plan_pair_type = std::pair<std::string, std::shared_ptr<CufftPlan>>;
        std::deque<plan_pair_type> m_queue;
        size_t m_max_size{4};
        size_t m_workspace_size{};
    };

    // Since a cufft plan can only be used by one thread at a time, for simplicity,
    // have a per-host-thread cache. Each GPU has its own cache, of course.
    auto get_cache_(noa::cuda::Device device) -> CufftCache& {
        constexpr i32 MAX_DEVICES = 64;
        noa::cuda::check(device.id() < MAX_DEVICES,
                         "Internal buffer for caching cufft plans is limited to 64 visible devices");

        thread_local std::unique_ptr<CufftCache> g_cache[MAX_DEVICES];
        std::unique_ptr<CufftCache>& cache = g_cache[device.id()];
        if (not cache)
            cache = std::make_unique<CufftCache>();
        return *cache;
    }

    // Offset the type if double precision.
    auto to_cufft_type_(noa::cuda::fft::Type type, bool is_single_precision) noexcept -> cufftType_t {
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

    auto get_plan_(
        std::string&& hash,
        noa::cuda::Device device,
        noa::cuda::fft::Type type,
        bool is_single_precision,
        int rank,
        long long int* n,
        long long int* inembed,
        long long int istride,
        long long int idist,
        long long int* onembed,
        long long int ostride,
        long long int odist,
        long long int batch,
        bool save_in_cache,
        bool plan_only,
        bool record_workspace
    ) -> std::shared_ptr<void> {
        CufftCache& cache = get_cache_(device);

        // Set the workspace for previous calls that had record_workspace=true and needed a temp buffer.
        if (not record_workspace)
            cache.allocate_and_share_workspace(device);

        if (plan_only or record_workspace) {
            // In the case where the plan just needs to be cached, if the cache is turned off or
            // if the plan is already cached, we have nothing to do. Since the plan is not meant
            // for direct execution, we can return nullptr.
            if (not save_in_cache or cache.limit() == 0 or cache.has(hash))
                return nullptr;
        } else {
            // Look for a cached plan.
            if (std::shared_ptr handle_ptr = cache.find(hash))
                return handle_ptr;
        }

        // We are about to create a new plan and save it into the cache,
        // so if the cache is full, we can free the oldest plan to potentially
        // release some memory early.
        if (save_in_cache)
            cache.reserve_one();

        // Create the plan.
        cufftHandle handle{};
        cufftResult_t err{};
        err = ::cufftCreate(&handle);
        check(err == CUFFT_SUCCESS, "Failed to create cufftHandle. {}", noa::cuda::error2string(err));

        if (record_workspace) {
            err = cufftSetAutoAllocation(handle, 0);
            check(err == CUFFT_SUCCESS, "Failed to turn off cufft auto allocation. {}", noa::cuda::error2string(err));
        }

        // If the allocation fails, release some cached plans (with the hope to release some memory) and try again.
        // If record_workspace=true, the workspace isn't allocated so this is unlikely to help...
        size_t work_size = 0;
        const i32 n_trials = cache.size() + 1;
        for (i32 i{}; i < n_trials; ++i) {
            if (i > 0)
                cache.pop();
            err = ::cufftMakePlanMany64(
                handle, rank, n, inembed, istride, idist, onembed, ostride, odist,
                to_cufft_type_(type, is_single_precision), batch, &work_size
            );
            if (err == CUFFT_SUCCESS)
                break;
        }
        check(err == CUFFT_SUCCESS, "Failed to make the plan. {}", noa::cuda::error2string(err));

        if (save_in_cache or cache.limit() > 0) {
            cache.push(std::move(hash), handle, record_workspace ? work_size : 0);
            return plan_only or record_workspace ? nullptr : cache.m_queue.front().second;
        }

        // If the plan isn't saved in the cache, we know it's about to be executed and simply wrap it.
        return std::make_shared<CufftPlan>(handle, true);
    }

    template<typename Real>
    bool is_aligned_to_complex_(Real* ptr) {
        // This is apparently not guaranteed to work by the C++ standard.
        // But this should work in all modern and mainstream platforms.
        constexpr size_t ALIGNMENT = alignof(Complex<Real>);
        return not(reinterpret_cast<std::uintptr_t>(ptr) % ALIGNMENT);
    }
}

namespace noa::cuda::fft::guts {
    auto get_plan(
        Type type, bool is_single_precision,
        const Shape4<i64>& shape, Device device,
        bool save_in_cache, bool plan_only, bool record_workspace
    ) -> std::shared_ptr<void> {
        auto [batch, shape_3d] = shape.as<long long int>().split_batch();
        const auto rank = static_cast<int>(shape_3d.ndim());
        check(rank == 1 or not ni::is_vector(shape_3d));
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

        return get_plan_(
            std::move(hash), device, type, is_single_precision, rank,
            shape_3d.data() + 3 - rank, nullptr, 1, 0, nullptr, 1, 0, batch,
            save_in_cache, plan_only, record_workspace
        );
    }

    auto get_plan(
        Type type, bool is_single_precision,
        Strides4<i64> input_strides, Strides4<i64> output_strides,
        const Shape4<i64>& shape, Device device,
        bool save_in_cache, bool plan_only, bool record_workspace
    ) -> std::shared_ptr<void> {
        using lli = long long int;
        auto [batch, shape_3d] = shape.as_safe<lli>().split_batch();
        const auto rank = static_cast<int>(shape_3d.ndim());

        check(rank == 1 or not ni::is_vector(shape_3d));
        if (rank == 1 and shape_3d[2] == 1) { // column vector -> row vector
            std::swap(shape_3d[1], shape_3d[2]);
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
        }

        const auto i_strides = input_strides.as<lli>();
        const auto o_strides = output_strides.as<lli>();
        Shape3<lli> i_pitch = i_strides.physical_shape();
        Shape3<lli> o_pitch = o_strides.physical_shape();
        const int offset = 3 - rank;

        const auto i_type = noa::to_underlying(type);
        std::string hash;
        {
            std::ostringstream tmp;
            tmp << rank << ':';
            for (int i{}; i < rank; ++i)
                tmp << shape_3d[offset + i] << ',';

            for (int i{}; i < rank; ++i)
                tmp << i_pitch[offset + i] << ',';
            tmp << i_strides[3] << ':';
            tmp << i_strides[0] << ':';

            for (int i{}; i < rank; ++i)
                tmp << o_pitch[offset + i] << ',';
            tmp << o_strides[3] << ':';
            tmp << o_strides[0] << ':';

            tmp << i_type << ':';
            tmp << batch;

            hash = tmp.str();
        }

        return get_plan_(
            std::move(hash), device, type, is_single_precision, rank,
            shape_3d.data() + offset,
            i_pitch.data() + offset, i_strides[3], i_strides[0],
            o_pitch.data() + offset, o_strides[3], o_strides[0],
            batch, save_in_cache, plan_only, record_workspace
        );
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
        return get_cache_(device).clear();
    }

    auto cache_limit(Device device) noexcept -> i32 {
        return get_cache_(device).limit();
    }

    auto cache_size(Device device) noexcept -> i32 {
        return get_cache_(device).size();
    }

    auto set_cache_limit(Device device, i32 count) noexcept -> i32 {
        return get_cache_(device).set_limit(count);
    }

    auto workspace_left_to_allocate(Device device) noexcept -> size_t {
        return get_cache_(device).workspace_size();
    }

    template<typename T>
    void Plan<T>::execute(T* input, Complex<T>* output, Stream& stream) && {
        if (m_plan == nullptr)
            return;
        check(is_aligned_to_complex_(input),
              "cufft requires both the input and output to be aligned to the complex type. This might not "
              "be the case for the real input when operating on a subregion starting at an odd offset. "
              "Hint: copy the real array to a new array or add adequate padding.");

        auto plan = static_cast<CufftPlan*>(m_plan.get());
        check(plan->has_workspace, "Trying to run a plan without a workspace");
        check(cufftSetStream(plan->handle, stream.get()));
        if constexpr (is_single_precision)
            check(cufftExecR2C(plan->handle, input, reinterpret_cast<cufftComplex*>(output)));
        else
            check(cufftExecD2Z(plan->handle, input, reinterpret_cast<cufftDoubleComplex*>(output)));
        stream.enqueue_attach(std::move(m_plan));
    }

    template<typename T>
    void Plan<T>::execute(Complex<T>* input, T* output, Stream& stream) && {
        if (m_plan == nullptr)
            return;
        check(is_aligned_to_complex_(output),
              "cufft requires both the input and output to be aligned to the complex type. This might not "
              "be the case for the real output when operating on a subregion starting at an odd offset. "
              "Hint: copy the real array to a new array or add adequate padding.");

        auto plan = static_cast<CufftPlan*>(m_plan.get());
        check(cufftSetStream(plan->handle, stream.get()));
        if constexpr (is_single_precision)
            check(cufftExecC2R(plan->handle, reinterpret_cast<cufftComplex*>(input), output));
        else
            check(cufftExecZ2D(plan->handle, reinterpret_cast<cufftDoubleComplex*>(input), output));
        stream.enqueue_attach(std::move(m_plan));
    }

    template<typename T>
    void Plan<T>::execute(Complex<T>* input, Complex<T>* output, noa::fft::Sign sign, Stream& stream) && {
        if (m_plan == nullptr)
            return;

        auto plan = static_cast<CufftPlan*>(m_plan.get());
        check(plan->has_workspace, "Trying to run a plan without a workspace");
        check(cufftSetStream(plan->handle, stream.get()));
        if constexpr (is_single_precision) {
            check(cufftExecC2C(
                plan->handle,
                reinterpret_cast<cufftComplex*>(input),
                reinterpret_cast<cufftComplex*>(output),
                to_underlying(sign)));
        } else {
            check(cufftExecZ2Z(
                plan->handle,
                reinterpret_cast<cufftDoubleComplex*>(input),
                reinterpret_cast<cufftDoubleComplex*>(output),
                to_underlying(sign)));
        }
        stream.enqueue_attach(std::move(m_plan));
    }

    template class Plan<f32>;
    template class Plan<f64>;
}
