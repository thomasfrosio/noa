#pragma once

#include <forward_list>
#include <mutex>
#include <atomic>

#include "noa/base/Vec.hpp"
#include "noa/runtime/core/Shareable.hpp"
#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/cuda/Error.hpp"
#include "noa/runtime/cuda/Device.hpp"
#include "noa/runtime/cuda/Runtime.hpp"

#if (defined(NOA_ENABLE_CUDA) && defined(__CUDACC__)) || \
    (defined(NOA_ENABLE_HIP) && defined(__HIPCC__)) || \
     defined(NOA_ENABLE_SYCL)
#   include "noa/runtime/cuda/ComputeHandle.hpp"
#endif

namespace noa::gpu::details {
    // Registry to attach a shared_ptr to a stream, using a FIFO buffer.
    //
    // Kernel execution is asynchronous relative to the host. As such, we need a way to know when the
    // kernel is running and when it's done, so that we can make sure the resources used by this kernel
    // will never be deleted while the kernel is running. As a solution to this, this registry can attach
    // resources to a stream and can release these resources (using callbacks) when the kernel is done executing.
    //
    // Any type can be passed to try_insert(), but only std::shared_ptr(s) can and will be added to the registry.
    // Types with a .share() method returning a type that can be converted to a std::shared_ptr (e.g., Array or Texture)
    // can also be passed, and the returned std::shared_ptr will be added to the registry.
    //
    // 1) try_insert() should be called after the kernel launch. It forwards the shared_ptr(s) at the back
    //    of the registry. Resources are attached to a unique key, and the key is per try_insert() call, meaning that
    //    all resources passed to the same try_insert() call have the same lifetime.
    // 2) Then, a callback is enqueued to the stream. When the kernel is done execution, the callback is
    //    called. Note that the callback isn't allowed to call the CUDA API, so it cannot remove the registered
    //    shared_ptr(s) in case one of them calls its deleter (which may use the CUDA API). Instead, the callback
    //    simply increments the "callback_count", signaling the stream reached this point and the callback was called.
    // 3) At the next try_insert() or clear() call, "callback_count" is decremented back to zero. For every decrement,
    //    the resources with the same keys as the first in-used resource at the front of the registry are flagged
    //    as unused. Unused resources are then cleared from the registry, effectively releasing them.
    //
    // try_insert() is meant to be called with a bunch of different types at once, and as mentioned above, it will
    // filter the types that can be added to the registry from the types that cannot. It is indeed more efficient
    // to call try_insert() once with many types than to call it once for every type.
    //
    // TODO cudaFree(Async) is synchronizing the device/stream or is stream-ordered, so this shouldn't be necessary
    //      for CUDA-managed memory. For unregistered memory that the stream depends on (e.g. CPU<->GPU copies), this
    //      should be useful. Update the documentation to reflect that?
    class StreamResourceRegistry {
    private:
        std::mutex m_mutex;
        std::vector<std::pair<isize, std::shared_ptr<const void>>> m_registry;
        isize m_key{};

    public:
        std::atomic<i32> callback_count{0};

    public:
        // Adds one or multiple shared_ptr to the back of the registry.
        // The shared_ptr reference count is increased by one.
        // This function also deletes the registry from unused shared_ptr.
        template<typename ...Args>
        auto try_insert(Args&& ... args) -> bool {
            const std::scoped_lock lock(m_mutex);
            clear_unused_();

            if constexpr (((nt::shareable<Args> or nt::shareable_using_share<Args>) or ...)) {
                isize key = [this] {
                    if (m_key == std::numeric_limits<isize>::max())
                        m_key = 0;
                    return m_key++;
                }();
                ([&, this]<typename T>(T&& arg) {
                    if constexpr (nt::shareable<T>)
                        m_registry.emplace_back(key, NOA_FWD(arg));
                    else if constexpr (nt::shareable_using_share<T>)
                        m_registry.emplace_back(key, NOA_FWD(arg).share());
                    // else: do nothing
                }(NOA_FWD(args)), ...);
                return true;
            } else {
                return false;
            }
        }

        // Removes from the registry the shared_ptr(s) flagged as unused.
        // The shared_ptr reference count is decreased by one, thus their deleter can be called.
        void clear() {
            const std::scoped_lock lock(m_mutex);
            clear_unused_();
        }

        void clear_after_sync() {
            const std::scoped_lock lock(m_mutex);
            // This is unsafe because we may have a callback incrementing the count while registry.clear() is called,
            // which would create a mismatch between the count and the registry. That's why this should only be
            // called just after synchronization, i.e. when the CUDA stream is empty/idle.
            callback_count.store(0);
            m_key = 0; // reset the key (this is optional)
            m_registry.clear();
        }

    private:
        // !! Make sure to lock before calling this function !!
        void clear_unused_() {
            // 1. Flag unused resources as such.
            // When a callback is called, callback_count is incremented.
            // For each increment, it means the oldest valid resources are now unused and can be flagged as such.
            i32 count = callback_count.load();
            while (count != 0 and callback_count.compare_exchange_weak(count, count - 1)) {
                // Flag as unused (key == -1) the first valid resources at the front of the registry.
                // Multiple resources can have the same key (they were enqueued in the same try_insert() call),
                // so we need to flag the first (from the front) _valid_ resource and flag any other resource with
                // the same key.
                isize key = -1;
                for (auto& p: m_registry) {
                    if (key == -1 and p.first != -1)
                        key = p.first; // oldest valid resource
                    if (key != -1 and p.first == key)
                        p.first = -1; // is now flagged as unused
                }
                --count;
            }

            // 2. Resources with a key == -1 are unused and should be deleted.
            // If it was the last record of that shared_ptr, the reference count goes to zero,
            // and the deleter is called (which can in turn call the CUDA API).
            bool something_was_deleted{false};
            for (auto& p: m_registry) {
                if (p.first == -1) {
                    p.second = nullptr;
                    something_was_deleted = true;
                }
            }

            // 3. Erase nullptrs from the registry, otherwise it would keep growing.
            if (something_was_deleted)
                std::erase_if(m_registry, [](const auto& pair) { return pair.second == nullptr; });
        }
    };
}

#if (defined(NOA_ENABLE_CUDA) && defined(__CUDACC__)) || (defined(NOA_ENABLE_HIP) && defined(__HIPCC__))
namespace noa::gpu::details {
    template<typename Handle, typename Operator>
    __global__ __launch_bounds__(Handle::block_type::block_size)
    void compute_kernel(Handle handle, Operator op) {
        op(handle);
    }
}
#endif

namespace noa::gpu {
    struct LaunchConfig {
        Vec<u32, 3> grid_shape{}; // ZYX
        Vec<u32, 3> block_shape{}; // ZYX
        usize n_bytes_of_scratch{};
        bool is_cooperative{};
    };

    // A reference-counted GPU stream (and its associated device).
    class Stream {
    public:
        enum class Mode : u32 {
            // Work running in the created stream is implicitly synchronized with the null stream.
            // In SYCL, sync streams are implicitly synchronized when the null stream is synchronized,
            // but may still overlap execution with the null stream.
            SYNC = 0,

            // Work running in the created stream may run concurrently with the null stream and
            // there is no implicit synchronization performed between it and the null stream.
            ASYNC = 1,

            // Default (aka null) stream. One per device.
            DEFAULT = 2,
        };
        using enum Mode;

        struct Core {
            Core(Device device, Mode mode, bool enable_profiling);
            ~Core();

            details::StreamResourceRegistry resource_registry{};
#if defined(NOA_ENABLE_CUDA)
            cudaStream_t stream_handle{};
#elif defined(NOA_ENABLE_ROCM)
            hipStream_t stream_handle{};
#elif defined(NOA_ENABLE_SYCL)
            sycl::queue* queue;
            std::optional<sycl::event> last_event{};
            std::mutex last_event_mutex{};
            Mode queue_mode{};

            // For enqueue_attach, the CUDA/HIP implementation uses host callbacks to notify the registry that the
            // attachments can be dropped. SYCL does have a host_task we could use, but AdaptiveCpp doesn't support it
            // (apparently for good reasons, https://github.com/AdaptiveCpp/AdaptiveCpp/issues/1589). So instead,
            // store the last_event from the queue when enqueue_attach is called, and check its completion status.
            std::deque<sycl::event> pending_events{};

            void poll_events();
#endif
        };

    public:
        /// Creates an empty (invalid) stream.
        constexpr explicit Stream() = default;

        /// Creates a new stream on the current device.
        explicit Stream(Mode mode, bool enable_profiling) : Stream(Device::current(), mode, enable_profiling) {}

        /// Creates a new stream on a given device.
        explicit Stream(Device device, Mode mode, bool enable_profiling)
            : m_core(std::make_shared<Core>(device, mode, enable_profiling)), m_device(device) {}

    public:
        /// Enqueues a kernel to the stream.
        template<nt::compute_handle_gpu Handle, typename Func>
        void enqueue(
            const LaunchConfig& config,
            Handle handle,
            Func&& function
        ) {
            check(m_core);
            if (config.is_cooperative)
                panic("Cooperative kernels are not supported");

#if defined(NOA_ENABLE_CUDA)
#   ifndef __CUDACC__
            panic("To launch kernels, the compilation must be steered by nvcc or nvc++, in CUDA mode");
#   else
            const auto guard = DeviceGuard(m_device);
            auto grid_shape = dim3(config.grid_shape[2], config.grid_shape[1], config.grid_shape[0]); // XYZ
            auto block_shape = dim3(config.block_shape[2], config.block_shape[1], config.block_shape[0]); // XYZ
            details::compute_kernel<<<grid_shape, block_shape, config.n_bytes_of_scratch, m_core->stream_handle>>>(
                handle, NOA_FWD(function));
            check(cudaGetLastError());
#   endif
#elif defined(NOA_ENABLE_ROCM)
#   ifndef __HIPCC__
            panic("To launch kernels, the compilation must be steered by hipcc, in HIP mode");
#   else
            const auto guard = DeviceGuard(m_device);
            auto grid_shape = dim3(config.grid_shape[2], config.grid_shape[1], config.grid_shape[0]); // XYZ
            auto block_shape = dim3(config.block_shape[2], config.block_shape[1], config.block_shape[0]); // XYZ
            handle.set_hip_total_scratch_size(config.n_bytes_of_scratch);
            details::compute_kernel<<<grid_shape, block_shape, config.n_bytes_of_scratch, m_core->stream_handle>>>(
                handle, NOA_FWD(function));
            check(hipGetLastError());
#   endif
#elif defined(NOA_ENABLE_SYCL)
            // Dimensions that can't be expressed with N dims must be trivial.
            // ComputeHandle already static asserts BLOCK_NDIM > GRID_NDIM.
            static constexpr usize N = Handle::GRID_NDIM;
            if constexpr (N < 3) {
                check(config.block_shape[0] == 1 and config.grid_shape[0] == 1);
                if constexpr (N < 2)
                    check(config.block_shape[1] == 1 and config.grid_shape[1] == 1);
            }

            auto local = sycl::range<N>{};
            auto global = sycl::range<N>{};
            for (usize i{}; i < N; ++i) {
                const usize j = 3 - N + i;
                local[static_cast<i32>(i)] = config.block_shape[j];
                global[static_cast<i32>(i)] = config.grid_shape[j] * config.block_shape[j];
            }

            const auto nd_range = sycl::nd_range<N>(global, local);
            const auto n_bytes = static_cast<u32>(config.n_bytes_of_scratch);

            const std::scoped_lock lock(m_core->last_event_mutex);
            m_core->last_event = m_core->queue->submit([&](sycl::handler& h) {
                // TODO https://adaptivecpp.github.io/AdaptiveCpp/extensions/#acpp_ext_coarse_grained_events
                // TODO https://adaptivecpp.github.io/AdaptiveCpp/extensions/#acpp_ext_cg_property_prefer_group_size
                // TODO Turn off shared memory at compile-time to save space in kernel function (32 bytes).
                //      This could be an option to the compute function (IwiseCompileOptions::disable_scratch),
                //      but it could also be deduced from the user operator (if it doesn't take the compute handle)
                //      and/or the presence of a type alias.
                auto shared = sycl::local_accessor<std::byte>{};
                if (n_bytes > 0) {
                    shared = sycl::local_accessor<std::byte>(sycl::range(n_bytes), h);
                    handle.set_sycl_total_scratch_size(n_bytes);
                }

                h.parallel_for(nd_range, [=, func = std::decay_t<Func>(NOA_FWD(function))](sycl::nd_item<N> item) {
                    // SYCL requires the kernel lambda/functor operator() to be const.
                    // We want to support mutable operator() for func, so create
                    // non-const local copies, hoping they'll be optimized out.
                    // TODO try mutable member variable in struct
                    auto handle_ = handle;
                    auto func_ = func;

                    // Store the nd_item, and the scratch pointer if scratch was allocated.
                    handle_.set_sycl_nd_item(&item);
                    if (handle_.total_scratch_size() != 0)
                        handle_.set_sycl_scratch_pointer(
                            shared.get_multi_ptr<sycl::access::decorated::no>().get());

                    func_(handle_);
                });
            });
#endif
        }

        /// Enqueues attachments to the stream.
        /// \details
        ///     Attachments are resources that should be kept alive until the stream reaches this point. A resource is
        ///     anything that is convertible to `std::shared_ptr<const void>` or a type that has a .share() member
        ///     function returning a `std::shared_ptr`. Passing anything else in \p attachments is also valid but will
        ///     be ignored. In practice, these shared_ptr are copied into an internal buffer, thereby incrementing
        ///     their reference count. When the stream reaches this point, the internal buffer flags these shared_ptr
        ///     as unused and ready to be destructed. Due to implementation limitations (mostly from the driver), the
        ///     buffer cannot destroy these shared_ptr right away (see .update_registry_callback_()). Instead, at the
        ///     next synchronization or enqueueing, the flagged shared_ptrs will be destroyed, thereby decrementing
        ///     their reference count. The flagged shared_ptr can also be explicitly cleared using .clear_attachments().
        template<typename ...Args>
        void enqueue_attach(Args&& ... args) {
            NOA_ASSERT(m_core);
            #ifdef NOA_ENABLE_SYCL
            if (not m_core->last_event)
                return; // the stream has no enqueued work
            m_core->poll_events();
            #endif

            if (m_core->resource_registry.try_insert(NOA_FWD(args)...)) {
                // Something was inserted, so enqueue a "cleaning" callback.
#if defined(NOA_ENABLE_CUDA)
                void (* fun_ptr)(void*) = &update_registry_callback_;
                check(cudaLaunchHostFunc(m_core->stream_handle, fun_ptr, &m_core->resource_registry));
#elif defined(NOA_ENABLE_ROCM)
                void (* fun_ptr)(void*) = &update_registry_callback_;
                check(hipLaunchHostFunc(m_core->stream_handle, fun_ptr, &m_core->resource_registry));
#elif defined(NOA_ENABLE_SYCL)
                // Save the last_event as marker for this attachment.
                // When completed, the attachments can be released.
                const std::scoped_lock lock(m_core->last_event_mutex);
                m_core->pending_events.push_back(*m_core->last_event);
#endif
            }
        }

#ifdef NOA_ENABLE_SYCL
        [[nodiscard]] auto enqueue_sycl_event_barrier() const -> std::optional<sycl::event> {
            check(m_core);
            auto l = std::scoped_lock(m_core->last_event_mutex);
            if (m_core->last_event)
                return m_core->last_event;

            m_core->last_event = m_core->queue->submit([](sycl::handler& h) { h.single_task([]{}); });
            return m_core->last_event;
        }

        [[nodiscard]] auto enqueue_sycl_memcpy(void* dst, const void* src, usize n_bytes) const {
            auto l = std::scoped_lock(m_core->last_event_mutex);
            m_core->last_event = m_core->queue->memcpy(dst, src, n_bytes);
        }

        [[nodiscard]] auto enqueue_sycl_memset(void* ptr, int value, usize n_bytes) const {
            auto l = std::scoped_lock(m_core->last_event_mutex);
            m_core->last_event = m_core->queue->memset(ptr, value, n_bytes);
        }

        [[nodiscard]] auto enqueue_sycl_prefetch_gpu(const void* ptr, usize n_bytes) const {
            auto l = std::scoped_lock(m_core->last_event_mutex);
            m_core->last_event = m_core->queue->prefetch(ptr, n_bytes);
        }

        [[nodiscard]] auto enqueue_sycl_prefetch_cpu(const void* ptr, usize n_bytes) const {
#   ifdef ACPP_EXT_PREFETCH_HOST
            auto l = std::scoped_lock(m_core->last_event_mutex);
            m_core->last_event = m_core->queue->prefetch_host(ptr, n_bytes);
#   else
            // TODO DPC++
#   endif
        }
#endif

        /// Clears the registry from any unused attached data.
        /// This is forcing a clean, removing all resources regardless of whether they are still in use.
        void clear_attachments() const {
            #ifdef NOA_ENABLE_SYCL
            m_core->poll_events();
            #endif
            m_core->resource_registry.clear();
        }

        /// Whether the stream has completed all operations.
        [[nodiscard]] auto is_busy() const -> bool;

        /// Blocks until the stream has completed all operations.
        void synchronize() const;

#if defined(NOA_ENABLE_CUDA)
        [[nodiscard]] auto get() const noexcept -> cudaStream_t { NOA_ASSERT(m_core); return m_core->stream_handle; }
        [[nodiscard]] auto id() const noexcept -> cudaStream_t { return get(); }
#elif defined(NOA_ENABLE_ROCM)
        [[nodiscard]] auto get() const noexcept -> hipStream_t { NOA_ASSERT(m_core); return m_core->stream_handle; }
        [[nodiscard]] auto id() const noexcept -> hipStream_t { return get(); }
#elif defined(NOA_ENABLE_SYCL)
        [[nodiscard]] auto get() const noexcept -> const sycl::queue& { NOA_ASSERT(m_core); return *m_core->queue; }
        [[nodiscard]] auto get() noexcept -> sycl::queue& { NOA_ASSERT(m_core); return *m_core->queue; }
        [[nodiscard]] auto id() const noexcept -> usize {
            // This is used to identify the "NULL"/default stream,
            // similarly to CUDA where stream.id() returns nullptr.
            return m_core->queue_mode == DEFAULT ? 0 : std::hash<sycl::queue>{}(get());
        }
#endif
        [[nodiscard]] auto core() const noexcept -> const std::shared_ptr<Core>& { return m_core; }
        [[nodiscard]] auto device() const noexcept -> Device { return m_device; }
        [[nodiscard]] auto is_empty() const noexcept -> bool { return m_core == nullptr; }

    private:
#if defined(NOA_ENABLE_CUDA)
        // We need to make sure this callback:
        // 1) doesn't call the CUDA API. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-callbacks
        // 2) doesn't lock the StreamResourceRegistry.
        // The latter was actually a surprise: cudaFreeArray seems to wait/use the thread that calls the callbacks.
        // As such, if the StreamResourceRegistry calls the CUDA API as part of insert() or clear(), since these
        // function lock, the CUDA host thread can deadlock...
        static void CUDART_CB update_registry_callback_(void* object) {
            auto* registry = static_cast<details::StreamResourceRegistry*>(object);
            ++(registry->callback_count);
        }
#elif defined(NOA_ENABLE_ROCM)
        static void update_registry_callback_(void* object) { // no HIPRT_CB-like macro needed
            auto* registry = static_cast<details::StreamResourceRegistry*>(object);
            ++(registry->callback_count);
        }
#endif

    private:
        std::shared_ptr<Core> m_core{};
        Device m_device{0, Unchecked{}};
    };
}
