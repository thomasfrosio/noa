#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include <list>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "noa/core/Traits.hpp"
#include "noa/core/utils/Misc.hpp"
#include "noa/core/utils/ShareHandles.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/Device.hpp"

// TODO cudaFree(Async) is synchronizing the device/stream or is stream-ordered, so this shouldn't be necessary
//      for CUDA-managed memory. For unregistered memory that the stream depends on (e.g. CPU<->GPU copies), this
//      should be useful. Update the documentation to reflect that?

namespace noa::cuda::guts {
    // Registry to attach a shared_ptr to a stream, using a FIFO buffer.
    //
    // Kernel execution is asynchronous relative to the host. As such, we need a way to know when the
    // kernel is running and when it's done, so that we can make sure the resources used by this kernel
    // will never be deleted while the kernel is running. As a solution to this, this registry can attach
    // resources to a stream and can release (using callbacks) these resources when the kernel is done executing.
    //
    // Any type can be passed to try_insert(), but only std::shared_ptr(s) can and will be added to the registry.
    // Types with a .share() method returning a type that can be converted to a std::shared_ptr (e.g., Array or Texture)
    // can also be passed, and the returned std::shared_ptr will be added to the registry.
    //
    // 1) try_insert() should be called after the kernel launch. It forwards the shared_ptr(s) at the back
    //    of the registry. Resources are attached to a unique key, and the key is per try_insert() call, meaning that
    //    all resources passed to the same try_insert() call have the same lifetime.
    // 2) then a callback is enqueued to the stream. When the kernel is done execution, the callback is
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
    class StreamResourceRegistry {
    private:
        std::mutex m_mutex;
        std::vector<std::pair<i64, std::shared_ptr<const void>>> m_registry;
        i64 m_key{0};

    public:
        std::atomic<i32> callback_count{0};

    public:
        // Adds one or multiple shared_ptr to the back of the registry.
        // The shared_ptr reference count is increased by one.
        // This function also deletes the registry from unused shared_ptr.
        template<typename ...Args>
        bool try_insert(Args&& ... args) {
            const std::scoped_lock lock(m_mutex);
            clear_unused_();

            if constexpr (nt::bool_or<(nt::is_shareable_v<Args> || nt::has_share_v<Args>)...>::value) {
                i64 key = [this]() {
                    if (m_key == std::numeric_limits<i64>::max())
                        m_key = 0;
                    return m_key++;
                }();
                ([&, this]<typename T>(T&& arg) {
                    if constexpr (nt::is_shareable_v<T>)
                        m_registry.emplace_back(key, std::forward<T>(arg));
                    else if constexpr (nt::has_share_v<T>)
                        m_registry.emplace_back(key, std::forward<T>(arg).share());
                    // else: do nothing?
                }(std::forward<Args>(args)), ...);
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
                i64 key = -1;
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

namespace noa::cuda {
    enum class StreamMode : u32 {
        // Work running in the created stream is implicitly synchronized with the NULL stream.
        SERIAL = cudaStreamDefault,

        // Work running in the created stream may run concurrently with work in stream 0 (the
        // NULL stream) and there is no implicit synchronization performed between it and stream 0.
        ASYNC = cudaStreamNonBlocking,

        // Default (NULL) stream.
        DEFAULT = 2
    };

    // A CUDA stream (and its associated device).
    class Stream {
    public:
        struct Core {
            guts::StreamResourceRegistry resource_registry{};
            cudaStream_t stream_handle{};

            ~Core() {
                [[maybe_unused]] cudaError_t err{};
                err = cudaStreamSynchronize(stream_handle);
                NOA_ASSERT(err == cudaSuccess);
                resource_registry.clear_after_sync();
                if (stream_handle) {
                    err = cudaStreamDestroy(stream_handle);
                    NOA_ASSERT(err == cudaSuccess);
                }
            }
        };

    public:
        /// Creates an empty (invalid) stream.
        constexpr explicit Stream() = default;

        /// Creates a new stream on the current device.
        explicit Stream(StreamMode mode = StreamMode::ASYNC)
                : m_core(std::make_shared<Core>()), m_device(Device::current()) {
            if (mode != StreamMode::DEFAULT)
                check(cudaStreamCreateWithFlags(&m_core->stream_handle, to_underlying(mode)));
        }

        /// Creates a new stream on a given device.
        explicit Stream(Device device, StreamMode mode = StreamMode::ASYNC)
                : m_core(std::make_shared<Core>()), m_device(device) {
            if (mode != StreamMode::DEFAULT) {
                const DeviceGuard guard(m_device);
                check(cudaStreamCreateWithFlags(&m_core->stream_handle, to_underlying(mode)));
            }
        }


    public:
        /// Enqueues a kernel to the stream.
        template<typename K, typename ...Args>
        void enqueue([[maybe_unused]] K kernel,
                     [[maybe_unused]] LaunchConfig config,
                     [[maybe_unused]] Args&& ... args) {
            #ifndef __CUDACC__
            panic("To launch kernels, the compilation must be steered by nvcc or nvc++, "
                  "i.e. this function should be called from CUDA C++ files");
            #else
            NOA_ASSERT(m_core);
            // Cooperative kernels are not supported by the triple-chevron syntax.
            const DeviceGuard guard(m_device);
            if (config.is_cooperative) {
                panic("Cooperative kernels are not supported yet");
            } else {
                kernel<<<config.n_blocks, config.n_threads, config.n_bytes_of_shared_memory, m_core->stream_handle>>>(::std::forward<Args>(args)...);
                check(cudaGetLastError());
            }
            #endif
        }

        // TODO
//        void enqueue(
//                std::string_view kernel,
//                LaunchConfig config,
//                void** arguments
//        );

        /// Enqueues attachments to the stream.
        /// \details Attachments are resources that should be kept alive until the CUDA stream reaches this point.
        ///          A resource is anything that is convertible to `std::shared_ptr<const void>` or a type that has a
        ///          .share() member function returning a `std::shared_ptr`. Passing anything else in \p attachments is
        ///          also valid but will be ignored. In practice, these shared_ptr are copied into an internal buffer,
        ///          thereby incrementing their reference count. When the stream reaches this point, the internal
        ///          buffer flags these shared_ptr as unused and ready to be destructed. Due to some CUDA limitations,
        ///          the buffer cannot destroy these shared_ptr right away (see .update_registry_callback_()).
        ///          Instead, at the next synchronization or enqueueing, the flagged shared_ptr will be destroyed,
        ///          thereby decrementing the reference count. The flagged shared_ptr can also be explicitly cleared
        ///          using .clear().
        template<typename ...Args>
        void enqueue_attach(Args&& ... args) {
            NOA_ASSERT(m_core);
            if (m_core->resource_registry.try_insert(std::forward<Args>(args)...)) {
                // Something was inserted, so enqueue a "cleaning" callback.
                void (* fun_ptr)(void*) = &update_registry_callback_;
                check(cudaLaunchHostFunc(m_core->stream_handle, fun_ptr, &m_core->resource_registry));
            }
        }

        /// Clears the registry from any unused attached data.
        /// The registry can be "forced"-clean, removing all resources regardless of whether they are still in use.
        void clear() const {
            m_core->resource_registry.clear();
        }

        /// Whether the stream has completed all operations.
        [[nodiscard]] bool is_busy() const {
            NOA_ASSERT(m_core);
            const DeviceGuard guard(m_device);
            const cudaError_t status = cudaStreamQuery(m_core->stream_handle);
            if (status == cudaError_t::cudaSuccess)
                return false;
            else if (status == cudaError_t::cudaErrorNotReady)
                return true;
            else
                panic_runtime(error2string(status));
        }

        // Blocks until the stream has completed all operations. See Device::synchronize().
        void synchronize() const {
            NOA_ASSERT(m_core);
            const DeviceGuard guard(m_device);
            check(cudaStreamSynchronize(m_core->stream_handle));
            m_core->resource_registry.clear_after_sync();
        }

        [[nodiscard]] cudaStream_t get() const noexcept {
            NOA_ASSERT(m_core);
            return m_core->stream_handle;
        }

        [[nodiscard]] const std::shared_ptr<Core>& core() const noexcept {
            return m_core;
        }

        [[nodiscard]] cudaStream_t id() const noexcept { return get(); }
        [[nodiscard]] Device device() const noexcept { return m_device; }
        [[nodiscard]] bool is_empty() const noexcept { return m_core == nullptr; }

    private:
        // We need to make sure this callback:
        // 1) doesn't call the CUDA API. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-callbacks
        // 2) doesn't lock the StreamResourceRegistry.
        // The latter was actually a surprise: cudaFreeArray seems to wait/use the thread that calls the callbacks.
        // As such, if the StreamResourceRegistry calls the CUDA API as part of insert() or clear(), since these
        // function lock, the CUDA host thread can deadlock...
        static void CUDART_CB update_registry_callback_(void* object) {
            auto* registry = static_cast<guts::StreamResourceRegistry*>(object);
            ++(registry->callback_count);
        }

    private:
        std::shared_ptr<Core> m_core{};
        Device m_device{0, Device::DeviceUnchecked{}};
    };
}
#endif
