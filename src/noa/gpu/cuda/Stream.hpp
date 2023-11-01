#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/Device.hpp"

#if defined(NOA_IS_OFFLINE)
#include <list>
#include <mutex>
#include <condition_variable>
#include <atomic>

// TODO cudaFree(Async) is synchronizing the device/stream or is stream-ordered, so this shouldn't be necessary
//      for CUDA-managed memory. For unregistered memory that the stream depends on (e.g. CPU<->GPU copies), this
//      should be useful. Update the documentation to reflect that?

namespace noa::cuda::guts {
    template<typename T> struct proclaim_is_shared_ptr : std::false_type {};
    template<typename T> struct proclaim_is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
    template<typename T> constexpr bool is_shared_ptr_v = std::bool_constant<proclaim_is_shared_ptr<nt::remove_ref_cv_t<T>>::value>::value;

    template<typename T> using has_share = decltype(std::declval<T>().share());
    template<typename T> constexpr bool has_share_v = nt::is_detected_convertible_v<std::shared_ptr<const void>, has_share, T>;

    template<typename... Ts> constexpr bool is_registrable_v = nt::bool_or<(is_shared_ptr_v<Ts> || has_share_v<Ts>)...>::value;

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
            if constexpr (is_registrable_v<Args...>) {
                const std::scoped_lock lock(m_mutex);

                purge_callback_count_(); // flag unused resources
                clear_(); // erase unused resources

                const auto key = get_unique_key_();
                ([&key, this](auto&& input) { this->add_to_registry_(key, input); }(std::forward<Args>(args)), ...);
                return true;
            } else {
                return false;
            }
        }

        // Removes from the registry the shared_ptr(s) flagged as unused.
        // The shared_ptr reference count is decreased by one, thus their deleter can be called.
        void clear(bool force = false) {
            const std::scoped_lock lock(m_mutex);
            if (force)
                callback_count.store(0);
            else
                purge_callback_count_();
            clear_(force);
        }

    private: // !! Make sure to lock before calling these functions !!
        i64 get_unique_key_() {
            if (m_key == std::numeric_limits<i64>::max())
                m_key = 0;
            return m_key++;
        }

        // Flag unused resources as such.
        void purge_callback_count_() {
            // When a callback is called, callback_count is incremented.
            // For each increment, it means the oldest valid resources are now unused and can be flagged as such.
            i32 count = callback_count.load();
            while (count != 0 && callback_count.compare_exchange_weak(count, count - 1)) {
                flag_front_as_unused_();
                --count;
            }
        }

        // Flags as unused (key == -1) the valid shared_ptr(s) at the front of the registry.
        // This function can be called from a CUDA callback.
        void flag_front_as_unused_() {
            i64 key = -1;
            // Multiple resources can have the same key (they were enqueued in the same try_insert() call),
            // so we need to flag the first (from the front) _valid_ resource and flag any other resource with
            // the same key.
            for (auto& p: m_registry) {
                if (key == -1 && p.first != -1)
                    key = p.first; // oldest valid resource
                if (key != -1 && p.first == key)
                    p.first = -1; // is now flagged as unused
            }
        }

        // Removes the unused elements from the registry.
        // !! Should not be called from a CUDA callback !!
        void clear_(bool force = false) {
            if (force) {
                m_key = 0; // reset the key (this is optional)
                return m_registry.clear();
            }

            // Resources with a key == -1 are deleted. If it was the last record of that shared-pointer,
            // the reference count goes to 0, and the deleter of that resource is called (which can in
            // turn call the CUDA API).
            bool something_was_deleted{false};
            for (auto& p: m_registry) {
                if (p.first == -1) {
                    p.second = nullptr;
                    something_was_deleted = true;
                }
            }

            if (something_was_deleted)
                erase_if(m_registry, [](const auto& pair) { return pair.second == nullptr; });
        }

        template<typename T>
        void add_to_registry_(i64 key, T&& ptr) {
            if constexpr (is_shared_ptr_v<T>)
                m_registry.emplace_back(key, std::forward<T>(ptr));
            else if constexpr (has_share_v<T>)
                m_registry.emplace_back(key, std::forward<T>(ptr).share());
            // else do nothing
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
            guts::StreamResourceRegistry registry{};
            cudaStream_t handle{};

            ~Core() {
                [[maybe_unused]] cudaError_t err{};
                err = cudaStreamSynchronize(handle);
                NOA_ASSERT(err == cudaSuccess);
                registry.clear(/*force*/ true);
                if (handle) {
                    err = cudaStreamDestroy(handle);
                    NOA_ASSERT(err == cudaSuccess);
                }
            }
        };

    public:
        // Creates a new stream on the current device.
        explicit Stream(StreamMode mode = StreamMode::ASYNC)
                : m_core(std::make_shared<Core>()), m_device(Device::current()) {
            if (mode != StreamMode::DEFAULT)
                NOA_THROW_IF(cudaStreamCreateWithFlags(&m_core->handle, noa::to_underlying(mode)));
        }

        // Creates a new stream on a given device.
        explicit Stream(Device device, StreamMode mode = StreamMode::ASYNC)
                : m_core(std::make_shared<Core>()), m_device(device) {
            if (mode != StreamMode::DEFAULT) {
                const DeviceGuard guard(m_device);
                NOA_THROW_IF(cudaStreamCreateWithFlags(&m_core->handle, noa::to_underlying(mode)));
            }
        }

        // Empty constructor.
        // Creates an empty instance that is meant to be reset using one of the operator assignment.
        // Calling empty() returns true, but any other member function call will fail. Passing an
        // empty stream is never allowed (and will result in segfault) unless specified otherwise.
        constexpr explicit Stream(std::nullptr_t) {}

    public:
        // Enqueues a kernel launch to the stream.
        template<typename K, typename ...Args>
        void enqueue([[maybe_unused]] const char* kernel_name,
                     [[maybe_unused]] K kernel,
                     [[maybe_unused]] LaunchConfig config,
                     [[maybe_unused]] Args&& ... args) {
            #ifndef __CUDACC__
            NOA_THROW("To launch kernels, the compilation must be steered by NVCC "
                      "(i.e. this function should be called from CUDA C/C++ .cu files)");
            #else
            NOA_ASSERT(m_core);
            // Cooperative kernels are not supported by the triple-chevron syntax.
            const DeviceGuard guard(m_device);
            if (config.cooperative) {
                NOA_THROW("Cooperative kernels are not supported yet");
            } else {
                kernel<<<config.blocks, config.threads, config.bytes_shared_memory, m_core->handle>>>(::std::forward<Args>(args)...);
                const auto err = cudaGetLastError();
                if (err)
                    NOA_THROW_FUNC(kernel_name, "Failed to launch the kernel, with message: {}", error2string(err));
            }
            #endif
        }

        // Enqueues a kernel launch to the stream.
        template<typename K, typename ...Args>
        void enqueue([[maybe_unused]] K kernel,
                     [[maybe_unused]] LaunchConfig config,
                     [[maybe_unused]] Args&& ... args) {
            #ifndef __CUDACC__
            NOA_THROW("To launch kernels, the compilation must be steered by NVCC "
                      "(i.e. this function should be called from CUDA C/C++ .cu files)");
            #else
            NOA_ASSERT(m_core);
            // Cooperative kernels are not supported by the triple-chevron syntax.
            const DeviceGuard guard(m_device);
            if (config.cooperative) {
                NOA_THROW("Cooperative kernels are not supported yet");
            } else {
                kernel<<<config.blocks, config.threads, config.bytes_shared_memory, m_core->handle>>>(::std::forward<Args>(args)...);
                const auto err = cudaGetLastError();
                if (err)
                    NOA_THROW("Failed to launch the kernel, with message: {}", error2string(err));
            }
            #endif
        }

        // Copies some resources into the stream resource registry. A resource is anything that is a `std::shared_ptr`
        // or a type that has a .share() method returning a `std::shared_ptr`. Anything else is valid but is ignored.
        // This is used to enforce stream-ordering to resource lifetimes, by incrementing the reference count of
        // a resource, this function indeed guarantees that the memory managed by the shared_ptr(s) stays valid
        // until the stream reaches this point. The attached memory is implicitly released by .synchronize() or
        // next .enqueue_attach() calls, but it can also be explicitly cleared with .clear();
        template<typename ...Args>
        void enqueue_attach(Args&& ... args) {
            NOA_ASSERT(m_core);
            if (m_core->registry.try_insert(std::forward<Args>(args)...)) {
                // Something was inserted, so enqueue a "cleaning" callback.
                void (* fun_ptr)(void*) = &updateRegistryCallback_;
                NOA_THROW_IF(cudaLaunchHostFunc(m_core->handle, fun_ptr, &m_core->registry));
            }
        }

        // Clears the registry from any unused attached data.
        // The registry can be "forced"-clean, removing all resources regardless of whether they are still in use.
        void clear(bool force = false) const {
            m_core->registry.clear(force);
        }

        // Whether the stream has completed all operations.
        [[nodiscard]] bool is_busy() const {
            NOA_ASSERT(m_core);
            const DeviceGuard guard(m_device);
            const cudaError_t status = cudaStreamQuery(m_core->handle);
            if (status == cudaError_t::cudaSuccess)
                return false;
            else if (status == cudaError_t::cudaErrorNotReady)
                return true;
            else
                NOA_THROW(error2string(status));
        }

        // Blocks until the stream has completed all operations. See Device::synchronize().
        void synchronize() const {
            NOA_ASSERT(m_core);
            const DeviceGuard guard(m_device);
            NOA_THROW_IF(cudaStreamSynchronize(m_core->handle));
            clear(true);
        }

        [[nodiscard]] cudaStream_t get() const noexcept {
            NOA_ASSERT(m_core);
            return m_core->handle;
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
        static void CUDART_CB updateRegistryCallback_(void* object) {
            auto* registry = static_cast<guts::StreamResourceRegistry*>(object);
            ++(registry->callback_count);
        }

    private:
        std::shared_ptr<Core> m_core{};
        Device m_device{0, Device::DeviceUnchecked{}};
    };
}
#endif
