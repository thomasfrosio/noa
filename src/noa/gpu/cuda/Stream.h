#pragma once

#include <cuda_runtime.h>
#include <list>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Device.h"

namespace noa::cuda::details {
    // Registry to attach a stream and a shared_ptr, using a FIFO buffer.
    // Kernel execution is asynchronous relative to the host. As such, we need a way to know when the
    // kernel is running and when it's done, so that we can make sure the device memory used by this kernel
    // will never be deleted while the kernel is running. To solve this problem, this registry attaches some
    // memory regions to a stream and, using callbacks, releases them when the kernel is done executing.
    // In practice, insert() should be called after the kernel launch and copies the shared_ptr(s) at the
    // back of the registry. Then a callback is enqueued to the stream. When the kernel is done execution,
    // the callback is called. The callback is then updating the "callback_count", letting the registry know
    // that the shared_ptr(s) at the front can be deleted.
    class StreamMemoryRegistry {
    private:
        std::mutex m_mutex;
        std::vector<std::pair<int, std::shared_ptr<const void>>> m_registry;
    public:
        std::atomic<int> callback_count{0};

    public:
        // Adds one or multiple shared_ptr to the back of the registry.
        // The shared_ptr reference count is increased by one.
        // This function also deletes the registry from unused shared_ptr.
        template<typename ...Args>
        void insert(Args&& ... args) {
            std::scoped_lock lock(m_mutex);

            int count = callback_count.load();
            while (count != 0 && callback_count.compare_exchange_weak(count, count - 1)) {
                update_();
                --count;
            }
            clear_();

            const auto key = static_cast<int>(m_registry.size());
            ([&key, this](auto&& input) { this->pushBack_(key, input); }(std::forward<Args>(args)), ...);
        }

        // Removes from the registry the shared_ptr(s) flagged as unused.
        // The shared_ptr reference count is decreased by one, thus their deleter can be called.
        void clear(bool force = false) {
            std::scoped_lock lock(m_mutex);
            clear_(force);
            callback_count.store(0);
        }

    private:
        // Flags as unused, and ready to be deleted from the register, the shared_ptr(s) at the front of registry.
        // This function needs to make sure to take the "first in", excluding the elements already marked as unused
        // that haven't been cleared yet.
        void update_() {
            int key = -1;
            for (auto& p: m_registry) {
                if (key == -1 && p.first != -1)
                    key = p.first;
                if (key != -1 && p.first == key)
                    p.first = -1;
            }
        }

        // Removes the unused elements from the registry.
        void clear_(bool force = false) {
            if (force)
                return m_registry.clear();

            // Everything marked with -1 (by update_) is deleted. If it was the last record of that pointer,
            // the reference count goes to 0 and the deleter of that managed (shared) object is called.
            // As such, the CUDA API can be called (e.g. cudaFree).
            bool something_was_deleted{false};
            for (auto& p: m_registry) {
                if (p.first == -1) {
                    p.second = nullptr;
                    something_was_deleted = true;
                }
            }

            if (something_was_deleted) {
                // Now that the shared_ptr are set to nullptr, we need to remove them from the registry by
                // squeezing the remaining elements, i.e. we need to stack valid elements contiguously at
                // the front of the vector while keeping their original order.
                size_t last_available = 0;
                for (size_t i = 0; i < m_registry.size(); ++i) {
                    if (m_registry[i].second) {
                        if (i > last_available)
                            std::swap(m_registry[i], m_registry[last_available]);
                        ++last_available;
                    }
                }
                // Additionally, we need to reset the keys to make sure they stay unique.
                size_t count = 0;
                int old_key = m_registry[0].first;
                int new_key = 0;
                for (; count < m_registry.size(); ++count) {
                    if (!m_registry[count].second)
                        break;

                    if (m_registry[count].first != old_key)
                        new_key = static_cast<int>(count);
                    old_key = std::exchange(m_registry[count].first, new_key);
                }
                m_registry.resize(count);
            }
        }

        template<typename T>
        void pushBack_(int key, const std::shared_ptr<T>& ptr) {
            // TODO When moving to C++20, use std::forward to remove extra copy if original shared_ptr is a rvalue.
            m_registry.emplace_back(key, std::reinterpret_pointer_cast<const void>(ptr));
        }
    };
}

namespace noa::cuda {
    // A CUDA stream (and its associated device).
    class Stream {
    public:
        struct Core {
            details::StreamMemoryRegistry registry{};
            cudaStream_t handle{};

            ~Core() {
                [[maybe_unused]] cudaError_t err;
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
        enum Mode : uint {
            // Work running in the created stream is implicitly synchronized with the NULL stream.
            SERIAL = cudaStreamDefault,

            // Work running in the created stream may run concurrently with work in stream 0 (the
            // NULL stream) and there is no implicit synchronization performed between it and stream 0.
            ASYNC = cudaStreamNonBlocking,

            // Default (NULL) stream.
            DEFAULT = 2
        };

    public:
        // Creates a new stream on the current device.
        explicit Stream(Mode mode = Stream::ASYNC)
                : m_core(std::make_shared<Core>()), m_device(Device::current()) {
            if (mode != Stream::DEFAULT)
                NOA_THROW_IF(cudaStreamCreateWithFlags(&m_core->handle, mode));
        }

        // Creates a new stream on a given device.
        explicit Stream(Device device, Mode mode = Stream::ASYNC)
                : m_core(std::make_shared<Core>()), m_device(device) {
            if (mode != Stream::DEFAULT) {
                DeviceGuard guard(m_device);
                NOA_THROW_IF(cudaStreamCreateWithFlags(&m_core->handle, mode));
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
        void enqueue(const char* kernel_name, K kernel, LaunchConfig config, Args&& ... args) {
            #ifndef __CUDACC__
            NOA_THROW("To launch kernels, the compilation must be steered by NVCC "
                      "(i.e. this function should be called from CUDA C/C++ .cu files)");
            #else
            NOA_ASSERT(m_core);
            // Cooperative kernels are not supported by the triple-chevron syntax.
            DeviceGuard guard(m_device);
            if (config.cooperative) {
                NOA_THROW("Cooperative kernels are not supported yet");
            } else {
                kernel<<<config.blocks, config.threads, config.bytes_shared_memory, m_core->handle>>>(::std::forward<Args>(args)...);
                const auto err = cudaGetLastError();
                if (err)
                    NOA_THROW_FUNC(kernel_name, "Failed to launch the kernel, with message: {}", toString(err));
            }
            #endif
        }

        // Attach some shared_ptr to the stream. By incrementing the reference count this function guarantees
        // that the memory managed by the shared_ptr(s) can be accessed by kernels until the stream reaches this point.
        // The attached memory is implicitly released by synchronize() or next attach() calls, but it can also be
        // explicitly cleared with clear();
        template<typename ...Args>
        void attach(Args&& ... args) {
            NOA_ASSERT(m_core);
            m_core->registry.insert(std::forward<Args>(args)...);
            void (*fun_ptr)(void*) = &updateRegistryCallback_;
            NOA_THROW_IF(cudaLaunchHostFunc(m_core->handle, fun_ptr, &m_core->registry));
        }

        // Whether the stream has completed all operations.
        [[nodiscard]] bool busy() const {
            NOA_ASSERT(m_core);
            DeviceGuard guard(m_device);
            cudaError_t status = cudaStreamQuery(m_core->handle);
            if (status == cudaError_t::cudaSuccess)
                return false;
            else if (status == cudaError_t::cudaErrorNotReady)
                return true;
            else
                NOA_THROW(toString(status));
        }

        // Blocks until the stream has completed all operations. See Device::synchronize().
        void synchronize() const {
            NOA_ASSERT(m_core);
            DeviceGuard guard(m_device);
            NOA_THROW_IF(cudaStreamSynchronize(m_core->handle));
            clear(true);
        }

        // Clears the registry from any unused attached data.
        void clear(bool force = false) const {
            m_core->registry.clear(force);
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
        [[nodiscard]] bool empty() const noexcept { return m_core == nullptr; }

    private:
        // We need to make sure this callback:
        // 1) doesn't call the CUDA API. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-callbacks
        // 2) doesn't lock the StreamMemoryRegistry.
        // The latter was actually a surprise: cudaFreeArray seems to wait/use the thread that calls the callbacks.
        // As such, if the StreamMemoryRegistry calls the CUDA API as part of insert() or clear(), since these
        // function look, the CUDA host thread can deadlock...
        static void CUDART_CB updateRegistryCallback_(void* object) {
            auto* registry = static_cast<details::StreamMemoryRegistry*>(object);
            ++(registry->callback_count);
        }

    private:
        std::shared_ptr<Core> m_core{};
        Device m_device{0, true};
    };
}
