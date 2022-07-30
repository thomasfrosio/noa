/// \file noa/gpu/cuda/Stream.h
/// \brief CUDA streams.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021
#pragma once

#include <cuda_runtime.h>
#include <list>
#include <mutex>
#include <condition_variable>

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Device.h"

namespace noa::cuda::details {
    /// Registry to attach a stream and a shared_ptr.
    /// \details This object tries to solve the following issue: when a kernel is enqueued to a stream,
    ///          we assume the device pointers will stay valid until the kernel computation is completed.
    ///          This assumption can be easily broken if the delete functions are called prematurely,
    ///          e.g. during stack unrolling. This registry attaches some memory regions to a stream and
    ///          releases them when the kernel is done executing. In practice, insert() copies the shared_ptr(s)
    ///          into an internal buffer, update() marks the shared_ptr(s) at the back of the FIFO registry
    ///          to be ready for delete and clear() removes everything marked by update(). As a result,
    ///          if update() is called by a callback enqueued just after the kernel launch, it guarantees
    ///          that the reference count of the inserted pointers never reaches zero while the kernel runs.
    ///          In terms of performance, it seems to add an overhead of ~0.01 to 0.02 ms per callback.
    class StreamMemoryRegistry {
    public:
        /// Adds one or multiple shared_ptr to the front of the registry.
        /// The shared_ptr reference count is increased by one.
        template<typename ...Args>
        void insert(Args&& ... args) {
            std::scoped_lock lock(m_mutex);
            clear_();
            const auto key = static_cast<int>(m_registry.size());
            ([&key, this](auto&& input) { this->insertOne_(key, input); }(std::forward<Args>(args)), ...);
        }

        /// Removes from the registry the shared_ptr(s) flagged as unused.
        /// The shared_ptr reference count is decreased by one, thus their deleter can be called.
        /// This function should not be called from a CUDA callback if the deleter makes a CUDA API call.
        /// \see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-callbacks
        void clear(bool force = false) {
            std::scoped_lock lock(m_mutex);
            clear_(force);
        }

        /// Flags as unused, and ready to be deleted from the register, the shared_ptr(s) at the back of registry.
        /// This function can and should be called from a CUDA callback.
        void update() {
            std::scoped_lock lock(m_mutex);
            // Make sure to take the "first in", excluding the elements already
            // marked as unused that haven't been cleared yet.
            int key = -1;
            for (auto& p: m_registry) {
                if (key == -1 && p.first != -1)
                    key = p.first;
                if (key != -1 && p.first == key)
                    p.first = -1;
            }
        }

    private:
        void clear_(bool force = false) {
            if (force) {
                m_registry.clear();
            } else {
                for (auto& p: m_registry)
                    if (p.first == -1)
                        p.second = nullptr;

                // Stack elements contiguously at the front while keeping the order and resize:
                size_t last_available = 0;
                for (size_t i = 0; i < m_registry.size(); ++i) {
                    if (m_registry[i].second) {
                        if (i > last_available)
                            std::swap(m_registry[i], m_registry[last_available]);
                        ++last_available;
                    }
                }
                size_t count = 0;
                for (auto& p: m_registry) {
                    if (p.second)
                        ++count;
                    else
                        break;
                }
                m_registry.resize(count);
            }
        }

        template<typename T>
        void insertOne_(int key, const std::shared_ptr<T>& ptr) {
            m_registry.emplace_back(key, std::reinterpret_pointer_cast<const void>(ptr));
        }

        std::mutex m_mutex;
        std::vector<std::pair<int, std::shared_ptr<const void>>> m_registry;
    };
}

namespace noa::cuda {
    /// A CUDA stream (and its associated device).
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
            /// Work running in the created stream is implicitly synchronized with the NULL stream.
            SERIAL = cudaStreamDefault,

            /// Work running in the created stream may run concurrently with work in stream 0 (the
            /// NULL stream) and there is no implicit synchronization performed between it and stream 0.
            ASYNC = cudaStreamNonBlocking,

            /// Default (NULL) stream.
            DEFAULT = 2
        };

    public:
        /// Creates a new stream on the current device.
        explicit Stream(Mode mode = Stream::ASYNC)
                : m_core(std::make_shared<Core>()), m_device(Device::current()) {
            if (mode != Stream::DEFAULT)
                NOA_THROW_IF(cudaStreamCreateWithFlags(&m_core->handle, mode));
        }

        /// Creates a new stream on a given device.
        explicit Stream(Device device, Mode mode = Stream::ASYNC)
                : m_core(std::make_shared<Core>()), m_device(device) {
            if (mode != Stream::DEFAULT) {
                DeviceGuard guard(m_device);
                NOA_THROW_IF(cudaStreamCreateWithFlags(&m_core->handle, mode));
            }
        }

        /// Empty constructor.
        /// \details Creates an empty instance that is meant to be reset using one of the operator assignment.
        ///          Calling empty() returns true, but any other member function call will fail. Passing an
        ///          empty stream is never allowed (and will result in segfault) unless specified otherwise.
        constexpr explicit Stream(std::nullptr_t) {}

    public:
        /// Enqueues a kernel launch to the stream.
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

        /// Attach some shared_ptr to the stream. By incrementing the reference count this function guarantees
        /// that the memory managed by the shared_ptr(s) can be accessed by kernels until the stream reaches this point.
        /// The attached memory is implicitly released by synchronize() or next attach() calls, but it can also be
        /// explicitly cleared with clear();
        template<typename ...Args>
        void attach(Args&& ... args) {
            NOA_ASSERT(m_core);
            m_core->registry.insert(std::forward<Args>(args)...);
            void (*fun_ptr)(void*) = &updateRegistryCallback_;
            NOA_THROW_IF(cudaLaunchHostFunc(m_core->handle, fun_ptr, &m_core->registry));
        }

        /// Whether or not the stream has completed all operations.
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

        /// Blocks until the stream has completed all operations. \see Device::synchronize().
        void synchronize() const {
            NOA_ASSERT(m_core);
            DeviceGuard guard(m_device);
            NOA_THROW_IF(cudaStreamSynchronize(m_core->handle));
            clear(true);
        }

        /// Clears the registry from any unused attached data.
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
        static void CUDART_CB updateRegistryCallback_(void* object) {
            auto* registry = static_cast<details::StreamMemoryRegistry*>(object);
            registry->update();
            // TODO Add a profiler call?
        }

    private:
        std::shared_ptr<Core> m_core{};
        Device m_device{0, true};
    };
}
