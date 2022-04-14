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
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Device.h"

namespace noa::cuda::details {
    /// Registry to attach a stream and a shared_ptr.
    /// \details This object tries to solve the following issue: when a kernel is enqueued to a stream,
    ///          we assume the device pointers will stay valid until the kernel is done. This assumption
    ///          can be easily broken if the delete functions are called prematurely, e.g. during stack unrolling.
    ///          The registry allows to attach some memory regions to a stream and releases them when the
    ///          kernel is done executing. In practice, insert() copies the shared_ptr(s), update() marks the
    ///          shared_ptr(s) at the back of the registry to be ready for delete (FIFO) and clear() removes
    ///          everything marked by update().
    class StreamMemoryRegistry {
    public:
        /// Adds one or multiple shared_ptr to the front of registry.
        /// The shared_ptr reference count is increased by one.
        template<typename ...Args>
        void insert(Args&& ... args) {
            std::scoped_lock lock(m_mutex);
            clear_();
            const auto key = static_cast<int>(m_registry.size());
            ([&](auto& input) { insertOne_(key, input); }(std::forward<Args>(args)), ...);
        }

        /// Removes from the registry the shared_ptr(s) flagged as unused.
        /// The shared_ptr reference count is decreased by one, thus their deleter can be called.
        /// This function should not be called from a CUDA callback if the deleter makes a CUDA API call.
        /// \see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-callbacks
        void clear() {
            std::scoped_lock lock(m_mutex);
            clear_();
        }

        /// Flags as unused, and ready to be deleted from the register, the shared_ptr(s) at the back of registry.
        /// This function can be called from a CUDA callback.
        void update() {
            std::scoped_lock lock(m_mutex);
            const int key = m_registry.back().first;
            for (auto& p: m_registry)
                if (p.first == key)
                    p.first = -1;
        }

    private:
        void clear_() {
            m_registry.remove_if([](auto& p) { return p.first == -1; });
        }

        template<typename T>
        void insertOne_(int key, const std::shared_ptr<T>& ptr) {
            m_registry.emplace_front(key, std::reinterpret_pointer_cast<const void>(ptr)); // aliasing ctor
        }

        std::mutex m_mutex;
        std::list<std::pair<int, std::shared_ptr<const void>>> m_registry;
    };
}

namespace noa::cuda {
    /// A CUDA stream (and its associated device).
    class Stream {
    private:
        struct StreamImp {
            details::StreamMemoryRegistry registry{};
            cudaStream_t handle{};

            ~StreamImp() {
                cudaStreamSynchronize(handle);
                if (handle)
                    cudaStreamDestroy(handle);
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
                : m_imp(std::make_unique<StreamImp>()), m_device(Device::current()) {
            if (mode != Stream::DEFAULT)
                NOA_THROW_IF(cudaStreamCreateWithFlags(&m_imp->handle, mode));
        }

        /// Creates a new stream on a given device.
        explicit Stream(Device device, Mode mode = Stream::ASYNC)
                : m_imp(std::make_unique<StreamImp>()), m_device(device) {
            if (mode != Stream::DEFAULT) {
                DeviceGuard guard(m_device);
                NOA_THROW_IF(cudaStreamCreateWithFlags(&m_imp->handle, mode));
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
            NOA_ASSERT(m_imp);
            // Cooperative kernels are not supported by the triple-chevron syntax.
            DeviceGuard guard(m_device);
            if (config.cooperative) {
                NOA_THROW("Cooperative kernels are not supported yet");
            } else {
                kernel<<<config.blocks, config.threads, config.bytes_shared_memory, m_imp->handle>>>(::std::forward<Args>(args)...);
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
            NOA_ASSERT(m_imp);
            m_imp->registry.insert(std::forward<Args>(args)...);
            void (*fun_ptr)(void*) = &updateRegistryCallback_;
            NOA_THROW_IF(cudaLaunchHostFunc(m_imp->handle, fun_ptr, &m_imp->registry));
        }

        /// Whether or not the stream has completed all operations.
        [[nodiscard]] bool busy() const {
            NOA_ASSERT(m_imp);
            DeviceGuard guard(m_device);
            cudaError_t status = cudaStreamQuery(m_imp->handle);
            if (status == cudaError_t::cudaSuccess)
                return false;
            else if (status == cudaError_t::cudaErrorNotReady)
                return true;
            else
                NOA_THROW(toString(status));
        }

        /// Blocks until the stream has completed all operations. \see Device::synchronize().
        void synchronize() const {
            NOA_ASSERT(m_imp);
            NOA_PROFILE_FUNCTION();
            DeviceGuard guard(m_device);
            NOA_THROW_IF(cudaStreamSynchronize(m_imp->handle));
            clear();
        }

        /// Clears the registry from any unused attached data.
        void clear() const {
            m_imp->registry.clear();
        }

        [[nodiscard]] cudaStream_t get() const noexcept {
            NOA_ASSERT(m_imp);
            return m_imp->handle;
        }

        [[nodiscard]] cudaStream_t id() const noexcept { return get(); }
        [[nodiscard]] Device device() const noexcept { return m_device; }
        [[nodiscard]] bool empty() const noexcept { return m_imp == nullptr; }

    private:
        static void CUDART_CB updateRegistryCallback_(void* object) {
            auto registry = static_cast<details::StreamMemoryRegistry*>(object);
            registry->update();
            // TODO Add a profiler call?
        }

    private:
        std::shared_ptr<StreamImp> m_imp{};
        Device m_device{0, true};
    };
}
