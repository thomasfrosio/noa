/// \file noa/gpu/cuda/Stream.h
/// \brief CUDA streams.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021
#pragma once

#include <cuda_runtime.h>

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Device.h"

namespace noa::cuda {
    /// A CUDA stream (and its associated device).
    class Stream {
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
        explicit Stream(Mode mode = Stream::ASYNC) : m_device(Device::current()) {
            if (mode != Stream::DEFAULT)
                NOA_THROW_IF(cudaStreamCreateWithFlags(&m_stream, mode));
        }

        /// Creates a new stream on a given device.
        explicit Stream(Device device, Mode mode = Stream::ASYNC)
                : m_device(device) {
            if (mode != Stream::DEFAULT) {
                DeviceGuard guard(m_device);
                NOA_THROW_IF(cudaStreamCreateWithFlags(&m_stream, mode));
            }
        }

        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;

        Stream(Stream&& to_move) noexcept
                : m_stream(std::exchange(to_move.m_stream, nullptr)),
                  m_device(to_move.m_device) {}

        Stream& operator=(Stream&& to_move) noexcept {
            std::swap(m_stream, to_move.m_stream); // let to_move dtor destroy the old data if any.
            std::swap(m_device, to_move.m_device);
            return *this;
        }

        ~Stream() {
            if (m_stream) {
                DeviceGuard guard(m_device);
                cudaStreamDestroy(m_stream); // ignore any potential error to keep destructor noexcept
            }
        }

    public:
        /// Enqueues a kernel launch to the stream.
        template<typename K, typename ...Args>
        NOA_HOST void enqueue(const char* kernel_name, K kernel, LaunchConfig config, Args&& ... args) {
            #ifndef __CUDACC__
            NOA_THROW("To launch kernels, the compilation must be steered by NVCC "
                      "(i.e. this function should be called from CUDA C/C++ .cu files)");
            #else
            // Cooperative kernels are not supported by the triple-chevron syntax.
            DeviceGuard guard(m_device);
            if (config.cooperative) {
                NOA_THROW("Cooperative kernels are not supported yet");
            } else {
                kernel<<<config.blocks, config.threads, config.bytes_shared_memory, m_stream>>>(::std::forward<Args>(args)...);
                const auto err = cudaGetLastError();
                if (err)
                    NOA_THROW_FUNC(kernel_name, "Failed to launch the kernel, with message: {}", toString(err));
            }
            #endif
        }

        /// Whether or not the stream has completed all operations.
        [[nodiscard]] bool busy() const {
            DeviceGuard guard(m_device);
            cudaError_t status = cudaStreamQuery(m_stream);
            if (status == cudaError_t::cudaSuccess)
                return true;
            else if (status == cudaError_t::cudaErrorNotReady)
                return false;
            else
                NOA_THROW(toString(status));
        }

        /// Blocks until the stream has completed all operations. \see Device::synchronize().
        void synchronize() const {
            NOA_PROFILE_FUNCTION();
            DeviceGuard guard(m_device);
            NOA_THROW_IF(cudaStreamSynchronize(m_stream));
        }

        [[nodiscard]] NOA_HOST cudaStream_t get() const noexcept { return m_stream; }
        [[nodiscard]] NOA_HOST cudaStream_t id() const noexcept { return m_stream; }
        [[nodiscard]] NOA_HOST Device device() const noexcept { return m_device; }

    private:
        cudaStream_t m_stream{nullptr}; // Uses the default (NULL) stream.
        Device m_device;
    };
}
