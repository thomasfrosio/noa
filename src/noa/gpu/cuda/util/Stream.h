/// \file noa/gpu/cuda/util/Stream.h
/// \brief CUDA streams.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include <cuda_runtime.h>

#include "noa/Definitions.h"
#include "noa/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Device.h"

namespace noa::cuda {
    /// A CUDA stream (and its associated device).
    class Stream {
    private:
        cudaStream_t m_stream{nullptr}; // Uses the default (NULL) stream.
        Device m_device;

    public:
        /// Blocks until stream has completed all operations. \see Device::synchronize().
        NOA_HOST static void synchronize(const Stream& stream) {
            NOA_PROFILE_FUNCTION();
            DeviceCurrentScope scope_device(stream.m_device);
            NOA_THROW_IF(cudaStreamSynchronize(stream.m_stream));
        }

        /// Whether or not the stream has completed all operations.
        NOA_HOST static bool hasCompleted(const Stream& stream) {
            DeviceCurrentScope scope_device(stream.m_device);
            cudaError_t status = cudaStreamQuery(stream.m_stream);
            if (status == cudaError_t::cudaSuccess)
                return true;
            else if (status == cudaError_t::cudaErrorNotReady)
                return false;
            else
                NOA_THROW(toString(status));
        }

    public:
        /// Use the CUDA runtime default stream.
        NOA_HOST Stream() : m_device(Device::getCurrent()) {}

        /// Creates a new stream on the current device.
        /// \param flag     Either \c STREAM_SERIAL or \c STREAM_CONCURRENT.
        /// \note Streams are associated with a specific device. Use device() to retrieve the device.
        NOA_HOST explicit Stream(StreamMode flag) : m_device(Device::getCurrent()) {
            NOA_THROW_IF(cudaStreamCreateWithFlags(&m_stream, flag));
        }

        /// Creates a new stream on a device.
        /// \param device   Device on which the stream should be created.
        /// \param flag     Either \c STREAM_SERIAL or \c STREAM_CONCURRENT.
        /// \note Streams are associated with a specific device. Use device() to retrieve the device.
        NOA_HOST explicit Stream(Device device, StreamMode flag) : m_device(device) {
            DeviceCurrentScope scope_device(m_device);
            NOA_THROW_IF(cudaStreamCreateWithFlags(&m_stream, flag));
        }

        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;

        NOA_HOST Stream(Stream&& to_move) noexcept
                : m_stream(std::exchange(to_move.m_stream, nullptr)), m_device(to_move.m_device) {}

        NOA_HOST Stream& operator=(Stream&& to_move) noexcept {
            std::swap(m_stream, to_move.m_stream); // let to_move dtor destroy the old data if any.
            std::swap(m_device, to_move.m_device);
            return *this;
        }

        NOA_HOST ~Stream() {
            if (m_stream) {
                DeviceCurrentScope scope_device(m_device);
                cudaError_t err = cudaStreamDestroy(m_stream);
                if (err != cudaSuccess && std::uncaught_exceptions() == 0)
                    NOA_THROW(toString(err));
            }
        }

        NOA_HOST cudaStream_t get() const noexcept { return m_stream; }
        NOA_HOST cudaStream_t id() const noexcept { return m_stream; }
        NOA_HOST Device device() const noexcept { return m_device; }
    };
}
