#pragma once

#include <cuda_runtime.h>

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/util/string/Format.h"
#include "noa/gpu/cuda/util/Device.h"

namespace Noa::CUDA {
    /** A CUDA stream (and its associated device) and a namespace-like. */
    class Stream {
    public:
        /**
         * @a concurrent: work running in the created stream may run concurrently with work in stream 0 (the NULL
         *                stream), and there is no implicit synchronization performed between it and stream 0.
         * @a serial:     work running in the created stream is implicitly synchronized with the NULL stream.
         */
        enum : uint { CONCURRENT = cudaStreamNonBlocking, SERIAL = cudaStreamDefault };

    private:
        cudaStream_t m_stream{nullptr}; // Uses the default (NULL) stream.
        Device m_device;

    public:
        /** Blocks until stream has completed all operations. @see Device::synchronize(). */
        NOA_IH static void synchronize(const Stream& stream) {
            DeviceCurrentScope scope_device(stream.m_device);
            NOA_THROW_IF(cudaStreamSynchronize(stream.m_stream));
        }

        /** Whether or not the stream has completed all operations. */
        NOA_IH static bool hasCompleted(const Stream& stream) {
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
        /** Use the CUDA runtime default stream. */
        NOA_IH Stream() : m_device(Device::getCurrent()) {}

        /**
         * Creates a new stream on the current device.
         * @param flag  Stream::concurrent or Stream::serial.
         * @note Streams are associated with a specific device. Use getDevice() to retrieve the device.
         */
        NOA_IH explicit Stream(uint flag) : m_device(Device::getCurrent()) {
            NOA_THROW_IF(cudaStreamCreateWithFlags(&m_stream, flag));
        }

        /**
         * Creates a new stream on a device.
         * @param device        Device on which the stream should be created.
         * @param flag          Stream::concurrent or Stream::serial.
         * @note Streams are associated with a specific device. Use getDevice() to retrieve the device.
         */
        NOA_IH explicit Stream(Device device, uint flag) : m_device(device) {
            DeviceCurrentScope scope_device(m_device);
            NOA_THROW_IF(cudaStreamCreateWithFlags(&m_stream, flag));
        }

        NOA_IH Stream(const Stream&) = delete;
        NOA_IH Stream& operator=(const Stream&) = delete;

        NOA_IH Stream(Stream&& to_move) noexcept
                : m_stream(std::exchange(to_move.m_stream, nullptr)), m_device(to_move.m_device) {}

        NOA_IH Stream& operator=(Stream&& to_move) noexcept {
            std::swap(m_stream, to_move.m_stream); // let to_move dtor destroy the old data if any.
            std::swap(m_device, to_move.m_device);
            return *this;
        }

        NOA_IH ~Stream() {
            if (m_stream) {
                DeviceCurrentScope scope_device(m_device);
                NOA_THROW_IF(cudaStreamDestroy(m_stream));
            }
        }

        NOA_IH cudaStream_t get() const noexcept { return m_stream; }
        NOA_IH cudaStream_t id() const noexcept { return m_stream; }
        NOA_IH Device device() const noexcept { return m_device; }
    };

    /** Retrieves the device's human readable name. */
    NOA_IH static std::string toString(const Stream& stream) {
        return String::format("CUDA stream (address: {}, device: {})",
                              static_cast<void*>(stream.id()), stream.device().id());
    }
}

template<>
struct fmt::formatter<Noa::CUDA::Stream> : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const Noa::CUDA::Stream& stream, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(Noa::CUDA::toString(stream), ctx);
    }
};

inline std::ostream& operator<<(std::ostream& os, const Noa::CUDA::Stream& stream) {
    os << Noa::CUDA::toString(stream);
    return os;
}
