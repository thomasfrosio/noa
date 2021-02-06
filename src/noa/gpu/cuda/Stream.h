#pragma once

#include "noa/gpu/Base.h"
#include "noa/gpu/cuda/Device.h"

namespace Noa::CUDA {

    class Stream {
    public:
        using id_t = cudaStream_t; // this is a pointer
    private:
        Stream::id_t m_stream{};
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
            cudaError_t status = cudaStreamQuery(stream.m_stream));
            if (status == cudaError_t::cudaSuccess)
                return true;
            else if (status == cudaError_t::cudaErrorNotReady)
                return false;
            else
                NOA_THROW(toString(status));
        }

        NOA_IH static Device getDevice(const Stream& stream) noexcept {
            return stream.m_device();
        }

    public:
        NOA_IH Stream() = default;

        NOA_IH explicit Stream(bool non_blocking = false) : m_device(Device::getCurrent()) {
            uint flags = non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;
            NOA_THROW_IF(cudaStreamCreateWithFlags(&m_stream, flags));
        }

        NOA_IH explicit Stream(Device device, bool non_blocking = false) : m_device(device) {
            DeviceCurrentScope scope_device(m_device);
            uint flags = non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;
            NOA_THROW_IF(cudaStreamCreateWithFlags(&m_stream, flags));
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
                NOA_THROW_IF(cudaStreamDestroy(&m_stream));
            }
        }

        NOA_IH Stream::id_t get() const noexcept { return m_stream; }
        NOA_IH Stream::id_t id() const noexcept { return m_stream; }
    };
}
