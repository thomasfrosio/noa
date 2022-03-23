/// \file noa/gpu/cuda/Event.h
/// \brief CUDA events.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include <cuda_runtime.h>

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Device.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda {
    /// A CUDA event (and its associated device).
    class Event {
    public:
        enum Mode : uint {
            /// Default behavior, i.e. record time and busy-wait on synchronization.
            BUSY_TIMER = 0U,

            /// When synchronizing on this event, shall a thread block?
            BLOCK_WHILE_WAITING = cudaEventBlockingSync,

            /// Can this event be used to record time values (e.g. duration between events)?
            DISABLE_TIMING = cudaEventDisableTiming,

            /// Can multiple processes work with the constructed event?
            INTERPROCESS = cudaEventInterprocess
        };

    public:
        /// Creates an event on the current device.
        NOA_HOST explicit Event(Mode flags = BUSY_TIMER) : m_device(Device::current()) {
            NOA_THROW_IF(cudaEventCreateWithFlags(&m_event, flags));
        }

        /// Creates an event on a specific device.
        NOA_HOST explicit Event(Device device, Mode flags = BUSY_TIMER) : m_device(device) {
            DeviceGuard stream_device(m_device);
            NOA_THROW_IF(cudaEventCreateWithFlags(&m_event, flags));
        }

    public:
        /// Waits until the completion of all work currently captured in event.
        /// \note    Waiting for an event that was created with the \c BLOCK_WHILE_WAITING flag will cause the
        ///          calling CPU thread to block until the event has been completed by the device. Otherwise, the CPU
        ///          thread will busy-wait until the event has been completed by the device.
        NOA_HOST void synchronize() {
            NOA_PROFILE_FUNCTION();
            DeviceGuard stream_device(m_device);
            NOA_THROW_IF(cudaEventSynchronize(m_event));
        }

        /// Whether or not the event has completed all operations.
        NOA_HOST bool busy() {
            DeviceGuard scope_device(m_device);
            cudaError_t status = cudaEventQuery(m_event);
            if (status == cudaError_t::cudaSuccess)
                return true;
            else if (status == cudaError_t::cudaErrorNotReady)
                return false;
            else
                NOA_THROW(toString(status));
        }

        /// Records an already existing \p event into a \p stream. They must be on the same device.
        NOA_HOST void record(const Stream& stream) {
            if (stream.device() != m_device)
                NOA_THROW("Stream and event are associated to different devices. Got device {} and device {}",
                          stream.device().id(), m_device.id());
            DeviceGuard scope_device(m_device);
            NOA_THROW_IF(cudaEventRecord(m_event, stream.id()));
        }

        /// Computes the elapsed time between events.
        /// \note    Both events should be completed and both events should have recorded times (i.e. created without
        ///          \c DISABLE_TIMING). Note that this measurement can be quite inaccurate.
        NOA_HOST static double elapsed(const Event& start, const Event& end) {
            if (start.m_device != end.m_device)
                NOA_THROW("Events are associated to different devices. Got device {} and device {}",
                          start.m_device.id(), end.m_device.id());
            float milliseconds{};
            NOA_THROW_IF(cudaEventElapsedTime(&milliseconds, start.m_event, end.m_event));
            return static_cast<double>(milliseconds);
        }

    public:
        Event(const Event&) = delete;
        Event(Event&&) = delete;
        Event& operator=(const Event&) = delete;
        Event& operator=(Event&&) = delete;

        NOA_HOST ~Event() noexcept(false) {
            cudaError_t err = cudaEventDestroy(m_event); // no need to be on the current device, apparently.
            if (err != cudaSuccess && std::uncaught_exceptions() == 0)
                NOA_THROW(toString(err));
        }

        [[nodiscard]] NOA_HOST cudaEvent_t get() const noexcept { return m_event; }
        [[nodiscard]] NOA_HOST cudaEvent_t id() const noexcept { return m_event; }
        [[nodiscard]] NOA_HOST Device device() const noexcept { return m_device; }

    private:
        cudaEvent_t m_event{nullptr};
        Device m_device{};
    };
}
