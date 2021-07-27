/// \file noa/gpu/cuda/util/Event.h
/// \brief CUDA events.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include <cuda_runtime.h>

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Device.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda {
    /// A CUDA event (and its associated device).
    class Event {
    private:
        cudaEvent_t m_event{nullptr};
        Device m_device{};

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
        /// Waits until the completion of all work currently captured in event.
        /// \note    Waiting for an event that was created with the \c BLOCK_WHILE_WAITING flag will cause the
        ///          calling CPU thread to block until the event has been completed by the device. Otherwise, the CPU
        ///          thread will busy-wait until the event has been completed by the device.
        NOA_HOST static void synchronize(const Event& event) {
            NOA_PROFILE_FUNCTION();
            DeviceCurrentScope stream_device(event.m_device);
            NOA_THROW_IF(cudaEventSynchronize(event.m_event));
        }

        /// Whether or not the event has completed all operations.
        NOA_HOST static bool hasCompleted(const Event& event) {
            DeviceCurrentScope scope_device(event.m_device);
            cudaError_t status = cudaEventQuery(event.m_event);
            if (status == cudaError_t::cudaSuccess)
                return true;
            else if (status == cudaError_t::cudaErrorNotReady)
                return false;
            else
                NOA_THROW(toString(status));
        }

        /// Records an already existing \p event into a \p stream. They must be on the same device.
        NOA_HOST static void record(const Stream& stream, const Event& event) {
            if (stream.device() != event.m_device)
                NOA_THROW("Stream and event are associated to different devices. Got device {} and device {}",
                          stream.device().id(), event.m_device.id());
            DeviceCurrentScope scope_device(event.m_device);
            NOA_THROW_IF(cudaEventRecord(event.m_event, stream.id()));
        }

        /// Computes the elapsed time between events.
        /// \note    Both events should be completed and both events should have recorded times (i.e. created without
        ///          \c DISABLE_TIMING). Note that this measurement can be quite inaccurate.
        NOA_HOST static float elapsedTime(const Event& start, const Event& end) {
            if (start.m_device != end.m_device)
                NOA_THROW("Events are associated to different devices. Got device {} and device {}",
                          start.m_device.id(), end.m_device.id());
            float milliseconds{};
            NOA_THROW_IF(cudaEventElapsedTime(&milliseconds, start.m_event, end.m_event));
            return milliseconds;
        }

    public:
        /// Creates an event on the current device.
        NOA_HOST explicit Event(Mode flags = BUSY_TIMER) : m_device(Device::getCurrent()) {
            NOA_THROW_IF(cudaEventCreateWithFlags(&m_event, flags));
        }

        /// Creates an event on a specific device.
        NOA_HOST explicit Event(Device device, Mode flags = BUSY_TIMER) : m_device(device) {
            DeviceCurrentScope stream_device(m_device);
            NOA_THROW_IF(cudaEventCreateWithFlags(&m_event, flags));
        }

        Event(const Event&) = delete;
        Event& operator=(const Event&) = delete;

        NOA_HOST Event(Event&& to_move) noexcept
                : m_event(std::exchange(to_move.m_event, nullptr)), m_device(to_move.m_device) {}

        NOA_HOST Event& operator=(Event&& to_move) noexcept {
            std::swap(m_event, to_move.m_event); // let to_move dtor destroy the old data if any.
            std::swap(m_device, to_move.m_device);
            return *this;
        }

        NOA_HOST ~Event() {
            cudaError_t err = cudaEventDestroy(m_event); // no need to be on the current device, apparently.
            if (err != cudaSuccess && std::uncaught_exceptions() == 0)
                NOA_THROW(toString(err));
        }

        NOA_HOST cudaEvent_t get() const noexcept { return m_event; }
        NOA_HOST cudaEvent_t id() const noexcept { return m_event; }
        NOA_HOST Device device() const noexcept { return m_device; }
    };
}
