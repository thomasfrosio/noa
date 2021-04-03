#pragma once

#include <cuda_runtime.h>

#include "noa/Definitions.h"
#include "noa/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Device.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace Noa::CUDA {

    class Event {
    public:
        enum flag_t : uint {
            busy_timer = 0U, // Default behavior, i.e. record time and busy-wait on synchronization.
            block_while_waiting = cudaEventBlockingSync, // When synchronizing on this event, shall a thread block?
            disable_timing = cudaEventDisableTiming, // Can this event be used to record time values (e.g. duration between events)?
            interprocess = cudaEventInterprocess // Can multiple processes work with the constructed event?
        };

    private:
        cudaEvent_t m_event{nullptr};
        Device m_device{};

    public:
        /**
         * Waits until the completion of all work currently captured in event.
         * @see     Event::record() for details on what is captured by an event.
         * @note    Waiting for an event that was created with the Event::block_while_waiting flag will cause the
         *          calling CPU thread to block until the event has been completed by the device. Otherwise, the CPU
         *          thread will busy-wait until the event has been completed by the device.
         */
        NOA_IH static void synchronize(const Event& event) {
            NOA_PROFILE_FUNCTION();
            DeviceCurrentScope stream_device(event.m_device);
            NOA_THROW_IF(cudaEventSynchronize(event.m_event));
        }

        /** Whether or not the event has completed all operations. */
        NOA_IH static bool hasCompleted(const Event& event) {
            DeviceCurrentScope scope_device(event.m_device);
            cudaError_t status = cudaEventQuery(event.m_event);
            if (status == cudaError_t::cudaSuccess)
                return true;
            else if (status == cudaError_t::cudaErrorNotReady)
                return false;
            else
                NOA_THROW(toString(status));
        }

        /** Records an already existing @a event into a @a stream. They must be on the same device. */
        NOA_IH static void record(const Stream& stream, const Event& event) {
            if (stream.device() != event.m_device)
                NOA_THROW("Stream and event are associated to different devices. Got device {} and device {}",
                          stream.device().id(), event.m_device.id());
            DeviceCurrentScope scope_device(event.m_device);
            NOA_THROW_IF(cudaEventRecord(event.m_event, stream.id()));
        }

        /**
         * Computes the elapsed time between events.
         * @note    Both events should be completed and both events should have recorded times (i.e. created without
         *          Event::disable_timing). Note that this measurement can be quite inaccurate.
         */
        NOA_IH static float elapsedTime(const Event& start, const Event& end) {
            if (start.m_device != end.m_device)
                NOA_THROW("Events are associated to different devices. Got device {} and device {}",
                          start.m_device.id(), end.m_device.id());
            float milliseconds{};
            NOA_THROW_IF(cudaEventElapsedTime(&milliseconds, start.m_event, end.m_event));
            return milliseconds;
        }

    public:
        /** Creates an event on the current device. */
        NOA_IH explicit Event(Event::flag_t flags = Event::busy_timer) : m_device(Device::getCurrent()) {
            NOA_THROW_IF(cudaEventCreateWithFlags(&m_event, flags));
        }

        /** Creates an event on a specific device. */
        NOA_IH explicit Event(Device device, Event::flag_t flags = Event::busy_timer) : m_device(device) {
            DeviceCurrentScope stream_device(m_device);
            NOA_THROW_IF(cudaEventCreateWithFlags(&m_event, flags));
        }

        NOA_IH Event(const Event&) = delete;
        NOA_IH Event& operator=(const Event&) = delete;

        NOA_IH Event(Event&& to_move) noexcept
                : m_event(std::exchange(to_move.m_event, nullptr)), m_device(to_move.m_device) {}

        NOA_IH Event& operator=(Event&& to_move) noexcept {
            std::swap(m_event, to_move.m_event); // let to_move dtor destroy the old data if any.
            std::swap(m_device, to_move.m_device);
            return *this;
        }

        ~Event() {
            NOA_THROW_IF(cudaEventDestroy(m_event)); // no need to be on the current device, apparently.
        }

        NOA_IH cudaEvent_t get() const noexcept { return m_event; }
        NOA_IH cudaEvent_t id() const noexcept { return m_event; }
        NOA_IH Device device() const noexcept { return m_device; }
    };
}
