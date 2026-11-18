#pragma once

#include "noa/runtime/cuda/Runtime.hpp"
#include "noa/runtime/cuda/Error.hpp"
#include "noa/runtime/cuda/Device.hpp"
#include "noa/runtime/cuda/Stream.hpp"

namespace noa::gpu {

    // An event (and its associated device) used for timing.
    class Event {
    public:
        // Creates an event on the current device.
        explicit Event(bool busy_wait = true) : Event(Device::current(), busy_wait) {}

        // Creates an event on a specific device.
        explicit Event(Device device, bool busy_wait = true) : m_device(device) {
            const DeviceGuard stream_device(m_device);
            #if defined(NOA_ENABLE_CUDA)
            check(cudaEventCreateWithFlags(&m_event, busy_wait ? 0 : cudaEventBlockingSync));
            #elif defined(NOA_ENABLE_HIP)
            check(hipEventCreateWithFlags(&m_event, busy_wait ? 0 : hipEventBlockingSync));
            #elif defined(NOA_ENABLE_SYCL)
            (void)busy_wait;
            #endif
        }

    public:
        // Waits until the completion of all work currently captured in event.
        // Waiting for an event that was created with the BLOCK_WHILE_WAITING flag will cause the
        // calling CPU thread to block until the event has been completed by the device. Otherwise, the CPU
        // thread will busy-wait until the event has been completed by the device.
        void synchronize() {
            #if defined(NOA_ENABLE_CUDA)
            const DeviceGuard stream_device(m_device);
            check(cudaEventSynchronize(m_event));
            #elif defined(NOA_ENABLE_HIP)
            const DeviceGuard guard(m_device);
            check(hipEventSynchronize(m_event));
            #elif defined(NOA_ENABLE_SYCL)
            if (m_event)
                m_event->wait_and_throw();
            #endif
        }

        // Whether the event has completed all operations.
        auto is_busy() -> bool {
            #if defined(NOA_ENABLE_CUDA)
            const DeviceGuard scope_device(m_device);
            const cudaError_t status = cudaEventQuery(m_event);
            if (status == cudaError_t::cudaSuccess) return false;
            if (status == cudaError_t::cudaErrorNotReady) return true;
            panic_runtime(error2string(status));
            #elif defined(NOA_ENABLE_HIP)
            const DeviceGuard guard(m_device);
            const hipError_t status = hipEventQuery(m_event);
            if (status == hipError_t::hipSuccess) return false;
            if (status == hipError_t::hipErrorNotReady) return true;
            panic_runtime(error2string(status));
            #elif defined(NOA_ENABLE_SYCL)
            if (not m_event)
                return false;
            return m_event->get_info<sycl::info::event::command_execution_status>() !=
                   sycl::info::event_command_status::complete;
            #endif
        }

        // Records an already existing event into a stream. They must be on the same device.
        void record(Stream& stream) {
            if (stream.device() != m_device) {
                panic("Stream and event are associated to different devices. Got device {} and device {}",
                      stream.device().id(), m_device.id());
            }
            #if defined(NOA_ENABLE_CUDA)
            const DeviceGuard scope_device(m_device);
            check(cudaEventRecord(m_event, stream.id()));
            #elif defined(NOA_ENABLE_HIP)
            const DeviceGuard scope_device(m_device);
            check(hipEventRecord(m_event, stream.id()));
            #elif defined(NOA_ENABLE_SYCL)
            m_event = stream.enqueue_sycl_event_barrier();
            #endif
        }

        // Computes the elapsed time between events. Both events should be completed.
        // Note that this measurement can be inaccurate. In CUDA, prefer the default/null stream.
        static auto elapsed(const Event& start, const Event& end) {
            if (start.m_device != end.m_device) {
                panic("Events are associated to different devices. Got device {} and device {}",
                      start.m_device.id(), end.m_device.id());
            }

            #if defined(NOA_ENABLE_CUDA)
            f32 milliseconds{};
            check(cudaEventElapsedTime(&milliseconds, start.m_event, end.m_event));
            return std::chrono::duration<f64, std::milli>(static_cast<f64>(milliseconds));
            #elif defined(NOA_ENABLE_HIP)
            f32 milliseconds{};
            check(hipEventElapsedTime(&milliseconds, start.m_event, end.m_event));
            return std::chrono::duration<f64, std::milli>(static_cast<f64>(milliseconds));
            #elif defined(NOA_ENABLE_SYCL)
            check(start.m_event and end.m_event, "Both events must have been recorded");
            const auto t0 = start.m_event->get_profiling_info<sycl::info::event_profiling::command_end>();
            const auto t1 = end.m_event->get_profiling_info<sycl::info::event_profiling::command_end>();
            return std::chrono::duration<f64, std::milli>(static_cast<f64>(t1 - t0) * 1e-6); // ns -> ms
            #endif
        }

    public:
        Event(const Event&) = delete;
        Event(Event&&) = delete;
        Event& operator=(const Event&) = delete;
        Event& operator=(Event&&) = delete;

        #if defined(NOA_ENABLE_CUDA)
        ~Event() noexcept(false) {
            const cudaError_t err = cudaEventDestroy(m_event); // no need to be on the current device, apparently.
            if (err != cudaSuccess and std::uncaught_exceptions() == 0)
                panic_runtime(error2string(err));
        }
        #elif defined(NOA_ENABLE_HIP)
        ~Event() noexcept(false) {
            const hipError_t err = hipEventDestroy(m_event);
            if (err != hipSuccess and std::uncaught_exceptions() == 0)
                panic_runtime(error2string(err));
        }
        #elif defined(NOA_ENABLE_SYCL)
        ~Event() = default;
        #endif

        [[nodiscard]] auto device() const noexcept -> Device { return m_device; }

    private:
#if defined(NOA_ENABLE_SYCL)
        std::optional<sycl::event> m_event{};
#else
        cudaEvent_t m_event{nullptr};
#endif
        Device m_device{};
    };
}
