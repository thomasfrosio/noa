#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/Error.hpp"
#include "noa/cpu/Stream.hpp"

namespace noa::cpu {
    // Simple event that can be enqueued in streams.
    // Elapsed time can be measured between events.
    class Event {
    public:
        Event() = default;

        // Waits ("busy sleep") until the completion of the event.
        void synchronize() const {
            while (m_event.load(std::memory_order_acquire) != Status::COMPLETED)
                std::this_thread::yield();
        }

        // Whether the event is completed.
        [[nodiscard]] auto is_busy() const -> bool {
            return m_event.load(std::memory_order_acquire) == Status::COMPLETED;
        }

        // Records (enqueue) the event into a stream.
        void record(Stream& stream) {
            m_event.store(QUEUED, std::memory_order_release);
            stream.enqueue(
                [this]() noexcept {
                    this->m_event.store(COMPLETED);
                    this->m_time = std::chrono::steady_clock::now();
                });
        }

        // Computes the elapsed time between completed events.
        static auto elapsed(const Event& start, const Event& end) {
            const Status status_start = start.m_event.load(std::memory_order_acquire);
            const Status status_end = end.m_event.load(std::memory_order_acquire);

            if (status_start == COMPLETED and status_end == COMPLETED) {
                std::chrono::duration<f64, std::milli> diff = end.m_time - start.m_time;
                return diff;
            } else if (status_start == QUEUED or status_end == QUEUED) {
                panic("At least one event has not been completed");
            } else {
                panic("At least one event has not been recorded");
            }
        }

    public:
        Event(const Event&) = delete;
        Event(Event&&) = delete;
        Event& operator=(const Event&) = delete;
        Event& operator=(Event&&) = delete;

    private:
        enum class Status : i32 {
            CREATED, QUEUED, COMPLETED
        };
        using enum Status;

        std::atomic<Status> m_event{CREATED};
        std::chrono::time_point<std::chrono::steady_clock> m_time;
    };
}
#endif
