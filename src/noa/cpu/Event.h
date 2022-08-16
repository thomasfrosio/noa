#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu {
    // Simple event, that can be enqueued in streams.
    // Elapsed time can be measured between events.
    class Event {
    public:
        // Waits ("busy sleep") until the completion of the event.
        void synchronize() {
            while (m_event.load(std::memory_order_acquire) != Status::COMPLETED)
                std::this_thread::yield();
        }

        // Whether the event is completed.
        bool busy() {
            return m_event.load(std::memory_order_acquire) == Status::COMPLETED;
        }

        // Records (enqueue) the event into a stream.
        void record(Stream& stream) {
            m_event.store(Status::QUEUED, std::memory_order_release);
            stream.enqueue(
                    [this]() noexcept {
                        this->m_event.store(Status::COMPLETED);
                        this->m_time = std::chrono::steady_clock::now();
                    });
        }

        // Computes the elapsed time between completed events.
        static double elapsed(const Event& start, const Event& end) {
            const Status status_start = start.m_event.load(std::memory_order_acquire);
            const Status status_end = end.m_event.load(std::memory_order_acquire);

            if (status_start == COMPLETED && status_end == COMPLETED) {
                std::chrono::duration<double, std::milli> diff = end.m_time - start.m_time;
                return diff.count();
            } else if (status_start == QUEUED || status_end == QUEUED) {
                NOA_THROW("At least one event has not been completed");
            } else {
                NOA_THROW("At least one event has not been recorded");
            }
        }

    public:
        Event(const Event&) = delete;
        Event(Event&&) = delete;
        Event& operator=(const Event&) = delete;
        Event& operator=(Event&&) = delete;

    private:
        enum Status : int {
            CREATED, QUEUED, COMPLETED
        };

        std::atomic<Status> m_event{CREATED};
        std::chrono::time_point<std::chrono::steady_clock> m_time;
    };
}
