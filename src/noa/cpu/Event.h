#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Timer.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu {
    /// Simple event, that can be enqueued in streams.
    /// Elapsed time can be measured between events.
    class Event {
    public:
        /// Waits ("busy sleep") until the completion of the event.
        NOA_HOST void synchronize() {
            while (m_event.load(std::memory_order_acquire) != Status::COMPLETED)
                std::this_thread::yield();
        }

        /// Whether or not the event is completed.
        NOA_HOST bool hasCompleted() {
            return m_event.load(std::memory_order_acquire) == Status::COMPLETED;
        }

        /// Records (enqueue) the event into a \p stream.
        NOA_HOST void record(Stream& stream) {
            m_event.store(Status::QUEUED, std::memory_order_release);
            stream.enqueue(
                    [this]() noexcept {
                        this->m_event.store(Status::COMPLETED);
                        this->m_time = std::chrono::steady_clock::now();
                    });
        }

        /// Computes the elapsed time between \e completed events.
        NOA_HOST static double elapsedTime(const Event& start, const Event& end) {
            std::chrono::duration<double, std::milli> diff = end.m_time - start.m_time;
            return diff.count();
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
