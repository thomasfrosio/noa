#pragma once

#include <chrono>

#include "noa/common/Definitions.h"

namespace noa {
    class Timer {
    public:
        /// Starts the timer.
        void start() noexcept {
            m_start = std::chrono::steady_clock::now();
            m_running = true;
        }

        /// Stops the timer.
        void stop() noexcept {
            m_end = std::chrono::steady_clock::now();
            m_running = false;
        }

        /// Returns the elapsed time, in milliseconds, between the last start() and stop().
        /// \note If the timer is still running (stop() was not called), elapsed() takes the current time
        ///       and uses it, but does not stop the timer.
        double elapsed() noexcept {
            time_point_t end = m_running ? std::chrono::steady_clock::now() : m_end;
            duration_t diff = end - m_start;
            return diff.count();
        }

    private:
        using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
        using duration_t = std::chrono::duration<double, std::milli>;
        using microseconds_t = std::chrono::milliseconds;

        time_point_t m_start;
        time_point_t m_end;
        bool m_running{false};
    };
}
