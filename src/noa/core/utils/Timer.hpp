#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include <chrono>
#else
#include <cuda/std/chrono>
#endif

namespace noa {
    class Timer {
    public:
        using time_point_type = std::chrono::time_point<std::chrono::steady_clock>;
        using duration_type = std::chrono::duration<double, std::milli>;

        /// Starts the timer.
        Timer& start() noexcept {
            m_start = std::chrono::steady_clock::now();
            m_running = true;
            return *this;
        }

        /// Stops the timer.
        Timer& stop() noexcept {
            m_end = std::chrono::steady_clock::now();
            m_running = false;
            return *this;
        }

        /// Returns the elapsed time, in milliseconds, between the last start() and stop().
        /// If the timer is still running (stop() was not called), elapsed() takes the current time
        /// and uses it, but does not stop the timer.
        [[nodiscard]] auto elapsed() const noexcept -> duration_type {
            time_point_type end = m_running ? std::chrono::steady_clock::now() : m_end;
            duration_type diff = end - m_start;
            return diff; // use .count() to get the double
        }

    private:
        time_point_type m_start;
        time_point_type m_end;
        bool m_running{};
    };
}
