#pragma once

#include <variant>

#include "noa/cpu/Event.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/Backend.hpp"
#include "noa/gpu/cuda/Event.hpp"
#else
namespace noa::gpu {
    class Event {
    public:
        static void record(auto&&) { panic(); }
        static void synchronize() { panic(); }
        static bool is_busy() { return false; }
        static auto elapsed(auto&&, auto&&) -> std::chrono::duration<f64, std::milli> { return {}; }
    };
}
#endif

#include "noa/unified/Device.hpp"
#include "noa/unified/Stream.hpp"

namespace noa::inline types {
    /// Simple event that can be enqueued.
    /// Elapsed time can be measured between events.
    class Event {
    public:
        using cpu_event_t = noa::cpu::Event;
        using gpu_event_t = noa::gpu::Event;
        using variant_type = std::variant<cpu_event_t, gpu_event_t>;

    public:
        /// Creates a new event on the given device.
        Event() = default;

        /// Waits ("busy sleep") until the completion of the event.
        void synchronize() {
            std::visit([](auto&& e) { e.synchronize(); }, m_event);
        }

        /// Whether the event is completed.
        [[nodiscard]] auto is_busy() -> bool {
            return std::visit([](auto&& e) { return e.is_busy(); }, m_event);
        }

        /// Records (enqueue) the event into a stream.
        void record(Stream& stream) {
            if (stream.device().is_cpu()) {
                m_event.emplace<cpu_event_t>();
                std::get<cpu_event_t>(m_event).record(stream.cpu());
            } else {
                m_event.emplace<gpu_event_t>();
                std::get<gpu_event_t>(m_event).record(stream.gpu());
            }
        }

        /// Computes the elapsed time between completed events.
        /// \example
        /// \code
        /// Event start, end;
        /// start.record(stream);
        /// ...
        /// end.record(stream);
        /// end.synchronize();
        /// fmt::print("... took {}", Event::elapsed(start, end));
        /// \endcode
        static auto elapsed(
            const Event& start,
            const Event& end
        ) -> std::chrono::duration<f64, std::milli> {
            return std::visit([&]<typename T>(const T& s) {
                auto* e = std::get_if<T>(&end.m_event);
                check(e, "Events are not of the same device type");
                return T::elapsed(s, *e);
            }, start.m_event);
        }

    private:
        variant_type m_event;
    };
}
