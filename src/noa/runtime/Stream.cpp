#include <forward_list>
#include "noa/runtime/Stream.hpp"

namespace {
    // Adding, removing and moving the elements within the list does not invalidate
    // the iterators currently referring to other elements in the list.
    thread_local std::forward_list<noa::Stream> g_current{};
}

namespace noa {
    auto Stream::current(Device device) -> Stream& {
        for (auto& current_stream: g_current)
            if (current_stream.device() == device)
                return current_stream;
        // Since the current for this device doesn't exist, create the stream and set it as current.
        g_current.emplace_front(device, Stream::DEFAULT);
        return g_current.front();
    }

    void Stream::set_current(const Stream& stream) {
        for (auto& current_stream: g_current) {
            if (current_stream.device() == stream.device()) {
                current_stream = stream;
                return;
            }
        }
        // At this point, we know there's no current stream for that device.
        g_current.push_front(stream);
    }
}
