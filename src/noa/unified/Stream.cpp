#include <forward_list>

#include "Stream.h"

namespace {
    // Adding, removing and moving the elements within the list does not invalidate
    // the iterators currently referring to other elements in the list.
    thread_local std::forward_list<noa::Stream> g_current{};
}

namespace noa {
    Stream& Stream::current(Device device) {
        for (auto& current_stream: g_current)
            if (current_stream.device() == device)
                return current_stream;
        // Since the current for this device doesn't exist, create the stream and set it as current.
        g_current.push_front(Stream(device, Stream::DEFAULT));
        return g_current.front();
    }

    void Stream::current(Stream& stream) {
        for (auto& current_stream: g_current) {
            if (current_stream.device() == stream.device()) {
                current_stream = stream;
                return;
            }
        }
        // At this point we know there's no current stream for that device.
        g_current.push_front(stream);
    }
}
