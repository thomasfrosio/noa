#include <forward_list>
#include <list>

#include "Stream.h"

namespace {
    // "origin" keeps track of the location in g_buffer.
    // If the stream was created by the user and passed to Stream::current(), origin == end().
    struct CurrentStream {
        noa::Stream* stream;
        std::list<noa::Stream>::iterator origin;
    };

    // Adding, removing and moving the elements within the list does not invalidate
    // the iterators currently referring to other elements in the list.
    thread_local std::list<noa::Stream> g_buffer{};
    thread_local std::forward_list<CurrentStream> g_current{};
}

namespace noa {
    Stream& Stream::current(Device device) {
        for (auto current_stream: g_current)
            if (current_stream.stream->device() == device)
                return *current_stream.stream;

        // Since the current for this device doesn't exist, create the stream and set it as current.
        g_buffer.emplace_front(device, Stream::DEFAULT);
        g_current.push_front({&g_buffer.front(), // the stream we just created
                              g_buffer.begin()}); // its position in the buffer so that we can destruct it later
        return g_buffer.front();
    }

    void Stream::current(Stream& stream) {
        for (auto& current_stream: g_current) {
            if (current_stream.stream->device() == stream.device()) {
                if (current_stream.stream == &stream)
                    return;
                if (current_stream.origin != g_buffer.end()) {// do we need to destruct the old current stream?
                    g_buffer.erase(current_stream.origin);
                    current_stream.origin = g_buffer.end();
                }
                current_stream.stream = &stream;
                return;
            }
        }
        // At this point we know there's no current stream for that device.
        g_current.push_front({&stream, g_buffer.end()}); // origin = end(), i.e. we don't own this stream
    }
}
