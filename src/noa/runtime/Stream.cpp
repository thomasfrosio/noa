#include <forward_list>
#include "noa/runtime/Stream.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/fft/cuda/Plan.hpp"
#endif

namespace {
    // Adding, removing and moving the elements within the list does not invalidate
    // the iterators currently referring to other elements in the list.
    thread_local std::forward_list<noa::Stream> g_current{};

    #ifdef NOA_ENABLE_CUDA
        void clear_current_streams(noa::i32 id) {
            g_current.remove_if([&](const noa::Stream& stream) { return stream.device().id() == id; });
        }
    #endif
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
        #ifdef NOA_ENABLE_CUDA
        static bool is_initialized = false;
        if (not is_initialized) {
            Device::add_reset_callback(clear_current_streams);
            is_initialized = true;
        }
        #endif

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
