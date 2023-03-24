#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/unified/Device.hpp"

#include "noa/cpu/Stream.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/Backend.hpp"
#else
namespace noa::cuda {
    class Stream {};
}
#endif

namespace noa::gpu {
    using Stream = noa::cuda::Stream;
}

namespace noa {
    /// Stream mode. DEFAULT refers to the NULL stream. ASYNC creates an actual asynchronous queue.
    /// On the CPU, ASYNC launches a new thread which waits to execute work. On the GPU, ASYNC launches
    /// a new "concurrent" stream which is not implicitly synchronized with the NULL stream.
    enum class StreamMode {
        DEFAULT,
        ASYNC
    };

    /// Unified stream, i.e. (asynchronous) dispatch queue, and its associated device.
    /// \details
    ///   - Streams are reference counted. While they can be moved and copied around, the actual
    ///     machinery stays at the same location, unaffected by these changes, and is destructed
    ///     once the reference count drops to zero.
    ///   - Each device has a "current" stream (this state is per host-thread). The current stream
    ///     is queried very frequently by the library to know where to enqueue function calls. By default,
    ///     the current streams are the NULL streams. For the CPU, this refers to the host thread being the
    ///     worker (all execution using the CPU device is synchronous, like if there was no streams at all).
    ///   - While streams allow for asynchronous execution relative to the host thread, the streams
    ///     are not thread-safe (this is also why the "current" stream is per host-thread). Enqueuing
    ///     or querying a stream, or any of its references, should be done in a thread-safe manner.
    class Stream {
    public:
        /// Creates a new stream on the given device.
        explicit Stream(Device device, StreamMode mode = StreamMode::ASYNC) : m_device(device) {
            if (m_device.is_cpu()) {
                const auto cpu_mode = mode == StreamMode::ASYNC ? cpu::StreamMode::ASYNC : cpu::StreamMode::DEFAULT;
                m_stream = cpu::Stream(cpu_mode);
            } else {
                #ifdef NOA_ENABLE_CUDA
                const auto cuda_mode = mode == StreamMode::ASYNC ? cuda::StreamMode::ASYNC : cuda::StreamMode::DEFAULT;
                m_stream = cuda::Stream(cuda::Device(m_device.id()), cuda_mode);
                #else
                NOA_THROW("No GPU backend detected");
                #endif
            }
        }

        /// Encapsulates the given backend stream into a new unified stream.
        template<typename BackendStream, typename = std::enable_if_t<
                 noa::traits::is_almost_any_v<BackendStream, cpu::Stream, gpu::Stream>>>
        explicit Stream(BackendStream&& stream) {
            if constexpr (noa::traits::is_almost_same_v<BackendStream, cpu::Stream>) {
                m_device = Device{};
            } else {
                #ifdef NOA_ENABLE_CUDA
                m_device = Device(DeviceType::GPU, stream.device().id(), Device::DeviceUnchecked{});
                #else
                NOA_THROW("No GPU backend detected");
                #endif
            }
            m_stream(std::forward<BackendStream>(stream));
        }

    public:
        /// Retrieves the current stream of a given device.
        /// \note If the device has no current stream, a new "default" stream is created.
        static Stream& current(Device device);

        /// Sets the stream as the current stream for its device.
        /// \note If the previous current stream was created by current(), any reference of it becomes invalidated.
        ///       However, since streams are reference counted, if the reference was copied into a new instance,
        ///       this instance will still be valid.
        static void set_current(const Stream& stream);

        /// Blocks until the stream has completed all operations.
        /// \note This function may also return error codes from previous, asynchronous launches.
        void synchronize() {
            std::visit([](auto&& stream) { stream.synchronize(); }, m_stream);
        }

        /// Whether or not the stream is busy.
        /// \note This function may also return error codes from previous, asynchronous launches.
        bool is_busy() {
            return std::visit([](auto&& stream) -> bool { return stream.is_busy(); }, m_stream);
        }

        /// Gets the underlying device.
        [[nodiscard]] Device device() const noexcept {
            return m_device;
        }

        /// Gets the underlying stream, assuming it is a CPU stream (i.e. device is CPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] cpu::Stream& cpu() {
            auto* cpu_stream = std::get_if<cpu::Stream>(&m_stream);
            NOA_CHECK(cpu_stream != nullptr, "The stream is not a CPU stream");
            return *cpu_stream;
        }

        /// Gets the underlying stream, assuming it is a GPU stream (i.e. device is GPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] gpu::Stream& gpu() {
            #ifdef NOA_ENABLE_CUDA
            return cuda();
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }

        /// Gets the underlying stream, assuming it is a CUDA stream (i.e. device is a CUDA-capable GPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] cuda::Stream& cuda() {
            #ifdef NOA_ENABLE_CUDA
            auto* cuda_stream = std::get_if<cuda::Stream>(&m_stream);
            NOA_CHECK(cuda_stream != nullptr, "The stream is not a GPU stream");
            return *cuda_stream;
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }

    private:
        std::variant<cpu::Stream, gpu::Stream> m_stream{};
        Device m_device;
    };
}

namespace noa {
    /// A stream that sets itself as the current stream for the remainder of the scope.
    class StreamGuard : public Stream {
    public:
        template<typename ... Args>
        explicit StreamGuard(Args&& ...args)
                : Stream(std::forward<Args>(args)...),
                  m_previous_current(Stream::current(this->device())) {
            Stream::set_current(*static_cast<Stream*>(this));
        }

        ~StreamGuard() {
            Stream::set_current(m_previous_current);
        }

    private:
        Stream m_previous_current;
    };
}
