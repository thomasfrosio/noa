#pragma once

#include "noa/unified/Device.hpp"
#include "noa/unified/Session.hpp"
#include "noa/cpu/Stream.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/Backend.hpp"
#else
namespace noa::gpu {
    class Stream {
    public:
        static void synchronize() {}
        static auto is_busy() -> bool { return false; }
    };
}
#endif

namespace noa::inline types {
    /// Unified stream, i.e. shared (asynchronous) dispatch queue, and its associated device.
    /// \details
    ///   - Streams are reference counted. While they can be moved and copied around, the actual
    ///     machinery stays at the same location, unaffected by these changes, and is destructed
    ///     once the reference count drops to zero.
    ///   - Each device has a "current" stream (this state is per host-thread). The library frequently queries
    ///     the current stream to know where to enqueue functions. By default, the current streams are the
    ///     NULL streams. For the CPU, this refers to the host thread (all executions using the CPU device
    ///     are synchronous, like if there were no streams at all).
    ///   - While streams allow for asynchronous execution relative to the host thread, the streams
    ///     are not thread-safe (this is also why the "current" stream is per host-thread). Enqueuing
    ///     or querying a stream should be done in a thread-safe manner.
    class Stream {
    public:
        using cpu_stream = noa::cpu::Stream;
        using gpu_stream = noa::gpu::Stream;

        /// Stream mode.
        /// For CPU devices: DEFAULT/SYNC refers to the current thread. ASYNC launches a new thread which waits
        ///                  to execute work, in which case the execution will be asynchronous relative to the
        ///                  current thread - ASYNC creates an asynchronous queue.
        /// For CUDA devices: DEFAULT refers to the NULL stream. SYNC is a new stream implicitly synchronized with
        ///                   the NULL stream. ASYNC creates a new "concurrent" stream which is not implicitly
        ///                   synchronized with the NULL stream.
        enum class Mode {
            DEFAULT = 0,
            SYNC = 1,
            ASYNC = 2,
        };
        using enum Mode;

    public:
        /// Creates a new stream on the given device.
        explicit Stream(Device device, Mode mode = Mode::ASYNC) : m_device(device) {
            if (m_device.is_cpu()) {
                m_stream = cpu_stream(
                    mode == Mode::ASYNC ? noa::cpu::Stream::ASYNC : noa::cpu::Stream::SYNC,
                    Session::thread_limit());
            } else {
                #ifdef NOA_ENABLE_CUDA
                m_stream = gpu_stream(
                    noa::cuda::Device(m_device.id()),
                    mode == Mode::ASYNC ? noa::cuda::Stream::ASYNC :
                    mode == Mode::SYNC ? noa::cuda::Stream::SYNC :
                    noa::cuda::Stream::DEFAULT);
                #else
                panic();
                #endif
            }
        }

    public:
        /// Retrieves the current stream of a given device.
        /// \note If the device has no current stream, a new "default" stream is created.
        static auto current(Device device) -> Stream&;

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
        [[nodiscard]] auto is_busy() -> bool {
            return std::visit([](auto&& stream) -> bool { return stream.is_busy(); }, m_stream);
        }

        /// Gets the underlying device.
        [[nodiscard]] auto device() const noexcept -> Device {
            return m_device;
        }

        /// Sets the maximum number of threads the CPU backend is allowed to use.
        void set_thread_limit(i32 n_threads) noexcept {
            auto* stream = std::get_if<cpu_stream>(&m_stream);
            if (stream)
                stream->set_thread_limit(n_threads);
        }

        [[nodiscard]] auto thread_limit() const noexcept -> i32 {
            const auto* stream = std::get_if<cpu_stream>(&m_stream);
            if (stream)
                return stream->thread_limit();
            return 1;
        }

    public: // Access backend stream.
        /// Gets the underlying stream, assuming it is a CPU stream (i.e. device is CPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] auto cpu() -> cpu_stream& {
            auto* stream = std::get_if<cpu_stream>(&m_stream);
            check(stream != nullptr, "The stream is not a CPU stream");
            return *stream;
        }

        /// Gets the underlying stream, assuming it is a GPU stream (i.e. device is GPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] auto gpu() -> gpu_stream& {
            #ifdef NOA_ENABLE_CUDA
            return cuda();
            #else
            panic();
            #endif
        }

#ifdef NOA_ENABLE_CUDA
        /// Gets the underlying stream, assuming it is a CUDA stream (i.e. device is a CUDA-capable GPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] auto cuda() -> gpu_stream& {
            auto* stream = std::get_if<gpu_stream>(&m_stream);
            check(stream != nullptr, "The stream is not a GPU stream");
            return *stream;
        }
#endif

    private:
        std::variant<cpu_stream, gpu_stream> m_stream{};
        Device m_device;
    };

    /// A stream that sets itself as the current stream for the remainder of the scope.
    class StreamGuard : public Stream {
    public:
        template<typename ... Args>
        explicit StreamGuard(Args&& ...args) :
            Stream(std::forward<Args>(args)...),
            m_previous_current(current(this->device()))
        {
            set_current(*static_cast<Stream*>(this));
        }

        ~StreamGuard() {
            set_current(m_previous_current);
        }

    private:
        Stream m_previous_current;
    };
}
