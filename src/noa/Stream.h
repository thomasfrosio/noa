#pragma once

#include "noa/common/Definitions.h"
#include "noa/Device.h"

#include "noa/cpu/Stream.h"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/Backend.h"
#else
namespace noa::cuda {
    class Stream {};
}
#endif

namespace noa::gpu {
    using Stream = noa::cuda::Stream;
}

namespace noa {
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
        /// Stream mode. DEFAULT refers to the NULL stream. ASYNC creates an actual asynchronous queue.
        /// On the CPU, ASYNC launches a new thread which waits to execute work. On the GPU, ASYNC launches
        /// a new "concurrent" stream which is not implicitly synchronized with the NULL stream.
        enum Mode {
            DEFAULT,
            ASYNC
        };

        template<typename T>
        static constexpr bool is_valid_stream_v =
                std::bool_constant<std::is_same_v<T, cpu::Stream> ||
                                   std::is_same_v<T, cuda::Stream>>::value;

    public:
        /// Retrieves the current stream of a given device.
        /// \note If the device has no current stream, a new "default" stream is created.
        static Stream& current(Device device);

        /// Sets the stream as the current stream for its device.
        /// \note If the previous current stream was created by current(), any reference of it becomes invalidated.
        ///       However, since streams are reference counted, if the reference was copied into a new instance,
        ///       this instance will still be valid.
        static void current(Stream& stream);

    public:
        /// Creates a new stream on the given device.
        explicit Stream(Device device, Mode mode = Stream::ASYNC);

        /// Encapsulates (i.e. move) the given stream.
        template<typename T, typename = std::enable_if_t<is_valid_stream_v<T>>>
        explicit Stream(T&& stream);

    public:
        /// Blocks until the stream has completed all operations.
        /// \note This function may also return error codes from previous, asynchronous launches.
        void synchronize();

        /// Whether or not the stream is busy.
        /// \note This function may also return error codes from previous, asynchronous launches.
        bool busy();

        /// Gets the underlying device.
        [[nodiscard]] Device device() const noexcept;

        /// Gets the underlying stream, assuming it is a CPU stream (i.e. device is CPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] cpu::Stream& cpu();

        /// Gets the underlying stream, assuming it is a GPU stream (i.e. device is GPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] gpu::Stream& gpu();

        /// Gets the underlying stream, assuming it is a CUDA stream (i.e. device is a CUDA-capable GPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] cuda::Stream& cuda();

    private: // Type erasure
        struct StreamConcept {
            virtual ~StreamConcept() = default;
            virtual const void* addr() const noexcept = 0;
            virtual void* addr() noexcept = 0;
            virtual void synchronize() = 0;
            virtual bool busy() = 0;
        };

        template<typename T>
        struct StreamModel : public StreamConcept {
            T stream;
            explicit StreamModel(T value) noexcept : stream(std::move(value)) {};
            const void* addr() const noexcept override;
            void* addr() noexcept override;
            void synchronize() override;
            bool busy() override;
        };

    private: // Small buffer optimization
        template<size_t N, size_t A>
        class alignas(A) StreamStorage {
        public:
            StreamStorage() = default;
            StreamStorage(const StreamStorage<N, A>& src);
            StreamStorage(StreamStorage<N, A>&& src) noexcept;
            StreamStorage& operator=(const StreamStorage<N, A>& src);
            StreamStorage& operator=(StreamStorage<N, A>&& src) noexcept;
            ~StreamStorage();

        public:
            StreamConcept* stream() noexcept;
            const StreamConcept* stream() const noexcept;
            void clear() noexcept;

            template<typename T, typename... Args>
            void emplace(Args&& ... args);

        private:
            std::byte storage[N]{};
            bool is_allocated{};
        };

    private:
        static constexpr size_t BUFFER_SIZE =
                noa::math::max(sizeof(StreamModel<cpu::Stream>), sizeof(StreamModel<cuda::Stream>));
        static constexpr size_t BUFFER_ALIGN =
                noa::math::max(alignof(StreamModel<cpu::Stream>), alignof(StreamModel<cuda::Stream>));
        StreamStorage<BUFFER_SIZE, BUFFER_ALIGN> m_storage{};
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
            Stream::current(*static_cast<Stream*>(this));
        }

        ~StreamGuard() {
            Stream::current(m_previous_current);
        }

    private:
        Stream m_previous_current;
    };
}

#define NOA_UNIFIED_STREAM_
#include "noa/details/Stream.inl"
#undef NOA_UNIFIED_STREAM_
