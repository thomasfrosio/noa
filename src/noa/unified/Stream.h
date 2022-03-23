#pragma once

#include "noa/common/Definitions.h"
#include "noa/unified/Device.h"

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
    class Stream {
    public:
        enum Mode {
            DEFAULT,
            ASYNC
        };

        template<typename T>
        static constexpr bool is_valid_stream_v =
                std::bool_constant<std::is_same_v<T, cpu::Stream> ||
                                   std::is_same_v<T, cuda::Stream>>::value;

    private: // Type erasure
        struct StreamConcept {
            virtual ~StreamConcept() = default;
            virtual void* addr() noexcept = 0;
            virtual void synchronize() = 0;
            virtual bool busy() = 0;
        };

        template<typename T>
        struct StreamModel : public StreamConcept {
            T stream;
            explicit StreamModel(T&& value) : stream(std::forward<T>(value)) {};
            void* addr() noexcept override;
            void synchronize() override;
            bool busy() override;
        };

    private: // Small buffer optimization
        template<size_t N, size_t A>
        class alignas(A) StreamStorage {
        public:
            StreamStorage() = default;
            StreamStorage(StreamStorage<N, A>&& src) noexcept ;
            StreamStorage& operator=(StreamStorage<N, A>&& src) noexcept ;
            ~StreamStorage();

        public:
            StreamConcept* stream() noexcept;
            void clear() noexcept;

            template<typename T, typename... Args>
            void emplace(Args&& ... args);

        private:
            std::byte storage[N]{};
            bool is_allocated{};
        };

    public:
        /// Retrieves the current stream of a given device.
        /// \note If the device has no current stream, a new "default" stream is created and its reference is returned.
        static Stream& current(Device device = Device::current());

        /// Sets the stream as the current stream for its device.
        /// \note If the previous current stream was created by current(), any reference of it becomes invalidated.
        static void current(Stream& stream);

    public:
        /// Creates a new stream on a given device.
        explicit Stream(Device device, Mode mode = Stream::ASYNC);

        /// Creates a new stream on the current device.
        explicit Stream(Mode mode = Stream::ASYNC);

        /// Encapsulates (i.e. move) the given stream.
        template<typename T, typename = std::enable_if_t<is_valid_stream_v<T>>>
        explicit Stream(T&& stream);

    public: // Move-only
        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;

        Stream(Stream&& to_copy) noexcept = default;
        Stream& operator=(Stream&&) noexcept = default;

    public:
        /// Blocks until the stream has completed all operations.
        /// \note This function may also return error codes from previous, asynchronous launches.
        void synchronize();

        /// Whether or not the stream is busy.
        /// \note This function may also return error codes from previous, asynchronous launches.
        bool busy();

        /// Gets the underlying device.
        [[nodiscard]] Device device() const noexcept;

        /// Gets the underlying stream, assuming it is a CPU stream. Otherwise, throws an exception.
        [[nodiscard]] cpu::Stream& cpu();

        /// Gets the underlying stream, assuming it is a GPU stream. Otherwise, throws an exception.
        [[nodiscard]] gpu::Stream& gpu();

        /// Gets the underlying stream, assuming it is a CUDA stream. Otherwise, throws an exception.
        [[nodiscard]] cuda::Stream& cuda();

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
                  m_previous_current(&Stream::current()) {
            Stream::current(*static_cast<Stream*>(this));
        }

        ~StreamGuard() {
            Stream::current(*m_previous_current);
        }

    private:
        Stream* m_previous_current;
    };
}


#define NOA_UNIFIED_STREAM_
#include "noa/unified/Stream.inl"
#undef NOA_UNIFIED_STREAM_
