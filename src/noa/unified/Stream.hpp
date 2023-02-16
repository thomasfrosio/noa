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
        static void set_current(Stream& stream);

    public:
        /// Creates a new stream on the given device.
        explicit Stream(Device device, StreamMode mode = StreamMode::ASYNC) : m_device(device) {
            if (m_device.is_cpu()) {
                const auto cpu_mode = mode == StreamMode::ASYNC ? cpu::StreamMode::ASYNC : cpu::StreamMode::DEFAULT;
                m_storage.emplace<StreamModel<cpu::Stream>>(cpu::Stream(cpu_mode));
            } else {
                #ifdef NOA_ENABLE_CUDA
                cuda::Device cuda_device(m_device.id());
                const auto cuda_mode = mode == StreamMode::ASYNC ? cuda::StreamMode::ASYNC : cuda::StreamMode::DEFAULT;
                m_storage.emplace<StreamModel<cuda::Stream>>(cuda::Stream(cuda_device, cuda_mode));
                #else
                NOA_THROW("No GPU backend detected");
                #endif
            }
        }

        /// Encapsulates (i.e. move) the given stream.
        template<typename T, typename = std::enable_if_t<is_valid_stream_v<T>>>
        explicit Stream(T&& stream) {
            m_storage.emplace<StreamModel<T>>(std::forward<T>(stream));
        }

    public:
        /// Blocks until the stream has completed all operations.
        /// \note This function may also return error codes from previous, asynchronous launches.
        void synchronize() {
            m_storage.stream()->synchronize();
        }

        /// Whether or not the stream is busy.
        /// \note This function may also return error codes from previous, asynchronous launches.
        bool is_busy() {
            return m_storage.stream()->is_busy();
        }

        /// Gets the underlying device.
        [[nodiscard]] Device device() const noexcept {
            return m_device;
        }

        /// Gets the underlying stream, assuming it is a CPU stream (i.e. device is CPU).
        /// Otherwise, throws an exception.
        [[nodiscard]] cpu::Stream& cpu() {
            if (!m_device.is_cpu())
                NOA_THROW("The stream is not a CPU stream");

            StreamConcept* tmp = m_storage.stream();
            auto* d = static_cast<StreamModel<cpu::Stream>*>(tmp); // dynamic_cast?
            return *reinterpret_cast<cpu::Stream*>(d->addr());
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
            if (!m_device.is_gpu())
                NOA_THROW("The stream is not a GPU stream");

            #ifdef NOA_ENABLE_CUDA
            StreamConcept* tmp = m_storage.stream();
            auto* d = static_cast<StreamModel<cuda::Stream>*>(tmp); // dynamic_cast?
            return *reinterpret_cast<cuda::Stream*>(d->addr());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }

    private: // Type erasure
        struct StreamConcept {
            virtual ~StreamConcept() = default;
            virtual const void* addr() const noexcept = 0;
            virtual void* addr() noexcept = 0;
            virtual void synchronize() = 0;
            virtual bool is_busy() = 0;
        };

        template<typename T>
        struct StreamModel : public StreamConcept {
            T stream;
            explicit StreamModel(T value) noexcept : stream(std::move(value)) {};
            [[nodiscard]] const void* addr() const noexcept override { return &stream; }
            void* addr() noexcept override { return &stream; }

            void synchronize() override {
                if constexpr (std::is_same_v<T, cpu::Stream>)
                    stream.synchronize();
                else {
                    #ifdef NOA_ENABLE_CUDA
                    stream.synchronize();
                    #endif
                }
            }

            bool is_busy() override  {
                if constexpr (std::is_same_v<T, cpu::Stream>)
                    return stream.is_busy();
                else {
                    #ifdef NOA_ENABLE_CUDA
                    return stream.is_busy();
                    #endif
                }
                return false;
            }
        };

    private: // Small buffer optimization
        template<size_t N, size_t A>
        class alignas(A) StreamStorage {
        public:
            StreamStorage() = default;

            StreamStorage(const StreamStorage<N, A>& src) {
                if (!src.is_allocated)
                    return;

                // FIXME Is there a better way of doing this?
                const StreamConcept* tmp = src.stream();
                if (const auto* ptr = dynamic_cast<const StreamModel<cpu::Stream>*>(tmp)) {
                    const cpu::Stream& src_stream = *reinterpret_cast<const cpu::Stream*>(ptr->addr());
                    this->emplace<StreamModel<cpu::Stream>>(src_stream); // copy
                    return;
                }
                if (const auto* ptr = dynamic_cast<const StreamModel<cuda::Stream>*>(tmp)) {
                    const cuda::Stream& src_stream = *reinterpret_cast<const cuda::Stream*>(ptr->addr());
                    this->emplace<StreamModel<cuda::Stream>>(src_stream); // copy
                    return;
                }
            }

            StreamStorage(StreamStorage<N, A>&& src) noexcept {
                if (src.is_allocated) {
                    std::copy(src.storage, src.storage + N, storage);
                    src.is_allocated = false;
                    is_allocated = true;
                }
            }

            StreamStorage& operator=(const StreamStorage<N, A>& src) {
                if (this != &src)
                    *this = StreamStorage(src); // move
                return *this;
            }

            StreamStorage& operator=(StreamStorage<N, A>&& src) noexcept {
                if (this != &src) {
                    clear();
                    if (src.is_allocated) {
                        std::copy(src.storage, src.storage + N, storage);
                        src.is_allocated = false;
                        is_allocated = true;
                    }
                }
                return *this;
            }

            ~StreamStorage() {
                clear();
            }

        public:
            StreamConcept* stream() noexcept {
                return reinterpret_cast<StreamConcept*>(&storage);
            }

            [[nodiscard]] const StreamConcept* stream() const noexcept {
                return reinterpret_cast<const StreamConcept*>(&storage);
            }

            void clear() noexcept {
                if (is_allocated) {
                    stream()->~StreamConcept();
                    is_allocated = false;
                }
            }

            template<typename T, typename... Args>
            void emplace(Args&& ... args) {
                static_assert(sizeof(T) <= N);
                static_assert(alignof(T) <= A);
                clear();
                new(this->stream()) T(std::forward<Args>(args)...);
                is_allocated = true;
            }

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
            Stream::set_current(*static_cast<Stream*>(this));
        }

        ~StreamGuard() {
            Stream::set_current(m_previous_current);
        }

    private:
        Stream m_previous_current;
    };
}
