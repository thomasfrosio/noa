#pragma once

#include "noa/runtime/core/Span.hpp"
#include "noa/io/OS.hpp"

namespace noa::io {
    class BinaryFile {
    public:
        struct Parameters {
            /// New size, in bytes, of the opened file.
            /// - This is ignored in read-only mode (the file should exist and cannot be modified).
            /// - If writing is allowed (e.g., write or read|write mode), the file will be resized
            ///   if a positive new_size is provided. Note that if the file doesn't exist and a new size isn't
            ///   provided (new_size=-1), the stream is simply opened, no memory-mapping can be done (an error
            ///   will be thrown if memory_map=true), and size() returns -1. This behavior is intended for cases
            ///   where the stream() is to be manipulated directly.
            isize new_size{-1};

            /// Whether to memory map the file after opening it.
            /// This allows accessing the file as an array of bytes, using the as_bytes function.
            bool memory_map{false};

            /// Create a private copy-on-write mapping. Updates to the mapping are not visible to
            /// other processes mapping the same file, and are not carried through to the underlying file.
            /// This is ignored if memory_map is false.
            bool keep_private{false};
        };
        static constexpr Parameters DEFAULT_PARAMS = Parameters{-1, false, false}; // required for default value in ctor

    public: // RAII
        BinaryFile() = default;

        /// Opens the file and memory map the entire file.
        BinaryFile(const Path& path, Open mode, Parameters parameters = DEFAULT_PARAMS) {
            open(path, mode, parameters);
        }

        BinaryFile(const BinaryFile& other) = delete;
        BinaryFile& operator=(const BinaryFile& other) = delete;

        BinaryFile(BinaryFile&& other) noexcept :
            m_path{std::move(other.m_path)},
            m_file{std::exchange(other.m_file, nullptr)},
            m_data{std::exchange(other.m_data, nullptr)},
            m_size{other.m_size},
            m_open{other.m_open} {}

        BinaryFile& operator=(BinaryFile&& other) noexcept {
            std::swap(m_path, other.m_path);
            std::swap(m_file, other.m_file);
            std::swap(m_data, other.m_data);
            std::swap(m_size, other.m_size);
            std::swap(m_open, other.m_open);
            return *this;
        }

        ~BinaryFile() noexcept(false) {
            try {
                close();
            } catch (...) {
                if (not std::uncaught_exceptions()) {
                    std::rethrow_exception(std::current_exception());
                }
            }
        }

    public:
        /// Opens and memory maps the file.
        /// \param path         File path.
        /// \param mode         Open mode. Append is not supported, but the file can be resized in read|write mode.
        /// \param parameters   File and mapping parameters.
        void open(const Path& path, Open mode, Parameters parameters = DEFAULT_PARAMS);
        void close();

        [[nodiscard]] auto is_open() const -> bool { return m_file != nullptr; }
        [[nodiscard]] auto path() const -> const Path& { return m_path; }

        /// Returns the file size.
        /// \warning This is the size when opening the file, after resizing.
        ///          If the file size was changed after that, the returned value will not be correct.
        [[nodiscard]] auto ssize() const -> isize { return m_size; }
        [[nodiscard]] auto size() const -> usize { return static_cast<usize>(m_size); }

        /// Retrieves the memory-mapped range.
        /// This function requires the file to be memory-mapped.
        /// Accesses must be compatible with the open-mode.
        /// \warning Closing the file invalidates this range.
        [[nodiscard]] auto as_bytes() const -> SpanContiguous<std::byte, 1> {
            check(is_open() and m_data != nullptr, "The file is not open or not memory mapped");
            return {static_cast<std::byte*>(m_data), m_size};
        }

        /// Retrieves the file stream.
        /// \warning The stream should not be closed. Any other access, e.g., std::fread/fwrite, if fine.
        [[nodiscard]] auto stream() const -> std::FILE* { return m_file; }

        /// Advise the kernel regarding accesses.
        void optimize_for_sequential_access(isize offset = 0, isize size = -1) const;
        void optimize_for_random_access(isize offset = 0, isize size = -1) const;
        void optimize_for_no_access(isize offset = 0, isize size = -1) const;
        void optimize_for_normal_access(isize offset = 0, isize size = -1) const;

    private:
        Path m_path{};
        std::FILE* m_file{};
        void* m_data{};
        isize m_size{};
        Open m_open{};
    };
}
