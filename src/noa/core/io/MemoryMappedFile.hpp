#pragma once

#include "noa/core/io/IO.hpp"
#include "noa/core/types/Span.hpp"

namespace noa::io {
   /// Memory mapped file.
    class MemoryMappedFile {
    public:
        struct Parameters {
            /// Size, in bytes, of the opened file. This is ignored in read-only mode.
            /// If the file already exists (read|write mode), the file is resized to this size.
            i64 new_size{-1};

            /// Create a private copy-on-write mapping. Updates to the mapping are not visible to
            /// other processes mapping the same file, and are not carried through to the underlying file.
            bool keep_private{false};
        };

    public: // RAII
        MemoryMappedFile() = default;

        /// Opens the file and memory map the entire file.
        MemoryMappedFile(const Path& path, Open mode, Parameters parameters = {.new_size = -1, .keep_private = false}) {
            open(path, mode, parameters);
        }

        MemoryMappedFile(const MemoryMappedFile& other) = delete;
        MemoryMappedFile& operator=(const MemoryMappedFile& other) = delete;

        MemoryMappedFile(MemoryMappedFile&& other) noexcept :
            m_path{std::move(other.m_path)},
            m_data{std::exchange(other.m_data, nullptr)},
            m_size{other.m_size},
            m_fd{std::exchange(other.m_fd, -1)},
            m_open{other.m_open} {}

        MemoryMappedFile& operator=(MemoryMappedFile&& other) noexcept {
            std::swap(m_path, other.m_path);
            std::swap(m_data, other.m_data);
            std::swap(m_size, other.m_size);
            std::swap(m_fd, other.m_fd);
            std::swap(m_open, other.m_open);
            return *this;
        }

        ~MemoryMappedFile() noexcept(false) {
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
        void open(const Path& path, Open mode, Parameters parameters = {.new_size = -1, .keep_private = false});

        void close();

        [[nodiscard]] auto is_open() const -> bool { return m_fd != -1; }
        [[nodiscard]] auto path() const -> const Path& { return m_path; }

        /// Retrieve the memory mapped range. Accesses must be compatible with the open-mode.
        /// \warning Closing the file invalidates this range.
        [[nodiscard]] auto as_bytes() const -> SpanContiguous<std::byte, 1> {
            check(is_open());
            return {static_cast<std::byte*>(m_data), m_size};
        }

        /// Advise the kernel regarding accesses.
        void optimize_for_sequential_access(i64 offset = 0, i64 size = -1) const;
        void optimize_for_random_access(i64 offset = 0, i64 size = -1) const;
        void optimize_for_no_access(i64 offset = 0, i64 size = -1) const;
        void optimize_for_normal_access(i64 offset = 0, i64 size = -1) const;

        [[nodiscard]] auto ssize() const -> i64 { return m_size; }
        [[nodiscard]] auto size() const -> u64 { return static_cast<u64>(m_size); }

    private:
        Path m_path{};
        void* m_data{};
        i64 m_size{};
        i32 m_fd{-1};
        Open m_open{};
    };
}
