#pragma once

#include "noa/core/io/IO.hpp"
#include "noa/core/types/Span.hpp"

namespace noa::io {
    class BinaryFile {
    public:
        struct Parameters {
            /// Size, in bytes, of the opened file. This is ignored in read-only mode.
            /// If the file already exists (read|write mode), the file is resized to this size.
            i64 new_size{-1};

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

        /// Retrieves the memory mapped range.
        /// This function requires the file to be memory mapped.
        /// Accesses must be compatible with the open-mode.
        /// \warning Closing the file invalidates this range.
        [[nodiscard]] auto as_bytes() const -> SpanContiguous<std::byte, 1> {
            check(is_open() and m_data != nullptr, "The file is not open or not memory mapped");
            return {static_cast<std::byte*>(m_data), m_size};
        }

        /// Retrieves the file stream.
        /// \warning The stream should not be closed. Any other access, e.g. std::fread/fwrite, if fine.
        [[nodiscard]] auto stream() const -> std::FILE* { return m_file; }

        /// Advise the kernel regarding accesses.
        void optimize_for_sequential_access(i64 offset = 0, i64 size = -1) const;
        void optimize_for_random_access(i64 offset = 0, i64 size = -1) const;
        void optimize_for_no_access(i64 offset = 0, i64 size = -1) const;
        void optimize_for_normal_access(i64 offset = 0, i64 size = -1) const;

        [[nodiscard]] auto ssize() const -> i64 { return m_size; }
        [[nodiscard]] auto size() const -> u64 { return static_cast<u64>(m_size); }

    private:
        Path m_path{};
        std::FILE* m_file{};
        void* m_data{};
        i64 m_size{};
        Open m_open{};
    };
}
