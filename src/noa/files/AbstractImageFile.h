/**
 * @file AbstractImageFile.h
 * @brief AbstractImageFile class. The handle to specialized image files.
 * @author Thomas - ffyr2w
 * @date 19 Dec 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/IO.h"
#include "noa/util/OS.h"
#include "noa/util/Vectors.h"


namespace Noa {
    NOA_API class AbstractImageFile {
    protected:
        fs::path m_path{};
        errno_t m_state{Errno::good};

    public:
        AbstractImageFile() = default;
        virtual ~AbstractImageFile() = default;

        template<typename T, typename = std::enable_if_t<std::is_convertible_v<T, std::filesystem::path>>>
        explicit AbstractImageFile(T&& path) : m_path(std::forward<T>(path)) {}

        inline bool exists() noexcept { return !m_state && OS::existsFile(m_path, m_state); }
        inline size_t size() noexcept { return !m_state ? OS::size(m_path, m_state) : 0U; }

        [[nodiscard]] inline const fs::path* path() const noexcept { return &m_path; }
        [[nodiscard]] inline errno_t state() const { return m_state; }
        inline void resetState() { m_state = Errno::good; }

        // Below are all the functions that derived classes should override.
        //  ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
    public:
        virtual errno_t open(std::ios_base::openmode, bool) = 0;
        virtual errno_t open(const fs::path&, std::ios_base::openmode, bool) = 0;
        virtual errno_t open(fs::path&&, std::ios_base::openmode, bool) = 0;
        [[nodiscard]] virtual bool isOpen() const = 0;

        virtual errno_t close() = 0;

        [[nodiscard]] virtual explicit operator bool() const noexcept = 0;

        [[nodiscard]] virtual Int3<size_t> getShape() const = 0;
        virtual errno_t setShape(Int3<size_t>) = 0;

        [[nodiscard]] virtual Float3<float> getPixelSize() const = 0;
        virtual errno_t setPixelSize(Float3<float>) = 0;

        [[nodiscard]] virtual std::string toString(bool) const = 0;

        virtual errno_t readAll(float*) = 0;
        virtual errno_t readSlice(float*, size_t, size_t) = 0;

        virtual errno_t setDataType(IO::DataType) = 0;
        virtual errno_t writeAll(float*) = 0;
        virtual errno_t writeSlice(float*, size_t, size_t) = 0;
    };
}
