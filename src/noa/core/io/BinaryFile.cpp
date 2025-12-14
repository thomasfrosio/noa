/// TODO This is Linux-only but support for Windows could be added in the future.
#include "noa/core/Config.hpp"
#include "noa/core/io/BinaryFile.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#ifdef NOA_PLATFORM_APPLE
#   include <unistd.h>
#endif

namespace {
    using namespace noa::types;

    void optimize_for_access_(isize offset, isize size, void* file, isize file_size, i32 flag) {
        if (file == nullptr or file_size == 0)
            return;

        offset = noa::clamp(offset, 0, file_size - 1);
        if (size < 0)
            size = file_size;
        size = noa::clamp(size, 0, file_size - offset);
        noa::check(::madvise(static_cast<std::byte*>(file) + offset, static_cast<usize>(size), flag) != -1,
                   "Failed to madvise. {}", std::strerror(errno));
    }
}

namespace noa::io {
    void BinaryFile::open(const Path& path, Open mode, Parameters parameters) {
        close();
        check(mode.is_valid() and not mode.append, "Invalid open mode {} (append is not supported)", mode);
        m_path = path;
        expand_user(m_path);
        m_open = mode;

        const char* oflags{};
        if (m_open.write) {
            // read|write -> rb+
            // read|write|truncate -> wb+
            // write(|truncate) -> wb
            const bool overwrite = m_open.truncate or not m_open.read;
            oflags = overwrite ? "wb+" : "rb+"; // mmap seems to require reading, so use wb+ instead of wb
            try {
                if (is_file(m_path) and mode.backup)
                    backup(m_path, overwrite);
                else if (overwrite)
                    mkdir(m_path.parent_path());
            } catch (...) {
                panic("File: {}. {}. Could not open the file because of an OS failure", m_path, mode);
            }
        } else if (m_open.read) {
            oflags = "rb";
        }

        for (i32 it{}; it < 3; ++it) {
            m_file = std::fopen(m_path.c_str(), oflags);
            if (m_file != nullptr)
                break;
            check(it < 2, "Failed to open file {} with error: {}", m_path, std::strerror(errno));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        int fd = ::fileno(m_file);
        check(fd != -1, "Failed to retrieve the file descriptor, {}", std::strerror(errno));

        if (not m_open.write or (m_open.read and not m_open.truncate and parameters.new_size < 0 and is_file(m_path))) {
            // Fetch the file size if:
            // - read-only
            // - read|write + file exists + no resize.
            struct stat s{};
            check(::fstat(fd, &s) != -1, "Failed to stat file");
            m_size = s.st_size;
        } else {
            // Resize the file if a new size is provided and:
            // - read|write
            // - read|write|truncate
            // - write(|truncate)
            check(parameters.new_size <= 0 or ::ftruncate(fd, parameters.new_size) != -1,
                  "Failed to resize the file. {}", std::strerror(errno));
            m_size = parameters.new_size;
        }

        if (not parameters.memory_map)
            return;

        // We are about to mmap, so the file should have a valid size by now.
        check(m_size > 0, "Memory mapping isn't allowed on files without a valid size");

        // Align the size to the next page.
        int pagesize = ::getpagesize();
        isize msize = m_size;
        msize += pagesize - (msize % pagesize);

        int mprot{};
        if (m_open.read)
            mprot |= PROT_READ;
        if (m_open.write)
            mprot |= PROT_WRITE;
        int mflags = parameters.keep_private or not mode.write ? MAP_PRIVATE : MAP_SHARED;

        // Memory map the entire file.
        m_data = ::mmap(nullptr, static_cast<usize>(msize), mprot, mflags, fd, 0);
        noa::check(m_data != reinterpret_cast<void*>(-1), "Failed to mmap: {}", std::strerror(errno));
    }

    void BinaryFile::close() {
        if (m_file == nullptr)
            return;

        if (m_data) {
            int pagesize = ::getpagesize();
            isize psize = m_size;
            psize += pagesize - (psize % pagesize);
            check(::munmap(m_data, static_cast<usize>(psize)) != -1, "Failed to munmap: {}", std::strerror(errno));
            m_data = nullptr;
        }

        check(std::fclose(m_file) == 0, "Failed to close file");
        m_file = nullptr;
    }

    void BinaryFile::optimize_for_sequential_access(isize offset, isize size) const {
        optimize_for_access_(offset, size, m_data, m_size, MADV_SEQUENTIAL);
    }
    void BinaryFile::optimize_for_random_access(isize offset, isize size) const {
        optimize_for_access_(offset, size, m_data, m_size, MADV_RANDOM);
    }
    void BinaryFile::optimize_for_no_access(isize offset, isize size) const {
        optimize_for_access_(offset, size, m_data, m_size, MADV_DONTNEED);
    }
    void BinaryFile::optimize_for_normal_access(isize offset, isize size) const {
        optimize_for_access_(offset, size, m_data, m_size, MADV_NORMAL);
    }
}
