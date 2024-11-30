/// TODO This is Linux-only but support for Windows could be added in the future.
#include <sys/mman.h>
#include <sys/stat.h>

#include "noa/core/io/MemoryMappedFile.hpp"

namespace {
    using namespace noa::types;

    void optimize_for_access_(i64 offset, i64 size, void* file, i64 file_size, i32 flag) {
        if (file == nullptr or file_size == 0)
            return;

        offset = noa::clamp(offset, 0, file_size - 1);
        if (size < 0)
            size = file_size;
        size = noa::clamp(size, 0, file_size - offset);
        noa::check(madvise(file + offset, static_cast<size_t>(size), flag) != -1,
                   "Failed to madvise. {}", strerror(errno));
    }
}

namespace noa::io {
    void MemoryMappedFile::open(const Path& path, Open mode, Parameters parameters) {
        close();
        check(mode.is_valid() and not mode.append, "Invalid open mode {} (append is not supported)", mode);
        m_path = path;
        m_open = mode;

        int oflags{};
        mode_t omode{};
        if (m_open.write) {
            oflags = O_RDWR | O_CREAT; // memory mapping requires the file to be open for reading
            omode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH; // rw-rw-r--
            const bool overwrite = m_open.truncate or not m_open.read;
            const bool exists = is_file(m_path);
            try {
                if (exists and mode.backup)
                    backup(m_path, overwrite);
                else if (overwrite) {
                    oflags |= O_TRUNC;
                    mkdir(m_path.parent_path());
                }
            } catch (...) {
                panic("File: {}. {}. Could not open the file because of an OS failure", m_path, mode);
            }
        } else if (m_open.read) {
            oflags = O_RDONLY;
        }

        for (i32 it{}; it < 3; ++it) {
            m_fd = ::open(m_path.c_str(), oflags, omode);
            if (m_fd != -1)
                break;
            check(it < 2, "Failed to open file {} with error: {}", m_path, std::strerror(errno));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Resize the file.
        if (m_open.write) {
            if (m_open.read and not m_open.truncate and parameters.new_size < 0) {
                // In read|write only, if the size isn't specified, do nothing.
            } else {
                check(parameters.new_size >= 0,
                      "When creating a new file or overwriting an existing one ({}), "
                      "a valid file size should be provided, but got {}",
                      m_open, parameters.new_size);
                check(::ftruncate(m_fd, parameters.new_size) != -1,
                      "Failed to resize the file. {}", std::strerror(errno));
                m_size = parameters.new_size;
            }
        } else {
            struct stat s{};
            check(fstat(m_fd, &s) != -1, "Failed to stat file");
            m_size = s.st_size;
        }

        // Align the size to the next page.
        int pagesize = ::getpagesize();
        i64 msize = m_size;
        msize += pagesize - (msize % pagesize);

        int mprot{};
        if (m_open.read)
            mprot |= PROT_READ;
        if (m_open.write)
            mprot |= PROT_WRITE;
        int mflags = parameters.keep_private or not mode.write ? MAP_PRIVATE : MAP_SHARED;

        // Memory map the entire file.
        m_data = ::mmap(nullptr, static_cast<size_t>(msize), mprot, mflags, m_fd, 0);
        noa::check(m_data != reinterpret_cast<void*>(-1), "Failed to mmap: {}", strerror(errno));
    }

    void MemoryMappedFile::close() {
        if (m_fd == -1)
            return;

        int pagesize = ::getpagesize();
        i64 psize = m_size;
        psize += pagesize - (psize % pagesize);

        check(::munmap(m_data, static_cast<size_t>(psize)) != -1, "Failed to munmap: {}", strerror(errno));
        check(::close(m_fd) != -1, "Failed to close file {}", strerror(errno));

        m_fd = -1;
        m_data = nullptr;
    }

    void MemoryMappedFile::optimize_for_sequential_access(i64 offset, i64 size) const {
        optimize_for_access_(offset, size, m_data, m_size, MADV_SEQUENTIAL);
    }
    void MemoryMappedFile::optimize_for_random_access(i64 offset, i64 size) const {
        optimize_for_access_(offset, size, m_data, m_size, MADV_RANDOM);
    }
    void MemoryMappedFile::optimize_for_no_access(i64 offset, i64 size) const {
        optimize_for_access_(offset, size, m_data, m_size, MADV_DONTNEED);
    }
    void MemoryMappedFile::optimize_for_normal_access(i64 offset, i64 size) const {
        optimize_for_access_(offset, size, m_data, m_size, MADV_NORMAL);
    }
}
