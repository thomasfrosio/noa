#include "noa/core/Config.hpp"
#include "noa/core/Exception.hpp"
#include "noa/core/io/IO.hpp"
#include "noa/core/io/TextFile.hpp"
#include "noa/core/io/OS.hpp"
#include "noa/core/string/Format.hpp"
#include "noa/core/string/Parse.hpp"
#include "noa/cpu/Device.hpp"

// Internal data to reset:
#include "noa/cpu/fft/Plan.hpp"

// TODO maybe use something a bit more robust like pytorch/cpuinfo?
// FIXME should this be in noa/core ?

#if defined(NOA_PLATFORM_WINDOWS)
#include <windows.h>
#include <bitset>
// From https://github.dev/ThePhD/infoware/tree/main/include/infoware
namespace {
    using namespace noa;

    cpu::DeviceMemory get_memory_info_windows() {
        cpu::DeviceMemory out{};
        MEMORYSTATUSEX mem;
        mem.dwLength = sizeof(mem);
        if (GlobalMemoryStatusEx(&mem)) {
            out.total = mem.ullAvailPhys;
            out.free = mem.ullTotalPhys;
        }
        return out;
    }

    std::string get_cpu_name_windows() {
        HKEY hkey;
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, R"(HARDWARE\DESCRIPTION\System\CentralProcessor\0)", 0, KEY_READ, &hkey))
            return {};

        char identifier[64 + 1];
        DWORD identifier_len = sizeof(identifier);
        LPBYTE lpdata = static_cast<LPBYTE>(static_cast<void*>(&identifier[0]));
        if (RegQueryValueExA(hkey, "ProcessorNameString", nullptr, nullptr, lpdata, &identifier_len))
            return {};

        return identifier;
    }

    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> getCPUInfoBuffer() {
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer;

        DWORD byte_count = 0;
        GetLogicalProcessorInformation(nullptr, &byte_count);
        buffer.resize(byte_count / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        GetLogicalProcessorInformation(buffer.data(), &byte_count);

        return buffer;
    }

    cpu::DeviceCore get_cpu_core_count_windows() {
        cpu::DeviceCore out{};
        for (auto&& info: getCPUInfoBuffer()) {
            if (info.Relationship == RelationProcessorCore) {
                ++out.physical;
                // A hyperthreaded core supplies more than one logical processor.
                out.logical += static_cast<std::uint32_t>(std::bitset<sizeof(ULONG_PTR) * 8>(
                        static_cast<std::uintptr_t>(info.ProcessorMask)).count());
            }
        }
        return out;
    }

    cpu::DeviceCache get_cpu_cache_windows(int level) {
        for (auto&& info: getCPUInfoBuffer()) {
            if (info.Relationship == RelationCache) {
                // Cache data is in ptr->Cache, one CACHE_DESCRIPTOR structure for each cache.
                if (info.Cache.Level == level)
                    return {info.Cache.Size, info.Cache.LineSize};
            }
        }
        return {};
    }
}
#elif defined(NOA_PLATFORM_LINUX)
#include <unistd.h>
namespace {
    using namespace noa;

    size_t parse_size_from_line(const std::string& line) {
        const size_t colon_id = line.find_first_of(':');
        std::string value{line.c_str() + colon_id + 1};
        return ns::parse<size_t>(value);
    }

    cpu::DeviceMemory get_memory_info_linux() {
        cpu::DeviceMemory ret{};

        std::string line;
        io::InputTextFile mem_info("/proc/meminfo", io::OpenMode{.read=true});
        while (mem_info.get_line(line)) {
            if (ns::starts_with(line, "MemTotal"))
                ret.total = parse_size_from_line(line) * 1024; // in bytes
            else if (ns::starts_with(line, "MemAvailable"))
                ret.free = parse_size_from_line(line) * 1024; // in bytes
        }
        check(not mem_info.bad(), "Error while reading {}", mem_info.path());
        return ret;
    }

    std::string get_cpu_name_linux() {
        io::InputTextFile cpu_info("/proc/cpuinfo", io::OpenMode{.read=true});
        std::string line;
        while (cpu_info.get_line(line)) {
            if (ns::starts_with(line, "model name")) {
                const size_t colon_id = line.find_first_of(':');
                const size_t nonspace_id = line.find_first_not_of(" \t", colon_id + 1);
                return line.c_str() + nonspace_id; // assume right trimmed
            }
        }
        panic("Could not retrieve CPU name from {}", cpu_info.path());
    }

    cpu::DeviceCore get_cpu_core_count_linux() {
        cpu::DeviceCore out{};
        bool got_logical{};
        bool got_physical{};

        io::InputTextFile cpu_info("/proc/cpuinfo", io::OpenMode{.read=true});
        std::string line;
        while (cpu_info.get_line(line)) {
            if (!got_logical && ns::starts_with(line, "siblings")) {
                out.logical = parse_size_from_line(line);
                got_logical = true;
            } else if (!got_physical && ns::starts_with(line, "cpu cores")) {
                out.physical = parse_size_from_line(line);
                got_physical = true;
            }
            if (got_logical && got_physical)
                break;
        }
        check(not cpu_info.bad() and got_logical and got_physical,
              "Could not retrieve CPU name from {}", cpu_info.path());
        return out;
    }

    cpu::DeviceCache get_cpu_cache_linux(int level) {
        cpu::DeviceCache out{};
        io::InputTextFile cache_info;
        const Path prefix = fmt::format("/sys/devices/system/cpu/cpu0/cache/index{}", level);
        // FIXME index1 is the instruction L1 cache on my machine...

        Path cache_size_path = prefix / "size";
        if (io::is_file(cache_size_path)) {
            cache_info.open(cache_size_path, io::OpenMode{.read=true});
            char suffix;
            cache_info.fstream() >> out.size >> suffix;
            switch (suffix) {
                case 'G':
                    out.size *= 1024;
                    [[fallthrough]];
                case 'M':
                    out.size *= 1024;
                    [[fallthrough]];
                case 'K':
                    out.size *= 1024;
            }
        }

        cache_size_path = prefix / "coherency_line_size";
        if (io::is_file(cache_size_path)) {
            cache_info.open(cache_size_path, io::OpenMode{.read=true});
            cache_info.fstream() >> out.line_size;
        }
        return out;
    }
}
#endif

namespace noa::cpu {
    DeviceMemory Device::memory() {
        #if defined(NOA_PLATFORM_LINUX)
        return get_memory_info_linux();
        #elif defined(NOA_PLATFORM_WINDOWS)
        return get_memory_info_windows();
        #else
        return {};
        #endif
    }

    DeviceCore Device::cores() {
        #if defined(NOA_PLATFORM_LINUX)
        return get_cpu_core_count_linux();
        #elif defined(NOA_PLATFORM_WINDOWS)
        return get_cpu_core_count_windows();
        #else
        return {};
        #endif
    }

    DeviceCache Device::cache(int level) {
        #if defined(NOA_PLATFORM_LINUX)
        return get_cpu_cache_linux(level);
        #elif defined(NOA_PLATFORM_WINDOWS)
        return get_cpu_cache_windows(level);
        #else
        return {};
        #endif
    }

    std::string Device::name() {
        #if defined(NOA_PLATFORM_LINUX)
        return std::string(ns::trim(get_cpu_name_linux()));
        #elif defined(NOA_PLATFORM_WINDOWS)
        return std::string{ns::trim(get_cpu_name_windows())};
        #else
        return {};
        #endif
    }

    std::string Device::summary() {
        const std::string name = Device::name();
        const DeviceCore core_count = Device::cores();
        const DeviceCache cache1 = Device::cache(1);
        const DeviceCache cache2 = Device::cache(2);
        const DeviceCache cache3 = Device::cache(3);
        const DeviceMemory sysmem = Device::memory();

        return fmt::format("cpu:\n"
                           "    Name: {}\n"
                           "    Cores: {}t, {}c\n"
                           "    L1 cache: {}KB, line:{}\n"
                           "    L2 cache: {}KB, line:{}\n"
                           "    L3 cache: {}KB, line:{}\n"
                           "    Memory: {}MB / {}MB\n"
                           "    Endianness: {}\n",
                           name, core_count.logical, core_count.physical,
                           cache1.size / 1024, cache1.line_size,
                           cache2.size / 1024, cache2.line_size,
                           cache3.size / 1024, cache3.line_size,
                           (sysmem.total - sysmem.free) / 1048576, sysmem.total / 1048576,
                           io::is_big_endian() ? "big" : "little");
    }

    void Device::reset() {
        // Reset all internal data created and managed automatically by the CPU backend:
        noa::cpu::fft::clear_caches();
    }
}
