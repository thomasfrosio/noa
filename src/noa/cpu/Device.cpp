#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/io/IO.h"
#include "noa/common/io/TextFile.h"
#include "noa/common/OS.h"
#include "noa/common/string/Format.h"
#include "noa/common/string/Parse.h"
#include "noa/cpu/Device.h"

// Internal data to reset:
#include "noa/cpu/fft/Plan.h"

// TODO maybe use something a bit more robust like pytorch/cpuinfo?
// FIXME should this be in noa/common ?

#if defined(NOA_PLATFORM_WINDOWS)
#include <windows.h>
#include <bitset>
using namespace noa;
// From https://github.dev/ThePhD/infoware/tree/main/include/infoware
namespace {
    cpu::DeviceMemory getMemoryInfoWindows() {
        cpu::DeviceMemory out{};
        MEMORYSTATUSEX mem;
        mem.dwLength = sizeof(mem);
        if (GlobalMemoryStatusEx(&mem)) {
            out.total = mem.ullAvailPhys;
            out.free = mem.ullTotalPhys;
        }
        return out;
    }

    std::string getCPUNameWindows() {
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

    cpu::DeviceCore getCPUCoreCountWindows() {
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

    cpu::DeviceCache getCPUCacheWindows(int level) {
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

    size_t parseSizeFromLine(const std::string& line) {
        const size_t colon_id = line.find_first_of(':');
        std::string value{line.c_str() + colon_id + 1};
        return string::toInt<size_t>(value);
    }

    cpu::DeviceMemory getMemoryInfoLinux() {
        cpu::DeviceMemory ret{};

        std::string line;
        io::TextFile<std::ifstream> mem_info("/proc/meminfo", io::READ);
        while (mem_info.getLine(line)) {
            if (string::startsWith(line, "MemTotal"))
                ret.total = parseSizeFromLine(line) * 1024; // in bytes
            else if (string::startsWith(line, "MemAvailable"))
                ret.free = parseSizeFromLine(line) * 1024; // in bytes
        }
        if (mem_info.bad())
            NOA_THROW("Error while reading {}", mem_info.path());
        return ret;
    }

    std::string getCPUNameLinux() {
        io::TextFile<std::ifstream> cpu_info("/proc/cpuinfo", io::READ);
        std::string line;
        while (cpu_info.getLine(line)) {
            if (string::startsWith(line, "model name")) {
                const size_t colon_id = line.find_first_of(':');
                const size_t nonspace_id = line.find_first_not_of(" \t", colon_id + 1);
                return line.c_str() + nonspace_id; // assume right trimmed
            }
        }
        NOA_THROW("Could not retrieve CPU name from {}", cpu_info.path());
    }

    cpu::DeviceCore getCPUCoreCountLinux() {
        cpu::DeviceCore out{};
        bool got_logical{}, got_physical{};

        io::TextFile<std::ifstream> cpu_info("/proc/cpuinfo", io::READ);
        std::string line;
        while (cpu_info.getLine(line)) {
            if (!got_logical && string::startsWith(line, "siblings")) {
                out.logical = parseSizeFromLine(line);
                got_logical = true;
            } else if (!got_physical && string::startsWith(line, "cpu cores")) {
                out.physical = parseSizeFromLine(line);
                got_physical = true;
            }
            if (got_logical && got_physical)
                break;
        }
        if (cpu_info.bad() || !got_logical || !got_physical)
            NOA_THROW("Could not retrieve CPU name from {}", cpu_info.path());

        return out;
    }

    cpu::DeviceCache getCPUCacheLinux(int level) {
        cpu::DeviceCache out{};
        io::TextFile<std::ifstream> cache_info;
        const path_t prefix = string::format("/sys/devices/system/cpu/cpu0/cache/index{}", level);
        // FIXME index1 is the instruction L1 cache on my machine...

        path_t cache_size_path = prefix / "size";
        if (os::existsFile(cache_size_path)) {
            cache_info.open(cache_size_path, io::READ);
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
        if (os::existsFile(cache_size_path)) {
            cache_info.open(cache_size_path, io::READ);
            cache_info.fstream() >> out.line_size;
        }

        return out;
    }
}
#endif

namespace noa::cpu {
    DeviceMemory Device::memory() {
        #if defined(NOA_PLATFORM_LINUX)
        return getMemoryInfoLinux();
        #elif defined(NOA_PLATFORM_WINDOWS)
        return getMemoryInfoWindows();
        #else
        return {};
        #endif
    }

    DeviceCore Device::cores() {
        #if defined(NOA_PLATFORM_LINUX)
        return getCPUCoreCountLinux();
        #elif defined(NOA_PLATFORM_WINDOWS)
        return getCPUCoreCountWindows();
        #else
        return {};
        #endif
    }

    DeviceCache Device::cache(int level) {
        #if defined(NOA_PLATFORM_LINUX)
        return getCPUCacheLinux(level);
        #elif defined(NOA_PLATFORM_WINDOWS)
        return getCPUCacheWindows(level);
        #else
        return {};
        #endif
    }

    std::string Device::name() {
        #if defined(NOA_PLATFORM_LINUX)
        return std::string(string::trim(getCPUNameLinux()));
        #elif defined(NOA_PLATFORM_WINDOWS)
        return std::string{string::trim(getCPUNameWindows())};
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

        return string::format("cpu:\n"
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
                              io::isBigEndian() ? "big" : "little");
    }

    void Device::reset() {
        // Reset all internal data created and managed automatically by the CPU backend:
        fft::Plan<float>::cleanup();
        fft::Plan<double>::cleanup();
    }
}
