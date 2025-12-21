#include "noa/base/Config.hpp"
#include "noa/base/Error.hpp"
#include "noa/base/Strings.hpp"
#include "noa/base/SafeCast.hpp"
#include "noa/runtime/cpu/Device.hpp"

// TODO maybe use something a bit more robust like pytorch/cpuinfo?
// FIXME should this be in noa/core ?
// FIXME For unsupported plateforms, this still compiles but returns zeros for everything.
//       For MacOS, sysctlbyname should be used.

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
namespace {
    using namespace noa;

    usize parse_size_from_line(const std::string& line) {
        const usize colon_id = line.find_first_of(':');
        const auto start = std::string_view(line.c_str() + colon_id + 1);
        auto size = nd::parse<usize>(start);
        check(size, "Could not retrieve file size. Line={}", start);
        return *size;
    }

    cpu::DeviceMemory get_memory_info_linux() {
        cpu::DeviceMemory ret{};

        std::string line;
        std::ifstream mem_info("/proc/meminfo");
        while (std::getline(mem_info, line)) {
            if (nd::starts_with(line, "MemTotal"))
                ret.total = parse_size_from_line(line) * 1024; // in bytes
            else if (nd::starts_with(line, "MemAvailable"))
                ret.free = parse_size_from_line(line) * 1024; // in bytes
        }
        check(not mem_info.bad(), "Error while reading {}", "/proc/meminfo");
        return ret;
    }

    std::string get_cpu_name_linux() {
        std::ifstream cpu_info("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpu_info, line)) {
            if (nd::starts_with(line, "model name")) {
                const usize colon_id = line.find_first_of(':');
                const usize nonspace_id = line.find_first_not_of(" \t", colon_id + 1);
                return line.c_str() + nonspace_id; // assume right trimmed
            }
        }
        panic("Could not retrieve CPU name from {}", "/proc/cpuinfo");
    }

    cpu::DeviceCore get_cpu_core_count_linux() {
        cpu::DeviceCore out{};
        bool got_logical{};
        bool got_physical{};

        std::ifstream cpu_info("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpu_info, line)) {
            if (not got_logical and nd::starts_with(line, "siblings")) {
                out.logical = parse_size_from_line(line);
                got_logical = true;
            } else if (not got_physical and nd::starts_with(line, "cpu cores")) {
                out.physical = parse_size_from_line(line);
                got_physical = true;
            }
            if (got_logical and got_physical)
                break;
        }
        check(not cpu_info.bad() and got_logical and got_physical,
              "Could not retrieve CPU name from {}", "/proc/cpuinfo");
        return out;
    }

    cpu::DeviceCache get_cpu_cache_linux(int level) {
        cpu::DeviceCache out{};
        std::ifstream cache_info;
        const Path prefix = fmt::format("/sys/devices/system/cpu/cpu0/cache/index{}", level);
        // FIXME index1 is the instruction L1 cache on my machine...

        Path cache_size_path = prefix / "size";
        if (std::filesystem::is_regular_file(cache_size_path)) {
            cache_info.open(cache_size_path.string());
            char suffix;
            cache_info >> out.size >> suffix;
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
            cache_info.close();
        }

        cache_size_path = prefix / "coherency_line_size";
        if (std::filesystem::is_regular_file(cache_size_path)) {
            cache_info.open(cache_size_path.string());
            cache_info >> out.line_size;
            cache_info.close();
        }
        return out;
    }
}
#elif defined (NOA_PLATFORM_APPLE)
#include <unistd.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>

namespace {
    using namespace noa;

    auto get_memory_info_macos() -> cpu::DeviceMemory {
        cpu::DeviceMemory ret{};

        u64 memsize{};
        usize len = sizeof(memsize);
        if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) == 0) {
            ret.total = static_cast<usize>(memsize);
        }

        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        vm_statistics64_data_t vmstat;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vmstat), &count) == KERN_SUCCESS) {
            const auto page_size = safe_cast<usize>(sysconf(_SC_PAGESIZE));
            const auto free_bytes = safe_cast<usize>(vmstat.free_count * page_size);
            const auto inactive_bytes = safe_cast<usize>(vmstat.inactive_count * page_size);
            ret.free = free_bytes + inactive_bytes;
        }

        return ret;
    }

    auto get_cpu_name_macos() -> std::string {
        char buffer[256];
        usize size = sizeof(buffer);
        check(sysctlbyname("machdep.cpu.brand_string", buffer, &size, nullptr, 0) == 0,
              "Could not retrieve CPU name");
        return {buffer, size - 1};
    }

    auto get_cpu_core_count_macos() -> cpu::DeviceCore {
        cpu::DeviceCore out{};

        u32 logical = 0, physical = 0;
        usize size = sizeof(u32);
        check(sysctlbyname("hw.logicalcpu", &logical, &size, nullptr, 0) == 0,
              "Could not retrieve logical core count");
        check(sysctlbyname("hw.physicalcpu", &physical, &size, nullptr, 0) == 0,
              "Could not retrieve physical core count");

        out.logical = logical;
        out.physical = physical;
        return out;
    }

    auto get_cpu_cache_macos(int level) -> cpu::DeviceCache {
        cpu::DeviceCache out{};

        const char* cache_sysctls[] = {
            "hw.l1dcachesize", // data cache L1
            "hw.l2cachesize",
            "hw.l3cachesize"
        };
        check(level > 0 and level <= 3);

        u64 cache_size{};
        usize size = sizeof(cache_size);
        if (sysctlbyname(cache_sysctls[level - 1], &cache_size, &size, nullptr, 0) == 0) {
            out.size = static_cast<usize>(cache_size);
        } else if (level == 3) {
            // L3 isn't exposed, use SLC instead. From https://en.wikipedia.org/wiki/Apple_silicon
            out.size = [&]() -> usize {
                constexpr usize MB = 1024 * 1024;
                const auto name = get_cpu_name_macos();
                if (name.find("M1 Ultra") != std::string::npos) return 96 * MB;
                if (name.find("M1 Max")   != std::string::npos) return 48 * MB;
                if (name.find("M1 Pro")   != std::string::npos) return 24 * MB;
                if (name.find("M1")       != std::string::npos) return 8  * MB;
                if (name.find("M2 Ultra") != std::string::npos) return 96 * MB;
                if (name.find("M2 Max")   != std::string::npos) return 48  * MB;
                if (name.find("M2 Pro")   != std::string::npos) return 24  * MB;
                if (name.find("M2")       != std::string::npos) return 8   * MB;
                if (name.find("M3 Ultra") != std::string::npos) return 96 * MB;
                if (name.find("M3 Max")   != std::string::npos) return 48  * MB;
                if (name.find("M3 Pro")   != std::string::npos) return 24  * MB;
                if (name.find("M3")       != std::string::npos) return 8   * MB;
                return {};
            }();
        }

        // Line size (same across all levels)
        u64 line_size{};
        size = sizeof(line_size);
        if (sysctlbyname("hw.cachelinesize", &line_size, &size, nullptr, 0) == 0)
            out.line_size = static_cast<usize>(line_size);

        return out;
    }
}
#endif

namespace {
    std::mutex g_mutex{};
    std::vector<noa::cpu::Device::reset_callback_type> g_callbacks{};
}

namespace noa::cpu {
    void Device::add_reset_callback(Device::reset_callback_type callback) {
        if (callback == nullptr)
            return;

        auto lock = std::lock_guard(g_mutex);
        for (auto p: g_callbacks)
            if (p == callback)
                return;
        g_callbacks.push_back(callback);
    }

    void Device::remove_reset_callback(Device::reset_callback_type callback) {
        auto lock = std::lock_guard(g_mutex);
        std::erase(g_callbacks, callback);
    }

    auto Device::memory() -> DeviceMemory {
        #if defined(NOA_PLATFORM_LINUX)
        return get_memory_info_linux();
        #elif defined(NOA_PLATFORM_WINDOWS)
        return get_memory_info_windows();
        #elif defined(NOA_PLATFORM_APPLE)
        return get_memory_info_macos();
        #else
        return {};
        #endif
    }

    auto Device::cores() -> DeviceCore {
        #if defined(NOA_PLATFORM_LINUX)
        return get_cpu_core_count_linux();
        #elif defined(NOA_PLATFORM_WINDOWS)
        return get_cpu_core_count_windows();
        #elif defined(NOA_PLATFORM_APPLE)
        return get_cpu_core_count_macos();
        #else
        return {};
        #endif
    }

    auto Device::cache(int level) -> DeviceCache {
        #if defined(NOA_PLATFORM_LINUX)
        return get_cpu_cache_linux(level);
        #elif defined(NOA_PLATFORM_WINDOWS)
        return get_cpu_cache_windows(level);
        #elif defined(NOA_PLATFORM_APPLE)
        return get_cpu_cache_macos(level);
        #else
        (void) level;
        return {};
        #endif
    }

    auto Device::name() -> std::string {
        #if defined(NOA_PLATFORM_LINUX)
        return std::string(nd::trim(get_cpu_name_linux()));
        #elif defined(NOA_PLATFORM_WINDOWS)
        return std::string{nd::trim(get_cpu_name_windows())};
        #elif defined(NOA_PLATFORM_APPLE)
        return std::string{nd::trim(get_cpu_name_macos())};
        #else
        return {};
        #endif
    }

    auto Device::summary() -> std::string {
        const std::string name = Device::name();
        const DeviceCore core_count = Device::cores();
        const DeviceCache cache1 = Device::cache(1);
        const DeviceCache cache2 = Device::cache(2);
        const DeviceCache cache3 = Device::cache(3);
        const DeviceMemory sysmem = Device::memory();

        return fmt::format(
            "cpu:\n"
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
            std::endian::native == std::endian::big ? "big" : "little"
        );
    }

    void Device::reset() const {
        auto lock = std::lock_guard(g_mutex);
        for (auto p: g_callbacks)
            p(*this);
    }
}
