#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/Version.hpp"
#include "noa/core/Types.hpp"
#include "noa/unified/Device.hpp"

namespace noa::inline types {
    /// The session is used to initialise and control the library initialization and static data.
    /// Note that these utilities are entirely optional, already managed automatically by the library,
    /// and can therefore be ignored by users. However, in some cases, users may want to tweaks a few
    /// things that are internally managed by the library and the Session offers a common control
    /// point to do just that.
    ///
    /// \details \b Threads:
    /// The CPU backend uses OpenMP's multithreading to divide large arrays and speed up computation.
    /// Streams control the maximum amount OpenMP can use for a given task. This value can be changed
    /// at any point during the program's lifetime and is per stream. If a stream has a thread limit of 1,
    /// multithreading is turned off for tasks enqueued to that specific stream. When creating a stream,
    /// the initial thread limit is set by the current Session. By default, when creating the session,
    /// the thread limit is retrieved from the environmental variable \c NOA_THREADS or \c OMP_NUM_THREADS
    /// (the former has precedence). If these variables are both empty or not defined, Session tries to deduce
    /// the number of available threads on the machine and uses this number has thread limit. Note that
    /// this value can be changed using set_threads(). New streams will use whatever value is currently
    /// set in the Session.\n
    ///
    /// \details \b CUDA-contexts:
    /// The library uses the CUDA runtime, and doesn't explicitly set CUDA contexts. The CUDA runtime,
    /// and by extension this library, will use the current context (which doesn't need to be the primary
    /// context). This usually goes unnoticed for CUDA runtime users, but is important to note for CUDA
    /// driver users creating and managing their own contexts.
    ///
    /// \details \b CUDA-module-loading:
    /// In CUDA 11.7, CUDA module loading can be lazy, which greatly improves load times.
    /// See https://docs.nvidia.com/cuda/cuda-c-programming-guide/lazy-loading.html.
    /// When the Session is created, it can check that lazy loading is enabled (the default in CUDA 12.1)
    /// and can also enable it, so the library module is loaded lazily.
    ///
    /// \details \b FFT-plans:
    /// The backends use the external libraries to compute DFTs. These libraries often save a lot of data, referred
    /// to as "plans". The backends save these plans so that users don't need to keep track of them. The cache is
    /// per device (for CUDA, it is also per host-thread). Since these plans can take a lot of memory, and users
    /// may want to control the maximum number of plans that can be cached or may want to reset the cache. Note that
    /// the library FFT API also offers a more granular control of what transforms should be cached.
    ///
    /// \details \b CUDA's-cuBLAS:
    /// The CUDA backend uses the cuBLAS library for matrix-matrix multiplication. The library caches cuBLAS
    /// handles (one per device). While there's not much point to clear this cache, users can still explicitly
    /// clear this cache.
    class Session {
    public: // CPU backend
        /// Sets the maximum number of internal threads used by a session.
        /// \param n_threads    Maximum number of threads. If 0, retrieve value from environmental variable
        ///                     \c NOA_THREADS or \c OMP_NUM_THREADS (the former takes precedence). If these
        ///                     variables are empty or not defined, try to deduce the number of available
        ///                     threads and use this number instead.
        /// \note This is the maximum number of internal threads. Users can of course create additional threads
        ///       using tools from the library, e.g. ThreadPool or asynchronous streams.
        static void set_thread_limit(i64 n_threads);

        /// Returns the maximum number of internal threads.
        static i64 thread_limit() noexcept {
            if (m_thread_limit <= 0) // if not initialized, do it now
                set_thread_limit(0);
            return m_thread_limit;
        }

    public: // CUDA backend
        /// Tries to enable CUDA's lazy module loading (only affects the CUDA backend).
        /// \details If the driver is already initialized or the environment is explicitly set to eager mode,
        ///          this function will not be able to enable lazy loading. As such, it is meant to be called
        ///          before any interaction with the CUDA driver.
        ///          Note that while this function initialises the CUDA driver, it doesn't load the library
        ///          module nor does it call the CUDA runtime.
        /// \return Whether lazy loading is actually enabled.
        static bool set_cuda_lazy_loading();

        /// Clears the FFT cache for a given device.
        /// Returns how many plans were cleared from the cache.
        /// \warning This function doesn't synchronize before clearing the cache, so the caller should make sure
        ///          that none of plans are being used. This can be easily done by synchronizing the relevant
        ///          streams or the device.
        static i64 clear_fft_cache(Device device = Device::current(DeviceType::GPU));

        /// Sets the maximum number of plans the cache can hold on a given device.
        static void set_fft_cache_limit(i64 count, Device device = Device::current(DeviceType::GPU));

        /// Clears the cuBLAS cache for a given device.
        /// \warning This function doesn't synchronize before clearing the cache, so the caller should make sure
        ///          that none of plans are being used. This can be easily done by synchronizing the relevant
        ///          streams or the device.
        static void clear_cublas_cache(Device device = Device::current(DeviceType::GPU));

    public:
        static std::string version() { return NOA_VERSION; }
        static std::string url() { return NOA_URL; }

    private:
        static i64 m_thread_limit;
    };
}

#endif
