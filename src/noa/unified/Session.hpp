#pragma once

#include "noa/unified/Device.hpp"
#include "noa/unified/Traits.hpp"

namespace noa::inline types {
    /// The session is used to initialize and control the library initialization and static data.
    /// Note that these utilities are entirely optional, already managed automatically by the library,
    /// and can therefore be ignored by users. However, in some cases, users may want to tweaks a few
    /// things that are internally managed by the library and Session does just that.
    ///
    /// \details \b Threads:
    /// The CPU backend uses OpenMP's multithreading to process large arrays and increase hardware usage.
    /// Streams control the maximum number of threads OpenMP can use for a given task. This value can be changed
    /// at any point during the program's lifetime and is per stream. If a stream has a thread limit of 1,
    /// multithreading is turned off for tasks enqueued to that specific stream. When creating a stream,
    /// the initial thread limit is set by the Session. By default, when creating the session, the thread limit
    /// is retrieved from the environmental variable \c NOA_THREADS or \c OMP_NUM_THREADS (the former has precedence).
    /// If these variables are both empty or not defined, Session tries to deduce the number of available threads
    /// on the machine and uses this number has thread limit. This value can be changed explicitly using
    /// set_thread_limit().
    ///
    /// \details \b CUDA-contexts:
    /// The library uses the CUDA runtime, and doesn't explicitly set CUDA contexts. The CUDA runtime,
    /// and by extension this library, will use the current context (which doesn't need to be the primary
    /// context). This usually goes unnoticed for CUDA runtime users, but is important to note for CUDA
    /// driver users creating and managing their own contexts.
    ///
    /// \details \b CUDA-module-loading:
    /// In CUDA 11.7, CUDA module loading can be lazy, which can improves load times.
    /// See https://docs.nvidia.com/cuda/cuda-c-programming-guide/lazy-loading.html.
    /// Session can check that lazy loading is enabled (the default in CUDA 12.1) and can also enable it directly.
    ///
    /// \details \b FFT-plans:
    /// Backends guarantee that FFT plans are cached so that the transform functions (e.g. noa::fft::r2c)
    /// are efficient. However, since these plans can take a lot of memory, we provide ways for users to control and
    /// query this cache. Some backends (and the libraries they use) may be more flexible than others.
    /// CUDA:
    ///     - We manage the caching system explicitly, which offers maximum flexibility. The API below allows turning
    ///       on and off the cache, resizing it, and querying its current state. The free-functions we provide in
    ///       noa::fft also allow turning on and off the cache temporarily for a particular transform (see
    ///       noa::fft::FFTOptions.cache_plan).
    ///     - Plan creation (and the cuFFT APIs in general) is thread safe. However, plans and output data should only
    ///       be accessed by one (host) thread at a time. As such and for simplicity, we hold a per host-thread and
    ///       per-device cache. In cuFFT, a single plan encodes both the forward and backward transform. For instance,
    ///       noa::r2c and noa::c2r on the same arrays only require (and cache) a single plan.
    ///     - Plans often allocate a workspace on the device. Holding many plans in the cache quite often results
    ///       in significant memory usage. If plans are to be executed sequentially (e.g., on the same stream), the
    ///       workspace can be shared across multiple plans (see noa::fft::FFTOptions.record_and_share_workspace).
    /// CPU-FFTW3:
    ///     - The cache is handled by FFTW3's wisdom. We don't manage it, and its flexibility is limited.
    ///       The user can only clear the cache or query its size.
    ///     - The cache is per device (CPU), globally shared within the application, and has no fixed capacity.
    ///       Forward and backward transforms are considered different plans in FFTW3. In fact, a single transform
    ///       can generate many (>16) entries in the cache.
    ///     - FFTW3's wisdom is well-optimized and doesn't require a lot of memory.
    ///
    /// \details \b CUDA's-cuBLAS:
    /// The CUDA backend uses the cuBLAS library for matrix-matrix multiplication. The library caches cuBLAS
    /// handles (one per device). While there's not much point to clear this cache, users can still explicitly
    /// clear it.
    class Session {
    public:
        /// Sets the maximum number of internal threads used in this session.
        /// \param n_threads    Maximum number of threads. If 0, retrieve value from environmental variable
        ///                     \c NOA_THREADS or \c OMP_NUM_THREADS (the former takes precedence). If these
        ///                     variables are empty or not defined, try to deduce the number of available
        ///                     threads and use this number instead.
        /// \note This is the maximum number of internal threads stream can use. Users can of course create additional
        ///       threads using tools from the library, e.g. ThreadPool or asynchronous streams.
        static void set_thread_limit(i32 n_threads);

        /// Returns the maximum number of internal threads.
        [[nodiscard]] static auto thread_limit() noexcept -> i32 {
            if (m_thread_limit <= 0) // if not initialized, do it now
                set_thread_limit(0);
            return m_thread_limit;
        }

        /// Tries to enable GPU lazy module loading.
        /// \details If the driver is already initialized or the environment is explicitly set to eager mode,
        ///          this function will not be able to enable lazy loading. As such, it is meant to be called
        ///          before any interaction with the GPU driver. Note that while this function initialises the
        ///          GPU driver, it doesn't load the library module nor does it call the GPU runtime.
        /// \return Whether lazy loading is actually enabled.
        static bool set_gpu_lazy_loading();

        /// Clears the FFT cache for a given device.
        /// Returns how many "plans" were cleared from the cache, or returns zero.
        /// \warning This function doesn't synchronize before clearing the cache, so the caller should make sure
        ///          that none of the plans are being used. This can be easily done by synchronizing the relevant
        ///          streams or the device.
        static auto clear_fft_cache(Device device) -> i32;

        /// Sets the maximum number of plans the FFT cache can hold on a given device.
        /// Returns how many plans were cleared from the resizing of the cache.
        /// \note Some backends may not allow setting this parameter, in which case -1 is returned.
        static auto set_fft_cache_limit(i32 count, Device device) -> i32;

        /// Gets the maximum number of plans the FFT cache can hold on a given device.
        /// \note Some backends may not support retrieving this parameter or may have a
        ///       dynamic cache without a fixed capacity. In these cases, -1 is returned.
        [[nodiscard]] static auto fft_cache_limit(Device device) -> i32;

        /// Returns the current cache size, i.e. how many plans are currently cached.
        /// \note This number may be different from the number of transforms that have been launched.
        ///       Indeed, FFT planners (like FFTW) may create and cache many plans for a single transformation.
        ///       Similarly, transforms may be divided internally into multiple transforms.
        /// \note Some backends may not support retrieving this parameter, in which case, -1 is returned.
        [[nodiscard]] static auto fft_cache_size(Device device) -> isize;

        /// Returns the number of bytes that are left to allocate from previous plan creations.
        /// \see noa::fft::FFTOptions.record_and_share_workspace for more details.
        /// \note Some backends (e.g., FFTW) do not support explicit workspace management and always return 0.
        [[nodiscard]] static auto fft_workspace_left_to_allocate(Device device) -> isize;

        /// Assigns cached plans without a workspace to this buffer.
        /// \details The buffer should be on the device and contiguous, otherwise an error will be thrown. The number of
        ///          plans that have been assigned to the given buffer is returned. If the size of the buffer is less than
        ///          workspace_left_to_allocate(device), the buffer cannot be used and 0 is returned.
        /// \see noa::fft::FFTOptions.record_and_share_workspace for more details.
        /// \note Some backends (e.g., FFTW) do not support explicit workspace management and always return 0.
        template<nt::varray_decay T>
        static auto fft_set_workspace(Device device, const T& buffer) -> i32 {
            check(buffer.device() == device,
                  "The buffer should be on the device, but got device={} and buffer:device={}",
                  device, buffer.device());
            check(buffer.are_contiguous(), "The workspace should be a contiguous array");
            return fft_set_workspace(
                device,
                std::reinterpret_pointer_cast<std::byte[]>(buffer.share()),
                buffer.ssize() * static_cast<isize>(sizeof(nt::value_type_t<T>))
            );
        }

        /// Clears the BLAS cache for a given device.
        /// \warning This function doesn't synchronize before clearing the cache, so the caller should make sure
        ///          that none of the plans are being used. This can be easily done by synchronizing the relevant
        ///          streams or the device.
        static void clear_blas_cache(Device device);

    private:
        static auto fft_set_workspace(Device device, const std::shared_ptr<std::byte[]>& buffer, isize buffer_bytes) -> i32;

    private:
        static i32 m_thread_limit;
    };
}
