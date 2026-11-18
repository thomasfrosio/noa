#include "noa/runtime/gpu/Stream.hpp"

#ifdef NOA_ENABLE_SYCL
namespace {
    using namespace noa::types;
    using namespace noa::gpu;

    std::vector<std::optional<sycl::queue>> g_null_queues{};
    std::forward_list<Pair<sycl::queue, bool>> g_queues{};
    std::mutex g_queues_mutex;

    // Synchronizes all the queues from that device.
    void stream_synchronize_callback(i32 device_id) {
        const auto device = Device(device_id, Unchecked{}).get_sycl_device();
        const auto lock = std::scoped_lock(g_queues_mutex);
        const auto synchronize = [&](sycl::queue& queue) {
            if (device == queue.get_device())
                queue.wait_and_throw();
        };
        std::ranges::for_each(g_queues, [&](auto& pair) {
            synchronize(pair.first);
        });
        std::ranges::for_each(g_null_queues, [&](auto& optional) {
            if (optional.has_value())
                synchronize(*optional);
        });
    }

    // Create a SYCL queue and add it to our register.
    auto create_queue(Device device, Stream::Mode mode, bool enable_profiling) -> sycl::queue* {
        Device::add_synchronize_callback(stream_synchronize_callback);
        auto make_queue = [&] {
            if (enable_profiling) {
                return sycl::queue(
                    details::sycl_device_context(device.id()),
                    details::sycl_devices()[static_cast<usize>(device.id())],
                    details::sycl_async_error_handler,
                    sycl::property_list{
                        sycl::property::queue::in_order(),
                        sycl::property::queue::enable_profiling(),
                    }
                );
            }
            return sycl::queue(
                details::sycl_device_context(device.id()),
                details::sycl_devices()[static_cast<usize>(device.id())],
                details::sycl_async_error_handler,
                sycl::property_list{sycl::property::queue::in_order()}
            );
        };

        if (mode == Stream::Mode::DEFAULT) {
            const auto lock = std::scoped_lock(g_queues_mutex);
            if (g_null_queues.empty())
                g_null_queues.resize(details::sycl_devices().size());

            auto& queue = g_null_queues[static_cast<usize>(device.id())];
            if (not queue.has_value())
                queue = make_queue();
            return &queue.value();
        }

        auto queue = make_queue();
        const auto lock = std::scoped_lock(g_queues_mutex);
        g_queues.emplace_front(std::move(queue), mode == Stream::SYNC);
        return &g_queues.front().first;
    }

    auto wait_on_queue(sycl::queue& queue, Stream::Mode mode) {
        queue.wait_and_throw();

        if (mode == Stream::DEFAULT) {
            const auto sycl_device = queue.get_device();
            const auto lock = std::scoped_lock(g_queues_mutex);
            std::ranges::for_each(g_queues, [&](auto& pair) {
                // Synchronizes all the sync streams from that device.
                if (pair.second and pair.first.get_device() == sycl_device)
                    queue.wait_and_throw();
            });
        }
    }

    auto delete_queue(sycl::queue& queue, Stream::Mode mode) {
        wait_on_queue(queue, mode);

        const auto lock = std::scoped_lock(g_queues_mutex);
        if (mode == Stream::Mode::DEFAULT) {
            const auto sycl_device = queue.get_device();
            for (auto& e: g_null_queues) {
                if (e and e->get_device() == sycl_device)
                    e = std::nullopt;
            }
        } else {
            g_queues.remove_if([&](auto& pair) {
                return pair.first == queue;
            });
        }
    }
}
#endif

namespace noa::gpu {
    Stream::Core::Core(Device device, Mode mode, bool enable_profiling) {
        #if defined(NOA_ENABLE_CUDA)
        (void)enable_profiling;
        if (mode != Mode::DEFAULT) {
            const DeviceGuard guard(device);
            const u32 type = mode == Stream::SYNC ? cudaStreamDefault : cudaStreamNonBlocking;
            check(cudaStreamCreateWithFlags(&stream_handle, type));
        }
        #elif defined(NOA_ENABLE_ROCM)
        (void)enable_profiling;
        if (mode != Mode::DEFAULT) {
            const DeviceGuard guard(device);
            const hipStream_t type = mode == Stream::SYNC ? hipStreamDefault : hipStreamNonBlocking;
            check(hipStreamCreateWithFlags(&stream_handle, type));
        }
        #elif defined(NOA_ENABLE_SYCL)
        queue = create_queue(device, mode, enable_profiling);
        queue_mode = mode;
        #endif
    }

    Stream::Core::~Core() {
#if defined(NOA_ENABLE_CUDA)
        [[maybe_unused]] cudaError_t err{};
        err = cudaStreamSynchronize(stream_handle);
        NOA_ASSERT(err == cudaSuccess);
        resource_registry.clear_after_sync();
        if (stream_handle) {
            err = cudaStreamDestroy(stream_handle);
            NOA_ASSERT(err == cudaSuccess);
        }
#elif defined(NOA_ENABLE_ROCM)
        [[maybe_unused]] hipError_t err{};
        err = hipStreamSynchronize(stream_handle);
        NOA_ASSERT(err == hipSuccess);
        resource_registry.clear_after_sync();
        if (stream_handle) {
            err = hipStreamDestroy(stream_handle);
            NOA_ASSERT(err == hipSuccess);
        }
#elif defined(NOA_ENABLE_SYCL)
        NOA_ASSERT(queue); // always valid pointer
        try {
            delete_queue(*queue, queue_mode);
        } catch (...) {
            NOA_ASSERT(false);
        }
        resource_registry.clear_after_sync();
#endif
    }

#if defined(NOA_ENABLE_SYCL)
    void Stream::Core::poll_events() {
        const std::scoped_lock lock(last_event_mutex);
        while (not pending_events.empty() and
               pending_events.front().get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete)
        {
            pending_events.pop_front();
            ++resource_registry.callback_count; // notify registry
        }
    }
#endif

    [[nodiscard]] auto Stream::is_busy() const -> bool {
        #if defined(NOA_ENABLE_CUDA)
        const auto guard = DeviceGuard(m_device);
        const cudaError_t status = cudaStreamQuery(m_core->stream_handle);
        if (status == cudaSuccess) return false;
        if (status == cudaErrorNotReady) return true;
        panic_runtime(error2string(status));
        #elif defined(NOA_ENABLE_ROCM)
        const auto guard = DeviceGuard(m_device);
        const hipError_t status = hipStreamQuery(m_core->stream_handle);
        if (status == hipSuccess) return false;
        if (status == hipErrorNotReady) return true;
        panic_runtime(error2string(status));
        #elif defined(NOA_ENABLE_SYCL)
        const auto lock = std::scoped_lock(m_core->last_event_mutex);
        if (m_core->last_event)
            return m_core->last_event->get_info<sycl::info::event::command_execution_status>() !=
                sycl::info::event_command_status::complete;
        return false;
        #endif
    }

    void Stream::synchronize() const {
        NOA_ASSERT(m_core);
        #if defined(NOA_ENABLE_CUDA)
        const auto guard = DeviceGuard(m_device);
        check(cudaStreamSynchronize(m_core->stream_handle));
        #elif defined(NOA_ENABLE_ROCM)
        const auto guard = DeviceGuard(m_device);
        check(hipStreamSynchronize(m_core->stream_handle));
        #elif defined(NOA_ENABLE_SYCL)
        wait_on_queue(*m_core->queue, m_core->queue_mode);
        {
            const auto lock = std::scoped_lock(m_core->last_event_mutex);
            m_core->pending_events.clear(); // the queue is idle, so these are all complete

            // We could clear the last event, but keep it instead so that if Event::record(stream) is called
            // we don't need to launch a empty task and can just return the last event. Clearing it just offers
            // a small advantage for enqueue_attach as the entire call is skipped if there's no last event. It's
            // nice, but not relevant since we always call enqueue_attach after calling enqueue.
        }
        #endif
        m_core->resource_registry.clear_after_sync(); // resets callback_count to 0
    }
}
