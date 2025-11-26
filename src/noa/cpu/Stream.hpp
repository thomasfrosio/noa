#pragma once

#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <tuple>
#include <functional>

#include "noa/core/Error.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/utils/Misc.hpp"

namespace noa::cpu::details {
    // (Asynchronous) dispatch queue. Enqueued tasks are executed in order.
    struct DispatchQueue {
        explicit DispatchQueue(bool async) {
            if (async)
                m_thread = std::thread(&DispatchQueue::waiting_room_, this); // spawn thread
        }

        ~DispatchQueue() {
            if (is_sync())
                return;

            // Send notification to despawn when all tasks are done.
            // Writing to "stop" should be protected by the mutex.
            {
                const std::scoped_lock lock(m_mutex);
                m_stop = true;
                m_condition_work.notify_one();
            }
            m_thread.join();
            // Ignore any potential exception to keep destructor noexcept.
            // Because of this, it is best to synchronize the stream before calling the dtor.
        }

// Suppress spurious visibility warning with NVCC-GCC when a lambda captures anonymous types.
#ifdef __CUDACC__
#   if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#       pragma GCC diagnostic push
#       pragma GCC diagnostic ignored "-Wattributes"
#   elif defined(NOA_COMPILER_MSVC)
#       pragma warning(push, 0)
#   endif
#endif

        template<typename F, typename... Args>
        void enqueue(F&& func, Args&&... args) {
            if (is_sync()) {
                std::forward<F>(func)(std::forward<Args>(args)...);
                return;
            }

            // Copy/Move both the func and args into this lambda, thereby ensuring
            // that these objects will stay alive until the task is completed.
            auto no_args_func = [f = std::forward<F>(func), ...a = std::forward<Args>(args)]() mutable {
                forward_like<F>(f)(forward_like<Args>(a)...);
            };

            const std::scoped_lock lock(m_mutex);
            if (m_exception) {
                while (not m_queue.empty())
                    m_queue.pop();
                std::rethrow_exception(std::exchange(m_exception, nullptr));
            }
            m_queue.push(std::move(no_args_func));
            m_condition_work.notify_one();
        }

#ifdef __CUDACC__
#   if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#       pragma GCC diagnostic pop
#   elif defined(NOA_COMPILER_MSVC)
#       pragma warning(pop)
#   endif
#endif

        bool is_busy() {
            if (is_sync())
                return false;

            const std::scoped_lock lock_worker(m_mutex);
            if (m_exception) {
                while (not m_queue.empty())
                    m_queue.pop();
                std::rethrow_exception(std::exchange(m_exception, nullptr));
            }
            return not m_queue.empty() or m_is_busy;
        }

        void synchronize() {
            if (is_sync())
                return;

            std::unique_lock lock(m_mutex);
            m_condition_sync.wait(lock, [this] { return m_queue.empty() and not m_is_busy; });
            if (m_exception)
                std::rethrow_exception(std::exchange(m_exception, nullptr));
        }

        [[nodiscard]] std::thread::id thread_id() const noexcept {
            return m_thread.get_id();
        }

        [[nodiscard]] bool is_sync() const noexcept {
            return not m_thread.joinable();
        }

    private:
        // The working thread is launched into the "waiting room". The thread waits for a task to pop into the
        // queue or for the destructor to be called (i.e. stop=true). Once a task is added and the waiting thread
        // receives the notification, it extracts it from the queue and launches the task.
        void waiting_room_() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock lock(m_mutex);
                    if (m_queue.empty()) {
                        m_is_busy = false;
                        m_condition_sync.notify_one();
                    }
                    // If the predicate is false, we release the lock and go to sleep.
                    // If we get notified (or spurious awakenings), we'll lock again and check the predicate.
                    // If the predicate is true, we continue while still holding the lock.
                    m_condition_work.wait(lock, [this] { return m_stop or not m_queue.empty(); });

                    if (m_queue.empty()) {
                        // The predicate was true and the lock was acquired this entire time, so at this point
                        // if the queue is empty, it is because the destructor was called, so despawn.
                        NOA_ASSERT(m_stop == true);
                        break;
                    } else {
                        // Retrieve task and change status to busy. This is actually important because
                        // the queue can be empty but the task isn't done yet. is_busy() and synchronize()
                        // need to wait for the status to go back to not-busy.
                        m_is_busy = true;
                        std::swap(m_queue.front(), task);
                        m_queue.pop();

                        // If there's an exception that was thrown by the previous task,
                        // ignore the remaining tasks, which is effectively emptying the queue.
                        if (m_exception) {
                            continue;
                        }
                    }
                }

                // At this point, the lock is released and new enquires can be made to the stream.
                // Meanwhile, the working thread will execute the task.
                try {
                    if (task)
                        task();
                } catch (...) {
                    const std::scoped_lock lock(m_mutex);
                    m_exception = std::current_exception();
                }
            }
        }

    private:
        // TODO Switch to `std::move_only_function` to allow move-only objects?
        std::queue<std::function<void()>> m_queue;
        std::thread m_thread;
        std::exception_ptr m_exception;

        // Synchronization to communicate back and forth with the worker.
        // Every access to member variables is protected by a single mutex.
        // Notifications are send while holding the lock, as explained here:
        // https://stackoverflow.com/a/66162551
        // We may be okay using the same condition variable, but just to be sure,
        // use a different condition variable for synchronization and enqueueing.
        std::condition_variable m_condition_work;
        std::condition_variable m_condition_sync;
        std::mutex m_mutex;
        bool m_is_busy{false};
        bool m_stop{false};
    };
}

namespace noa::cpu {
    // Shared (a)synchronous dispatch queue.
    class Stream {
    public:
        enum class Mode {
            // Uses the current thread as working thread. Task-execution is synchronous.
            SYNC = 0,

            // Spawns a new thread when the stream is created. Enqueued tasks are sent to this thread.
            // The same thread is reused throughout the lifetime of the stream, and order of execution
            // is of course guaranteed. The stream is automatically synchronized when destructed,
            // but potential captured exceptions are ignored. To properly rethrow exceptions, it is
            // thus best to explicitly synchronize the stream before the destructor is called.
            ASYNC = 1
        };
        using enum Mode;

        struct Core {
            Core(bool async, i64 thread_limit) : worker(async), omp_thread_limit(thread_limit) {}

            details::DispatchQueue worker;

            // Number of "internal" threads that OpenMP is allowed to use.
            // This has nothing to do with the number of workers (there's only one worker).
            i64 omp_thread_limit;
        };

    public:
        // Creates a stream.
        explicit Stream(Mode mode = SYNC, i64 omp_thread_limit = 1) :
            m_core(std::make_shared<Core>(mode == ASYNC, omp_thread_limit)) {}

    public:
        // Enqueues a task.
        // While the current implementation relies on std::function, which requires copyable objects,
        // perfect forwarding is guaranteed (the function and its arguments are not copied).
        //
        // If the stream uses Stream::SYNC, the function is immediately executed by the current thread.
        // If the stream uses Stream::ASYNC, the function is executed asynchronously, on the working thread.
        // If an enqueued task throws an exception, the stream flushes its queue. The exception will be
        // correctly rethrown on the current thread making the next enquiry (e.g. enqueue, synchronization).
        // As such, this call may also rethrow exceptions from previous asynchronous tasks.
        //
        // WARNING: In Stream::ASYNC, the function should not capture the stream as it could create a scenario
        // where the function becomes the last owner of the stream, which may result in a segfault.
        template<typename F, typename... Args>
        constexpr void enqueue(F&& func, Args&&... args) {
            m_core->worker.enqueue(std::forward<F>(func), std::forward<Args>(args)...);
        }

        [[nodiscard]] auto is_sync() const -> bool { return m_core->worker.is_sync(); }
        [[nodiscard]] auto is_async() const -> bool { return not is_sync(); }

        // Whether the stream is busy running tasks.
        // This function may also throw an exception from previous asynchronous tasks.
        [[nodiscard]] bool is_busy() const {
            return m_core->worker.is_busy();
        }

        // Blocks until the stream has completed all operations.
        // This function may also throw an exception from previous asynchronous tasks.
        void synchronize() const {
            m_core->worker.synchronize();
        }

        // Sets the number of internal threads that enqueued functions are allowed to use.
        void set_thread_limit(i64 n_threads) const noexcept {
            m_core->omp_thread_limit = n_threads ? n_threads : 1;
        }

        // Returns the number of internal threads that enqueued functions are allowed to use.
        [[nodiscard]] i64 thread_limit() const noexcept {
            return m_core->omp_thread_limit;
        }

    private:
        std::shared_ptr<Core> m_core;
    };
}
