#pragma once

#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <queue>
#include <tuple>
#include <functional>

#include "noa/core/Assert.hpp"
#include "noa/core/Definitions.hpp"
#include "noa/core/Exception.hpp"
#include "noa/core/Session.hpp"
#include "noa/core/Types.hpp"

namespace noa::cpu::details {
    // Asynchronous dispatch queue. Enqueued tasks are executed in order.
    struct AsyncDispatchQueue {
        AsyncDispatchQueue() {
            // Spawn the thread. It is important to have the member variable already initialized
            // correctly before spawning the thread, so don't use the member initializer list here.
            m_thread = std::thread(&AsyncDispatchQueue::waitingRoom_, this);
        }

        ~AsyncDispatchQueue() {
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

        template<class Functor, class... Args>
        void enqueue(Functor&& functor, Args&&... args) {
            auto no_arg_functor =
                    [functor_ = std::forward<Functor>(functor),
                     args_ = std::make_tuple(std::forward<Args>(args)...)]()
                     mutable { std::apply(std::move(functor_), std::move(args_)); };
            {
                const std::scoped_lock lock(m_mutex);
                if (m_exception) {
                    while (!m_queue.empty())
                        m_queue.pop();
                    std::rethrow_exception(std::exchange(m_exception, nullptr));
                }
                m_queue.emplace(std::move(no_arg_functor));
                m_condition_work.notify_one();
            }
        }

        bool is_busy() {
            const std::scoped_lock lock_worker(m_mutex);
            if (m_exception) {
                while (!m_queue.empty())
                    m_queue.pop();
                std::rethrow_exception(std::exchange(m_exception, nullptr));
            }
            return !m_queue.empty() || m_is_busy;
        }

        void synchronize() {
            std::unique_lock lock(m_mutex);
            m_condition_sync.wait(lock, [this] { return m_queue.empty() && !m_is_busy; });
            if (m_exception)
                std::rethrow_exception(std::exchange(m_exception, nullptr));
        }

        [[nodiscard]] auto threadID() const noexcept {
            return m_thread.get_id();
        }

    private:
        // The working thread is launched into the "waiting room". The thread waits for a task to pop into the
        // queue or for the destructor to be called (i.e. stop=true). Once a task is added and the waiting thread
        // receives the notification, it extracts it from the queue and launches the task.
        void waitingRoom_() {
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
                    m_condition_work.wait(lock, [this] { return m_stop || !m_queue.empty(); });

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
                        task = std::move(m_queue.front());
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
        // Every access to member variables are protected by a single mutex.
        // Notifications are send while holding the lock, as explained here:
        // https://stackoverflow.com/a/66162551
        // We may be OK using the same condition variable, but just to be sure,
        // use a different condition variable for synchronization and enqueueing.
        std::condition_variable m_condition_work;
        std::condition_variable m_condition_sync;
        std::mutex m_mutex;
        bool m_is_busy{false};
        bool m_stop{false};
    };
}

namespace noa::cpu {
    // (A)synchronous dispatch queue.
    class Stream {
    public:
        enum ThreadMode {
            // Uses the current thread as working thread. Task-execution is synchronous. Creating a stream with
            // this mode is trivial, doesn't require any dynamic allocation, and doesn't spawn any thread.
            CURRENT = 0,
            DEFAULT = 0, // deprecated

            // Spawns a new thread when the stream is created. Enqueued tasks are sent to this thread.
            // This mode is more expensive, but allows asynchronous execution of enqueued tasks.
            // The same thread is reused throughout the lifetime of the stream and order of execution
            // is of course guaranteed. The stream is automatically synchronized when destructed,
            // but potential captured exceptions are ignored. To properly rethrow exceptions, it is
            // then best to explicitly synchronize the stream before the destructor being called.
            ASYNC = 1
        };

    public:
        // Creates a stream.
        explicit Stream(ThreadMode mode = ThreadMode::ASYNC) : m_omp_thread_count(Session::threads()) {
            if (mode == ThreadMode::ASYNC)
                m_worker = std::make_shared<details::AsyncDispatchQueue>();
        }

    public:
        // Enqueues a task.
        // While the current implementation relies on std::function, which requires copyable objects,
        // perfect forwarding is guaranteed (the functor and its arguments are not copied).
        //
        // If the stream uses Mode::CURRENT, the functor is immediately executed by the current thread.
        // If the stream uses Mode::ASYNC, the functor will be executed asynchronously, on the working thread.
        // If an enqueued task throws an exception, the stream flushes its queue. The exception will be
        // correctly rethrown on the current thread making the next enquiry (e.g. enqueue, synchronization).
        // As such, this call may also rethrow exceptions from previous asynchronous tasks.
        //
        // WARNING: In Mode::ASYNC, it is NOT allowed for the functor to capture the stream. The reason is that
        // it can create a scenario where the functor becomes the last owner of the stream, making it difficult
        // to safely despawn the work thread. Furthermore, the only case where capturing a stream could be
        // useful is when the functor needs to call a function taking a stream. In this case, the best solution
        // is to create a new Mode::CURRENT stream in the functor and use this stream instead. This has no
        // overhead and has also a clearer intent.
        template<class Functor, class... Args>
        constexpr void enqueue(Functor&& functor, Args&&... args) {
            if (!m_worker) {
                functor(std::forward<Args>(args)...);
            } else if (m_worker->threadID() == std::this_thread::get_id()) {
                NOA_THROW("The asynchronous stream was captured by an enqueued task, which is now trying to "
                          "enqueue another task. This is currently not supported and it is usually better to create "
                          "a new synchronous stream inside the original task and use this new stream instead");
            } else {
                m_worker->enqueue(std::forward<Functor>(functor), std::forward<Args>(args)...);
            }
        }

        // Whether the stream is busy running tasks.
        // This function may also throw an exception from previous asynchronous tasks.
        bool is_busy() {
            if (!m_worker)
                return false;
            return m_worker->is_busy();
        }

        // Blocks until the stream has completed all operations.
        // This function may also throw an exception from previous asynchronous tasks.
        void synchronize() {
            if (!m_worker)
                return;
            m_worker->synchronize();
        }

        // Sets the number of internal threads that enqueued functions are allowed to use.
        // When the stream is created, this value is set to the corresponding value of the current session.
        void set_threads(i64 threads) noexcept {
            m_omp_thread_count = threads ? threads : 1;
        }

        // Returns the number of internal threads that enqueued functions are allowed to use.
        // When the stream is created, this value is set to the corresponding value of the current session.
        [[nodiscard]] i64 threads() const noexcept {
            return m_omp_thread_count;
        }

    private:
        std::shared_ptr<details::AsyncDispatchQueue> m_worker;

        // Number of "internal" threads that OpenMP is allowed to use.
        // This has nothing to do with the number of workers (there's only one worker).
        i64 m_omp_thread_count{1};
    };
}
