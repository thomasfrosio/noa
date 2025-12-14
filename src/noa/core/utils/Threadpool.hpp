#pragma once

#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <queue>

#include "noa/core/Error.hpp"
#include "noa/core/utils/Misc.hpp"

namespace noa {
    /// Simple thread pool.
    /// \note Similarly to std::async, exceptions thrown inside the pool are correctly propagated
    ///       to the parent thread via std::future. See enqueue() for more details.
    class ThreadPool {
    private:
        std::vector<std::thread> workers;
        std::queue<std::packaged_task<void()>> tasks;

        // synchronization
        std::mutex queue_mutex;
        std::condition_variable condition;
        bool stop{};

    public:
        /// Launches the thread pool.
        /// \details Threads are launched into the "waiting room". The first thread to arrive waits
        ///          for a task to pop-up into the queue or for the destructor to be called.
        ///          The area is guarded, so the first thread waits for the condition while the others
        ///          wait for the lock to be released. Once a task is added and the waiting thread receives
        ///          the notification, it extracts it from the queue, releases the lock so that another thread
        ///          can enter the waiting area, and launches the task.
        explicit ThreadPool(usize n_threads) {
            check(n_threads, "Threads should be a positive non-zero number, got 0");

            auto waiting_room = [this] {
                while (true) {
                    std::packaged_task<void()> task;
                    {
                        std::unique_lock lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop or not this->tasks.empty(); });
                        if (this->stop and this->tasks.empty()) // join only if there's no task left.
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    } // release the lock so that another thread can start to wait().
                    task();
                }
            };

            workers.reserve(n_threads);
            for (usize i{}; i < n_threads; ++i) {
                workers.emplace_back(waiting_room); // launch threads into the pool.
            }
        }

        /// Enqueues a task. The queue is asynchronous and returns immediately. As such, the return
        /// value is a std::future. Use get() to retrieve to output value. Even if the task returns
        /// void, it is recommended to get() to output so that exceptions are not lost in the working
        /// thread (exception_ptr can also be used to report exceptions).
        ///
        /// \example
        /// \code
        /// ThreadPool a(2);
        /// auto future = a.enqueue([]() -> int {
        ///     throw std::runtime_error("aie");
        ///     return 1;
        /// });
        /// future.get(); // throws std::runtime_error("aie")
        /// \endcode
        template<class F, class... Args>
        auto enqueue(F&& f, Args&& ... args) {
            using return_type = std::invoke_result_t<F, Args...>;

            std::packaged_task<return_type()> task(
                [f_ = std::forward<F>(f), ...args_ = std::forward<Args>(args)]()
                    mutable { return forward_like<F>(f_)(forward_like<Args>(args_)...); }
            );

            std::future<return_type> res = task.get_future(); // res.valid() will work as expected.
            {
                std::unique_lock lock(queue_mutex);
                tasks.emplace(std::move(task));
            }
            condition.notify_one();
            return res;
        }

        /// Ensures all tasks are done and then closes the pool.
        ~ThreadPool() {
            {
                std::unique_lock lock(queue_mutex);
                stop = true;
            }
            condition.notify_all();
            for (std::thread& worker : workers)
                worker.join();
        }
    };
}
