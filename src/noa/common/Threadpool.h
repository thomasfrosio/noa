/// \file noa/common/Threadpool.h
/// \brief A simple and safe threadpool.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <vector>
#include <queue>
#include <tuple>

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"

namespace noa {
    /// Thread pool.
    /// \note This is similar to std::async in the way that exceptions thrown inside the pool are
    ///       correctly propagated to the parent thread. See enqueue() for more details.
    /// \see Threadpool() to create a pool of workers.
    /// \see enqueue() to add an asynchronous tasks.
    /// \see ~Threadpool() to close the pool.
    class ThreadPool {
    private:
        std::vector<std::thread> workers;
        std::queue<std::packaged_task<void()>> tasks;

        // synchronization
        std::mutex queue_mutex;
        std::condition_variable condition;
        bool stop{false};

    public:
        /// Launches \a threads threads.
        /// \details Threads are launched into the "waiting room". The first thread to arrive waits
        ///          for a task to pop-up into the queue or for the destructor to be called (i.e. stop).
        ///          The area is guarded, so the "first" thread waits for the condition while the others
        ///          wait for the lock to be released.
        ///          Once a task is added and the waiting thread receives the notification, it extracts
        ///          it from the queue, release the lock so that another thread can enter the waiting
        ///          area, and launch the task. This launch starts outside of the locking area, so that
        ///          multiple tasks can be executed at the same time (by different threads).
        NOA_HOST explicit ThreadPool(size_t threads) {
            if (threads == 0)
                NOA_THROW("Threads should be a positive non-zero number, got 0");

            auto waiting_room = [this] {
                while (true) {
                    std::packaged_task<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                                             [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty())
                            return; // join only if there's no task left.
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    } // release the lock so that another thread can start to wait().
                    task();
                }
            };

            workers.reserve(threads);
            for (size_t i = 0; i < threads; ++i) {
                workers.emplace_back(waiting_room); // launch threads into the pool.
            }
        }

        /// Enqueue a tasks. The queue is asynchronous and returns immediately. As such, the return
        /// value is a std::future. Use get() to retrieve to output value. Even if the task returns
        /// void, it is recommended to get() to output so that exceptions are not lost in the working
        /// thread (exception_ptr can also be used to report exceptions).
        ///
        /// \example
        /// \code
        /// ThreadPool a(2);
        /// auto future = a.enqueue([]() -> int {
        ///                 throw std::runtime_error("aie");
        ///                 return 1;
        ///               });
        /// int result = future.get(); // throws std::runtime_error("aie")
        /// \endcode
        ///
        /// \note It looks like gcc and clang can see through std::bind and generate identical code to
        ///       the lambda version. Nevertheless, I'd rather use the lambda version since it is
        ///       recommended to not use std::bind (since C++14). Unfortunately, since perfect
        ///       capture with variadic lambdas are a C++20 feature, a workaround with std::make_tuple
        ///       and std::apply is required in C++17. See: https://stackoverflow.com/questions/47496358
        template<class F, class... Args>
        NOA_HOST decltype(auto) enqueue(F&& f, Args&& ... args) {
            using return_type = std::invoke_result_t<F, Args...>;

            std::packaged_task<return_type()> task(
                    [f_ = std::forward<F>(f), args_ = std::make_tuple(std::forward<Args>(args)...)]()
                            mutable { return std::apply(std::move(f_), std::move(args_)); }
            );

            std::future<return_type> res = task.get_future(); // res.valid() will work as expected.
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                tasks.emplace(std::move(task));
            }
            condition.notify_one();
            return res;
        }

        /// Ensures all tasks are done and then closes the pool.
        /// \note This should not be called explicitly.
        NOA_HOST ~ThreadPool() {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                stop = true;
            }
            condition.notify_all();
            for (std::thread& worker : workers)
                worker.join();
        }
    };
}
