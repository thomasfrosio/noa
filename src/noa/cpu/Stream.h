#pragma once

#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <queue>
#include <tuple>
#include <functional>

#include "noa/Session.h"
#include "noa/common/Definitions.h"
#include "noa/common/types/Constants.h"

namespace noa::cpu {
    /// Stream or (asynchronous) dispatch queue.
    /// \details A Stream is managing a working thread, which may be different than the main thread. In this case,
    ///          enqueued functions (referred to as tasks) are executed asynchronously. The order of execution is
    ///          sequential (it's a queue).
    ///          If a task through an exception, the stream becomes invalid and will flush its queue. The exception
    ///          will be correctly rethrown on the main thread at the next enquiry (e.g. enqueue, synchronization).
    ///          Before rethrowing, the stream resets itself and will be ready to work on new tasks.
    class Stream {
    public:
        enum StreamMode {
            DEFAULT,
            SERIAL
        };

    public:
        /// Creates the default stream.
        /// \details The working thread is the calling thread. Thus, all executions will be synchronous relative
        ///          to the calling thread.
        Stream() = default;

        /// Creates a stream.
        /// \param mode     Stream mode.
        ///                 STREAM_DEFAULT uses the calling thread as working thread.
        ///                 STREAM_SERIAL and STREAM_CONCURRENT have the same effect on this stream.
        NOA_HOST explicit Stream(StreamMode mode) {
            if (mode != DEFAULT)
                m_worker = std::thread(&Stream::waitingRoom_, this);
        }

        /// Enqueues a task.
        /// \details If created with STREAM_SERIAL or STREAM_CONCURRENT, the queue is asynchronous relative to the
        ///          calling thread and may return before completion.
        /// \param f Function to enqueue.
        /// \param args (optional) Parameters of \p f.
        /// \note This function may also return error codes from previous, asynchronous launches.
        template<class F, class... Args>
        NOA_HOST void enqueue(F&& f, Args&& ... args) {
            enqueue_<false>(std::forward<F>(f), std::forward<Args>(args)...);
        }

        /// Whether or not the stream has completed all operations.
        /// \note This function may also return error codes from previous, asynchronous launches.
        NOA_HOST bool hasCompleted() {
            if (m_exception)
                rethrow_();
            return m_queue.empty() && m_is_waiting;
        }

        /// Blocks until the stream has completed all operations.
        /// \note This function may also return error codes from previous, asynchronous launches.
        NOA_HOST void synchronize() {
            std::promise<void> p;
            std::future<void> fut = p.get_future();
            enqueue_<true>([](std::promise<void>& pr) { pr.set_value(); }, std::ref(p));
            fut.wait();
            if (m_exception)
                rethrow_();
        }

        /// Sets the number of internal threads that enqueued functions are allowed to use.
        /// \note When the stream is created, this value is set to the corresponding value of the current session.
        NOA_HOST void threads(size_t threads) noexcept {
            m_threads = threads ? static_cast<uint16_t>(threads) : 1;
        }

        /// Returns the number of internal threads that enqueued functions are allowed to use.
        /// \note When the stream is created, this value is set to the corresponding value of the current session.
        NOA_HOST [[nodiscard]] size_t threads() const noexcept {
            return m_threads;
        }

        /// Ensures all tasks are done and then closes the pool.
        /// \note This function may also return error codes from previous, asynchronous launches.
        NOA_HOST ~Stream() noexcept(false) {
            if (m_worker.joinable()) {
                m_stop = true;
                m_condition.notify_one();
                m_worker.join();
                if (m_exception && !std::uncaught_exceptions())
                    rethrow_();
            }
        }

    public:
        Stream(const Stream&) = delete;
        Stream(Stream&&) = delete;
        Stream& operator=(const Stream&) = delete;
        Stream& operator=(Stream&&) = delete;

    private:
        // The working thread is launched into the "waiting room". The thread waits for a task to pop into the
        // queue or for the destructor to be called (i.e. stop). Once a task is added and the waiting thread receives
        // the notification, it extracts it from the queue and launches the task.
        NOA_HOST void waitingRoom_() {
            // This mutex is only used to block the working thread until it is notified.
            // The working thread is the only one to use it, and it can be reused until despawn.
            std::mutex mutex_worker;
            while (true) {
                std::function<void()> task;
                bool sync_call;
                {
                    std::unique_lock<std::mutex> lock_worker(mutex_worker);
                    m_is_waiting = true;
                    this->m_condition.wait(lock_worker,
                                           [this] { return this->m_stop || !this->m_queue.empty(); });
                    m_is_waiting = false;
                }
                {
                    std::scoped_lock queue_lock(this->m_mutex_queue);
                    if (this->m_queue.empty()) {
                        if (this->m_stop)
                            break; // the dtor called, there's no tasks left, so despawn
                    } else {
                        std::tie(task, sync_call) = std::move(this->m_queue.front());
                        this->m_queue.pop();
                        if (m_exception && !sync_call)
                            continue; // don't even try to run the task
                    }
                }

                // At this point, the queue is released (new tasks can be added by enqueue)
                // and the working thread can execute the task.
                try {
                    if (task)
                        task();
                } catch (...) {
                    m_exception = std::current_exception();
                }
            }
        }

        // Enqueues the task to the queue.
        // To allow for synchronization, tasks are marked with a boolean flag: false means the task comes from the
        // user and should not be run if an exception was caught. true means the task comes from a synchronization
        // query and the task should be run to release the calling thread.
        template<bool SYNC, class F, class... Args>
        NOA_HOST void enqueue_(F&& f, Args&& ... args) {
            if (!m_worker.joinable()) {
                f(args...);
            } else {
                auto no_arg_func = [f_ = std::forward<F>(f), args_ = std::make_tuple(std::forward<Args>(args)...)]()
                        mutable { std::apply(std::move(f_), std::move(args_)); };
                {
                    std::scoped_lock lock(m_mutex_queue);
                    m_queue.emplace(std::move(no_arg_func), SYNC);
                }
                m_condition.notify_one();
                // If sync, don't throw and destruct the future, otherwise worker may segfault.
                if (!SYNC && m_exception)
                    rethrow_();
            }
        }

        // Rethrow the caught exception and reset the stream to a valid state.
        // Make sure the queue is emptied so the work thread is reset as well.
        NOA_HOST void rethrow_() {
            if (m_exception) {
                {
                    std::scoped_lock queue_lock(this->m_mutex_queue);
                    while (!m_queue.empty()) m_queue.pop();
                }
                std::rethrow_exception(std::exchange(m_exception, nullptr));
            }
        }

    private:
        // work
        std::queue<std::pair<std::function<void()>, bool>> m_queue;
        std::thread m_worker;
        std::exception_ptr m_exception;

        // synchronization
        std::condition_variable m_condition;
        std::mutex m_mutex_queue;
        uint16_t m_threads{static_cast<uint16_t>(Session::threads())};
        bool m_is_waiting{true};
        bool m_stop{};
    };
}
