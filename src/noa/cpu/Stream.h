#pragma once

#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <queue>
#include <tuple>
#include <functional>

#include "noa/common/Session.h"
#include "noa/common/Definitions.h"
#include "noa/common/types/Constants.h"

namespace noa::cpu {
    // Stream or (asynchronous) dispatch queue.
    // A Stream is managing a working thread, which may be different than the main thread. In this case,
    // enqueued functions (referred to as tasks) are executed asynchronously. The order of execution is
    // sequential (it's a queue).
    // If a task throws an exception, the stream becomes invalid and it will flush its queue. The exception
    // will be correctly rethrown on the main thread at the next enquiry (e.g. enqueue, synchronization).
    // Before rethrowing, the stream resets itself.
    class Stream {
    public:
        enum Mode {
            // The working thread is the calling thread and tasks execution is synchronous.
            DEFAULT,

            // Spawns a new thread when the stream is created and tasks execution is asynchronous.
            ASYNC
        };

    private:
        struct StreamImp {
            // work
            std::queue<std::pair<std::function<void()>, bool>> queue; // TODO std::move_only_function
            std::thread worker;
            std::exception_ptr exception;

            // synchronization
            std::condition_variable condition;
            std::mutex mutex_queue;
            dim_t threads{Session::threads()};
            bool is_waiting{true};
            bool stop{};

            ~StreamImp() {
                if (worker.joinable()) {
                    stop = true;
                    condition.notify_one();
                    worker.join();
                    // ignore any potential exception to keep destructor noexcept
                }
            }
        };

    public:
        // Creates a stream.
        explicit Stream(Mode mode = Mode::ASYNC) : m_imp(std::make_shared<StreamImp>()) {
            if (mode != DEFAULT)
                m_imp->worker = std::thread(Stream::waitingRoom_, m_imp.get());
        }

        // Empty constructor.
        // Creates an empty instance that is meant to be reset using one of the operator assignment.
        // Calling empty() returns true, but any other member function call will fail. Passing an
        // empty stream is never allowed (and will result in segfault) unless specified otherwise.
        constexpr explicit Stream(std::nullptr_t) {}

    public:
        // Enqueues a task: f is the function to enqueue, and (optional) args are the parameters of f.
        // Depending on the stream, this function may be asynchronous and may also return error codes
        // from previous, asynchronous launches.
        template<class F, class... Args>
        void enqueue(F&& f, Args&& ... args) {
            NOA_ASSERT(m_imp);
            enqueue_<false>(std::forward<F>(f), std::forward<Args>(args)...);
        }

        // Whether the stream is busy with some tasks.
        // This function may also return error codes from previous, asynchronous launches.
        bool busy() {
            NOA_ASSERT(m_imp);
            if (m_imp->exception)
                rethrow_();
            return !m_imp->queue.empty() || !m_imp->is_waiting;
        }

        // Blocks until the stream has completed all operations.
        // This function may also return error codes from previous, asynchronous launches.
        void synchronize() {
            NOA_ASSERT(m_imp);
            std::promise<void> p;
            std::future<void> fut = p.get_future();
            enqueue_<true>([](std::promise<void>& pr) { pr.set_value(); }, std::ref(p));
            fut.wait();
            if (m_imp->exception)
                rethrow_();
        }

        // Sets the number of internal threads that enqueued functions are allowed to use.
        // When the stream is created, this value is set to the corresponding value of the current session.
        void threads(dim_t threads) noexcept {
            NOA_ASSERT(m_imp);
            m_imp->threads = threads ? threads : 1;
        }

        // Returns the number of internal threads that enqueued functions are allowed to use.
        // When the stream is created, this value is set to the corresponding value of the current session.
        [[nodiscard]] dim_t threads() const noexcept {
            NOA_ASSERT(m_imp);
            return m_imp->threads;
        }

        // Whether the stream is an empty instance.
        [[nodiscard]] bool empty() const noexcept {
            return m_imp == nullptr;
        }

    private:
        // The working thread is launched into the "waiting room". The thread waits for a task to pop into the
        // queue or for the destructor to be called (i.e. stop=true). Once a task is added and the waiting thread
        // receives the notification, it extracts it from the queue and launches the task.
        static void waitingRoom_(StreamImp* imp) {
            while (true) {
                std::function<void()> task;
                bool sync_call;
                {
                    std::unique_lock<std::mutex> lock_worker(imp->mutex_queue);
                    imp->is_waiting = true;
                    imp->condition.wait(lock_worker, [imp] { return imp->stop || !imp->queue.empty(); });
                    imp->is_waiting = false;

                    if (imp->queue.empty()) {
                        // The predicate was true and the lock is acquired, so at this point if the queue
                        // is empty, it is because the destructor was called, so delete the thread.
                        NOA_ASSERT(imp->stop == true);
                        break;
                    } else {
                        std::tie(task, sync_call) = std::move(imp->queue.front());
                        imp->queue.pop();
                        if (imp->exception && !sync_call)
                            continue; // don't even try to run the task
                    }
                }

                // At this point, the queue is released (new tasks can be added by enqueue)
                // and the working thread can execute the task.
                try {
                    if (task)
                        task();
                } catch (...) {
                    imp->exception = std::current_exception();
                }
            }
        }

        // Enqueues the task to the queue.
        // To allow for synchronization, tasks are marked with a boolean flag: false means the task comes from the
        // user and should not be run if an exception was caught. true means the task comes from a synchronization
        // query and the task should be run to release the calling thread.
        template<bool SYNC, class F, class... Args>
        void enqueue_(F&& func, Args&& ... args) {
            if (!m_imp->worker.joinable() || m_imp->worker.get_id() == std::this_thread::get_id()) {
                func(std::forward<Args>(args)...);
            } else {
                auto no_arg_func = [f_ = std::forward<F>(func), args_ = std::make_tuple(std::forward<Args>(args)...)]()
                        mutable { std::apply(std::move(f_), std::move(args_)); };
                {
                    std::scoped_lock lock(m_imp->mutex_queue);
                    m_imp->queue.emplace(std::move(no_arg_func), SYNC);
                }
                m_imp->condition.notify_one();
                // If sync, don't throw and destruct the future, otherwise worker may segfault.
                if (!SYNC && m_imp->exception)
                    rethrow_();
            }
        }

        // Rethrow the caught exception and reset the stream to a valid state.
        // Make sure the queue is emptied so the work thread is reset as well.
        void rethrow_() {
            if (m_imp->exception) {
                {
                    std::scoped_lock queue_lock(m_imp->mutex_queue);
                    while (!m_imp->queue.empty()) m_imp->queue.pop();
                }
                std::rethrow_exception(std::exchange(m_imp->exception, nullptr));
            }
        }

    private:
        std::shared_ptr<StreamImp> m_imp;
    };
}
