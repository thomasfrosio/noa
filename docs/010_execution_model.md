## `Devices, memory resources and streams`


`Device` is the main object to refer to and to manipulate compute-devices. The library
currently supports two types of devices, CPUs and GPUs, and each device type is internally
handled by a backend. `Device` contains a unique identifier, which is simply an integer
used to uniquely refer to a compute device. While the `Device` type can be used to reset
internal resources (via `Device::reset()`) and enforce global synchronisation barriers (via
`Device::synchronize()`), it is mostly used as a “pointer” to a particular compute device,
e.g. to allocate an array on that specific device, and to query simple information about the
compute context, e.g. to list of GPUs available in the system.

`Device` only refer to the high level hardware that can execute code, e.g. the CPU device or
the GPU devices; it does not refer to the different memory resources available on the system.
`Allocator` is the type used to describe different memory resources, e.g. “async”, “unified”
or “pinned”. Each memory resource has its own set of advantages and disadvantages
and users should be aware of them to ensure correctness and better performance.

`Stream` is similar to a CUDA stream, i.e. it is an (asynchronous) dispatch queue
and is attached to a device. For the CPU, `Stream` defaults to the current thread, thus
all execution is by default synchronous. An asynchronous stream can be created too, which
redirects works to a thread managed by the stream. Importantly, the library defines a per-
thread and per-device stream, called the “current stream”. While users rarely need to pay
attention to it, the current streams play a major role in how the library works, notably how
operators are sent to the backends (see code snippet below).

`Device` and `Stream` are two fundamental types of the (frontend) library's API and both
interfaces with the corresponding backend types, e.g. `noa::cuda::Device` and `noa::cuda::Stream`,
and uses some kind of type-erasure (e.g. `std::variant`) to stay backend- and device-agnostic.

```c++
using namespace ::noa::types; // import Device and Stream

// By default:
// - the CPU current stream is the current thread.
// - the GPU current stream is the NULL stream (at least in CUDA).
auto device = Device("gpu:0");
auto my_stream = Stream(device);

// (Re)set the current stream for "gpu:0".
// This can be changed at any time.
Stream::set_current(my_stream); 

// After this point, the library will use "my_stream"
// as the asynchronous dispatch queue for arrays using
// the "gpu:0" device. Note that Stream is reference
// counted and Stream::set_current() keeps a copy of
// the stream.

// The library can indeed retrieve the current stream of
// any device at any point and dispatch work on that
// stream:
{
    // Library frontend (only a few functions actually
    // dispatch work to backends):
    Stream& current_stream = Stream::current(device);
    if (device.is_cpu()) {
        // This can expose backend specific code
        current_stream.cpu().enqueue(...); 
    } else if (device.is_gpu()) {
        // This can expose backend specific code
        current_stream.gpu().enqueue(...);
    }
}

// Users can also temporarily change the current stream
// using StreamGuard. StreamGuard derives from Stream,
// but:
// 1. Its constructor captures a reference of the current stream.
// 2. It then sets itself as the current stream.
// 3. Its destructor synchronises the queue making sure
//    work is completed (like Stream), and then resets
//    the captured stream back to the current stream.
{
	StreamGuard new_stream(device, Stream::ASYNC);
	// ... following work will on "new_stream" ...
}
// "new_stream" is destroyed, "my_stream" is back to
// being the current stream
```


## `Asynchronous eager execution`


The library currently only supports an (asynchronous) eager execution model (as opposed to a lazy-evaluation model),
that is, functions are dispatched to the underlying hardware as soon as possible. Depending on the
device and the stream, these functions are executed asynchronously on the underlying
hardware, meaning that the program execution is not blocked and that functions may return
before completion. One exception is for functions that return scalars, for instance,
`noa::sum(const Array<f64>&)->f64`. In this case, the current stream is synchronized
before the function returns. Note that alternative functions that don’t synchronize by returning
array(s) instead of scalars are often available.

```c++
// import i64, f64, Shape, Array, ArrayOption, Allocator, ReduceAxes
using namespace ::noa::types;

// Allocate an array of 256x256x256 double-precision
// floating-points. Then enqueue the Randomizer<Uniform>
// operator to initialise the values with random numbers
// with an uniform distribution, between -1 and 1.
const auto options = ArrayOption{.device="cuda:0", .allocator=Allocator::DEFAULT);
const auto shape = Shape4<i64>{1, 256, 256, 256};
const Array array = noa::random(noa::Uniform{-1., 1.}, shape, options);
// Note the use of class-template argument deduction
// (CTAD) from Uniform<f64> to Array<f64>.

// At this point, "array" should be seen as a promise:
// its elements may not be initialised or even allocated
// yet. The library simply enqueues operators to the
// current stream of the device cuda:0 and the device
// executes these operators eagerly.

// Then compute the sum of the elements of that
// randomised array. Internally, the library does
// Stream::current(array.device()) to query the relevant
// stream and then enqueues the ReduceSum operator.
// Note that no synchronisations have been performed so far.
// However, this function returns a scalar, so it must wait
// that all work is completed before returning. When this function
// returns, the current stream is synchronised.
const f64 sum_0 = noa::sum(array); 

// Then compute the sum again, but this time using the
// overload that returns an array. (Note that it also
// exists an overload that takes an existing output array).
// Since this function returns an array, as opposed to a
// scalar, it doesn't need to synchronise the stream
// before returning.
const Array<f64> sum_1 = noa::sum(array, sum_1, ReduceAxes::all());

// All axes are reduced, so sum_1 is an array with a single element.
assert(sum_1.size() == 1);

// To access this element, we should first synchronise the stream to
// make sure the device is done computing the sum. Array::eval()
// does just that and is equivalent to Stream::current(sum_1.device()).synchronize().
// Furthermore, the underlying memory may not be accessible to the CPU (this depends on
// the Allocator, see ArrayOptions::is_dereferenceable() for more details), so we may
// have to copy the array to the CPU.
// A special member function was added just for this case though:
const f64 sum_2 = sum_1.first();

assert(noa::allclose(sum_0, sum_2));
```

With this API, it is possible to write programs that rarely need to synchronise execution.
This approach scales extremely well with GPUs since the CPU time spent launching GPU
kernels can now be overlapped with GPU compute.
