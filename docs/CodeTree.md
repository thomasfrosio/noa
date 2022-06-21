## `Namespaces`

- `::noa` contains the `common` functionalities and types, e.g. files, strings, vectors and other utilities 
accessible and used by the backends. It also contains the unified API.
- `::noa::cpu` contains the CPU backend. This code runs on CPU threads.
- `::noa::cuda` contains the CUDA backend. This code is meant to be mostly called by CPU but runs on CUDA capable 
  devices.

Backends are independent of each other, meaning that the library can be built without the CPU or CUDA backend. Note
however that in order to build the unified API and/or test the CUDA backend via `noa_tests`, the CPU backend must be 
included as well.

## `Unified API`

This is meant to be the main user interface and allows writing "device-agnostic" code. It is much simpler and safer 
to use this API as opposed to the backends. The `Stream`, the `Device` and the `Array` are the central pieces of the 
API. For now, things are limited, especially relating to just-in-time compilation and lazy evaluation but a lot can be
achieved already.

## `CUDA backend`

The library currently only supports `nvcc` to compile CUDA device code. Any `.h` header can be included in `.cpp`
(compiled with GCC or Clang) or `. cu` (compiled with nvcc) files. These headers contain functions that can be called
from host threads, with some exceptions. For instance the static vectors, complex types and `noa::math` functions can
often be called from the host or the device. `.cuh` files should only be included by `.cu` files.

## `Directory hierarchy`

These are defined for each backend, as well as the unified API

- `signal`        : Filters and masks, e.g. convolutions, median filters, geometric shapes, etc.
- `signal::fft`   : Bandpass, Cross-correlation, phase-shifts, etc.
- `fft`           : Fourier specific functions, e.g. FFT plans, FFTs, Fourier padding/cropping, etc.
- `math`          : Math functions for arrays, e.g. arithmetics, reductions, element-wise transformation, BLAS, etc.
- `memory`        : Scoped-bound memory resources, array manipulations and initialization, casting, etc.
- `geometry`      : Linear/affine transforms and symmetries. Polar transforms.
- `geometry::fft` : Linear/affine transforms and symmetries, for Fourier transforms. Direct Fourier 
  insertion/extraction, etc.

Note: The directory hierarchy is the same as the namespace hierarchy. This might result in quite long signatures,
e.g. `::noa::cpu::geometry::rotate2D()` but is often easier, specially for newcomers, to understand where
functions are declared and defined. Moreover, the length of the signature can always be simplified by including
the namespace at the beginning of the scope, e.g. `using namespace ::noa::cpu`.
