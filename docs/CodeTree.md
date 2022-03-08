## `Namespaces`

- `::noa` contains the `common` functionalities and types, e.g. files, strings, vectors and other utilities 
accessible and used by the backends.
- `::noa::cpu` contains the CPU backend. This code is meant to be called by the host only.
- `::noa::cuda` contains the CUDA backend. This code is meant to be mostly called by the host, however host/device or
  device only functions are also available.

Backends are independent of each other, meaning that the library can be built without the CPU or CUDA backend. Note
however that in order to test the CUDA backend via `noa_tests`, the CPU backend must be included as well.

## `Device code`

The library currently only supports `nvcc` to compile device code. Any `.h` header can be included in `.cpp`
(compiled with GCC or Clang) or `. cu` (compiled with nvcc) files. These headers contain functions that can be called
from host threads, with some exceptions. For instance the static vectors, complex types and `noa::math` functions can
often be called from the host or the device. `.cuh` files should only be included by `.cu` files.

## `Directory hierarchy`

These are defined for each backend:

- `filter`        :   Filters and masks, e.g. convolutions, median filters, geometric shapes, etc.
- `fft`           :   Fourier specific functions, e.g. FFT plans, FFTs, bandpass filters, Fourier cropping, etc.
- `math`          :   Math functions for arrays, e.g. arithmetics, reductions, element-wise transformation, etc.
- `memory`        :   Scoped-bound memory resources, array manipulations and initialization, casting, etc.
- `reconstruct`   :   Reconstruction methods.
  - `fft`         :   Fourier reconstruction methods, e.g. backward and forward projections, etc.
- `geometry`      :   Linear/affine transforms and symmetries.
  - `fft`         :   Linear/affine transforms and symmetries, for Fourier transforms.

Note: The directory hierarchy is the same as the namespace hierarchy. This might result in quite long signatures,
e.g. `::noa::cpu::geometry::rotate2D()` but is often easier, specially for newcomers, to understand where
functions are declared and defined. Moreover, the length of the signature can always be simplified by including
the namespace at the beginning of the scope, e.g. `using namespace ::noa::cpu`.
