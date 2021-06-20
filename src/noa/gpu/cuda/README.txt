Error management
================

- Most if not all functions in the CUDA namespace can throw nested noa::Exception. This is not a perfect solution
  but it makes debugging easier (functions will report an error message, with file name, function name and line
  number) and keeps the client side cleaner.

Utilities
=========

- util/Device.h:
    -- Contains a Device object which also contains device related static functions.
    -- Contains the DeviceCurrentScope, which sets the current device for the remaining of the scope.
- util/Stream.h: a Stream object (which is associated to a Device) which also contains stream related static functions.
- util/Event.h: a Event object (which is associated to a Device) which also contains event related static functions.

Allocators
==========

These are never passed around (raw pointers are) and are simply there to make (de)allocation easier via RAII.
- PtrPinned.h: template class managing (host) pinned memory.
- PtrDevice.h: template class managing linear memory on the device.
- PtrDevicePadded.h: template class managing padded (with pitch) memory on the device.
- PtrArray.h: template class managing CUDA arrays.
- PtrTexture.h: class managing a texture object. Can be constructed from PtrDevice, PtrDevicePadded or PtrArray.

Memory namespace
================

- memory/Copy.h: Functions to copy memory regions (contiguous, padded or CUDA array) from, to and withing devices.
- memory/Random.h:
- memory/Set.h:

Fourier namespace
=================

- fourier/Exception.h: Overload the default CUDA exceptions with cufft error numbers.
- fourier/Plan.h: Create plans. The API is adapted from the CPU version (using FFTW) for cufft.
- fourier/Remap.h: CUDA version of cpu/fourier/Remap.h
- fourier/Resize.h: CUDA version of cpu/fourier/Resize.h
- fourier/Transforms.h: CUDA version of cpu/fourier/Transforms.h

Math namespace
==============


