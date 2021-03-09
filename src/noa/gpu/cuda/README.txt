Error management
================

-   Most if not all functions in the CUDA namespace can throw nested exception. This is not a perfect solution
    but it makes debugging easier (functions will report an error message, with file name, function name and line
    number) and keeps the client side cleaner.

Device and stream management
============================

-   Device: an object and a "namespace-like" gathering device related functions.
-   DeviceCurrentScope: an object to set the current device for the remaining of the scope.
-   Stream: an object and a "namespace-like" to work with stream. Streams are associated to a Device.
-   Event: an object and a "namespace-like" to work with events. Events are associated to a Device.

Memory management
=================

-   PtrPinned: template class managing (host) pinned memory.
-   PtrDevice: template class managing linear memory on the device.
-   PtrDevicePadded: template class managing padded (with pitch) memory on the device.
-   PtrArray: template class managing CUDA arrays.
-   PtrTexture: class managing a texture object. Can be constructed from PtrDevice, PtrDevicePadded or PtrArray.

-   copy: Set of functions to copy memory regions from, to and withing the device.
