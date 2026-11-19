list(APPEND NOA_HEADERS
    noa/runtime/core/Access.hpp
    noa/runtime/core/Accessor.hpp
    noa/runtime/core/Border.hpp
    noa/runtime/core/Interfaces.hpp
    noa/runtime/core/Iwise.hpp
    noa/runtime/core/Random.hpp
    noa/runtime/core/Reduce.hpp
    noa/runtime/core/Shape.hpp
    noa/runtime/core/Shareable.hpp
    noa/runtime/core/Span.hpp
    noa/runtime/core/Subregion.hpp
    noa/runtime/core/Traits.hpp
    noa/runtime/core/Utils.hpp

#    noa/runtime/Allocator.hpp
#    noa/runtime/Array.hpp
#    noa/runtime/ArrayOption.hpp
#    noa/runtime/Backend.hpp
#    noa/runtime/Blas.hpp
#    noa/runtime/Complex.hpp
#    noa/runtime/CopyBatches.hpp
#    noa/runtime/Device.hpp
#    noa/runtime/Event.hpp
#    noa/runtime/Ewise.hpp
#    noa/runtime/Factory.hpp
#    noa/runtime/Iwise.hpp
#    noa/runtime/Random.hpp
#    noa/runtime/Reduce.hpp
#    noa/runtime/ReduceAxesEwise.hpp
#    noa/runtime/ReduceAxesIwise.hpp
#    noa/runtime/ReduceEwise.hpp
#    noa/runtime/ReduceIwise.hpp
#    noa/runtime/Resize.hpp
#    noa/runtime/Session.hpp
#    noa/runtime/Sort.hpp
#    noa/runtime/Stream.hpp
#    noa/runtime/Subregion.hpp
#    noa/runtime/Traits.hpp
#    noa/runtime/Utils.hpp
)

list(APPEND NOA_SOURCES
#    noa/runtime/Allocator.cpp
#    noa/runtime/Device.cpp
#    noa/runtime/Session.cpp
#    noa/runtime/Stream.cpp
)

if (NOA_ENABLE_CPU)
    list(APPEND NOA_HEADERS
        noa/runtime/cpu/Allocators.hpp
        noa/runtime/cpu/Blas.hpp
        noa/runtime/cpu/ComputeHandle.hpp
        noa/runtime/cpu/Copy.hpp
        noa/runtime/cpu/Device.hpp
        noa/runtime/cpu/Event.hpp
        noa/runtime/cpu/Ewise.hpp
        noa/runtime/cpu/Iwise.hpp
        noa/runtime/cpu/Median.hpp
        noa/runtime/cpu/Permute.hpp
        noa/runtime/cpu/ReduceAxesEwise.hpp
        noa/runtime/cpu/ReduceAxesIwise.hpp
        noa/runtime/cpu/ReduceEwise.hpp
        noa/runtime/cpu/ReduceIwise.hpp
        noa/runtime/cpu/Set.hpp
        noa/runtime/cpu/Sort.hpp
        noa/runtime/cpu/Stream.hpp
    )

    list(APPEND NOA_SOURCES
        noa/runtime/cpu/Blas.cpp
        noa/runtime/cpu/Device.cpp
    )
endif ()

if (NOA_ENABLE_GPU)
    list(APPEND NOA_HEADERS
        noa/runtime/gpu/Allocators.hpp
#        noa/runtime/gpu/Blas.hpp
        noa/runtime/gpu/Block.hpp
        noa/runtime/gpu/ComputeHandle.hpp
        noa/runtime/gpu/Constants.hpp
        noa/runtime/gpu/Copy.hpp
        noa/runtime/gpu/Device.hpp
        noa/runtime/gpu/Event.hpp
        noa/runtime/gpu/Ewise.hpp
        noa/runtime/gpu/Error.hpp
        noa/runtime/gpu/IncludeGuard.hpp
        noa/runtime/gpu/Iwise.hpp
#        noa/runtime/gpu/Median.hpp
        noa/runtime/gpu/MemoryPool.hpp
        noa/runtime/gpu/Permute.hpp
        noa/runtime/gpu/Pointers.hpp
        noa/runtime/gpu/ReduceAxesEwise.hpp
        noa/runtime/gpu/ReduceAxesIwise.hpp
        noa/runtime/gpu/ReduceEwise.hpp
        noa/runtime/gpu/ReduceIwise.hpp
        noa/runtime/gpu/Runtime.hpp
#        noa/runtime/gpu/Sort.hpp
        noa/runtime/gpu/Stream.hpp
        noa/runtime/gpu/Utils.hpp
        noa/runtime/gpu/Warp.hpp
    )

    list(APPEND NOA_SOURCES
        noa/runtime/gpu/Device.cpp
        noa/runtime/gpu/Stream.cpp
#        noa/runtime/gpu/Blas.cu
    )
endif ()
