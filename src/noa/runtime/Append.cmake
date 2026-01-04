list(APPEND NOA_HEADERS
    noa/runtime/core/Access.hpp
    noa/runtime/core/Accessor.hpp
    noa/runtime/core/Atomic.hpp
    noa/runtime/core/Batch.hpp
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

    noa/runtime/Allocator.hpp
    noa/runtime/Array.hpp
    noa/runtime/ArrayOption.hpp
    noa/runtime/Backend.hpp
    noa/runtime/Blas.hpp
    noa/runtime/Complex.hpp
    noa/runtime/CopyBatches.hpp
    noa/runtime/Device.hpp
    noa/runtime/Event.hpp
    noa/runtime/Ewise.hpp
    noa/runtime/Factory.hpp
    noa/runtime/Indexing.hpp
    noa/runtime/Iwise.hpp
    noa/runtime/Layout.hpp
    noa/runtime/Random.hpp
    noa/runtime/Reduce.hpp
    noa/runtime/ReduceAxesEwise.hpp
    noa/runtime/ReduceAxesIwise.hpp
    noa/runtime/ReduceEwise.hpp
    noa/runtime/ReduceIwise.hpp
    noa/runtime/Resize.hpp
    noa/runtime/Session.hpp
    noa/runtime/Sort.hpp
    noa/runtime/Stream.hpp
    noa/runtime/Subregion.hpp
    noa/runtime/Traits.hpp
    noa/runtime/Utils.hpp
    noa/runtime/View.hpp
)

list(APPEND NOA_SOURCES
    noa/runtime/Allocator.cpp
    noa/runtime/Device.cpp
    noa/runtime/Session.cpp
    noa/runtime/Stream.cpp
)

if (NOA_ENABLE_CPU)
    list(APPEND NOA_HEADERS
        noa/runtime/cpu/Allocators.hpp
        noa/runtime/cpu/Blas.hpp
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

if (NOA_ENABLE_CUDA)
    list(APPEND NOA_HEADERS
        noa/runtime/cuda/Allocators.hpp
        noa/runtime/cuda/Blas.hpp
        noa/runtime/cuda/Block.cuh
        noa/runtime/cuda/Constants.hpp
        noa/runtime/cuda/Copy.cuh
        noa/runtime/cuda/Device.hpp
        noa/runtime/cuda/Event.hpp
        noa/runtime/cuda/Ewise.cuh
        noa/runtime/cuda/Error.hpp
        noa/runtime/cuda/IncludeGuard.cuh
        noa/runtime/cuda/Iwise.cuh
        noa/runtime/cuda/Median.cuh
        noa/runtime/cuda/MemoryPool.hpp
        noa/runtime/cuda/Permute.cuh
        noa/runtime/cuda/Pointers.hpp
        noa/runtime/cuda/ReduceAxesEwise.cuh
        noa/runtime/cuda/ReduceAxesIwise.cuh
        noa/runtime/cuda/ReduceEwise.cuh
        noa/runtime/cuda/ReduceIwise.cuh
        noa/runtime/cuda/Runtime.hpp
        noa/runtime/cuda/Sort.cuh
        noa/runtime/cuda/Stream.hpp
        noa/runtime/cuda/Version.hpp
        noa/runtime/cuda/Warp.cuh
    )

    list(APPEND NOA_SOURCES
        noa/runtime/cuda/Device.cpp
        noa/runtime/cuda/Blas.cu
    )
endif ()
