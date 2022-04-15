# Included files for noa/gpu/cuda:
if (NOT NOA_ENABLE_CUDA)
    return()
endif ()

set(NOA_CUDA_HEADERS
        # noa::cuda
        gpu/cuda/Device.h
        gpu/cuda/Event.h
        gpu/cuda/Exception.h
        gpu/cuda/Stream.h
        gpu/cuda/Types.h

        # noa::cuda::util
        gpu/cuda/util/Atomic.cuh
        gpu/cuda/util/Block.cuh
        gpu/cuda/util/EwiseBinary.cuh
        gpu/cuda/util/EwiseTrinary.cuh
        gpu/cuda/util/EwiseUnary.cuh
        gpu/cuda/util/Pointers.h
        gpu/cuda/util/Reduce.cuh
        gpu/cuda/util/Traits.h
        gpu/cuda/util/Version.h
        gpu/cuda/util/Warp.cuh

        # noa::cuda::memory
        gpu/cuda/memory/Arange.h
        gpu/cuda/memory/Cast.h
        gpu/cuda/memory/Copy.h
        gpu/cuda/memory/Index.h
        gpu/cuda/memory/Linspace.h
        gpu/cuda/memory/MemoryPool.h
        gpu/cuda/memory/PtrArray.h
        gpu/cuda/memory/PtrDevice.h
        gpu/cuda/memory/PtrDevicePadded.h
        gpu/cuda/memory/PtrManaged.h
        gpu/cuda/memory/PtrPinned.h
        gpu/cuda/memory/PtrTexture.h
        gpu/cuda/memory/Resize.h
        gpu/cuda/memory/Set.h
        gpu/cuda/memory/Transpose.h

        # noa::cuda::fft
        gpu/cuda/fft/Exception.h
        gpu/cuda/fft/Filters.h
        gpu/cuda/fft/Plan.h
        gpu/cuda/fft/Remap.h
        gpu/cuda/fft/Resize.h
        gpu/cuda/fft/Transforms.h

        # noa::cuda::math
        gpu/cuda/math/Complex.h
        gpu/cuda/math/Ewise.h
        gpu/cuda/math/Find.h
        gpu/cuda/math/Reduce.h

        # noa::cuda::filter
        gpu/cuda/filter/Convolve.h
        gpu/cuda/filter/Median.h
        gpu/cuda/filter/Shape.h

        # noa::cuda::geometry
        gpu/cuda/geometry/Interpolate.h
        gpu/cuda/geometry/Prefilter.h
        gpu/cuda/geometry/Rotate.h
        gpu/cuda/geometry/Scale.h
        gpu/cuda/geometry/Shift.h
        gpu/cuda/geometry/Symmetry.h
        gpu/cuda/geometry/Transform.h

        # noa::cuda::geometry::fft
        gpu/cuda/geometry/fft/Shift.h
        gpu/cuda/geometry/fft/Symmetry.h
        gpu/cuda/geometry/fft/Transform.h

        # # noa::cuda::reconstruct
        # gpu/cuda/reconstruct/ProjectBackward.h
        # gpu/cuda/reconstruct/ProjectForward.h

        )

set(NOA_CUDA_SOURCES
        gpu/cuda/Device.cpp

        # noa::cuda::fft
        gpu/cuda/fft/Exception.cpp
        gpu/cuda/fft/Filters.cu
        gpu/cuda/fft/Plan.cpp
        gpu/cuda/fft/Remap.cu
        gpu/cuda/fft/Resize.cu

        # noa::cuda::memory
        gpu/cuda/memory/Arange.cu
        gpu/cuda/memory/Cast.cu
        gpu/cuda/memory/Copy.cu
        gpu/cuda/memory/Index.cu
        gpu/cuda/memory/IndexSequence.cu
        gpu/cuda/memory/Linspace.cu
        gpu/cuda/memory/Resize.cu
        gpu/cuda/memory/Set.cu
        gpu/cuda/memory/Transpose0132.cu
        gpu/cuda/memory/Transpose0213.cu
        gpu/cuda/memory/Transpose0231.cu
        gpu/cuda/memory/Transpose0312.cu
        gpu/cuda/memory/Transpose0321.cu

        # noa::cuda::math
        gpu/cuda/math/Complex.cu
        gpu/cuda/math/EwiseBinary.cu
        gpu/cuda/math/EwiseTrinary.cu
        gpu/cuda/math/EwiseUnary.cu
        gpu/cuda/math/Find.cu
        gpu/cuda/math/Reduce.cu
        gpu/cuda/math/ReduceAxes.cu
        gpu/cuda/math/ReduceAxesVariance.cu

        # noa::cuda::filter
        gpu/cuda/filter/Convolve1.cu
        gpu/cuda/filter/Convolve2.cu
        gpu/cuda/filter/Convolve3.cu
        gpu/cuda/filter/ConvolveSeparable.cu
        gpu/cuda/filter/Median.cu
        gpu/cuda/filter/ShapeCylinder.cu
        gpu/cuda/filter/ShapeRectangle.cu
        gpu/cuda/filter/ShapeSphere.cu

        # noa::cuda::geometry
        gpu/cuda/geometry/Prefilter.cu
        gpu/cuda/geometry/Shift2D.cu
        gpu/cuda/geometry/Shift3D.cu
        gpu/cuda/geometry/Symmetry2D.cu
        gpu/cuda/geometry/Symmetry3D.cu
        gpu/cuda/geometry/Transform2D.cu
        gpu/cuda/geometry/Transform3D.cu
        gpu/cuda/geometry/TransformSymmetry2D.cu
        gpu/cuda/geometry/TransformSymmetry3D.cu

        # noa::cuda::geometry::fft
        gpu/cuda/geometry/fft/Shift.cu
        gpu/cuda/geometry/fft/Transform2D.cu
        gpu/cuda/geometry/fft/Transform2DSymmetry.cu
        gpu/cuda/geometry/fft/Transform3D.cu
        gpu/cuda/geometry/fft/Transform3DSymmetry.cu

        # # noa::cuda::reconstruct
        # gpu/cuda/reconstruct/ProjectBackward.cu
        # gpu/cuda/reconstruct/ProjectForward.cu

        )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_CUDA_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_CUDA_SOURCES})
