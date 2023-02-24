# Included files for noa/gpu/cuda:
if (NOT NOA_ENABLE_CUDA)
    return()
endif ()

set(NOA_CUDA_HEADERS
        # noa::cuda
        gpu/cuda/Device.hpp
        gpu/cuda/Event.hpp
        gpu/cuda/Ewise.hpp
        gpu/cuda/Exception.hpp
        gpu/cuda/Find.hpp
        gpu/cuda/Sort.hpp
        gpu/cuda/Stream.hpp
        gpu/cuda/Types.hpp

        # noa::cuda::utils
        gpu/cuda/utils/Block.cuh
        gpu/cuda/utils/EwiseBinary.cuh
        gpu/cuda/utils/EwiseTrinary.cuh
        gpu/cuda/utils/EwiseUnary.cuh
        gpu/cuda/utils/Iwise.cuh
        gpu/cuda/utils/Pointers.hpp
        gpu/cuda/utils/ReduceUnary.cuh
        gpu/cuda/utils/ReduceBinary.cuh
        gpu/cuda/utils/Version.hpp
        gpu/cuda/utils/Warp.cuh

        # noa::cuda::memory
        gpu/cuda/memory/Arange.hpp
        gpu/cuda/memory/Cast.hpp
        gpu/cuda/memory/Copy.hpp
        gpu/cuda/memory/Index.hpp
        gpu/cuda/memory/Iota.hpp
        gpu/cuda/memory/Linspace.hpp
        gpu/cuda/memory/MemoryPool.hpp
        gpu/cuda/memory/Permute.hpp
        gpu/cuda/memory/PtrArray.hpp
        gpu/cuda/memory/PtrDevice.hpp
        gpu/cuda/memory/PtrDevicePadded.hpp
        gpu/cuda/memory/PtrManaged.hpp
        gpu/cuda/memory/PtrPinned.hpp
        gpu/cuda/memory/PtrTexture.hpp
        gpu/cuda/memory/Resize.hpp
        gpu/cuda/memory/Set.hpp
        gpu/cuda/memory/Subregion.hpp

        # noa::cuda::fft
        gpu/cuda/fft/Exception.hpp
        gpu/cuda/fft/Plan.hpp
        gpu/cuda/fft/Remap.hpp
        gpu/cuda/fft/Resize.hpp
        gpu/cuda/fft/Transforms.hpp

        # noa::cuda::math
        gpu/cuda/math/Blas.hpp
        gpu/cuda/math/Complex.hpp
        gpu/cuda/math/Random.hpp
        gpu/cuda/math/Reduce.hpp

#        # noa::cuda::signal
#        gpu/cuda/signal/fft/Bandpass.h
#        gpu/cuda/signal/fft/Correlate.h
#        gpu/cuda/signal/fft/FSC.h
#        gpu/cuda/signal/fft/Shape.h
#        gpu/cuda/signal/fft/Shift.h
#        gpu/cuda/signal/fft/Standardize.h
#        gpu/cuda/signal/Convolve.h
#        gpu/cuda/signal/Median.h
#        gpu/cuda/signal/Shape.h

        # noa::cuda::geometry
        gpu/cuda/geometry/fft/Polar.h
        gpu/cuda/geometry/fft/Project.h
        gpu/cuda/geometry/fft/Transform.h
        gpu/cuda/geometry/Interpolator.h
        gpu/cuda/geometry/Polar.h
        gpu/cuda/geometry/Prefilter.h
        gpu/cuda/geometry/Transform.h

        )

set(NOA_CUDA_SOURCES
        gpu/cuda/Device.cpp

        gpu/cuda/EwiseBinaryArithmetic.cu
        gpu/cuda/EwiseBinaryComparison.cu
        gpu/cuda/EwiseTrinaryArithmetic.cu
        gpu/cuda/EwiseTrinaryComparison.cu
        gpu/cuda/EwiseTrinaryComplexDivide.cu
        gpu/cuda/EwiseTrinaryComplexMinus.cu
        gpu/cuda/EwiseTrinaryComplexMultiply.cu
        gpu/cuda/EwiseTrinaryComplexPlus.cu
        gpu/cuda/EwiseUnary.cu
        gpu/cuda/FindMin.cu
        gpu/cuda/FindMax.cu
        gpu/cuda/Sort.cu

        # noa::cuda::fft
        gpu/cuda/fft/Exception.cpp
        gpu/cuda/fft/Plan.cpp
        gpu/cuda/fft/Remap.cu
        gpu/cuda/fft/Resize.cu

        # noa::cuda::memory
        gpu/cuda/memory/Arange.cu
        gpu/cuda/memory/Cast.cu
        gpu/cuda/memory/Copy.cu
        gpu/cuda/memory/Index.cu
        gpu/cuda/memory/Iota.cu
        gpu/cuda/memory/Linspace.cu
        gpu/cuda/memory/Permute0132.cu
        gpu/cuda/memory/Permute0213.cu
        gpu/cuda/memory/Permute0231.cu
        gpu/cuda/memory/Permute0312.cu
        gpu/cuda/memory/Permute0321.cu
        gpu/cuda/memory/Resize.cu
        gpu/cuda/memory/Set.cu
        gpu/cuda/memory/Subregion.cu

        # noa::cuda::math
        gpu/cuda/math/Blas.cu
        gpu/cuda/math/Complex.cu
        gpu/cuda/math/Random.cu
        gpu/cuda/math/Reduce.cu
        gpu/cuda/math/ReduceAxes.cu
        gpu/cuda/math/ReduceAxesVariance.cu

#        # noa::cuda::signal
#        gpu/cuda/signal/fft/Bandpass.cu
#        gpu/cuda/signal/fft/Correlate.cu
#        gpu/cuda/signal/fft/CorrelatePeak.cu
#        gpu/cuda/signal/fft/FSC.cu
#        gpu/cuda/signal/fft/Shape2D.cu
#        gpu/cuda/signal/fft/Shape3D.cu
#        gpu/cuda/signal/fft/Shift.cu
#        gpu/cuda/signal/fft/Standardize.cu
#        gpu/cuda/signal/Convolve1.cu
#        gpu/cuda/signal/Convolve2.cu
#        gpu/cuda/signal/Convolve3.cu
#        gpu/cuda/signal/ConvolveSeparable.cu
#        gpu/cuda/signal/Median.cu

        # noa::cuda::geometry
        gpu/cuda/geometry/fft/Polar.cu
        gpu/cuda/geometry/fft/Project.cu
        gpu/cuda/geometry/fft/Transform.cu
        gpu/cuda/geometry/fft/TransformTexture.cu
        gpu/cuda/geometry/Polar.cu
        gpu/cuda/geometry/Prefilter.cu
        gpu/cuda/geometry/Transform.cu
        gpu/cuda/geometry/TransformTexture.cu

        )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_CUDA_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_CUDA_SOURCES})
