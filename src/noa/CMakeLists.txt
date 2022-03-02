# ---------------------------------------------------------------------------------------
# The library source and header files
# ---------------------------------------------------------------------------------------
set(NOA_HEADERS
        Session.h

        common/Assert.h
        common/Definitions.h
        common/Exception.h
        common/Functors.h
        common/Irange.h
        common/Logger.h
        common/Math.h
        common/Profiler.h
        common/Offset.h
        common/OS.h
        common/Threadpool.h
        common/Timer.h

        common/io/IO.h
        common/io/IO.inl
        common/io/TextFile.h
        common/io/TextFile.inl
        common/io/header/Header.h
        common/io/header/MRCHeader.h
        common/io/header/TIFFHeader.h
        common/io/BinaryFile.h
        common/io/ImageFile.h
        common/io/ImageFile.inl
        common/io/Stats.h

        common/traits/BaseTypes.h
        common/traits/Containers.h
        common/Traits.h

        common/string/Convert.h
        common/string/Format.h
        common/string/Parse.h
        common/String.h

        common/geometry/Euler.h
        common/geometry/Euler.cpp
        common/geometry/Symmetry.h
        common/geometry/Transform.h

        common/types/BoolX.h
        common/types/Bool2.h
        common/types/Bool3.h
        common/types/Bool4.h
        common/types/ClampCast.h
        common/types/Complex.h
        common/types/Constants.h
        common/types/FloatX.h
        common/types/Float2.h
        common/types/Float3.h
        common/types/Float4.h
        common/types/Half.h
        common/types/IntX.h
        common/types/Int2.h
        common/types/Int3.h
        common/types/Int4.h
        common/types/Mat22.h
        common/types/Mat23.h
        common/types/Mat33.h
        common/types/Mat34.h
        common/types/Mat44.h
        common/types/MatX.h
        common/Types.h
        )

set(NOA_SOURCES
        Session.cpp

        common/Exception.cpp
        common/Logger.cpp

        common/io/IO.cpp
        common/io/BinaryFile.cpp
        common/io/header/MRCHeader.cpp
        common/io/header/TIFFHeader.cpp

        common/types/Constants.cpp
        common/string/Convert.cpp
        common/geometry/Symmetry.cpp
        )

set(NOA_CPU_HEADERS
        # noa::cpu::fft
        cpu/fft/Transforms.h
        cpu/fft/Plan.h
        cpu/fft/Resize.h
        cpu/fft/Remap.h
        cpu/fft/Filters.h

        # noa::cpu::math
        cpu/math/Complex.h
        cpu/math/Ewise.h
        cpu/math/Ewise.inl
        cpu/math/Find.h
        cpu/math/Reduce.h
        cpu/math/Reduce.inl

        # noa::cpu::filter
        cpu/filter/Shape.h
        cpu/filter/Median.h
        cpu/filter/Convolve.h

        # noa::cpu::memory
        cpu/memory/Arange.h
        cpu/memory/Cast.h
        cpu/memory/Copy.h
        cpu/memory/PtrHost.h
        cpu/memory/Index.h
        cpu/memory/Index.inl
        cpu/memory/Linspace.h
        cpu/memory/Resize.h
        cpu/memory/Set.h
        cpu/memory/Transpose.h

        # noa::cpu::geometry
        cpu/geometry/Interpolate.h
        cpu/geometry/Interpolator.h
        cpu/geometry/Prefilter.h
        cpu/geometry/Rotate.h
        cpu/geometry/Scale.h
        cpu/geometry/Apply.h
        cpu/geometry/Shift.h
        cpu/geometry/Symmetry.h

#        # noa::cpu::transform::fft
#        cpu/transform/fft/Apply.h
#        cpu/transform/fft/Symmetry.h
#        cpu/transform/fft/Shift.h
#
#        # noa::cpu::reconstruct
#        cpu/reconstruct/ProjectBackward.h
#        cpu/reconstruct/ProjectForward.h
#
        # noa::cpu
        cpu/Stream.h
        cpu/Event.h

        )

set(NOA_CPU_SOURCES
        # noa::cpu::fft
        cpu/fft/Plan.cpp
        cpu/fft/Filters.cpp
        cpu/fft/Remap.cpp
        cpu/fft/Resize.cpp

        # noa::cpu::math
        cpu/math/Reduce.cpp
        cpu/math/Find.cpp

        # noa::cpu::filter
        cpu/filter/ShapeSphere.cpp
        cpu/filter/ShapeRectangle.cpp
        cpu/filter/ShapeCylinder.cpp
        cpu/filter/Median.cpp
        cpu/filter/Convolve.cpp

        # noa::cpu::memory
        cpu/memory/Index.cpp
        cpu/memory/Resize.cpp
        cpu/memory/Transpose.cpp

        # noa::cpu::geometry
        cpu/geometry/Apply.cpp
        cpu/geometry/ApplySymmetry.cpp
        cpu/geometry/Prefilter.cpp
        cpu/geometry/Shift.cpp
        cpu/geometry/Symmetry.cpp

#        # noa::cpu::transform::fft
#        cpu/transform/fft/Apply.cpp
#        cpu/transform/fft/ApplySymmetry.cpp
#        cpu/transform/fft/Shift.cpp
#
#        # noa::cpu::reconstruct
#        cpu/reconstruct/ProjectBackward.cpp
#        cpu/reconstruct/ProjectForward.cpp

        )

set(NOA_CUDA_HEADERS
        # noa::cuda
        gpu/cuda/Exception.h
        gpu/cuda/Types.h
        gpu/cuda/Device.h
        gpu/cuda/Event.h
        gpu/cuda/Stream.h

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
        gpu/cuda/memory/Linspace.h
        gpu/cuda/memory/MemoryPool.h
        gpu/cuda/memory/PtrArray.h
        gpu/cuda/memory/PtrDevice.h
        gpu/cuda/memory/PtrDevicePadded.h
        gpu/cuda/memory/PtrManaged.h
        gpu/cuda/memory/PtrPinned.h
        gpu/cuda/memory/PtrTexture.h
        gpu/cuda/memory/Index.h
        gpu/cuda/memory/Resize.h
        gpu/cuda/memory/Set.h
        gpu/cuda/memory/Transpose.h

        # noa::cuda::fft
        gpu/cuda/fft/Exception.h
        gpu/cuda/fft/Filters.h
        gpu/cuda/fft/Plan.h
        gpu/cuda/fft/Transforms.h
        gpu/cuda/fft/Resize.h
        gpu/cuda/fft/Remap.h

        # noa::cuda::math
        gpu/cuda/math/Complex.h
        gpu/cuda/math/Ewise.h
        gpu/cuda/math/Find.h
        gpu/cuda/math/Reduce.h

        # noa::cuda::filter
        gpu/cuda/filter/Shape.h
        gpu/cuda/filter/Median.h
        gpu/cuda/filter/Convolve.h

#        # noa::cuda::transform
#        gpu/cuda/transform/Apply.h
#        gpu/cuda/transform/Interpolate.h
#        gpu/cuda/transform/Rotate.h
#        gpu/cuda/transform/Scale.h
#        gpu/cuda/transform/Symmetry.h
#        gpu/cuda/transform/Translate.h
#
#        # noa::cuda::transform::fft
#        gpu/cuda/transform/fft/Apply.h
#        gpu/cuda/transform/fft/Shift.h
#        gpu/cuda/transform/fft/Symmetry.h
#
#        # noa::cuda::reconstruct
#        gpu/cuda/reconstruct/ProjectBackward.h
#        gpu/cuda/reconstruct/ProjectForward.h

        )

set(NOA_CUDA_SOURCES
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
        gpu/cuda/memory/Transpose0213.cu
        gpu/cuda/memory/Transpose0132.cu
        gpu/cuda/memory/Transpose0312.cu
        gpu/cuda/memory/Transpose0231.cu
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
        gpu/cuda/filter/ShapeSphere.cu
        gpu/cuda/filter/ShapeRectangle.cu
        gpu/cuda/filter/ShapeCylinder.cu
        gpu/cuda/filter/Median.cu
        gpu/cuda/filter/Convolve1.cu
        gpu/cuda/filter/Convolve2.cu
        gpu/cuda/filter/Convolve3.cu
        gpu/cuda/filter/ConvolveSeparable.cu

#        # noa::cuda::transform
#        gpu/cuda/transform/Apply2D.cu
#        gpu/cuda/transform/Apply3D.cu
#        gpu/cuda/transform/ApplySymmetry.cu
#        gpu/cuda/transform/Interpolate.cu
#        gpu/cuda/transform/Symmetry.cu
#        gpu/cuda/transform/Translate2D.cu
#        gpu/cuda/transform/Translate3D.cu
#
#        # noa::cuda::transform::fft
#        gpu/cuda/transform/fft/Apply2D.cu
#        gpu/cuda/transform/fft/Apply3D.cu
#        gpu/cuda/transform/fft/Apply2DSymmetry.cu
#        gpu/cuda/transform/fft/Apply3DSymmetry.cu
#        gpu/cuda/transform/fft/Shift.cu
#
#        # noa::cuda::reconstruct
#        gpu/cuda/reconstruct/ProjectBackward.cu
#        gpu/cuda/reconstruct/ProjectForward.cu

        )

set(NOA_UNIFIED_HEADER

        )

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa.cmake)
