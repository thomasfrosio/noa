# ---------------------------------------------------------------------------------------
# The library source and header files
# ---------------------------------------------------------------------------------------
set(NOA_HEADERS
        Session.h

        common/Assert.h
        common/Definitions.h
        common/Environment.h
        common/Exception.h
        common/Logger.h
        common/Math.h
        common/Profiler.h
        common/OS.h
        common/Threadpool.h

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

        common/traits/BaseTypes.h
        common/traits/Containers.h
        common/Traits.h

        common/string/Convert.h
        common/string/Format.h
        common/string/Parse.h
        common/String.h

        common/transform/Euler.h
        common/transform/Geometry.h
        common/transform/Symmetry.h

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
        common/types/Index.h
        common/types/Stats.h
        common/Types.h

        common/types/details/Bool2.inl
        common/types/details/Bool3.inl
        common/types/details/Bool4.inl
        common/types/details/Complex.inl
        common/types/details/Float2.inl
        common/types/details/Float3.inl
        common/types/details/Float4.inl
        common/types/details/Int2.inl
        common/types/details/Int3.inl
        common/types/details/Int4.inl
        common/types/details/Mat22.inl
        common/types/details/Mat23.inl
        common/types/details/Mat33.inl
        common/types/details/Mat34.inl
        common/types/details/Mat44.inl

        )

set(NOA_SOURCES
        Session.cpp

        common/Logger.cpp
        common/io/IO.cpp
        common/io/BinaryFile.cpp
        common/io/header/MRCHeader.cpp
        common/io/header/TIFFHeader.cpp

        common/string/Convert.cpp
        common/transform/Symmetry.cpp

        )

set(NOA_CPU_HEADERS
        # noa::cpu::fft
        cpu/fft/Transforms.h
        cpu/fft/Plan.h
        cpu/fft/Resize.h
        cpu/fft/Remap.h
        cpu/fft/Filters.h

        # noa::cpu::math
        cpu/math/Arithmetics.h
        cpu/math/ArithmeticsComposite.h
        cpu/math/Booleans.h
        cpu/math/Generics.h
        cpu/math/Indexes.h
        cpu/math/Reductions.h

        # noa::cpu::filter
        cpu/filter/Shape.h
        cpu/filter/Median.h
        cpu/filter/Convolve.h

        # noa::cpu::memory
        cpu/memory/Copy.h
        cpu/memory/PtrHost.h
        cpu/memory/Index.h
        cpu/memory/Resize.h
        cpu/memory/Set.h
        cpu/memory/Transpose.h

        # noa::cpu::transform
        cpu/transform/Interpolate.h
        cpu/transform/Interpolator.h
        cpu/transform/Rotate.h
        cpu/transform/Scale.h
        cpu/transform/Apply.h
        cpu/transform/Symmetry.h
        cpu/transform/Translate.h

        # noa::cpu::transform::fft
        cpu/transform/fft/Rotate.h
        cpu/transform/fft/Scale.h
        cpu/transform/fft/Apply.h
        cpu/transform/fft/Symmetry.h
        cpu/transform/fft/Shift.h

        # noa::cpu::reconstruct
        cpu/reconstruct/ProjectBackward.h
        cpu/reconstruct/ProjectForward.h

        # noa::cpu
        cpu/Execution.h

        )

set(NOA_CPU_SOURCES
        # noa::cpu::fft
        cpu/fft/Plan.cpp
        cpu/fft/Filters.cpp
        cpu/fft/Remap.cpp
        cpu/fft/Resize.cpp

        # noa::cpu::math
        cpu/math/Reductions.cpp
        cpu/math/Indexes.cpp
        cpu/math/ArithmeticsComposite.cpp
        cpu/math/Arithmetics.cpp

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

        # noa::cpu::transform
        cpu/transform/Apply.cpp
        cpu/transform/ApplySymmetry.cpp
        cpu/transform/Interpolate.cpp
        cpu/transform/Translate.cpp
        cpu/transform/Symmetry.cpp

        # noa::cpu::transform::fft
        cpu/transform/fft/Apply.cpp
        cpu/transform/fft/Shift.cpp

        # noa::cpu::reconstruct
        cpu/reconstruct/ProjectBackward.cpp
        cpu/reconstruct/ProjectForward.cpp

        )

set(NOA_CUDA_HEADERS
        # noa::cuda
        gpu/cuda/Exception.h
        gpu/cuda/Types.h
        gpu/cuda/util/Atomic.h
        gpu/cuda/util/Device.h
        gpu/cuda/util/Event.h
        gpu/cuda/util/ExternShared.h
        gpu/cuda/util/Stream.h

        # noa::cuda::memory
        gpu/cuda/memory/Copy.h
        gpu/cuda/memory/PtrDevice.h
        gpu/cuda/memory/PtrPinned.h
        gpu/cuda/memory/PtrDevicePadded.h
        gpu/cuda/memory/PtrArray.h
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
        gpu/cuda/math/Arithmetics.h
        gpu/cuda/math/ArithmeticsComposite.h
        gpu/cuda/math/Booleans.h
        gpu/cuda/math/Generics.h
        gpu/cuda/math/Indexes.h
        gpu/cuda/math/Reductions.h

        # noa::cuda::filter
        gpu/cuda/filter/Shape.h
        gpu/cuda/filter/Median.h
        gpu/cuda/filter/Convolve.h

        # noa::cuda::transform
        gpu/cuda/transform/Apply.h
        gpu/cuda/transform/Interpolate.h
        gpu/cuda/transform/Rotate.h
        gpu/cuda/transform/Scale.h
        gpu/cuda/transform/Symmetry.h
        gpu/cuda/transform/Translate.h

        # noa::cuda::reconstruct
        gpu/cuda/reconstruct/ProjectBackward.h
        gpu/cuda/reconstruct/ProjectForward.h

        )

set(NOA_CUDA_SOURCES
        # noa::cuda::fft
        gpu/cuda/fft/Filters.cu
        gpu/cuda/fft/Plan.cpp
        gpu/cuda/fft/Remap.cu
        gpu/cuda/fft/Resize.cu

        # noa::cuda::math
        gpu/cuda/math/Arithmetics.cu
        gpu/cuda/math/ArithmeticsComposite.cu
        gpu/cuda/math/Booleans.cu
        gpu/cuda/math/Generics.cu
        gpu/cuda/math/Indexes.cu
        gpu/cuda/math/reductions/Min_Max_SumMean.cu
        gpu/cuda/math/reductions/MinMaxSumMean.cu
        gpu/cuda/math/reductions/VarianceStddev.cu
        gpu/cuda/math/reductions/ReduceAdd.cu
        gpu/cuda/math/reductions/MinMax.cu

        # noa::cuda::filter
        gpu/cuda/filter/ShapeSphere.cu
        gpu/cuda/filter/ShapeRectangle.cu
        gpu/cuda/filter/ShapeCylinder.cu
        gpu/cuda/filter/Median.cu
        gpu/cuda/filter/Convolve1.cu
        gpu/cuda/filter/Convolve2.cu
        gpu/cuda/filter/Convolve3.cu
        gpu/cuda/filter/ConvolveSeparable.cu

        # noa::cuda::memory
        gpu/cuda/memory/Index.cu
        gpu/cuda/memory/Resize.cu
        gpu/cuda/memory/Set.cu
        gpu/cuda/memory/transpose/Transpose021.cu
        gpu/cuda/memory/transpose/Transpose102.cu
        gpu/cuda/memory/transpose/Transpose120.cu
        gpu/cuda/memory/transpose/Transpose201.cu
        gpu/cuda/memory/transpose/Transpose210.cu

        # noa::cuda::transform
        gpu/cuda/transform/Apply2D.cu
        gpu/cuda/transform/Apply3D.cu
        gpu/cuda/transform/ApplySymmetry.cu
        gpu/cuda/transform/Interpolate.cu
        gpu/cuda/transform/Symmetry.cu
        gpu/cuda/transform/Translate2D.cu
        gpu/cuda/transform/Translate3D.cu

        # noa::cuda::reconstruct
        gpu/cuda/reconstruct/ProjectBackward.cu
        gpu/cuda/reconstruct/ProjectForward.cu

        )

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa.cmake)
