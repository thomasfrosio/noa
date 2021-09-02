# ---------------------------------------------------------------------------------------
# The library source and header files
# ---------------------------------------------------------------------------------------
set(NOA_SOURCES
        Session.cpp

        common/Logger.cpp
        common/files/BinFile.cpp
        common/files/MRCFile.cpp
        common/files/ImageFile.cpp
        common/Inputs.cpp
        common/IO.cpp
        common/string/Convert.cpp

        )

set(NOA_HEADERS
        Session.h

        common/Assert.h
        common/Definitions.h
        common/Environment.h
        common/Exception.h
        common/Logger.h
        common/Math.h
        common/Profiler.h
        common/Inputs.h
        common/IO.h
        common/OS.h
        common/Threadpool.h

        common/files/BinFile.h
        common/files/ImageFile.h
        common/files/MRCFile.h
        common/files/TextFile.h

        common/traits/BaseTypes.h
        common/traits/Containers.h
        common/Traits.h

        common/string/Convert.h
        common/string/Format.h
        common/string/Parse.h
        common/String.h

        common/transform/Euler.h
        common/transform/Geometry.h

        common/types/BoolX.h
        common/types/Bool2.h
        common/types/Bool3.h
        common/types/Bool4.h
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
        common/types/Sizes.h
        common/types/Stats.h
        common/Types.h

        )

set(NOA_CPU_SOURCES
        # noa::cpu::fourier
        cpu/fourier/Plan.cpp
        cpu/fourier/Filters.cpp
        cpu/fourier/Remap.cpp
        cpu/fourier/Resize.cpp

        # noa::cpu::math
        cpu/math/Reductions.cpp
        cpu/math/Indexes.cpp
        cpu/math/ArithmeticsComposite.cpp
        cpu/math/Arithmetics.cpp

        # noa::cpu::filter
        cpu/filter/Sphere.cpp
        cpu/filter/Rectangle.cpp
        cpu/filter/Cylinder.cpp
        cpu/filter/Median.cpp
        cpu/filter/Convolve.cpp

        # noa::cpu::memory
        cpu/memory/Remap.cpp
        cpu/memory/Resize.cpp
        cpu/memory/Transpose.cpp

        # noa::cpu::transform
        cpu/transform/Apply.cpp
        cpu/transform/Interpolate.cpp
        cpu/transform/Translate.cpp

        # noa::cpu::project
        cpu/project/Backward.cpp
        cpu/project/Forward.cpp

        )

set(NOA_CPU_HEADERS
        # noa::cpu::fourier
        cpu/fourier/Transforms.h
        cpu/fourier/Plan.h
        cpu/fourier/Resize.h
        cpu/fourier/Remap.h
        cpu/fourier/Filters.h

        # noa::cpu::math
        cpu/math/Arithmetics.h
        cpu/math/ArithmeticsComposite.h
        cpu/math/Booleans.h
        cpu/math/Generics.h
        cpu/math/Indexes.h
        cpu/math/Reductions.h

        # noa::cpu::filter
        cpu/filter/Sphere.h
        cpu/filter/Rectangle.h
        cpu/filter/Cylinder.h
        cpu/filter/Median.h
        cpu/filter/Convolve.h

        # noa::cpu::memory
        cpu/memory/Copy.h
        cpu/memory/PtrHost.h
        cpu/memory/Remap.h
        cpu/memory/Resize.h
        cpu/memory/Set.h
        cpu/memory/Transpose.h

        # noa::cpu::transform
        cpu/transform/Interpolate.h
        cpu/transform/Interpolator.h
        cpu/transform/Rotate.h
        cpu/transform/Scale.h
        cpu/transform/Apply.h
        cpu/transform/Translate.h

        # noa::cpu::project
        cpu/project/Backward.h
        cpu/project/Forward.h

        # noa::cpu
        cpu/Execution.h

        )

set(NOA_CUDA_SOURCES
        # noa::cuda::fourier
        gpu/cuda/fourier/Filters.cu
        gpu/cuda/fourier/Remap.cu
        gpu/cuda/fourier/Resize.cu

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
        gpu/cuda/filter/Sphere.cu
        gpu/cuda/filter/Rectangle.cu
        gpu/cuda/filter/Cylinder.cu
        gpu/cuda/filter/Median.cu
        gpu/cuda/filter/Convolve1.cu
        gpu/cuda/filter/Convolve2.cu
        gpu/cuda/filter/Convolve3.cu
        gpu/cuda/filter/ConvolveSeparable.cu

        # noa::cuda::memory
        gpu/cuda/memory/Remap.cu
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
        gpu/cuda/transform/Interpolate.cu
        gpu/cuda/transform/Translate2D.cu
        gpu/cuda/transform/Translate3D.cu

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
        gpu/cuda/memory/Remap.h
        gpu/cuda/memory/Resize.h
        gpu/cuda/memory/Set.h
        gpu/cuda/memory/Transpose.h

        # noa::cuda::fourier
        gpu/cuda/fourier/Exception.h
        gpu/cuda/fourier/Filters.h
        gpu/cuda/fourier/Plan.h
        gpu/cuda/fourier/Transforms.h
        gpu/cuda/fourier/Resize.h
        gpu/cuda/fourier/Remap.h

        # noa::cuda::math
        gpu/cuda/math/Arithmetics.h
        gpu/cuda/math/ArithmeticsComposite.h
        gpu/cuda/math/Booleans.h
        gpu/cuda/math/Generics.h
        gpu/cuda/math/Indexes.h
        gpu/cuda/math/Reductions.h

        # noa::cuda::filter
        gpu/cuda/filter/Sphere.h
        gpu/cuda/filter/Cylinder.h
        gpu/cuda/filter/Rectangle.h
        gpu/cuda/filter/Median.h
        gpu/cuda/filter/Convolve.h

        # noa::cuda::transform
        gpu/cuda/transform/Apply.h
        gpu/cuda/transform/Interpolate.h
        gpu/cuda/transform/Rotate.h
        gpu/cuda/transform/Scale.h
        gpu/cuda/transform/Translate.h

        )

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa.cmake)
