# ---------------------------------------------------------------------------------------
# The library source and header files
# ---------------------------------------------------------------------------------------
set(NOA_SOURCES
        Logger.cpp
        Session.cpp

        # IO
        io/files/MRCFile.cpp
        io/files/ImageFile.cpp
        io/Inputs.cpp
        io/IO.cpp

        util/string/Convert.cpp

        )

set(NOA_HEADERS
        Angles.h
        Assert.h
        Definitions.h
        Environment.h
        Exception.h
        Logger.h
        Math.h
        Profiler.h
        Session.h
        Transform.h
        Types.h

        # IO
        io/files/ImageFile.h
        io/files/MRCFile.h
        io/files/TextFile.h
        io/Inputs.h
        io/IO.h
        io/OS.h

        # Utilities
        util/BoolX.h
        util/Bool2.h
        util/Bool3.h
        util/Bool4.h
        util/Complex.h
        util/Constants.h
        util/FloatX.h
        util/Float2.h
        util/Float3.h
        util/Float4.h
        util/IntX.h
        util/Int2.h
        util/Int3.h
        util/Int4.h
        util/Mat2.h
        util/Mat3.h
        util/Mat4.h
        util/MatX.h
        util/Traits.h
        util/traits/BaseTypes.h
        util/traits/Containers.h
        util/Sizes.h
        util/Stats.h
        util/String.h
        util/string/Convert.h
        util/string/Format.h
        util/string/Parse.h

        )

set(NOA_CPU_SOURCES
        cpu/fourier/Plan.cpp
        cpu/fourier/Filters.cpp
        cpu/fourier/Remap.cpp
        cpu/fourier/Resize.cpp

        cpu/math/Reductions.cpp
        cpu/math/Indexes.cpp
        cpu/math/ArithmeticsComposite.cpp
        cpu/math/Arithmetics.cpp

        cpu/mask/Sphere.cpp
        cpu/mask/Rectangle.cpp
        cpu/mask/Cylinder.cpp

        cpu/filter/Median.cpp
        cpu/filter/Convolve.cpp

        cpu/memory/Remap.cpp
        cpu/memory/Resize.cpp

        )

set(NOA_CPU_HEADERS
        # noa::fourier
        cpu/fourier/Transforms.h
        cpu/fourier/Plan.h
        cpu/fourier/Resize.h
        cpu/fourier/Remap.h
        cpu/fourier/Filters.h

        # noa::math
        cpu/math/Arithmetics.h
        cpu/math/ArithmeticsComposite.h
        cpu/math/Booleans.h
        cpu/math/Generics.h
        cpu/math/Indexes.h
        cpu/math/Reductions.h

        # noa::mask
        cpu/mask/Sphere.h
        cpu/mask/Rectangle.h
        cpu/mask/Cylinder.h

        # noa::filter
        cpu/filter/Median.h
        cpu/filter/Convolve.h

        # noa::memory
        cpu/memory/Copy.h
        cpu/memory/PtrHost.h
        cpu/memory/Remap.h
        cpu/memory/Resize.h
        cpu/memory/Set.h

        # noa
        cpu/Threadpool.h
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

        # noa::cuda::mask
        gpu/cuda/mask/Sphere.cu
        gpu/cuda/mask/Rectangle.cu
        gpu/cuda/mask/Cylinder.cu

        # noa::cuda::filter
        gpu/cuda/filter/Median.cu
        gpu/cuda/filter/convolve/Convolve1.cu
        gpu/cuda/filter/convolve/Convolve2.cu
        gpu/cuda/filter/convolve/Convolve3.cu
        gpu/cuda/filter/convolve/ConvolveSeparable.cu

        # noa::cuda::memory
        gpu/cuda/memory/Remap.cu
        gpu/cuda/memory/Resize.cu
        gpu/cuda/memory/Set.cu

        )

set(NOA_CUDA_HEADERS
        # noa::cuda
        gpu/cuda/Exception.h
        gpu/cuda/Types.h
        gpu/cuda/util/Device.h
        gpu/cuda/util/Event.h
        gpu/cuda/util/Stream.h
        gpu/cuda/util/Sizes.h

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
        gpu/cuda/memory/Shared.h

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

        # noa::cuda::mask
        gpu/cuda/mask/Sphere.h
        gpu/cuda/mask/Cylinder.h
        gpu/cuda/mask/Rectangle.h

        # noa::cuda::filter
        gpu/cuda/filter/Median.h
        gpu/cuda/filter/Convolve.h

        )

include(${PROJECT_SOURCE_DIR}/cmake/targets/noa.cmake)
