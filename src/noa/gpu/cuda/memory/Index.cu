#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"
#include "noa/common/Math.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/memory/Index.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/util/Pointers.h"

namespace {
    using namespace noa;
    constexpr uint BLOCK_SIZE = 256;
    constexpr dim3 BLOCK_SIZE_2D(32, BLOCK_SIZE / 32);
    constexpr uint ELEMENTS_PER_THREAD = 4;
    constexpr dim3 BLOCK_WORK_SIZE_2D(BLOCK_SIZE_2D.x * ELEMENTS_PER_THREAD, BLOCK_SIZE_2D.y);

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void extractOrNothing_(const T* __restrict__ input, uint4_t input_stride, int4_t input_shape,
                           T* __restrict__ subregions, uint4_t subregion_stride, int2_t subregion_shape,
                           const int4_t* __restrict__ origins, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        if (gid[2] >= subregion_shape[0])
            return;

        const int4_t origin = origins[gid[0]]; // TODO constant memory?
        const int ii = origin[0];
        const int ij = origin[1] + gid[1];
        const int ik = origin[2] + gid[2];
        if (ii < 0 || ii >= input_shape[0] ||
            ij < 0 || ij >= input_shape[1] ||
            ik < 0 || ik >= input_shape[2])
            return;

        input += indexing::at(ii, ij, ik, input_stride);
        subregions += indexing::at(gid[0], gid[1], gid[2], subregion_stride);

        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int ol = gid[3] + static_cast<int>(BLOCK_SIZE_2D.x) * i;
            const int il = origin[3] + ol;
            if (ol < subregion_shape[1] && il >= 0 && il < input_shape[3])
                subregions[ol * subregion_stride[3]] = input[il * input_stride[3]];
        }
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void extractOrValue_(const T* __restrict__ input, uint4_t input_stride, int4_t input_shape,
                         T* __restrict__ subregions, uint4_t subregion_stride, int2_t subregion_shape,
                         const int4_t* __restrict__ origins, T value, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        if (gid[2] >= subregion_shape[0])
            return;

        const int4_t origin = origins[gid[0]]; // TODO constant memory?
        const int ii = origin[0];
        const int ij = origin[1] + gid[1];
        const int ik = origin[2] + gid[2];
        const bool is_in = ii >= 0 && ii < input_shape[0] &&
                           ij >= 0 && ij < input_shape[1] &&
                           ik >= 0 && ik < input_shape[2];

        subregions += indexing::at(gid[0], gid[1], gid[2], subregion_stride);
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int ol = gid[3] + static_cast<int>(BLOCK_SIZE_2D.x) * i;
            if (ol >= subregion_shape[1])
                return;

            const int il = origin[3] + ol;
            if (is_in && il >= 0 && il < input_shape[3])
                subregions[ol * subregion_stride[3]] = input[indexing::at(ii, ij, ik, il, input_stride)];
            else
                subregions[ol * subregion_stride[3]] = value;
        }
    }

    template<BorderMode MODE, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void extract_(const T* __restrict__ input, uint4_t input_stride, int4_t input_shape,
                  T* __restrict__ subregions, uint4_t subregion_stride, int2_t subregion_shape,
                  const int4_t* __restrict__ origins, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        if (gid[2] >= subregion_shape[0])
            return;

        const int4_t origin = origins[gid[0]]; // TODO constant memory?
        const int ii = getBorderIndex<MODE>(origin[0], input_shape[0]);
        const int ij = getBorderIndex<MODE>(origin[1] + gid[1], input_shape[1]);
        const int ik = getBorderIndex<MODE>(origin[2] + gid[2], input_shape[2]);

        input += indexing::at(ii, ij, ik, input_stride);
        subregions += indexing::at(gid[0], gid[1], gid[2], subregion_stride);

        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int ol = gid[2] + static_cast<int>(BLOCK_SIZE_2D.x) * i;
            const int il = getBorderIndex<MODE>(origin[3] + ol, input_shape[3]);
            if (ol < subregion_shape[1])
                subregions[ol * subregion_stride[3]] = input[il * input_stride[3]];
        }
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void insert_(const T* __restrict__ subregions, uint4_t subregion_stride, int2_t subregion_shape,
                 T* __restrict__ output, uint4_t output_stride, int4_t output_shape,
                 const int4_t* __restrict__ origins, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        if (gid[2] >= subregion_shape[0])
            return;

        const int4_t origin = origins[gid[0]]; // TODO constant memory?
        const int oi = origin[0];
        const int oj = origin[1] + gid[1];
        const int ok = origin[2] + gid[2];
        if (oi < 0 || oi >= output_shape[0] ||
            oj < 0 || oj >= output_shape[1] ||
            ok < 0 || ok >= output_shape[2])
            return;

        output += indexing::at(oi, oj, ok, output_stride);
        subregions += indexing::at(gid[0], gid[1], gid[2], subregion_stride);

        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int il = gid[3] + static_cast<int>(BLOCK_SIZE_2D.x) * i;
            const int ol = origin[3] + il;
            if (il < subregion_shape[1] && ol >= 0 && ol < output_shape[3])
                output[ol * output_stride[3]] = subregions[il * subregion_stride[3]];
        }
    }
}

namespace noa::cuda::memory {
    template<typename T>
    void extract(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& subregions, size4_t subregion_stride, size4_t subregion_shape,
                 const shared_t<int4_t[]>& origins, BorderMode border_mode, T border_value, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != subregions);

        const shared_t<int4_t[]> d_origins = util::ensureDeviceAccess(origins, stream, subregion_shape[0]);
        const int4_t i_shape(input_shape);
        const int2_t o_shape(subregion_shape.get() + 2);

        const uint blocks_x = math::divideUp(static_cast<uint>(o_shape[1]), BLOCK_WORK_SIZE_2D.x);
        const uint blocks_y = math::divideUp(static_cast<uint>(o_shape[0]), BLOCK_WORK_SIZE_2D.y);
        const dim3 blocks(blocks_x * blocks_y, subregion_shape[1], subregion_shape[0]);
        switch (border_mode) {
            case BORDER_NOTHING:
                return stream.enqueue("memory::extractOrNothing", extractOrNothing_<T>, {blocks, BLOCK_SIZE_2D},
                                      input.get(), uint4_t{input_stride}, i_shape,
                                      subregions.get(), uint4_t{subregion_stride},
                                      o_shape, d_origins.get(), blocks_x);
            case BORDER_ZERO:
                return stream.enqueue("memory::extractOrValue", extractOrValue_<T>, {blocks, BLOCK_SIZE_2D},
                                      input.get(), uint4_t{input_stride}, i_shape,
                                      subregions.get(), uint4_t{subregion_stride},
                                      o_shape, d_origins.get(), static_cast<T>(0), blocks_x);
            case BORDER_VALUE:
                return stream.enqueue("memory::extractOrValue", extractOrValue_<T>, {blocks, BLOCK_SIZE_2D},
                                      input.get(), uint4_t{input_stride}, i_shape,
                                      subregions.get(), uint4_t{subregion_stride},
                                      o_shape, d_origins.get(), border_value, blocks_x);
            case BORDER_CLAMP:
                return stream.enqueue("memory::extract<CLAMP>", extract_<BORDER_CLAMP, T>, {blocks, BLOCK_SIZE_2D},
                                      input.get(), uint4_t{input_stride}, i_shape,
                                      subregions.get(), uint4_t{subregion_stride},
                                      o_shape, d_origins.get(), blocks_x);
            case BORDER_MIRROR:
                return stream.enqueue("memory::extract<MIRROR>", extract_<BORDER_MIRROR, T>, {blocks, BLOCK_SIZE_2D},
                                      input.get(), uint4_t{input_stride}, i_shape,
                                      subregions.get(), uint4_t{subregion_stride},
                                      o_shape, d_origins.get(), blocks_x);
            case BORDER_REFLECT:
                return stream.enqueue("memory::extract<REFLECT>", extract_<BORDER_REFLECT, T>, {blocks, BLOCK_SIZE_2D},
                                      input.get(), uint4_t{input_stride}, i_shape,
                                      subregions.get(), uint4_t{subregion_stride},
                                      o_shape, d_origins.get(), blocks_x);
            default:
                NOA_THROW("Border mode {} is not supported", border_mode);
        }
        stream.attach(input, subregions, d_origins);
    }

    template<typename T>
    void insert(const shared_t<T[]>& subregions, size4_t subregion_stride, size4_t subregion_shape,
                const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                const shared_t<int4_t[]>& origins, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(subregions != output);

        const shared_t<int4_t[]> d_origins = util::ensureDeviceAccess(origins, stream, subregion_shape[0]);
        const int2_t i_shape{subregion_shape.get() + 2};
        const uint blocks_x = math::divideUp(static_cast<uint>(i_shape[1]), BLOCK_WORK_SIZE_2D.x);
        const uint blocks_y = math::divideUp(static_cast<uint>(i_shape[0]), BLOCK_WORK_SIZE_2D.y);
        const dim3 blocks(blocks_x * blocks_y, subregion_shape[1], subregion_shape[0]);
        stream.enqueue("memory::insert", insert_<T>, {blocks, BLOCK_SIZE_2D},
                       subregions.get(), uint4_t{subregion_stride}, i_shape,
                       output.get(), uint4_t{output_stride}, int4_t{output_shape}, d_origins.get(), blocks_x);
        stream.attach(subregions, output, d_origins);
    }

    #define INSTANTIATE_EXTRACT_INSERT_(T)                                                                                                                          \
    template void extract<T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<int4_t[]>&, BorderMode, T, Stream&);    \
    template void insert<T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<int4_t[]>&, Stream&)

    INSTANTIATE_EXTRACT_INSERT_(int8_t);
    INSTANTIATE_EXTRACT_INSERT_(int16_t);
    INSTANTIATE_EXTRACT_INSERT_(int32_t);
    INSTANTIATE_EXTRACT_INSERT_(int64_t);
    INSTANTIATE_EXTRACT_INSERT_(uint8_t);
    INSTANTIATE_EXTRACT_INSERT_(uint16_t);
    INSTANTIATE_EXTRACT_INSERT_(uint32_t);
    INSTANTIATE_EXTRACT_INSERT_(uint64_t);
    INSTANTIATE_EXTRACT_INSERT_(half_t);
    INSTANTIATE_EXTRACT_INSERT_(float);
    INSTANTIATE_EXTRACT_INSERT_(double);
    INSTANTIATE_EXTRACT_INSERT_(chalf_t);
    INSTANTIATE_EXTRACT_INSERT_(cfloat_t);
    INSTANTIATE_EXTRACT_INSERT_(cdouble_t);

    // This a copied from noa/cpu/memory/Index.inl
    size4_t atlasLayout(size4_t subregion_shape, int4_t* origins) {
        const auto col = static_cast<size_t>(math::ceil(math::sqrt(static_cast<float>(subregion_shape[0]))));
        const size_t row = (subregion_shape[0] + col - 1) / col;
        const size4_t atlas_shape{1, subregion_shape[1], row * subregion_shape[2], col * subregion_shape[3]};
        for (size_t y = 0; y < row; ++y) {
            for (size_t x = 0; x < col; ++x) {
                const size_t idx = y * col + x;
                if (idx >= subregion_shape[0])
                    break;
                origins[idx] = {0, 0, y * subregion_shape[2], x * subregion_shape[3]};
            }
        }
        return atlas_shape;
    }
}
