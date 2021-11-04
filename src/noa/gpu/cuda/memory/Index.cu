#include <memory>

#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/Index.h"

namespace {
    using namespace noa;
    constexpr uint MAX_THREADS = 256;
    constexpr dim3 THREADS(32, MAX_THREADS / 32); // and kernels using 2D blocks are computing 2 elements per thread.

    __forceinline__ __host__ __device__ int3_t getCornerLeft_(int3_t subregion_shape, size3_t subregion_center) {
        return int3_t(subregion_center) - subregion_shape / 2;
    }

    template<typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void extractOrNothing_(const T* __restrict__ input, uint input_pitch, int3_t input_shape,
                           T* __restrict__ subregions, uint subregion_pitch, int3_t subregion_shape,
                           const size3_t* __restrict__ subregion_centers, uint blocks_x) {
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const int3_t gid(THREADS.x * idx.x * 2 + threadIdx.x,
                         THREADS.y * idx.y + threadIdx.y,
                         blockIdx.y);
        if (gid.y >= subregion_shape.y)
            return;

        const int3_t corner_left = getCornerLeft_(subregion_shape, subregion_centers[blockIdx.z]);
        const int i_y = corner_left.y + gid.y;
        const int i_z = corner_left.z + gid.z;
        if (i_z < 0 || i_z >= input_shape.z || i_y < 0 || i_y >= input_shape.y)
            return;

        input += noa::index<uint>(i_y, i_z, input_pitch, input_shape.y);
        subregions += noa::index<uint>(gid.y, gid.z, subregion_pitch, subregion_shape.y);
        subregions += blockIdx.z * subregion_pitch * rows(subregion_shape);

        for (int i = 0; i < 2; ++i) {
            const int o_x = gid.x + static_cast<int>(THREADS.x) * i;
            const int i_x = corner_left.x + o_x;
            if (o_x < subregion_shape.x && i_x >= 0 && i_x < input_shape.x)
                subregions[o_x] = input[i_x];
        }
    }

    template<typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void extractOrNothing_(const T* __restrict__ input, uint input_pitch, int3_t input_shape,
                           T* __restrict__ subregion, uint subregion_pitch, int3_t subregion_shape,
                           int3_t corner_left) {
        const int3_t gid(THREADS.x * blockIdx.x * 2 + threadIdx.x,
                         THREADS.y * blockIdx.y + threadIdx.y,
                         blockIdx.z);
        const int i_y = corner_left.y + gid.y;
        const int i_z = corner_left.z + gid.z;
        if (gid.y >= subregion_shape.y ||
            i_z < 0 || i_z >= input_shape.z || i_y < 0 || i_y >= input_shape.y)
            return;

        input += noa::index<uint>(i_y, i_z, input_pitch, input_shape.y);
        subregion += noa::index<uint>(gid.y, gid.z, subregion_pitch, subregion_shape.y);
        for (int i = 0; i < 2; ++i) {
            const int o_x = gid.x + static_cast<int>(THREADS.x) * i;
            const int i_x = corner_left.x + o_x;
            if (o_x < subregion_shape.x && i_x >= 0 && i_x < input_shape.x)
                subregion[o_x] = input[i_x];
        }
    }

    template<typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void extractOrValue_(const T* __restrict__ input, uint input_pitch, int3_t input_shape,
                         T* __restrict__ subregions, uint subregion_pitch, int3_t subregion_shape,
                         const size3_t* __restrict__ subregion_centers, T value, uint blocks_x) {
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const int3_t gid(THREADS.x * idx.x * 2 + threadIdx.x,
                         THREADS.y * idx.y + threadIdx.y,
                         blockIdx.y);
        if (gid.y >= subregion_shape.y)
            return;

        int3_t corner_left = getCornerLeft_(subregion_shape, subregion_centers[blockIdx.z]);

        int i_y = corner_left.y + gid.y;
        int i_z = corner_left.z + gid.z;
        bool is_in = i_z >= 0 && i_z < input_shape.z && i_y >= 0 && i_y < input_shape.y;

        subregions += blockIdx.z * subregion_pitch * rows(subregion_shape);
        subregions += noa::index<uint>(gid.y, gid.z, subregion_pitch, subregion_shape.y);
        for (int i = 0; i < 2; ++i) {
            const int o_x = gid.x + static_cast<int>(THREADS.x) * i;
            if (o_x >= subregion_shape.x)
                return;

            const int i_x = corner_left.x + o_x;
            if (is_in && i_x >= 0 && i_x < input_shape.x)
                subregions[o_x] = input[noa::index<uint>(i_x, i_y, i_z, input_pitch, input_shape.y)];
            else
                subregions[o_x] = value;
        }
    }

    template<typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void extractOrValue_(const T* __restrict__ input, uint input_pitch, int3_t input_shape,
                         T* __restrict__ subregion, uint subregion_pitch, int3_t subregion_shape,
                         int3_t corner_left, T value) {
        const int3_t gid(THREADS.x * blockIdx.x * 2 + threadIdx.x,
                         THREADS.y * blockIdx.y + threadIdx.y,
                         blockIdx.z);
        if (gid.y >= subregion_shape.y)
            return;

        const int i_y = corner_left.y + gid.y;
        const int i_z = corner_left.z + gid.z;
        const bool is_in = i_z >= 0 && i_z < input_shape.z && i_y >= 0 && i_y < input_shape.y;

        subregion += noa::index<uint>(gid.y, gid.z, subregion_pitch, subregion_shape.y);
        for (int i = 0; i < 2; ++i) {
            const int o_x = gid.x + static_cast<int>(THREADS.x) * i;
            if (o_x >= subregion_shape.x)
                return;

            const int i_x = corner_left.x + o_x;
            if (is_in && i_x >= 0 && i_x < input_shape.x)
                subregion[o_x] = input[noa::index<uint>(i_y, i_z, input_pitch, input_shape.y) + i_x];
            else
                subregion[o_x] = value;
        }
    }

    template<BorderMode MODE, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void extract_(const T* __restrict__ input, uint input_pitch, int3_t input_shape,
                  T* __restrict__ subregions, uint subregion_pitch, int3_t subregion_shape,
                  const size3_t* __restrict__ subregion_centers, uint blocks_x) {
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const int3_t gid(THREADS.x * idx.x * 2 + threadIdx.x,
                         THREADS.y * idx.y + threadIdx.y,
                         blockIdx.y);
        if (gid.y >= subregion_shape.y)
            return;

        const int3_t corner_left = getCornerLeft_(subregion_shape, subregion_centers[blockIdx.z]);
        const int i_y = getBorderIndex<MODE>(corner_left.y + gid.y, input_shape.y);
        const int i_z = getBorderIndex<MODE>(corner_left.z + gid.z, input_shape.z);

        subregions += blockIdx.z * subregion_pitch * rows(subregion_shape);
        subregions += noa::index<uint>(gid.y, gid.z, subregion_pitch, subregion_shape.y);
        input += noa::index<uint>(i_y, i_z, input_pitch, input_shape.y);
        for (int i = 0; i < 2; ++i) {
            const int o_x = gid.x + static_cast<int>(THREADS.x) * i;
            const int i_x = getBorderIndex<MODE>(corner_left.x + o_x, input_shape.x);
            if (o_x < subregion_shape.x)
                subregions[o_x] = input[i_x];
        }
    }

    template<BorderMode MODE, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void extract_(const T* __restrict__ input, uint input_pitch, int3_t input_shape,
                  T* __restrict__ subregion, uint subregion_pitch, int3_t subregion_shape,
                  int3_t corner_left) {
        const int3_t gid(THREADS.x * blockIdx.x * 2 + threadIdx.x,
                         THREADS.y * blockIdx.y + threadIdx.y,
                         blockIdx.z);
        if (gid.y >= subregion_shape.y)
            return;

        const int i_y = getBorderIndex<MODE>(corner_left.y + gid.y, input_shape.y);
        const int i_z = getBorderIndex<MODE>(corner_left.z + gid.z, input_shape.z);

        subregion += noa::index<uint>(gid.y, gid.z, subregion_pitch, subregion_shape.y);
        input += noa::index<uint>(i_y, i_z, input_pitch, input_shape.y);
        for (int i = 0; i < 2; ++i) {
            const int o_x = gid.x + static_cast<int>(THREADS.x) * i;
            const int i_x = getBorderIndex<MODE>(corner_left.x + o_x, input_shape.x);
            if (o_x < subregion_shape.x)
                subregion[o_x] = input[i_x];
        }
    }

    template<typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void insert_(const T* __restrict__ subregions, uint subregion_pitch, int3_t subregion_shape,
                 const size3_t* __restrict__ subregion_centers,
                 T* __restrict__ output, uint output_pitch, int3_t output_shape, uint blocks_x) {
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const int3_t gid(THREADS.x * idx.x * 2 + threadIdx.x,
                         THREADS.y * idx.y + threadIdx.y,
                         blockIdx.y);
        if (gid.y >= subregion_shape.y)
            return;

        const int3_t corner_left = getCornerLeft_(subregion_shape, subregion_centers[blockIdx.z]);
        const int o_y = corner_left.y + gid.y;
        const int o_z = corner_left.z + gid.z;
        if (o_z < 0 || o_z >= output_shape.z || o_y < 0 || o_y >= output_shape.y)
            return;

        output += noa::index<uint>(o_y, o_z, output_pitch, output_shape.y);
        subregions += noa::index<uint>(gid.y, gid.z, subregion_pitch, subregion_shape.y);
        subregions += blockIdx.z * subregion_pitch * rows(subregion_shape);

        for (int i = 0; i < 2; ++i) {
            const int i_x = gid.x + static_cast<int>(THREADS.x) * i;
            const int o_x = corner_left.x + i_x;
            if (i_x < subregion_shape.x && o_x >= 0 && o_x < output_shape.x)
                output[o_x] = subregions[i_x];
        }
    }

    template<typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void insert_(const T* __restrict__ subregion, uint subregion_pitch, int3_t subregion_shape, int3_t corner_left,
                 T* __restrict__ output, uint output_pitch, int3_t output_shape) {
        const int3_t gid(THREADS.x * blockIdx.x * 2 + threadIdx.x,
                         THREADS.y * blockIdx.y + threadIdx.y,
                         blockIdx.z);
        if (gid.y >= subregion_shape.y)
            return;

        const int o_y = corner_left.y + gid.y;
        const int o_z = corner_left.z + gid.z;
        if (o_z < 0 || o_z >= output_shape.z || o_y < 0 || o_y >= output_shape.y)
            return;

        output += noa::index<uint>(o_y, o_z, output_pitch, output_shape.y);
        subregion += noa::index<uint>(gid.y, gid.z, subregion_pitch, subregion_shape.y);

        for (int i = 0; i < 2; ++i) {
            const int i_x = gid.x + static_cast<int>(THREADS.x) * i;
            const int o_x = corner_left.x + i_x;
            if (i_x < subregion_shape.x && o_x >= 0 && o_x < output_shape.x)
                output[o_x] = subregion[i_x];
        }
    }

    template<typename T, typename I>
    __global__ __launch_bounds__(MAX_THREADS)
    void extractMap_(const T* __restrict__ i_sparse, uint i_sparse_elements,
                     T* __restrict__ o_dense, uint o_dense_elements,
                     const I* __restrict__ i_map) {
        const T* input = i_sparse + blockIdx.y * i_sparse_elements;
        T* output = o_dense + blockIdx.y * o_dense_elements;
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < o_dense_elements;
             idx += blockDim.x * gridDim.x)
            output[idx] = input[i_map[idx]];
    }

    template<typename T, typename I>
    __global__ __launch_bounds__(MAX_THREADS)
    void insertMap_(const T* __restrict__ i_dense, uint i_dense_elements,
                    T* __restrict__ o_sparse, uint o_sparse_elements,
                    const I* __restrict__ map) {
        const T* input = i_dense + blockIdx.y * i_dense_elements;
        T* output = o_sparse + blockIdx.y * o_sparse_elements;
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < i_dense_elements;
             idx += blockDim.x * gridDim.x)
            output[map[idx]] = input[idx];
    }
}

namespace noa::cuda::memory {
    template<typename T>
    void extract(const T* input, size_t input_pitch, size3_t input_shape,
                 T* subregions, size_t subregion_pitch, size3_t subregion_shape, const size3_t* subregion_centers,
                 size_t subregion_count, BorderMode border_mode, T border_value, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != subregions);

        int3_t i_shape(input_shape);
        int3_t o_shape(subregion_shape);

        uint blocks_x = math::divideUp(static_cast<uint>(o_shape.x), 2 * THREADS.x);
        uint blocks_y = math::divideUp(static_cast<uint>(o_shape.y), THREADS.y);
        dim3 blocks(blocks_x * blocks_y, subregion_shape.z, subregion_count);
        switch (border_mode) {
            case BORDER_NOTHING:
                extractOrNothing_<<<blocks, THREADS, 0, stream.id()>>>(input, input_pitch, i_shape,
                                                                       subregions, subregion_pitch, o_shape,
                                                                       subregion_centers, blocks_x);
                break;
            case BORDER_ZERO:
                extractOrValue_<<<blocks, THREADS, 0, stream.id()>>>(input, input_pitch, i_shape,
                                                                     subregions, subregion_pitch, o_shape,
                                                                     subregion_centers, static_cast<T>(0), blocks_x);
                break;
            case BORDER_VALUE:
                extractOrValue_<<<blocks, THREADS, 0, stream.id()>>>(input, input_pitch, i_shape,
                                                                     subregions, subregion_pitch, o_shape,
                                                                     subregion_centers, border_value, blocks_x);
                break;
            case BORDER_CLAMP:
                extract_<BORDER_CLAMP><<<blocks, THREADS, 0, stream.id()>>>(input, input_pitch, i_shape,
                                                                            subregions, subregion_pitch, o_shape,
                                                                            subregion_centers, blocks_x);
                break;
            case BORDER_MIRROR:
                extract_<BORDER_MIRROR><<<blocks, THREADS, 0, stream.id()>>>(input, input_pitch, i_shape,
                                                                             subregions, subregion_pitch, o_shape,
                                                                             subregion_centers, blocks_x);
                break;
            case BORDER_REFLECT:
                extract_<BORDER_REFLECT><<<blocks, THREADS, 0, stream.id()>>>(input, input_pitch, i_shape,
                                                                              subregions, subregion_pitch, o_shape,
                                                                              subregion_centers, blocks_x);
                break;
            default:
                NOA_THROW("Border mode {} is not supported", border_mode);
        }
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T>
    void extract(const T* input, size_t input_pitch, size3_t input_shape,
                 T* subregion, size_t subregion_pitch, size3_t subregion_shape, size3_t subregion_center,
                 BorderMode border_mode, T border_value, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != subregion);

        const int3_t i_shape(input_shape);
        const int3_t o_shape(subregion_shape);
        const int3_t corner_left = getCornerLeft_(o_shape, subregion_center);

        const uint blocks_x = math::divideUp(static_cast<uint>(o_shape.x), 2 * THREADS.x);
        const uint blocks_y = math::divideUp(static_cast<uint>(o_shape.y), THREADS.y);
        const dim3 blocks(blocks_x, blocks_y, subregion_shape.z);
        switch (border_mode) {
            case BORDER_NOTHING:
                extractOrNothing_<<<blocks, THREADS, 0, stream.id()>>>(
                        input, input_pitch, i_shape, subregion, subregion_pitch, o_shape, corner_left);
                break;
            case BORDER_ZERO:
                extractOrValue_<<<blocks, THREADS, 0, stream.id()>>>(
                        input, input_pitch, i_shape, subregion, subregion_pitch,
                        o_shape, corner_left, static_cast<T>(0));
                break;
            case BORDER_VALUE:
                extractOrValue_<<<blocks, THREADS, 0, stream.id()>>>(
                        input, input_pitch, i_shape, subregion, subregion_pitch, o_shape, corner_left, border_value);
                break;
            case BORDER_CLAMP:
                extract_<BORDER_CLAMP><<<blocks, THREADS, 0, stream.id()>>>(
                        input, input_pitch, i_shape, subregion, subregion_pitch, o_shape, corner_left);
                break;
            case BORDER_MIRROR:
                extract_<BORDER_MIRROR><<<blocks, THREADS, 0, stream.id()>>>(
                        input, input_pitch, i_shape, subregion, subregion_pitch, o_shape, corner_left);
                break;
            case BORDER_REFLECT:
                extract_<BORDER_REFLECT><<<blocks, THREADS, 0, stream.id()>>>(
                        input, input_pitch, i_shape, subregion, subregion_pitch, o_shape, corner_left);
                break;
            default:
                NOA_THROW("Border mode {} is not supported", border_mode);
        }
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T>
    void insert(const T* subregions, size_t subregion_pitch, size3_t subregion_shape,
                const size3_t* subregion_centers, size_t subregion_count,
                T* output, size_t output_pitch, size3_t output_shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(subregions != output);

        const uint blocks_x = math::divideUp(static_cast<uint>(subregion_shape.x), 2 * THREADS.x);
        const uint blocks_y = math::divideUp(static_cast<uint>(subregion_shape.y), THREADS.y);
        const dim3 blocks(blocks_x * blocks_y, subregion_shape.z, subregion_count);
        insert_<<<blocks, THREADS, 0, stream.id()>>>(subregions, subregion_pitch, int3_t(subregion_shape),
                                                     subregion_centers, output, output_pitch, int3_t(output_shape),
                                                     blocks_x);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T>
    void insert(const T* subregion, size_t subregion_pitch, size3_t subregion_shape, size3_t subregion_center,
                T* output, size_t output_pitch, size3_t output_shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(subregion != output);

        const int3_t i_shape(subregion_shape);
        const int3_t corner_left = getCornerLeft_(i_shape, subregion_center);

        const uint blocks_x = math::divideUp(static_cast<uint>(subregion_shape.x), 2 * THREADS.x);
        const uint blocks_y = math::divideUp(static_cast<uint>(subregion_shape.y), THREADS.y);
        const dim3 blocks(blocks_x, blocks_y, subregion_shape.z);
        insert_<<<blocks, THREADS, 0, stream.id()>>>(
                subregion, subregion_pitch, i_shape, corner_left, output, output_pitch, int3_t(output_shape));
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename I, typename T>
    std::pair<I*, size_t> where(const T* input, size_t elements, T threshold, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        // Copy to the CPU and compute the map there.
        std::unique_ptr<T[]> h_input = std::make_unique<T[]>(elements);
        copy(input, h_input.get(), elements, stream);

        std::vector<I> h_seq;
        h_seq.reserve(1000);
        stream.synchronize();
        for (size_t idx = 0; idx < elements; ++idx)
            if (h_input[idx] > threshold)
                h_seq.emplace_back(static_cast<I>(idx));

        // And back to the GPU.
        PtrDevice<I> d_seq(h_seq.size());
        copy(h_seq.data(), d_seq.get(), h_seq.size(), stream);
        h_input.reset(nullptr); // meanwhile, free the input on the host
        stream.synchronize(); // don't destruct h_seq until the copy is done
        return {d_seq.release(), h_seq.size()};
    }

    template<typename I, typename T>
    std::pair<I*, size_t> where(const T* input, size_t pitch, size3_t shape, size_t batches,
                                T threshold, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const size_t p_elements = pitch * shape.y * shape.z; // preserve the pitch
        std::unique_ptr<T[]> h_input = std::make_unique<T[]>(p_elements * batches);
        copy(input, h_input.get(), p_elements * batches, stream);

        std::vector<I> h_seq;
        h_seq.reserve(1000);
        stream.synchronize();
        for (size_t batch = 0; batch < batches; ++batch) {
            const size_t o_b = batch * p_elements;
            for (size_t z = 0; z < shape.z; ++z) {
                const size_t o_z = o_b + z * shape.y * pitch;
                for (size_t y = 0; y < shape.y; ++y) {
                    const size_t o_y = o_z + y * pitch;
                    for (size_t x = 0; x < shape.x; ++x) {
                        const size_t idx = o_y + x;
                        if (h_input[idx] > threshold)
                            h_seq.emplace_back(static_cast<I>(idx));
                    }
                }
            }
        }

        PtrDevice<I> d_seq(h_seq.size());
        copy(h_seq.data(), d_seq.get(), h_seq.size(), stream);
        h_input.reset(nullptr);
        stream.synchronize();
        return {d_seq.release(), h_seq.size()};
    }

    template<typename T, typename I>
    void extract(const T* sparse, size_t sparse_elements, T* dense, size_t dense_elements,
                 const I* sequence, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const dim3 blocks(math::min(math::divideUp(static_cast<uint>(dense_elements), MAX_THREADS), 32768U), batches);
        extractMap_<<<blocks, MAX_THREADS, 0, stream.id()>>>(
                sparse, sparse_elements, dense, dense_elements, sequence);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T, typename I>
    void insert(const T* dense, size_t dense_elements, T* sparse, size_t sparse_elements,
                const I* sequence, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const dim3 blocks(math::min(math::divideUp(static_cast<uint>(dense_elements), MAX_THREADS), 32768U), batches);
        insertMap_<<<blocks, MAX_THREADS, 0, stream.id()>>>(
                dense, dense_elements, sparse, sparse_elements, sequence);
        NOA_THROW_IF(cudaGetLastError());
    }

    #define INSTANTIATE_EXTRACT_INSERT(T)                                                                                       \
    template void extract<T>(const T*, size_t, size3_t, T*, size_t, size3_t, const size3_t*, size_t, BorderMode, T, Stream&);   \
    template void extract<T>(const T*, size_t, size3_t, T*, size_t, size3_t, size3_t, BorderMode, T, Stream&);                  \
    template void insert<T>(const T*, size_t, size3_t, const size3_t*, size_t, T*, size_t, size3_t, Stream&);                   \
    template void insert<T>(const T*, size_t, size3_t, size3_t, T*, size_t, size3_t, Stream&)

    INSTANTIATE_EXTRACT_INSERT(short);
    INSTANTIATE_EXTRACT_INSERT(int);
    INSTANTIATE_EXTRACT_INSERT(long);
    INSTANTIATE_EXTRACT_INSERT(long long);
    INSTANTIATE_EXTRACT_INSERT(unsigned short);
    INSTANTIATE_EXTRACT_INSERT(unsigned int);
    INSTANTIATE_EXTRACT_INSERT(unsigned long);
    INSTANTIATE_EXTRACT_INSERT(unsigned long long);
    INSTANTIATE_EXTRACT_INSERT(float);
    INSTANTIATE_EXTRACT_INSERT(double);

    #define NOA_INSTANTIATE_MAP1_(I, T)                                                         \
    template std::pair<I*, size_t> where<I, T>(const T*, size_t, T, Stream&);                   \
    template std::pair<I*, size_t> where<I, T>(const T*, size_t, size3_t, size_t, T, Stream&);  \
    template void extract<T, I>(const T*, size_t, T*, size_t, const I*, size_t, Stream&);       \
    template void insert<T, I>(const T*, size_t, T*, size_t, const I*, size_t, Stream&)

    #define NOA_INSTANTIATE_MAP_(T)             \
    NOA_INSTANTIATE_MAP1_(int, T);              \
    NOA_INSTANTIATE_MAP1_(long, T);             \
    NOA_INSTANTIATE_MAP1_(long long, T);        \
    NOA_INSTANTIATE_MAP1_(unsigned int, T);     \
    NOA_INSTANTIATE_MAP1_(unsigned long, T);    \
    NOA_INSTANTIATE_MAP1_(unsigned long long, T)

    NOA_INSTANTIATE_MAP_(short);
    NOA_INSTANTIATE_MAP_(int);
    NOA_INSTANTIATE_MAP_(long);
    NOA_INSTANTIATE_MAP_(long long);
    NOA_INSTANTIATE_MAP_(unsigned short);
    NOA_INSTANTIATE_MAP_(unsigned int);
    NOA_INSTANTIATE_MAP_(unsigned long);
    NOA_INSTANTIATE_MAP_(unsigned long long);
    NOA_INSTANTIATE_MAP_(float);
    NOA_INSTANTIATE_MAP_(double);

    // This a copied from noa/cpu/memory/Index.cpp
    size3_t atlasLayout(size3_t subregion_shape, size_t subregion_count, size3_t* o_subregion_centers) {
        auto sub_count = static_cast<uint>(subregion_count);
        uint col = static_cast<uint>(math::ceil(math::sqrt(static_cast<float>(sub_count))));
        uint row = (sub_count + col - 1) / col;
        size3_t atlas_shape(col * subregion_shape.x, row * subregion_shape.y, subregion_shape.z);
        size3_t half = subregion_shape / size_t{2};
        for (uint y = 0; y < row; ++y) {
            for (uint x = 0; x < col; ++x) {
                uint idx = y * col + x;
                if (idx >= sub_count)
                    break;
                o_subregion_centers[idx] = {x * subregion_shape.x + half.x,
                                            y * subregion_shape.y + half.y,
                                            half.z};
            }
        }
        return atlas_shape;
    }
}
