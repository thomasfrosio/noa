#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/memory/Remap.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/Remap.h"

namespace {
    using namespace noa;

    __forceinline__ __host__ __device__ int3_t getCornerLeft_(int3_t subregion_shape, size3_t subregion_center) {
        return int3_t(subregion_center) - subregion_shape / 2;
    }

    __forceinline__ __device__ size_t getOffset_(int3_t shape, size_t pitch, int idx_y, int idx_z) {
        return (static_cast<size_t>(idx_z) * static_cast<size_t>(shape.y) + static_cast<size_t>(idx_y)) * pitch;
    }

    template<typename T>
    __global__ void extractOrNothing_(const T* input, size_t input_pitch, int3_t input_shape,
                                      T* subregions, size_t subregion_pitch, size_t subregion_elements,
                                      int3_t subregion_shape, const size3_t* subregion_centers) {
        uint batch = blockIdx.z;
        int3_t corner_left = getCornerLeft_(subregion_shape, subregion_centers[batch]);

        uint o_y = blockIdx.x;
        uint o_z = blockIdx.y;
        int i_y = corner_left.y + static_cast<int>(o_y);
        int i_z = corner_left.z + static_cast<int>(o_z);
        if (i_z < 0 || i_z >= input_shape.z || i_y < 0 || i_y >= input_shape.y)
            return;

        input += getOffset_(input_shape, input_pitch, i_y, i_z);
        subregions += getOffset_(subregion_shape, subregion_pitch, o_y, o_z) + batch * subregion_elements;
        for (uint o_x = threadIdx.x; o_x < subregion_shape.x; o_x += blockDim.x) {
            int i_x = corner_left.x + static_cast<int>(o_x);
            if (i_x < 0 || i_x >= input_shape.x)
                continue;
            subregions[o_x] = input[i_x];
        }
    }

    template<typename T>
    __global__ void extractOrNothing_(const T* input, size_t input_pitch, int3_t input_shape,
                                      T* subregion, size_t subregion_pitch, int3_t subregion_shape,
                                      int3_t corner_left) {
        uint o_y = blockIdx.x;
        uint o_z = blockIdx.y;
        int i_y = corner_left.y + static_cast<int>(o_y);
        int i_z = corner_left.z + static_cast<int>(o_z);
        if (i_z < 0 || i_z >= input_shape.z || i_y < 0 || i_y >= input_shape.y)
            return;

        input += getOffset_(input_shape, input_pitch, i_y, i_z);
        subregion += getOffset_(subregion_shape, subregion_pitch, o_y, o_z);
        for (uint o_x = threadIdx.x; o_x < subregion_shape.x; o_x += blockDim.x) {
            int i_x = corner_left.x + static_cast<int>(o_x);
            if (i_x < 0 || i_x >= input_shape.x)
                continue;
            subregion[o_x] = input[i_x];
        }
    }

    template<typename T>
    __global__ void extractOrValue_(const T* input, size_t input_pitch, int3_t input_shape,
                                    T* subregions, size_t subregion_pitch, size_t subregion_elements,
                                    int3_t subregion_shape, const size3_t* subregion_centers, T value) {
        uint batch = blockIdx.z;
        int3_t corner_left = getCornerLeft_(subregion_shape, subregion_centers[batch]);

        uint o_y = blockIdx.x;
        uint o_z = blockIdx.y;
        int i_y = corner_left.y + static_cast<int>(o_y);
        int i_z = corner_left.z + static_cast<int>(o_z);
        bool is_out = i_z < 0 || i_z >= input_shape.z || i_y < 0 || i_y >= input_shape.y;

        subregions += getOffset_(subregion_shape, subregion_pitch, o_y, o_z) + batch * subregion_elements;
        for (uint o_x = threadIdx.x; o_x < subregion_shape.x; o_x += blockDim.x) {
            int i_x = corner_left.x + static_cast<int>(o_x);
            if (is_out || i_x < 0 || i_x >= input_shape.x)
                subregions[o_x] = value;
            else
                subregions[o_x] = input[getOffset_(input_shape, input_pitch, i_y, i_z) + i_x];
        }
    }

    template<typename T>
    __global__ void extractOrValue_(const T* input, size_t input_pitch, int3_t input_shape,
                                    T* subregion, size_t subregion_pitch,
                                    int3_t subregion_shape, int3_t corner_left, T value) {
        uint o_y = blockIdx.x;
        uint o_z = blockIdx.y;
        int i_y = corner_left.y + static_cast<int>(o_y);
        int i_z = corner_left.z + static_cast<int>(o_z);
        bool is_out = i_z < 0 || i_z >= input_shape.z || i_y < 0 || i_y >= input_shape.y;

        subregion += getOffset_(subregion_shape, subregion_pitch, o_y, o_z);
        for (uint o_x = threadIdx.x; o_x < subregion_shape.x; o_x += blockDim.x) {
            int i_x = corner_left.x + static_cast<int>(o_x);
            if (is_out || i_x < 0 || i_x >= input_shape.x)
                subregion[o_x] = value;
            else
                subregion[o_x] = input[getOffset_(input_shape, input_pitch, i_y, i_z) + i_x];
        }
    }

    template<BorderMode MODE, typename T>
    __global__ void extract_(const T* input, size_t input_pitch, int3_t input_shape,
                             T* subregions, size_t subregion_pitch, size_t subregion_elements,
                             int3_t subregion_shape, const size3_t* subregion_centers) {
        uint batch = blockIdx.z;
        int3_t corner_left = getCornerLeft_(subregion_shape, subregion_centers[batch]);

        uint o_y = blockIdx.x;
        uint o_z = blockIdx.y;
        int i_y = getBorderIndex<MODE>(corner_left.y + static_cast<int>(o_y), input_shape.y);
        int i_z = getBorderIndex<MODE>(corner_left.z + static_cast<int>(o_z), input_shape.z);

        subregions += getOffset_(subregion_shape, subregion_pitch, o_y, o_z) + batch * subregion_elements;
        for (uint o_x = threadIdx.x; o_x < subregion_shape.x; o_x += blockDim.x) {
            int i_x = getBorderIndex<MODE>(corner_left.x + static_cast<int>(o_x), input_shape.x);
            subregions[o_x] = input[getOffset_(input_shape, input_pitch, i_y, i_z) + i_x];
        }
    }

    template<BorderMode MODE, typename T>
    __global__ void extract_(const T* input, size_t input_pitch, int3_t input_shape,
                             T* subregion, size_t subregion_pitch,
                             int3_t subregion_shape, int3_t corner_left) {
        uint o_y = blockIdx.x;
        uint o_z = blockIdx.y;
        int i_y = getBorderIndex<MODE>(corner_left.y + static_cast<int>(o_y), input_shape.y);
        int i_z = getBorderIndex<MODE>(corner_left.z + static_cast<int>(o_z), input_shape.z);

        subregion += getOffset_(subregion_shape, subregion_pitch, o_y, o_z);
        for (uint o_x = threadIdx.x; o_x < subregion_shape.x; o_x += blockDim.x) {
            int i_x = getBorderIndex<MODE>(corner_left.x + static_cast<int>(o_x), input_shape.x);
            subregion[o_x] = input[getOffset_(input_shape, input_pitch, i_y, i_z) + i_x];
        }
    }

    template<typename T>
    __global__ void insert_(const T* subregions, size_t subregion_pitch, int3_t subregion_shape,
                            size_t subregion_elements, const size3_t* subregion_centers,
                            T* output, size_t output_pitch, int3_t output_shape) {
        uint batch = blockIdx.z;
        int3_t corner_left = getCornerLeft_(subregion_shape, subregion_centers[batch]);

        uint i_y = blockIdx.x;
        uint i_z = blockIdx.y;
        int o_y = corner_left.y + static_cast<int>(i_y);
        int o_z = corner_left.z + static_cast<int>(i_z);
        if (o_z < 0 || o_z >= output_shape.z || o_y < 0 || o_y >= output_shape.y)
            return;

        output += getOffset_(output_shape, output_pitch, o_y, o_z);
        subregions += getOffset_(subregion_shape, subregion_pitch, i_y, i_z) + batch * subregion_elements;
        for (uint i_x = threadIdx.x; i_x < subregion_shape.x; i_x += blockDim.x) {
            int o_x = corner_left.x + static_cast<int>(i_x);
            if (o_x < 0 || o_x >= output_shape.x)
                continue;
            output[o_x] = subregions[i_x];
        }
    }

    template<typename T>
    __global__ void insert_(const T* subregion, size_t subregion_pitch, int3_t subregion_shape, int3_t corner_left,
                            T* output, size_t output_pitch, int3_t output_shape) {
        uint i_y = blockIdx.x;
        uint i_z = blockIdx.y;
        int o_y = corner_left.y + static_cast<int>(i_y);
        int o_z = corner_left.z + static_cast<int>(i_z);
        if (o_z < 0 || o_z >= output_shape.z || o_y < 0 || o_y >= output_shape.y)
            return;

        output += getOffset_(output_shape, output_pitch, o_y, o_z);
        subregion += getOffset_(subregion_shape, subregion_pitch, i_y, i_z);
        for (uint i_x = threadIdx.x; i_x < subregion_shape.x; i_x += blockDim.x) {
            int o_x = corner_left.x + static_cast<int>(i_x);
            if (o_x < 0 || o_x >= output_shape.x)
                continue;
            output[o_x] = subregion[i_x];
        }
    }

    template<typename T, typename I>
    __global__ void extractMap_(const T* i_sparse, size_t i_sparse_elements, T* o_dense, size_t o_dense_elements,
                                const I* i_map, uint batches) {
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = i_sparse + batch * i_sparse_elements;
            T* output = o_dense + batch * o_dense_elements;
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < o_dense_elements;
                 idx += blockDim.x * gridDim.x)
                output[idx] = input[i_map[idx]];
        }
    }

    template<typename T, typename I>
    __global__ void insertMap_(const T* i_dense, size_t i_dense_elements, T* o_sparse, size_t o_sparse_elements,
                               const I* map, uint batches) {
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = i_dense + batch * i_dense_elements;
            T* output = o_sparse + batch * o_sparse_elements;
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < i_dense_elements;
                 idx += blockDim.x * gridDim.x)
                output[map[idx]] = input[idx];
        }
    }
}

namespace noa::cuda::memory {
    template<typename T>
    void extract(const T* input, size_t input_pitch, size3_t input_shape,
                 T* subregions, size_t subregion_pitch, size3_t subregion_shape, const size3_t* subregion_centers,
                 uint subregion_count, BorderMode border_mode, T border_value, Stream& stream) {
        int3_t i_shape(input_shape);
        int3_t o_shape(subregion_shape);
        size_t o_elements = subregion_pitch * getRows(subregion_shape);

        uint threads = math::min(256U, math::nextMultipleOf(static_cast<uint>(subregion_shape.x), 32U));
        dim3 blocks(o_shape.y, o_shape.z, subregion_count);
        switch (border_mode) {
            case BORDER_NOTHING:
                extractOrNothing_<<<blocks, threads, 0, stream.id()>>>(input, input_pitch, i_shape, subregions,
                                                                       subregion_pitch, o_elements, o_shape,
                                                                       subregion_centers);
                break;
            case BORDER_ZERO:
                extractOrValue_<<<blocks, threads, 0, stream.id()>>>(input, input_pitch, i_shape, subregions,
                                                                     subregion_pitch, o_elements, o_shape,
                                                                     subregion_centers, static_cast<T>(0));
                break;
            case BORDER_VALUE:
                extractOrValue_<<<blocks, threads, 0, stream.id()>>>(input, input_pitch, i_shape, subregions,
                                                                     subregion_pitch, o_elements, o_shape,
                                                                     subregion_centers, border_value);
                break;
            case BORDER_CLAMP:
                extract_<BORDER_CLAMP><<<blocks, threads, 0, stream.id()>>>(input, input_pitch, i_shape, subregions,
                                                                            subregion_pitch, o_elements, o_shape,
                                                                            subregion_centers);
                break;
            case BORDER_MIRROR:
                extract_<BORDER_MIRROR><<<blocks, threads, 0, stream.id()>>>(input, input_pitch, i_shape, subregions,
                                                                             subregion_pitch, o_elements, o_shape,
                                                                             subregion_centers);
                break;
            case BORDER_REFLECT:
                extract_<BORDER_REFLECT><<<blocks, threads, 0, stream.id()>>>(input, input_pitch, i_shape, subregions,
                                                                              subregion_pitch, o_elements, o_shape,
                                                                              subregion_centers);
                break;
            default:
                NOA_THROW("Border mode {} is not supported", border_mode);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void extract(const T* input, size_t input_pitch, size3_t input_shape,
                 T* subregion, size_t subregion_pitch, size3_t subregion_shape, size3_t subregion_center,
                 BorderMode border_mode, T border_value, Stream& stream) {
        int3_t i_shape(input_shape);
        int3_t o_shape(subregion_shape);
        int3_t corner_left = getCornerLeft_(o_shape, subregion_center);

        uint threads = math::min(256U, math::nextMultipleOf(static_cast<uint>(subregion_shape.x), 32U));
        dim3 blocks(o_shape.y, o_shape.z, 1);
        switch (border_mode) {
            case BORDER_NOTHING:
                extractOrNothing_<<<blocks, threads, 0, stream.id()>>>(
                        input, input_pitch, i_shape, subregion, subregion_pitch, o_shape, corner_left);
                break;
            case BORDER_ZERO:
                extractOrValue_<<<blocks, threads, 0, stream.id()>>>(
                        input, input_pitch, i_shape, subregion, subregion_pitch,
                        o_shape, corner_left, static_cast<T>(0));
                break;
            case BORDER_VALUE:
                extractOrValue_<<<blocks, threads, 0, stream.id()>>>(
                        input, input_pitch, i_shape, subregion, subregion_pitch, o_shape, corner_left, border_value);
                break;
            case BORDER_CLAMP:
                extract_<BORDER_CLAMP><<<blocks, threads, 0, stream.id()>>>(
                        input, input_pitch, i_shape, subregion, subregion_pitch, o_shape, corner_left);
                break;
            case BORDER_MIRROR:
                extract_<BORDER_MIRROR><<<blocks, threads, 0, stream.id()>>>(
                        input, input_pitch, i_shape, subregion, subregion_pitch, o_shape, corner_left);
                break;
            case BORDER_REFLECT:
                extract_<BORDER_REFLECT><<<blocks, threads, 0, stream.id()>>>(
                        input, input_pitch, i_shape, subregion, subregion_pitch, o_shape, corner_left);
                break;
            default:
                NOA_THROW("Border mode {} is not supported", border_mode);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void insert(const T* subregions, size_t subregion_pitch, size3_t subregion_shape,
                const size3_t* subregion_centers, uint subregion_count,
                T* output, size_t output_pitch, size3_t output_shape, Stream& stream) {
        int3_t tmp_subregion_shape(subregion_shape);
        int3_t tmp_output_shape(output_shape);
        size_t subregion_elements = subregion_pitch * getRows(subregion_shape);

        uint threads = math::min(256U, math::nextMultipleOf(static_cast<uint>(tmp_subregion_shape.x), 32U));
        dim3 blocks(tmp_subregion_shape.y, tmp_subregion_shape.z, subregion_count);
        insert_<<<blocks, threads, 0, stream.id()>>>(subregions, subregion_pitch, tmp_subregion_shape,
                                                     subregion_elements, subregion_centers, output, output_pitch,
                                                     tmp_output_shape);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void insert(const T* subregion, size_t subregion_pitch, size3_t subregion_shape, size3_t subregion_center,
                T* output, size_t output_pitch, size3_t output_shape, Stream& stream) {
        int3_t i_shape(subregion_shape);
        int3_t o_shape(output_shape);
        int3_t corner_left = getCornerLeft_(i_shape, subregion_center);
        uint threads = math::min(256U, math::nextMultipleOf(static_cast<uint>(i_shape.x), 32U));
        dim3 blocks(i_shape.y, i_shape.z, 1);
        insert_<<<blocks, threads, 0, stream.id()>>>(
                subregion, subregion_pitch, i_shape, corner_left, output, output_pitch, o_shape);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename I, typename T>
    std::pair<I*, size_t> getMap(const T* input, size_t elements, T threshold, Stream& stream) {
        // Copy to the CPU and compute the map there.
        noa::cpu::memory::PtrHost<T> h_input(elements);
        copy(input, h_input.get(), elements, stream);
        stream.synchronize();
        auto[h_free_map, elements_mapped] = noa::cpu::memory::getMap<I>(h_input.get(), elements, threshold);
        noa::cpu::memory::PtrHost<I> h_map(h_free_map, elements_mapped); // capture

        // Copy map to GPU
        PtrDevice<I> d_map(elements_mapped);
        copy(h_map.get(), d_map.get(), d_map.elements(), stream);
        stream.synchronize(); // don't destruct h_map until the copy is done.
        return {d_map.release(), elements_mapped};
    }

    template<typename I, typename T>
    std::pair<I*, size_t> getMap(const T* input, size_t pitch, size3_t shape,
                                     T threshold, Stream& stream) {
        // Back and forth to the CPU.
        size_t p_elements = pitch * shape.y * shape.z; // preserve the pitch
        noa::cpu::memory::PtrHost<T> h_input(p_elements);
        copy(input, h_input.get(), p_elements, stream);
        stream.synchronize();
        auto[h_free_map, elements_mapped] = noa::cpu::memory::getMap<I>(h_input.get(), pitch, shape, threshold);
        noa::cpu::memory::PtrHost<I> h_map(h_free_map, elements_mapped);

        // Copy map to GPU
        PtrDevice<I> d_map(elements_mapped);
        copy(h_map.get(), d_map.get(), d_map.elements(), stream);
        stream.synchronize();
        return {d_map.release(), elements_mapped};
    }

    template<typename T, typename I>
    void extract(const T* sparse, size_t sparse_elements, T* dense, size_t dense_elements,
                 const I* map, uint batches, Stream& stream) {
        uint threads = 192U;
        uint blocks = math::min((static_cast<uint>(dense_elements) + threads - 1) / threads, 32768U);
        extractMap_<<<blocks, threads, 0, stream.id()>>>(
                sparse, sparse_elements, dense, dense_elements, map, batches);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T, typename I>
    void insert(const T* dense, size_t dense_elements, T* sparse, size_t sparse_elements,
                const I* map, uint batches, Stream& stream) {
        uint threads = 192U;
        uint blocks = math::min((static_cast<uint>(dense_elements) + threads - 1) / threads, 32768U);
        insertMap_<<<blocks, threads, 0, stream.id()>>>(
                dense, dense_elements, sparse, sparse_elements, map, batches);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    #define INSTANTIATE_EXTRACT_INSERT(T)                                                                                   \
    template void extract<T>(const T*, size_t, size3_t, T*, size_t, size3_t, const size3_t*, uint, BorderMode, T, Stream&); \
    template void extract<T>(const T*, size_t, size3_t, T*, size_t, size3_t, size3_t, BorderMode, T, Stream&);              \
    template void insert<T>(const T*, size_t, size3_t, const size3_t*, uint, T*, size_t, size3_t, Stream&);                 \
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

    #define NOA_INSTANTIATE_MAP1_(I, T)                                         \
    template std::pair<I*, size_t> getMap<I, T>(const T*, size_t, T, Stream&);           \
    template std::pair<I*, size_t> getMap<I, T>(const T*, size_t, size3_t, T, Stream&);  \
    template void extract<T, I>(const T*, size_t, T*, size_t, const I*, uint, Stream&);  \
    template void insert<T, I>(const T*, size_t, T*, size_t, const I*, uint, Stream&)

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
}
