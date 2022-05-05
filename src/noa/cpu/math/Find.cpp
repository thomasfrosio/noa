#include <algorithm>
#include "noa/cpu/math/Find.h"

namespace noa::cpu::math {
    template<typename T, typename U, typename>
    void find(noa::math::min_t, const shared_t<T[]>& input, size4_t stride, size4_t shape,
              const shared_t<U[]>& offsets, bool batch, Stream& stream) {
        stream.enqueue([=]() {
            const size4_t shape_ = batch ? size4_t{1, shape[1], shape[2], shape[3]} : shape;
            const size_t batches = batch ? shape[0] : 1;

            for (size_t batch_ = 0; batch_ < batches; ++batch_) {
                const T* input_ = input.get() + batch_ * stride[0];
                U* offset_ = offsets.get() + batch_;
                size_t min_offset{};
                T min_value = *input_;

                for (size_t i = 0; i < shape_[0]; ++i) {
                    for (size_t j = 0; j < shape_[1]; ++j) {
                        for (size_t k = 0; k < shape_[2]; ++k) {
                            for (size_t l = 0; l < shape_[3]; ++l) {
                                const size_t offset = noa::indexing::at(i, j, k, l, stride);
                                if (input_[offset] < min_value) {
                                    min_value = input_[offset];
                                    min_offset = offset;
                                }
                            }
                        }
                    }
                }
                *offset_ = static_cast<U>(min_offset);
            }
        });
    }

    template<typename offset_t, typename T, typename>
    offset_t find(noa::math::min_t, const shared_t<T[]>& input, size_t elements, Stream& stream) {
        stream.synchronize();
        T* min_ptr = std::min_element(input.get(), input.get() + elements);
        return static_cast<offset_t>(min_ptr - input.get());
    }

    template<typename T, typename U, typename>
    void find(noa::math::max_t, const shared_t<T[]>& input, size4_t stride, size4_t shape,
              const shared_t<U[]>& offsets, bool batch, Stream& stream) {
        stream.enqueue([=]() {
            const size4_t shape_ = batch ? size4_t{1, shape[1], shape[2], shape[3]} : shape;
            const size_t batches = batch ? shape[0] : 1;

            for (size_t batch_ = 0; batch_ < batches; ++batch_) {
                const T* input_ = input.get() + batch_ * stride[0];
                U* offset_ = offsets.get() + batch_;
                size_t max_offset{};
                T max_value = *input_;

                for (size_t i = 0; i < shape_[0]; ++i) {
                    for (size_t j = 0; j < shape_[1]; ++j) {
                        for (size_t k = 0; k < shape_[2]; ++k) {
                            for (size_t l = 0; l < shape_[3]; ++l) {
                                const size_t offset = noa::indexing::at(i, j, k, l, stride);
                                if (input_[offset] > max_value) {
                                    max_value = input_[offset];
                                    max_offset = offset;
                                }
                            }
                        }
                    }
                }
                *offset_ = static_cast<U>(max_offset);
            }
        });
    }

    template<typename offset_t, typename T, typename>
    offset_t find(noa::math::max_t, const shared_t<T[]>& input, size_t elements, Stream& stream) {
        stream.synchronize();
        T* max_ptr = std::max_element(input.get(), input.get() + elements);
        return static_cast<offset_t>(max_ptr - input.get());
    }

    #define NOA_INSTANTIATE_INDEXES_(T, U)                                                                                          \
    template void find<T, U, void>(noa::math::min_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<U[]>&, bool, Stream&);  \
    template void find<T, U, void>(noa::math::max_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<U[]>&, bool, Stream&);  \
    template U find<U, T, void>(noa::math::min_t, const shared_t<T[]>&, size_t, Stream&);                                           \
    template U find<U, T, void>(noa::math::max_t, const shared_t<T[]>&, size_t, Stream&)

    #define NOA_INSTANTIATE_INDEXES_ALL_(T) \
    NOA_INSTANTIATE_INDEXES_(T, uint32_t);  \
    NOA_INSTANTIATE_INDEXES_(T, uint64_t);  \
    NOA_INSTANTIATE_INDEXES_(T, int32_t);   \
    NOA_INSTANTIATE_INDEXES_(T, int64_t)

    NOA_INSTANTIATE_INDEXES_ALL_(uint8_t);
    NOA_INSTANTIATE_INDEXES_ALL_(int8_t);
    NOA_INSTANTIATE_INDEXES_ALL_(uint16_t);
    NOA_INSTANTIATE_INDEXES_ALL_(int16_t);
    NOA_INSTANTIATE_INDEXES_ALL_(uint32_t);
    NOA_INSTANTIATE_INDEXES_ALL_(int32_t);
    NOA_INSTANTIATE_INDEXES_ALL_(uint64_t);
    NOA_INSTANTIATE_INDEXES_ALL_(int64_t);
    NOA_INSTANTIATE_INDEXES_ALL_(half_t);
    NOA_INSTANTIATE_INDEXES_ALL_(float);
    NOA_INSTANTIATE_INDEXES_ALL_(double);
}
