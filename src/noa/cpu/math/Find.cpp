#include <algorithm>
#include "noa/cpu/math/Find.h"

namespace {
    template<typename T>
    struct FirstMin {
        constexpr bool operator()(T current, T candidate) const noexcept { return current > candidate; }
    };

    template<typename T>
    struct LastMin {
        constexpr bool operator()(T current, T candidate) const noexcept { return current >= candidate; }
    };

    template<typename T>
    struct FirstMax {
        constexpr bool operator()(T current, T candidate) const noexcept { return current < candidate; }
    };

    template<typename T>
    struct LastMax {
        constexpr bool operator()(T current, T candidate) const noexcept { return current <= candidate; }
    };

    template<typename S, typename T>
    using op = std::conditional_t<std::is_same_v<S, noa::math::first_min_t>, FirstMin<T>,
               std::conditional_t<std::is_same_v<S, noa::math::first_max_t>, FirstMax<T>,
               std::conditional_t<std::is_same_v<S, noa::math::last_min_t>, LastMin<T>,
               std::conditional_t<std::is_same_v<S, noa::math::last_max_t>, LastMax<T>, void>>>>;
}

namespace noa::cpu::math {
    template<typename S, typename T, typename U, typename>
    void find(S, const shared_t<T[]>& input, size4_t strides, size4_t shape,
              const shared_t<U[]>& offsets, bool batch, Stream& stream) {
        stream.enqueue([=]() mutable {
            const size3_t order_ = indexing::order(size3_t(strides.get(1)), size3_t(shape.get(1))) + 1;
            const size4_t order{0, order_[0], order_[1], order_[2]};
            strides = indexing::reorder(strides, order);
            shape = indexing::reorder(shape, order);

            const size4_t shape_ = batch ? size4_t{1, shape[1], shape[2], shape[3]} : shape;
            const size_t batches = batch ? shape[0] : 1;
            const op<S, T> find_op{};

            for (size_t batch_ = 0; batch_ < batches; ++batch_) {
                const T* input_ = input.get() + batch_ * strides[0];
                U* offset_ = offsets.get() + batch_;
                size_t found_offset{};
                T found_value = *input_;

                for (size_t i = 0; i < shape_[0]; ++i) {
                    for (size_t j = 0; j < shape_[1]; ++j) {
                        for (size_t k = 0; k < shape_[2]; ++k) {
                            for (size_t l = 0; l < shape_[3]; ++l) {
                                const size_t offset = noa::indexing::at(i, j, k, l, strides);
                                if (find_op(found_value, input_[offset])) {
                                    found_value = input_[offset];
                                    found_offset = offset;
                                }
                            }
                        }
                    }
                }
                *offset_ = static_cast<U>(found_offset);
            }
        });
    }

    template<typename offset_t, typename S, typename T, typename>
    offset_t find(S, const shared_t<T[]>& input, size4_t strides, size4_t shape, Stream& stream) {
        const size4_t order = indexing::order(strides, shape);
        strides = indexing::reorder(strides, order);
        shape = indexing::reorder(shape, order);

        if (indexing::areContiguous(strides, shape))
            return find<offset_t>(S{}, input, shape.elements(), stream);

        stream.synchronize();
        const T* input_ = input.get();
        size_t found_offset{};
        T found_value = *input_;
        const op<S, T> find_op{};

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        const size_t offset = noa::indexing::at(i, j, k, l, strides);
                        if (find_op(found_value, input_[offset])) {
                            found_value = input_[offset];
                            found_offset = offset;
                        }
                    }
                }
            }
        }
        return static_cast<offset_t>(found_offset);
    }

    template<typename offset_t, typename S, typename T, typename>
    offset_t find(S, const shared_t<T[]>& input, size_t elements, Stream& stream) {
        stream.synchronize();
        if constexpr (std::is_same_v<S, noa::math::first_min_t>) {
            T* ptr = std::min_element(input.get(), input.get() + elements);
            return static_cast<offset_t>(ptr - input.get());
        } else if constexpr (std::is_same_v<S, noa::math::first_max_t>) {
            T* ptr = std::max_element(input.get(), input.get() + elements);
            return static_cast<offset_t>(ptr - input.get());
        } else { // not sure how to take the last occurrence using the STL, probably with reverse iterator?
            const T* input_ = input.get();
            size_t found_offset{};
            T found_value = *input_;
            const op<S, T> find_op{};

            for (size_t i = 0; i < elements; ++i) {
                if (find_op(found_value, input_[i])) {
                    found_value = input_[i];
                    found_offset = i;
                }
            }
            return static_cast<offset_t>(found_offset);
        }
    }

    #define NOA_INSTANTIATE_FIND_(S, T, U)                                                                              \
    template void find<S, T, U, void>(S, const shared_t<T[]>&, size4_t, size4_t, const shared_t<U[]>&, bool, Stream&);  \
    template U find<U, S, T, void>(S, const shared_t<T[]>&, size4_t, size4_t, Stream&);                                 \
    template U find<U, S, T, void>(S, const shared_t<T[]>&, size_t, Stream&)

    #define NOA_INSTANTIATE_FIND_OP_(T, U)               \
    NOA_INSTANTIATE_FIND_(noa::math::first_min_t, T, U); \
    NOA_INSTANTIATE_FIND_(noa::math::first_max_t, T, U); \
    NOA_INSTANTIATE_FIND_(noa::math::last_min_t, T, U);  \
    NOA_INSTANTIATE_FIND_(noa::math::last_max_t, T, U)

    #define NOA_INSTANTIATE_INDEXES_ALL_(T) \
    NOA_INSTANTIATE_FIND_OP_(T, uint32_t);  \
    NOA_INSTANTIATE_FIND_OP_(T, uint64_t);  \
    NOA_INSTANTIATE_FIND_OP_(T, int32_t);   \
    NOA_INSTANTIATE_FIND_OP_(T, int64_t)

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
