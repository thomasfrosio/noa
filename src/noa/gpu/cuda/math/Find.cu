#include "noa/common/Math.h"
#include "noa/gpu/cuda/math/Find.h"
#include "noa/gpu/cuda/util/Find.cuh"

namespace {
    using namespace ::noa;

    template<typename value_t, typename offset_t>
    struct FindFirstMin {
        using pair_t = Pair<value_t, offset_t>;
        NOA_FD pair_t operator()(pair_t current, pair_t candidate) noexcept {
            if (candidate.first < current.first ||
                (current.first == candidate.first && candidate.second < current.second))
                return candidate;
            return current;
        }
    };

    template<typename value_t, typename offset_t>
    struct FindFirstMax {
        using pair_t = Pair<value_t, offset_t>;
        NOA_FD pair_t operator()(pair_t current, pair_t candidate) noexcept {
            if (candidate.first > current.first ||
                (current.first == candidate.first && candidate.second < current.second))
                return candidate;
            return current;
        }
    };

    template<typename value_t, typename offset_t>
    struct FindLastMin {
        using pair_t = Pair<value_t, offset_t>;
        NOA_FD pair_t operator()(pair_t current, pair_t candidate) noexcept {
            if (candidate.first < current.first ||
                (current.first == candidate.first && candidate.second > current.second))
                return candidate;
            return current;
        }
    };

    template<typename value_t, typename offset_t>
    struct FindLastMax {
        using pair_t = Pair<value_t, offset_t>;
        NOA_FD pair_t operator()(pair_t current, pair_t candidate) noexcept {
            if (candidate.first > current.first ||
                (current.first == candidate.first && candidate.second > current.second))
                return candidate;
            return current;
        }
    };
}

namespace noa::cuda::math {
    template<typename S, typename T, typename U, typename>
    void find(S, const shared_t<T[]>& input, size4_t strides, size4_t shape,
              const shared_t<U[]>& offsets, bool batch, bool swap_layout, Stream& stream) {
        if constexpr (std::is_same_v<S, noa::math::first_min_t>) {
            util::find("math::find(first_min_t)", input.get(), uint4_t(strides), uint4_t(shape), noa::math::copy_t{},
                       FindFirstMin<T, U>{}, noa::math::Limits<T>::max(), offsets.get(), !batch, swap_layout, stream);
        } else if constexpr (std::is_same_v<S, noa::math::first_max_t>) {
            util::find("math::find(first_max_t)", input.get(), uint4_t(strides), uint4_t(shape), noa::math::copy_t{},
                       FindFirstMax<T, U>{}, noa::math::Limits<T>::lowest(), offsets.get(), !batch, swap_layout, stream);
        } else if constexpr (std::is_same_v<S, noa::math::last_min_t>) {
            util::find("math::find(last_min_t)", input.get(), uint4_t(strides), uint4_t(shape), noa::math::copy_t{},
                       FindLastMin<T, U>{}, noa::math::Limits<T>::max(), offsets.get(), !batch, swap_layout, stream);
        } else if constexpr (std::is_same_v<S, noa::math::last_max_t>) {
            util::find("math::find(last_max_t)", input.get(), uint4_t(strides), uint4_t(shape), noa::math::copy_t{},
                       FindLastMax<T, U>{}, noa::math::Limits<T>::lowest(), offsets.get(), !batch, swap_layout, stream);
        }
        stream.attach(input, offsets);
    }

    template<typename offset_t, typename S, typename T, typename>
    offset_t find(S, const shared_t<T[]>& input, size4_t strides, size4_t shape, bool swap_layout, Stream& stream) {
        offset_t offset;
        if constexpr (std::is_same_v<S, noa::math::first_min_t>) {
            util::find("math::find(first_min_t)", input.get(), uint4_t(strides), uint4_t(shape), noa::math::copy_t{},
                       FindFirstMin<T, offset_t>{}, noa::math::Limits<T>::max(), &offset, true, swap_layout, stream);
        } else if constexpr (std::is_same_v<S, noa::math::first_max_t>) {
            util::find("math::find(first_max_t)", input.get(), uint4_t(strides), uint4_t(shape), noa::math::copy_t{},
                       FindFirstMax<T, offset_t>{}, noa::math::Limits<T>::lowest(), &offset, true, swap_layout, stream);
        } else if constexpr (std::is_same_v<S, noa::math::last_min_t>) {
            util::find("math::find(last_min_t)", input.get(), uint4_t(strides), uint4_t(shape), noa::math::copy_t{},
                       FindLastMin<T, offset_t>{}, noa::math::Limits<T>::max(), &offset, true, swap_layout, stream);
        } else if constexpr (std::is_same_v<S, noa::math::last_max_t>) {
            util::find("math::find(last_max_t)", input.get(), uint4_t(strides), uint4_t(shape), noa::math::copy_t{},
                       FindLastMax<T, offset_t>{}, noa::math::Limits<T>::lowest(), &offset, true, swap_layout, stream);
        }
        stream.synchronize();
        return offset;
    }

    template<typename offset_t, typename S, typename T, typename>
    offset_t find(S searcher, const shared_t<T[]>& input, size_t elements, Stream& stream) {
        const size4_t shape_{1, 1, 1, elements};
        return find<offset_t>(searcher, input, shape_.strides(), shape_, false, stream);
    }

    #define NOA_INSTANTIATE_FIND_(S, T, U)                                                                                   \
    template void find<S, T, U, void>(S, const shared_t<T[]>&, size4_t, size4_t, const shared_t<U[]>&, bool, bool, Stream&); \
    template U find<U, S, T, void>(S, const shared_t<T[]>&, size4_t, size4_t, bool, Stream&);                                \
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

    NOA_INSTANTIATE_INDEXES_ALL_(uint32_t);
    NOA_INSTANTIATE_INDEXES_ALL_(int32_t);
    NOA_INSTANTIATE_INDEXES_ALL_(uint64_t);
    NOA_INSTANTIATE_INDEXES_ALL_(int64_t);
    NOA_INSTANTIATE_INDEXES_ALL_(half_t);
    NOA_INSTANTIATE_INDEXES_ALL_(float);
    NOA_INSTANTIATE_INDEXES_ALL_(double);
}
