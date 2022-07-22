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
}

namespace noa::cuda::math {
    template<typename T, typename U, typename>
    void find(noa::math::min_t, const shared_t<T[]>& input, size4_t stride, size4_t shape,
              const shared_t<U[]>& offsets, bool batch, Stream& stream) {
        if (batch) {
            util::find<false>("math::find(min_t)", input.get(), uint4_t{stride}, uint4_t{shape}, noa::math::copy_t{},
                              FindFirstMin<T, U>{}, noa::math::Limits<T>::max(), offsets.get(), stream);
        } else {
            util::find<true>("math::find(min_t)", input.get(), uint4_t{stride}, uint4_t{shape}, noa::math::copy_t{},
                             FindFirstMin<T, U>{}, noa::math::Limits<T>::max(), offsets.get(), stream);
        }
        stream.attach(input, offsets);
    }

    template<typename offset_t, typename T, typename>
    offset_t find(noa::math::min_t, const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        offset_t offset;
        util::find<true>("math::find(min_t)", input.get(), uint4_t{stride}, uint4_t{shape}, noa::math::copy_t{},
                         FindFirstMin<T, offset_t>{}, noa::math::Limits<T>::max(), &offset, stream);
        stream.synchronize();
        return offset;
    }

    template<typename offset_t, typename T, typename>
    offset_t find(noa::math::min_t, const shared_t<T[]>& input, size_t elements, Stream& stream) {
        offset_t offset;
        const uint4_t shape_{1, 1, 1, elements};
        util::find<true>("math::find(min_t)", input.get(), shape_.strides(), shape_, noa::math::copy_t{},
                         FindFirstMin<T, offset_t>{}, noa::math::Limits<T>::max(), &offset, stream);
        stream.synchronize();
        return offset;
    }

    template<typename T, typename U, typename>
    void find(noa::math::max_t, const shared_t<T[]>& input, size4_t stride, size4_t shape,
              const shared_t<U[]>& offsets, bool batch, Stream& stream) {
        if (batch) {
            util::find<false>("math::find(max_t)", input.get(), uint4_t{stride}, uint4_t{shape}, noa::math::copy_t{},
                              FindFirstMax<T, U>{}, noa::math::Limits<T>::lowest(), offsets.get(), stream);
        } else {
            util::find<true>("math::find(max_t)", input.get(), uint4_t{stride}, uint4_t{shape}, noa::math::copy_t{},
                             FindFirstMax<T, U>{}, noa::math::Limits<T>::lowest(), offsets.get(), stream);
        }
        stream.attach(input, offsets);
    }

    template<typename offset_t, typename T, typename>
    offset_t find(noa::math::max_t, const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        offset_t offset;
        util::find<true>("math::find(max_t)", input.get(), uint4_t{stride}, uint4_t{shape}, noa::math::copy_t{},
                         FindFirstMax<T, offset_t>{}, noa::math::Limits<T>::lowest(), &offset, stream);
        stream.synchronize();
        return offset;
    }

    template<typename offset_t, typename T, typename>
    offset_t find(noa::math::max_t, const shared_t<T[]>& input, size_t elements, Stream& stream) {
        offset_t offset;
        const uint4_t shape_{1, 1, 1, elements};
        util::find<true>("math::find(max_t)", input.get(), shape_.strides(), shape_, noa::math::copy_t{},
                         FindFirstMax<T, offset_t>{}, noa::math::Limits<T>::lowest(), &offset, stream);
        stream.synchronize();
        return offset;
    }

    #define NOA_INSTANTIATE_INDEXES_(T, U)                                                                                          \
    template void find<T, U, void>(noa::math::min_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<U[]>&, bool, Stream&);  \
    template void find<T, U, void>(noa::math::max_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<U[]>&, bool, Stream&);  \
    template U find<U, T, void>(noa::math::min_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);                                 \
    template U find<U, T, void>(noa::math::max_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);                                 \
    template U find<U, T, void>(noa::math::min_t, const shared_t<T[]>&, size_t, Stream&);                                           \
    template U find<U, T, void>(noa::math::max_t, const shared_t<T[]>&, size_t, Stream&)

    #define NOA_INSTANTIATE_INDEXES_ALL_(T) \
    NOA_INSTANTIATE_INDEXES_(T, int32_t);   \
    NOA_INSTANTIATE_INDEXES_(T, int64_t);   \
    NOA_INSTANTIATE_INDEXES_(T, uint32_t);  \
    NOA_INSTANTIATE_INDEXES_(T, uint64_t)

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
