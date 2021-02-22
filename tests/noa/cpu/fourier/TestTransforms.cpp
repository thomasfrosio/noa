#include <noa/cpu/fourier/Plan.h>
#include <noa/cpu/fourier/Transforms.h>
#include <noa/cpu/PtrHost.h>
#include "noa/util/files/MRCFile.h"

#include <catch2/catch.hpp>
#include "../../../Helpers.h"

using namespace Noa;

namespace Test {
    template<typename T>
    void initHostDataRandom(T* data, size_t elements, Test::IntRandomizer<size_t>& randomizer) {
        if constexpr (std::is_same_v<T, cfloat_t>) {
            for (size_t idx{0}; idx < elements; ++idx) {
                data[idx] = T{static_cast<float>(randomizer.get())};
            }
        } else if constexpr (std::is_same_v<T, cdouble_t>) {
            for (size_t idx{0}; idx < elements; ++idx) {
                data[idx] = T{static_cast<double>(randomizer.get())};
            }
        } else {
            for (size_t idx{0}; idx < elements; ++idx)
                data[idx] = static_cast<T>(randomizer.get());
        }
    }

    template<typename T>
    void initHostDataZero(T* data, size_t elements) {
        for (size_t idx{0}; idx < elements; ++idx)
            data[idx] = 0;
    }

    template<typename T>
    T getDifference(const T* in, const T* out, size_t elements) {
        T diff{0};
        for (size_t idx{0}; idx < elements; ++idx)
            diff += out[idx] - in[idx];
        return diff;
    }

    template<typename T>
    void normalize(T* array, size_t size, T scale) {
        for (size_t idx{0}; idx < size; ++idx) {
            array[idx] *= scale;
        }
    }
}

TEST_CASE("Fourier::Transforms") {
    Test::IntRandomizer<size_t> randomizer(2, 128);
    shape_t shape_real{randomizer.get(), randomizer.get(), randomizer.get()};
    shape_t shape_complex{shape_real.x / 2 + 1, shape_real.y, shape_real.z};
    size_t elements_real = Math::elements(shape_real);
    size_t elements_complex = Math::elements(shape_complex);

    PtrHost<float> input_in(elements_real);
    PtrHost<float> input_out(elements_real);
    PtrHost<cfloat_t> transform(elements_complex);

    AND_THEN("basic one shot transform") {
        Test::initHostDataRandom(input_in.get(), input_in.elements(), randomizer);
        Test::initHostDataZero(input_out.get(), input_out.elements());
        Fourier::r2c(transform.get(), input_in.get(), shape_real);
        Fourier::c2r(input_out.get(), transform.get(), shape_real);
        Test::normalize(input_out.get(), input_in.elements(), 1 / static_cast<float>(elements_real));
        float diff = Test::getDifference(input_in.get(), input_out.get(), elements_real);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-4));
    }

    AND_THEN("basic execute") {
        Fourier::Plan<float> plan_forward(transform.get(), input_in.get(), shape_real, Fourier::Flag::measure);
        Fourier::Plan<float> plan_backward(input_out.get(), transform.get(), shape_real, Fourier::Flag::measure);
        Test::initHostDataRandom(input_in.get(), input_in.elements(), randomizer);
        Test::initHostDataZero(input_out.get(), input_out.elements());
        Fourier::transform(plan_forward);
        Fourier::transform(plan_backward);
        Test::normalize(input_out.get(), input_in.elements(), 1 / static_cast<float>(elements_real));
        float diff = Test::getDifference(input_in.get(), input_out.get(), elements_real);
        REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-4));
    }
}
