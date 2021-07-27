//#include <noa/cpu/memory/PtrHost.h>
//#include <noa/cpu/transform/Interpolator.h>
//
//#include "Helpers.h"
//#include <catch2/catch.hpp>
//
//using namespace ::noa;
//
//TEST_CASE("cpu::transform::Interpolator, linear", "[noa][cpu][transform]") {
//    transform::Interpolator1D<double> interp;
//
//    constexpr int samples = 6;
//    double data[samples] = {1., 4., 3.2, -3.5, -1.5, 3};
//    interp.reset(data, samples);
//
//    AND_THEN("BORDER_ZERO") { // linear and border_zero should be the default
//        // Check coordinates at indexes match original data:
//        memory::PtrHost<double> filtered_data(samples);
//        transform::bspline::prefilter1D(data, filtered_data.get(), samples, 1);
//        interp.reset(filtered_data.get(), samples);
//
//        memory::PtrHost<double> values(samples);
//        for (size_t i = 0; i < samples; ++i)
//            values[i] = interp.get<INTERP_CUBIC_BSPLINE, BORDER_MIRROR>(static_cast<float>(i));
//        double diff = test::getDifference(data, values.get(), samples);
//        REQUIRE_THAT(diff, test::isWithinRel(0.f));
//
//        // Check (partial) OOB is correctly computed:
//        double out = interp.get<INTERP_COSINE, BORDER_REFLECT>(-0.5f);
//        REQUIRE_THAT(out, test::isWithinRel(0.5f));
//        out = interp.get(-1.f);
//        REQUIRE_THAT(out, test::isWithinRel(0.f));
//        out = interp.get(5.5f);
//        REQUIRE_THAT(out, test::isWithinRel(1.5f));
//        out = interp.get(6.f);
//        REQUIRE_THAT(out, test::isWithinRel(0.f));
//
//        // Within data:
//        out = interp.get(1.5f);
//        REQUIRE_THAT(out, test::isWithinRel(0.5f));
//        out = interp.get(2.5f);
//        REQUIRE_THAT(out, test::isWithinRel(0.f));
//        out = interp.get(4.5f);
//        REQUIRE_THAT(out, test::isWithinRel(1.5f));
//    }
//}
