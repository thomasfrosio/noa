#include <catch2/catch.hpp>
#include "../../Helpers.h"

#include "noa/structures/Vectors.h"

using namespace ::Noa;

#define REQUIRE_FOR_ALL_TYPES(type_trait)                       \
REQUIRE(type_trait<TestType>);                                  \
REQUIRE(type_trait<std::add_const_t<TestType>>);                \
REQUIRE(type_trait<std::add_volatile_t<TestType>>);             \
REQUIRE(type_trait<std::add_cv_t<TestType>>);                   \
REQUIRE(type_trait<std::add_lvalue_reference_t<TestType>>);     \
REQUIRE(type_trait<std::add_rvalue_reference_t<TestType>>)

#define REQUIRE_FALSE_FOR_ALL_TYPES(type_trait)                       \
REQUIRE_FALSE(type_trait<TestType>);                                  \
REQUIRE_FALSE(type_trait<std::add_const_t<TestType>>);                \
REQUIRE_FALSE(type_trait<std::add_volatile_t<TestType>>);             \
REQUIRE_FALSE(type_trait<std::add_cv_t<TestType>>);                   \
REQUIRE_FALSE(type_trait<std::add_lvalue_reference_t<TestType>>);     \
REQUIRE_FALSE(type_trait<std::add_rvalue_reference_t<TestType>>)


TEMPLATE_TEST_CASE("Vectors: Int2", "[noa][vectors]", int32_t, int64_t, uint32_t, uint64_t) {

    using Int = Int2<TestType>;

    //@CLION-formatter:off
    Int test{};
    REQUIRE((test == Int{0.f}));
    test = TestType{2};
    test += TestType{1}; REQUIRE(test == TestType{3});
    test -= TestType{2}; REQUIRE(test == TestType{1});
    test *= TestType{3}; REQUIRE(test == TestType{3});
    test /= TestType{2}; REQUIRE(test == TestType{1});

    test = TestType{30};
    REQUIRE((test + TestType{10} == 40));
    REQUIRE((test - TestType{5} == 25));
    REQUIRE((test * TestType{3} == 90));
    REQUIRE((test / TestType{2} == 15));

    test = {4, 10};
    REQUIRE_FALSE(test > TestType{5});
    REQUIRE(test < TestType{11});
    REQUIRE_FALSE(test >= TestType{7});
    REQUIRE_FALSE(test <= TestType{9});
    REQUIRE(test != TestType{4});

    test = Int2<int>{0, 2};
    test += Int(35, 20); REQUIRE((test == Int(35, 22)));
    test -= Int(22, 18); REQUIRE((test == Int(13, 4)));
    test *= Int(2, 6);   REQUIRE((test == Int(26, 24)));
    test /= Int(2, 9);   REQUIRE((test == Int(13, 2)));

    //@CLION-formatter:on
    test.x = 20;
    test.y = 50;
    REQUIRE((test + Int(10, 12) == Int(30, 62)));
    REQUIRE((test - Int(3, 2) == Int(17, 48)));
    REQUIRE((test * Int(1, 5) == Int(20, 250)));
    REQUIRE((test / Int(3, 2) == Int(6, 25)));

    test = {2, 4};
    REQUIRE(test > Int{1, 2});
    REQUIRE(test < Int{4, 5});
    REQUIRE(test >= Int{2, 4});
    REQUIRE(test <= Int{10, 4});
    REQUIRE(test != Int{4, 4});

    // Min & Max
    REQUIRE((min(Int{3, 4}, Int{5, 2}) == Int{3, 2}));
    REQUIRE((max(Int{3, 4}, Int{5, 2}) == Int{5, 4}));
    REQUIRE((min(Int{3, 6}, TestType{5}) == Int{3, 5}));
    REQUIRE((max(Int{9, 0}, TestType{2}) == Int{9, 2}));

    // .data()
    test = Float2<float>{3.4f, 90.6f}.data();
    REQUIRE(test == Int(3, 90));
    Int2<int> test1(test.data());
    REQUIRE((test1 == static_cast<Int2<int>>(test)));

    test.x = 23;
    test.y = 52;
    REQUIRE(test.sum() == 75);
    REQUIRE(test.prod() == 1196);
    REQUIRE(test.prodFFT() == 624);
    REQUIRE((test.toString() == std::string{"(23, 52)"}));
}


TEMPLATE_TEST_CASE("Vectors: Int3", "[noa][vectors]", int32_t, int64_t, uint32_t, uint64_t) {

    using Int = Int3<TestType>;

    //@CLION-formatter:off
    Int test{};
    REQUIRE((test == Int{0.f}));
    test = TestType{2};
    test += TestType{1}; REQUIRE(test == TestType{3});
    test -= TestType{2}; REQUIRE(test == TestType{1});
    test *= TestType{3}; REQUIRE(test == TestType{3});
    test /= TestType{2}; REQUIRE(test == TestType{1});

    test = TestType{30};
    REQUIRE((test + TestType{10} == 40));
    REQUIRE((test - TestType{5} == 25));
    REQUIRE((test * TestType{3} == 90));
    REQUIRE((test / TestType{2} == 15));

    test = {4, 10, 7};
    REQUIRE_FALSE(test > TestType{9});
    REQUIRE(test < TestType{11});
    REQUIRE_FALSE(test >= TestType{7});
    REQUIRE_FALSE(test <= TestType{9});
    REQUIRE(test != TestType{4});

    test = Int3<int>{0, 2, 130};
    test += Int(35, 20, 4);     REQUIRE((test == Int(35, 22, 134)));
    test -= Int(22, 18, 93);    REQUIRE((test == Int(13, 4, 41)));
    test *= Int(2, 6, 2);       REQUIRE((test == Int(26, 24, 82)));
    test /= Int(2, 9, 1);       REQUIRE((test == Int(13, 2, 82)));

    //@CLION-formatter:on
    test.x = 20;
    test.y = 50;
    test.z = 10;
    REQUIRE((test + Int(10, 12, 5) == Int(30, 62, 15)));
    REQUIRE((test - Int(3, 2, 7) == Int(17, 48, 3)));
    REQUIRE((test * Int(1, 5, 3) == Int(20, 250, 30)));
    REQUIRE((test / Int(3, 2, 2) == Int(6, 25, 5)));

    test = {2, 4, 4};
    REQUIRE(test > Int{1, 2, 3});
    REQUIRE(test < Int{4, 5, 6});
    REQUIRE(test >= Int{2, 4, 3});
    REQUIRE(test <= Int{10, 4, 6});
    REQUIRE(test != Int{4, 4, 4});


    // Min & Max
    REQUIRE((min(Int{3, 4, 8}, Int{5, 2, 10}) == Int{3, 2, 8}));
    REQUIRE((max(Int{3, 4, 1000}, Int{5, 2, 30}) == Int{5, 4, 1000}));
    REQUIRE((min(Int{3, 6, 4}, TestType{5}) == Int{3, 5, 4}));
    REQUIRE((max(Int{9, 0, 1}, TestType{2}) == Int{9, 2, 2}));

    // .data()
    test = Float3<float>{3.4f, 90.6f, 5.f}.data();
    REQUIRE(test == Int(3, 90, 5));
    Int3<long> test1(test.data());
    REQUIRE((test1 == static_cast<Int3<long>>(test)));

    test.x = 23;
    test.y = 52;
    test.z = 128;
    REQUIRE(test.sum() == 203);
    REQUIRE(test.prod() == 153088);
    REQUIRE(test.prodFFT() == 79872);
    REQUIRE(test.slice() == Int(23, 52, 1));
    REQUIRE(test.prodSlice() == 1196);

    REQUIRE((test.toString() == std::string{"(23, 52, 128)"}));
}


TEMPLATE_TEST_CASE("Vectors: Int4", "[noa][vectors]", int32_t, int64_t, uint32_t, uint64_t) {

    using Int = Int4<TestType>;

    //@CLION-formatter:off
    Int test{};
    REQUIRE((test == Int{0.f}));
    test = TestType{2};
    test += TestType{1}; REQUIRE(test == TestType{3});
    test -= TestType{2}; REQUIRE(test == TestType{1});
    test *= TestType{3}; REQUIRE(test == TestType{3});
    test /= TestType{2}; REQUIRE(test == TestType{1});

    test = TestType{30};
    REQUIRE((test + TestType{10} == 40));
    REQUIRE((test - TestType{5} == 25));
    REQUIRE((test * TestType{3} == 90));
    REQUIRE((test / TestType{2} == 15));

    test = {15, 130, 70, 2};
    REQUIRE_FALSE(test > TestType{9});
    REQUIRE(test < TestType{131});
    REQUIRE_FALSE(test >= TestType{7});
    REQUIRE_FALSE(test <= TestType{50});
    REQUIRE(test != TestType{15});

    test = Int4<long>{130, 5, 130, 120};
    test += Int(35, 20, 4, 20); REQUIRE(test == Int(165, 25, 134, 140));
    test -= Int(22, 1, 93, 2);  REQUIRE(test == Int(143, 24, 41, 138));
    test *= Int(2, 6, 2, 9);    REQUIRE(test == Int(286, 144, 82, 1242));
    test /= Int(2, 9, 1, 4);    REQUIRE(test == Int(143, 16, 82, 310));

    //@CLION-formatter:on
    test.x = 20;
    test.y = 50;
    test.z = 10;
    test.w = 3;
    REQUIRE((test + Int(10, 12, 5, 30) == Int(30, 62, 15, 33)));
    REQUIRE((test - Int(3, 2, 7, 1) == Int(17, 48, 3, 2)));
    REQUIRE((test * Int(1, 5, 3, 103) == Int(20, 250, 30, 309)));
    REQUIRE((test / Int(3, 2, 2, 10) == Int(6, 25, 5, 0)));

    test = {2, 4, 4, 13405};
    REQUIRE_FALSE(test > Int{5, 5, 5, 100});
    REQUIRE_FALSE(test < Int{5, 5, 6, 100});
    REQUIRE(test >= Int{2, 4, 3, 4});
    REQUIRE_FALSE(test <= Int{10, 4, 6, 2});
    REQUIRE_FALSE(test != Int{2, 4, 4, 13405});

    // Min & Max
    REQUIRE((min(Int{3, 4, 8, 1230}, Int{5, 2, 10, 312}) == Int{3, 2, 8, 312}));
    REQUIRE((max(Int{3, 4, 1000, 2}, Int{5, 2, 30, 1}) == Int{5, 4, 1000, 2}));
    REQUIRE((min(Int{3, 6, 4, 74}, TestType{5}) == Int{3, 5, 4, 5}));
    REQUIRE((max(Int{9, 0, 1, 4}, TestType{2}) == Int{9, 2, 2, 4}));

    // .data()
    test = Float4<double>{3.4, 90.6, 5., 12.99}.data();
    REQUIRE(test == Int(3, 90, 5, 12));
    Int4<int> test1(test.data());
    REQUIRE((test1 == static_cast<Int4<int>>(test)));

    test.x = 23;
    test.y = 52;
    test.z = 128;
    test.w = 4;
    REQUIRE(test.sum() == 207);
    REQUIRE(test.prod() == 612352);
    REQUIRE(test.prodFFT() == 319488);
    REQUIRE(test.slice() == Int(23, 52, 1, 1));
    REQUIRE(test.prodSlice() == 1196);

    REQUIRE((test.toString() == std::string{"(23, 52, 128, 4)"}));
}
