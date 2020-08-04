#include <gmock/gmock.h>
#include <gtest/gtest.h>

TEST(Foo, Bar) {
ASSERT_THAT(0, ::testing::Eq(1));
}

