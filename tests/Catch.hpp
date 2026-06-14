#pragma once

#include <noa/base/Config.hpp>

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wdouble-promotion"
#   if defined(NOA_COMPILER_CLANG)
#       pragma GCC diagnostic ignored "-Wc2y-extensions"
#       pragma GCC diagnostic ignored "-Wimplicit-int-float-conversion"
#   endif
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#   pragma GCC diagnostic pop
#endif

// FIXME temporary fix for Catch2 use of __COUNTER__ with apple-clang <= 22 on
#ifdef NOA_COMPILER_CLANG
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wc2y-extensions"
#endif
