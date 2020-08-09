/**
 * @file Core.h
 * @brief Contain the minimum files to include for the noa core.
 * @author Thomas - ffyr2w
 * @date 20 Jul 2020
 */

#pragma once

#define NOA_VERSION_MAJOR 0
#define NOA_VERSION_MINOR 1
#define NOA_VERSION_PATCH 0
#define NOA_VERSION (NOA_VERSION_MAJOR * 10000 + NOA_VERSION_MINOR * 100 + NOA_VERSION_PATCH)

#define NOA_WEBSITE "https://github.com/ffyr2w/noa"

// Basics:
#include <iostream>
#include <fstream>
#include <string>
#include <cctype>

// Containers:
#include <map>
#include <unordered_map>
#include <vector>
#include <array>
#include <tuple>

// Others STL:
#include <utility>
#include <algorithm>
#include <memory>
#include <type_traits>

// noa commons:
#include "Exception.h"
#include "Log.h"
