#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Atlas.h>
#include <noa/cpu/memory/Set.h>
#include <noa/io/files/ImageFile.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEST_CASE("Memory::atlas()", "[noa][cpu]") {
//    uint subregion_count = 2;
//    size3_t subregion_shape(50,50,1);
//    size_t elements = getElements(subregion_shape);
//    Memory::PtrHost<float> subregions(elements * subregion_count);
//
//    for (uint idx = 0; idx < subregion_count; ++idx)
//        Memory::set(subregions.get() + idx * elements, elements, 0.f);
//
//    Memory::PtrHost<size3_t> atlas_centers;
//    size3_t atlas_shape = Memory::getAtlasLayout(subregion_shape, subregion_count, atlas_centers.get());
//    Memory::PtrHost<float> atlas(getElements(atlas_shape));
//    Memory::insert(subregions.get(), subregion_shape, subregion_count, atlas.get(), atlas_shape, atlas_centers.get());
//    ImageFile::save(Test::PATH_TEST_DATA / "atlas.mrc", atlas.get(), atlas_shape);
}
