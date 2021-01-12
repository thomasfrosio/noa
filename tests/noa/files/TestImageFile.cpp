#include <catch2/catch.hpp>
#include "../../Helpers.h"

#include "noa/util/files/ImageFile.h"

using namespace ::Noa;

TEST_CASE("ImageFile:" , "[noa][file]") {
    fs::path fixture = NOA_TESTS_FIXTURE;
    fixture /= "TestImageFile_mrcfile.mrc";

//    ImageFile file(fixture);
//     ImageFile file(fixture, ImageFile::Type::MRC);
}
