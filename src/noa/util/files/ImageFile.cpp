#include "ImageFile.h"

// Must be included in the source file, as opposed to the header file,
// since MRCFile, etc. inherits from ImageFile.
#include "MRCFile.h"

using namespace ::Noa;

std::unique_ptr<ImageFile> ImageFile::get(const std::string& extension) {
    if (extension == ".mrc" || extension == ".st" || extension == ".rec" || extension == ".mrcs")
        return std::make_unique<MRCFile>();
        // else if (extension == ".tif" || extension == ".tiff")
        //    return std::make_unique<TIFFFile>();
        // else if (extension == ".eer")
        //    return std::make_unique<EERFile>();
    else
        return nullptr;
}
