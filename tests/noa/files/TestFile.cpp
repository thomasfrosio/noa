#include <catch2/catch.hpp>
#include "../../Helpers.h"

#include "noa/files/File.h"

using namespace ::Noa;
namespace fs = std::filesystem;

TEST_CASE("File:", "[noa][file]") {
    fs::path test_dir = "testFile";
    fs::path test_file1 = test_dir / "testFile1.txt";
    fs::path test_file2 = test_dir / "subdir/testFile2.txt";
    fs::remove_all(test_dir);

    errno_t err{Errno::good};
    std::fstream fstream;

    File::open(test_file1, fstream, std::ios::out);
    REQUIRE(fstream.is_open());

    fstream.write("123456789", 9);

    // Open a file that do not exist in read mode.
    err = File::open(test_file2, fstream, std::ios::in);
    REQUIRE(!fstream.is_open());
    REQUIRE(err == Errno::fail_open);

    err = Errno::good;
    REQUIRE(OS::size(test_file1, err) == 9);

    err = File::open(test_file2, fstream, std::ios::in | std::ios::out);
    REQUIRE(!fstream.is_open());
    REQUIRE(err == Errno::fail_open);

    err = File::open(test_file2, fstream, std::ios::out | std::ios::app);
    REQUIRE(fstream.is_open());
    REQUIRE(OS::existsFile(test_file2, err));
    REQUIRE_ERRNO_GOOD(err);

    err = File::close(fstream);
    REQUIRE(!fstream.is_open());
    REQUIRE_ERRNO_GOOD(err);

    // Backup copy
    fs::path test_file2_backup = test_file2.string() + '~';
    err = File::open(test_file2, fstream, std::ios::out);
    REQUIRE(fstream.is_open());
    REQUIRE(OS::existsFile(test_file2_backup, err));
    REQUIRE_ERRNO_GOOD(err);

    OS::remove(test_file2_backup);

    // Backup move
    err = File::open(test_file2, fstream, std::ios::out | std::ios::trunc);
    REQUIRE(fstream.is_open());
    REQUIRE(OS::existsFile(test_file2_backup, err));
    REQUIRE_ERRNO_GOOD(err);

    fs::remove_all(test_dir);
}
