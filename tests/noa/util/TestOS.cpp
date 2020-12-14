#include <catch2/catch.hpp>
#include "../../Helpers.h"

#include "noa/util/OS.h"

using namespace ::Noa;

#define CREATE_FILE(filename, string) {                                         \
std::ofstream ofstream_(filename, std::ios::out | std::ios::trunc);             \
ofstream_.write(string.data(), static_cast<std::streamsize>(string.length()));  \
ofstream_.close(); }


TEST_CASE("OS:", "[noa][file]") {
    // Get some fixtures
    std::string cwd = std::filesystem::current_path();
    std::string test_dir = cwd + "./testfiles/";
    std::filesystem::remove_all(test_dir);
    std::filesystem::create_directory(test_dir);

    // One empty file:
    std::string file1 = test_dir + "TestFile_1.txt";
    std::string file1_content{};
    fs::file_status file1_stat = fs::status(file1);
    CREATE_FILE(file1, file1_content);

    // An another file...
    std::string file2 = test_dir + "TestFile_2.txt";
    std::string file2_content = "Hello, this is just to create a file with a size of 60 bytes";
    fs::file_status file2_stat = fs::status(file2);
    CREATE_FILE(file2, file2_content);

    // And an another file...
    std::string file3 = test_dir + "TestFile_3.txt";
    std::string file3_content = "Hello world";
    fs::file_status file3_stat = fs::status(file3);
    CREATE_FILE(file3, file3_content);

    // And a symlink...
    std::string file4 = test_dir + "TestFile_3_symlink.txt";
    fs::file_status file4_stat = fs::status(file4);
    std::filesystem::create_symlink(test_dir + "TestFile_3.txt", file4);

    errno_t err{Errno::good};
    AND_GIVEN("size") {
        //@CLION-formatter:off
        REQUIRE(OS::size(file1, err) == file1_content.size());  REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::size(file2, err) == file2_content.size());  REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::size(file3, err) == file3_content.size());  REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::size(file4, err) == file3_content.size());  REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::size(test_dir + "foo.txt", err) == 0U);     REQUIRE(err == Errno::fail_os);
        //@CLION-formatter:on
    }

    AND_GIVEN("exist") {
        REQUIRE(OS::exist(file1, err));
        REQUIRE(OS::exist(file2, err));
        REQUIRE(OS::exist(file3, err));
        REQUIRE(OS::exist(file4, err));
        REQUIRE(!OS::exist("i_do_not_exist.foo", err));
    }

    WHEN("asking to remove files or directories") {
        std::filesystem::create_directory("./testfiles/dir1");
        std::filesystem::create_directory("./testfiles/dir2");
        ofstream.open("./testfiles/dir2/to_delete1.txt", std::ios::trunc);
        ofstream.close();
        ofstream.open("./testfiles/dir2/to_delete2.txt", std::ios::trunc);
        ofstream.close();

        REQUIRE(OS::exist("./testfiles/dir1"));
        OS::remove("./testfiles/dir1");
        REQUIRE(!OS::exist("./testfiles/dir1"));

        REQUIRE(OS::exist("./testfiles/dir2/to_delete1.txt"));
        OS::remove("./testfiles/dir2/to_delete1.txt");
        REQUIRE(!OS::exist("./testfiles/dir2/to_delete1.txt"));
        REQUIRE_THROWS_AS(OS::remove("./testfiles/dir2"), Error);
        OS::removeDirectory("./testfiles/dir2");
        REQUIRE(!OS::exist("./testfiles/dir2"));
    }

    WHEN("asking to rename files or directories") {
        OS::rename(file1, "./testfiles/file1_rename.txt");
        REQUIRE(!OS::exist(file1));
        REQUIRE(OS::exist("./testfiles/file1_rename.txt"));
    }
}
