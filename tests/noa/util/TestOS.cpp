/*
 * Test noa/files/Files.h
 */

#include <catch2/catch.hpp>
#include "noa/util/OS.h"


SCENARIO("Noa::Files: check the basic functionalities work", "[noa][file]") {
    using namespace Noa;

    std::filesystem::remove_all("./testfiles");
    std::filesystem::create_directory("./testfiles");

    // Generate some fixtures to work on...
    // One empty file:
    std::string filename1 = "./testfiles/TestFile_1.txt";
    std::ofstream ofstream(filename1, std::ios::out | std::ios::trunc);
    ofstream.close();

    // An another file with size of 60 bytes.
    std::string filename2 = "./testfiles/TestFile_2.txt";
    ofstream.open(filename2, std::ios::out | std::ios::trunc);
    ofstream.write("Hello, this is just to create a file with a size of 60 bytes", 60);
    ofstream.close();

    // Another file with size of 11 bytes.
    std::string filename3 = "./testfiles/TestFile_3.txt";
    ofstream.open(filename3, std::ios::out | std::ios::trunc);
    ofstream.write("Hello world", 11);
    ofstream.close();

    // And a symlink to this file.
    std::string filename4 = "./testfiles/TestFile_3_symlink.txt";
    std::string cwd = std::filesystem::current_path();
    std::filesystem::create_symlink(cwd + "/testfiles/TestFile_3.txt", filename4);

    WHEN("asking for the size of the file") {
        REQUIRE(OS::size(filename1) == 0U);
        REQUIRE(OS::size(filename2) == 60U);
        REQUIRE(OS::size(filename3) == 11U);
        REQUIRE(OS::size(filename4) == 11U);
        REQUIRE_THROWS_AS(OS::size("i_do_not_exist.foo"), ErrorCore);
    }

    WHEN("asking if the file exists") {
        REQUIRE(OS::exist(filename1));
        REQUIRE(OS::exist(filename2));
        REQUIRE(OS::exist(filename3));
        REQUIRE(OS::exist(filename4));
        REQUIRE(!OS::exist("i_do_not_exist.foo"));
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
        REQUIRE_THROWS_AS(OS::remove("./testfiles/dir2"), ErrorCore);
        OS::removeDirectory("./testfiles/dir2");
        REQUIRE(!OS::exist("./testfiles/dir2"));
    }

    WHEN("asking to rename files or directories") {
        OS::rename(filename1, "./testfiles/filename1_rename.txt");
        REQUIRE(!OS::exist(filename1));
        REQUIRE(OS::exist("./testfiles/filename1_rename.txt"));
    }
}
