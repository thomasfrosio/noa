/*
 * Test noa/files/Files.h
 */

#include <catch2/catch.hpp>
#include "noa/util/OS.h"

using namespace ::Noa;

//
//SCENARIO("OS: basics", "[noa][file]") {
//
//
//    std::filesystem::remove_all("./testfiles");
//    std::filesystem::create_directory("./testfiles");
//
//    // One empty file:
//    std::string file1 = "./testfiles/TestFile_1.txt";
//    fs::file_status file1_stat = fs::status(file1);
//    std::ofstream ofstream(file1, std::ios::out | std::ios::trunc);
//    ofstream.close();
//
//    // An another file with size of 60 bytes.
//    std::string file2 = "./testfiles/TestFile_2.txt";
//    fs::file_status file2_stat = fs::status(file2);
//    ofstream.open(file2, std::ios::out | std::ios::trunc);
//    ofstream.write("Hello, this is just to create a file with a size of 60 bytes", 60);
//    ofstream.close();
//
//    // Another file with size of 11 bytes.
//    std::string file3 = "./testfiles/TestFile_3.txt";
//    fs::file_status file3_stat = fs::status(file3);
//    ofstream.open(file3, std::ios::out | std::ios::trunc);
//    ofstream.write("Hello world", 11);
//    ofstream.close();
//
//    // And a symlink to this file.
//    std::string file4 = "./testfiles/TestFile_3_symlink.txt";
//    fs::file_status file4_stat = fs::status(file4);
//    std::string cwd = std::filesystem::current_path();
//    std::filesystem::create_symlink(cwd + "/testfiles/TestFile_3.txt", file4);
//
//    errno_t err{Errno::good};
//
//    WHEN("asking for the size of the file") {
//        REQUIRE(OS::size(file1, file1_stat, err) == 0U && err == Errno::good);
//        REQUIRE(OS::size(file2, file2_stat, err) == 60U && err == Errno::good);
//        REQUIRE(OS::size(file3, file3_stat, err) == 11U && err == Errno::good);
//        REQUIRE(OS::size(file4, file4_stat, err) == 11U && err == Errno::good);
//        REQUIRE(OS::size(file4, file4_stat, err) == 0U && err == Errno::fail_os);
//
//        REQUIRE(OS::size(file1, err) == 0U && err == Errno::good);
//        REQUIRE(OS::size(file2, err) == 60U && err == Errno::good);
//        REQUIRE(OS::size(file3, err) == 11U && err == Errno::good);
//        REQUIRE(OS::size(file4, err) == 11U && err == Errno::good);
//        REQUIRE(OS::size(file4, err) == 0U && err == Errno::fail_os);
//    }
//
//    WHEN("asking if the file exists") {
//        REQUIRE(OS::exist(file1, err));
//        REQUIRE(OS::exist(file2, err));
//        REQUIRE(OS::exist(file3, err));
//        REQUIRE(OS::exist(file4, err));
//        REQUIRE(!OS::exist("i_do_not_exist.foo", err));
//    }
//
//    WHEN("asking to remove files or directories") {
//        std::filesystem::create_directory("./testfiles/dir1");
//        std::filesystem::create_directory("./testfiles/dir2");
//        ofstream.open("./testfiles/dir2/to_delete1.txt", std::ios::trunc);
//        ofstream.close();
//        ofstream.open("./testfiles/dir2/to_delete2.txt", std::ios::trunc);
//        ofstream.close();
//
//        REQUIRE(OS::exist("./testfiles/dir1"));
//        OS::remove("./testfiles/dir1");
//        REQUIRE(!OS::exist("./testfiles/dir1"));
//
//        REQUIRE(OS::exist("./testfiles/dir2/to_delete1.txt"));
//        OS::remove("./testfiles/dir2/to_delete1.txt");
//        REQUIRE(!OS::exist("./testfiles/dir2/to_delete1.txt"));
//        REQUIRE_THROWS_AS(OS::remove("./testfiles/dir2"), Error);
//        OS::removeDirectory("./testfiles/dir2");
//        REQUIRE(!OS::exist("./testfiles/dir2"));
//    }
//
//    WHEN("asking to rename files or directories") {
//        OS::rename(file1, "./testfiles/file1_rename.txt");
//        REQUIRE(!OS::exist(file1));
//        REQUIRE(OS::exist("./testfiles/file1_rename.txt"));
//    }
//}
