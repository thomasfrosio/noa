#include <noa/core/io/MemoryMappedFile.hpp>
#include <catch2/catch.hpp>

using namespace ::noa::types;

TEST_CASE("core::io::MemoryMappedFile", "[noa]") {
    namespace fs = std::filesystem;

    const fs::path test_dir = "test_MemoryMappedFile";
    const fs::path test_file1 = test_dir / "file1";
    const fs::path test_file2 = test_dir / "subdir/file2";
    noa::io::remove_all(test_dir);

    AND_WHEN("file should exists") {
        noa::io::MemoryMappedFile file;
        REQUIRE_THROWS_AS(file.open(test_file2, {.read=true}), noa::Exception);
        REQUIRE_FALSE(file.is_open());

        REQUIRE_THROWS_AS(file.open(test_file2, {.read=true, .write=true}), noa::Exception);
        REQUIRE_FALSE(file.is_open());
    }

    AND_WHEN("read and write") {
        noa::io::MemoryMappedFile file;
        file.open(test_file1, {.write = true}, {.new_size = 32});
        auto s0 = file.as_bytes().as<char>();
        for (auto& e: s0)
            e = 'a';
        file.close();
        REQUIRE(noa::io::file_size(test_file1) == 32);

        file.open(test_file1, {.read = true});
        auto s1 = file.as_bytes().as<const char>();
        REQUIRE(s1.size() == 32);
        bool match = true;
        for (auto&e : s1) {
            if (e != 'a') {
                match = false;
                break;
            }
        }
        REQUIRE(match);
        file.close();

        // Append
        using noa::indexing::Slice;
        file.open(test_file1, {.read = true, .write = true}, {.new_size = 64});
        auto s2 = file.as_bytes().as<char>();
        REQUIRE(s2.size() == 64);
        match = true;
        for (auto& e: s2.subregion(Slice{0, 32})) {
            if (e != 'a') {
                match = false;
                break;
            }
        }
        REQUIRE(match);
        for (auto& e: s2.subregion(Slice{32, 64}))
            e = 'b';
        file.open(test_file1, {.read = true});
        REQUIRE(noa::io::file_size(test_file1) == 64);
        auto s3 = file.as_bytes().as<const char>();
        REQUIRE(s3.size() == 64);
        match = true;
        file.optimize_for_sequential_access();
        for (i32 i{}; auto& e: s2) {
            if ((i < 32 and e != 'a') or (i >= 32 and e != 'b')) {
                match = false;
                break;
            }
            i++;
        }
        REQUIRE(match);
    }

    AND_WHEN("backup copy and backup move") {
        noa::io::MemoryMappedFile file(test_file2, {.write = true}, {.new_size = 12});
        REQUIRE(file.size() == 12);
        file.close();

        // Backup copy: open an existing file in writing mode.
        const fs::path test_file2_backup = test_file2.string() + '~';
        file.open(test_file2, {.read = true, .write = true}, {.new_size = 6});
        REQUIRE(file.is_open());
        file.close();
        REQUIRE((noa::io::is_file(test_file2_backup) and
                 noa::io::file_size(test_file2_backup) == 12 and
                 noa::io::file_size(test_file2) == 6));

        noa::io::remove(test_file2_backup);

        // Backup move: open an existing file in overwriting mode.
        file.open(test_file2, {.write = true}, {.new_size = 0});
        REQUIRE(file.is_open());
        file.close();
        REQUIRE((noa::io::is_file(test_file2_backup) and
                 noa::io::file_size(test_file2_backup) == 6 and
                 noa::io::file_size(test_file2) == 0));
    }

    AND_WHEN("private doesn't update the file") {
        noa::io::MemoryMappedFile file(test_file1, {.write = true}, {.new_size = 32});
        auto s0 = file.as_bytes().as<char>();
        for (auto& e: s0)
            e = 'a';
        file.close();
        REQUIRE(noa::io::file_size(test_file1) == 32);

        file.open(test_file1, {.read = true, .write = true}, {.keep_private = true});
        auto s1 = file.as_bytes().as<char>();
        REQUIRE(s1.size() == 32);
        for (auto& e: s1)
            e = 'b';
        file.close();

        file.open(test_file1, {.read = true});
        auto s2 = file.as_bytes().as<const char>();
        REQUIRE(s2.size() == 32);
        bool match = true;
        for (auto& e: s2) {
            if (e != 'a') {
                match = false;
                break;
            }
        }
        REQUIRE(match);
        file.close();
    }

    noa::io::remove_all(test_dir);
}
