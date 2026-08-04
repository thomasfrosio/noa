#include <filesystem>
#include <fstream>
#include <string>

#include <noa/io/IO.hpp>

#include "Catch.hpp"

using namespace ::noa::types;
namespace io = ::noa::io;

#define CREATE_FILE(filename, string) {                                         \
std::ofstream ofstream_(filename, std::ios::out | std::ios::trunc);             \
ofstream_.write(string.data(), static_cast<std::streamsize>(string.length()));  \
ofstream_.close(); }

#define CREATE_EMPTY_FILE(filename) {                               \
std::ofstream ofstream_(filename, std::ios::out | std::ios::trunc); \
ofstream_.close(); }

TEST_CASE("io::os") {
    namespace fs = std::filesystem;

    // Get some data
    const fs::path cwd = std::filesystem::current_path();
    const fs::path test_dir = cwd / "testOS";
    std::filesystem::remove_all(test_dir);
    std::filesystem::create_directory(test_dir);

    // One empty file:
    const fs::path file1 = test_dir / "file1.txt";
    std::string file1_content{};
    CREATE_FILE(file1, file1_content);

    // An another file...
    const fs::path file2 = test_dir / "file2.txt";
    std::string file2_content = "Hello, this is just to create a file with a size of 60 bytes";
    CREATE_FILE(file2, file2_content);

    // And an another file...
    const fs::path file3 = test_dir / "file3.txt";
    std::string file3_content = "Hello world";
    CREATE_FILE(file3, file3_content);

    // And a symlink...
    const fs::path symlink1 = test_dir / "symlink1.txt";
    std::filesystem::create_symlink(file3, symlink1);

    const fs::path symlink2 = test_dir / "file3_symlink.txt";
    std::filesystem::create_symlink("i_do_not_exist.txt", symlink2);

    // Subdirectories
    const fs::path test_dir_subdir1 = test_dir / "subdir1";
    const fs::path test_dir_subdir2 = test_dir / "subdir2";
    std::filesystem::create_directory(test_dir_subdir1);
    std::filesystem::create_directory(test_dir_subdir2);

    const fs::path subdir2_file1 = test_dir_subdir2 / "file11.txt";
    const fs::path subdir2_file2 = test_dir_subdir2 / "file12.txt";
    const fs::path subdir2_symlink1 = test_dir_subdir2 / "symlink12.txt";
    CREATE_EMPTY_FILE(subdir2_file1);
    CREATE_EMPTY_FILE(subdir2_file2);
    fs::create_symlink(subdir2_file2, subdir2_symlink1);

    AND_THEN("is_file") {
        REQUIRE(io::is_file(file1));
        REQUIRE(io::is_file(file2));
        REQUIRE(io::is_file(file3));
        REQUIRE(io::is_file(symlink1));
        REQUIRE_FALSE(io::is_file("i_do_not_exist.txt"));
        REQUIRE_FALSE(io::is_file(symlink2));
        REQUIRE_FALSE(io::is_file(test_dir));
    }

    AND_THEN("exists") {
        REQUIRE(io::is_file_or_directory(file1));
        REQUIRE(io::is_file_or_directory(file2));
        REQUIRE(io::is_file_or_directory(file3));
        REQUIRE(io::is_file_or_directory(symlink1));
        REQUIRE_FALSE(io::is_file_or_directory("i_do_not_exist.txt"));
        REQUIRE_FALSE(io::is_file_or_directory(symlink2));
        REQUIRE(io::is_file_or_directory(test_dir));
    }

    AND_THEN("size") {
        REQUIRE(io::file_size(file1) == static_cast<i64>(file1_content.size()));
        REQUIRE(io::file_size(file2) == static_cast<i64>(file2_content.size()));
        REQUIRE(io::file_size(file3) == static_cast<i64>(file3_content.size()));
        REQUIRE(io::file_size(symlink1) == static_cast<i64>(file3_content.size()));
        REQUIRE_THROWS_AS(io::file_size(test_dir / "foo.txt"), noa::Exception);
    }

    AND_THEN("remove(All)") {
        // Remove empty dir.
        REQUIRE(io::is_file_or_directory(test_dir_subdir1));
        io::remove(test_dir_subdir1);
        REQUIRE_FALSE(io::is_file_or_directory(test_dir_subdir1));

        // Remove single file.
        REQUIRE(io::is_file(subdir2_file1));
        io::remove(subdir2_file1);
        REQUIRE_FALSE(io::is_file(subdir2_file1));

        // Remove symlink but not its target.
        REQUIRE(io::is_file(subdir2_symlink1));
        io::remove(subdir2_symlink1);
        REQUIRE_FALSE(io::is_file(subdir2_symlink1));
        REQUIRE(io::is_file(subdir2_file2));

        // Removing non-empty directories only works with removeAll().
        REQUIRE_THROWS_AS(io::remove(test_dir_subdir2), noa::Exception);
        io::remove_all(test_dir_subdir2);
        REQUIRE_FALSE(io::is_file_or_directory(test_dir_subdir2));
    }

    AND_THEN("move") {
        AND_THEN("regular files") {
            REQUIRE(io::is_file(file1));
            io::move(file1, file2);
            REQUIRE_FALSE(io::is_file(file1));
            const fs::path new_file = test_dir / "new_file.txt";
            REQUIRE_FALSE(io::is_file(new_file));
            io::move(file2, new_file);
            REQUIRE(io::is_file(new_file));

            // Moving file to a directory
            REQUIRE_THROWS_AS(io::move(file3, test_dir_subdir1), noa::Exception);
            io::move(file3, test_dir_subdir1 / file3.filename());
            REQUIRE_FALSE(io::is_file(file3));
            REQUIRE(io::is_file(test_dir_subdir1 / file3.filename()));
        }

        AND_THEN("symlinks") {
            REQUIRE(fs::is_symlink(symlink1));
            REQUIRE(io::is_file(symlink1));
            REQUIRE_FALSE(io::is_file(symlink2)); // invalid target

            io::move(symlink1, symlink2);
            REQUIRE_FALSE(io::is_file(symlink1));
            REQUIRE(io::is_file(symlink2));
            REQUIRE(io::is_file(file3));  // symlinks are not followed
            REQUIRE(fs::is_symlink(symlink2));

            const fs::path new_file = test_dir / "new_symlink.txt";
            REQUIRE_FALSE(io::is_file_or_directory(new_file));
            io::move(symlink2, new_file);
            REQUIRE(io::is_file_or_directory(new_file));
            REQUIRE(fs::is_symlink(new_file));
        }

        AND_THEN("directories") {
            const fs::path new_dir = test_dir / "new_subdir";
            REQUIRE(io::is_file_or_directory(test_dir_subdir1));
            io::move(test_dir_subdir1, new_dir);
            REQUIRE(io::is_file_or_directory(new_dir));
            REQUIRE_FALSE(io::is_file_or_directory(test_dir_subdir1));
        }
    }

    AND_THEN("copy(File|Symlink)") {
        AND_THEN("copy_file") {
            // Copy to non-existing file.
            REQUIRE_FALSE(io::is_file(test_dir_subdir2 / file1.filename()));
            REQUIRE(io::copy_file(file1, test_dir_subdir2 / file1.filename()));
            REQUIRE(io::is_file(test_dir_subdir2 / file1.filename()));

            // Copy to existing file.
            REQUIRE(io::copy_file(file1, file2));
            REQUIRE_FALSE(io::copy_file(file1, file2, fs::copy_options::skip_existing));

            // Copy directory is not allowed. Use copy().
            REQUIRE_THROWS_AS(io::copy_file(test_dir_subdir1, test_dir_subdir2), noa::Exception);
            REQUIRE_THROWS_AS(io::copy_file(file1, test_dir_subdir1), noa::Exception);
            REQUIRE_THROWS_AS(io::copy_file(test_dir_subdir1, file2), noa::Exception);

            // Copy symlink is copying the target not the link.
            REQUIRE_FALSE(io::is_file(test_dir_subdir2 / file2.filename()));
            REQUIRE(io::copy_file(symlink1, test_dir_subdir2 / file2.filename()));
            REQUIRE(io::is_file(test_dir_subdir2 / file2.filename()));
        }

        AND_THEN("copy_symlink") {
            io::copy_symlink(symlink1, test_dir_subdir2 / symlink1.filename());
            REQUIRE(io::is_file(test_dir_subdir2 / symlink1.filename()));
        }

        AND_THEN("copy") {
            // Copying files should be like copy_file
            REQUIRE_FALSE(io::is_file(test_dir_subdir2 / file1.filename()));
            io::copy(file1, test_dir_subdir2 / file1.filename());
            REQUIRE(io::is_file(test_dir_subdir2 / file1.filename()));

            // Copying directories, with subdirectories, etc.
            io::copy(test_dir_subdir2, test_dir_subdir1 / test_dir_subdir2.filename());

            io::mkdir(test_dir / "subdir3");
            io::mkdir(test_dir / "subdir4");
            CREATE_EMPTY_FILE(test_dir / "subdir3/file31.txt");

            io::copy(test_dir / "subdir3", test_dir / "subdir4");
            REQUIRE(io::is_file(test_dir / "subdir4/file31.txt"));
            io::copy(test_dir / "subdir3", test_dir / "subdir4/subdir3");
            io::copy(test_dir / "subdir3", test_dir / "subdir4/subdir3"); // overwrite by default
            REQUIRE(io::is_file(test_dir / "subdir4/subdir3/file31.txt"));
        }
    }

    AND_THEN("backup") {
        io::backup(file3, false);
        REQUIRE(io::is_file(file3));
        REQUIRE(io::is_file(file3.string() + '~'));
        REQUIRE(io::file_size(file3.string() + '~') == static_cast<i64>(file3_content.size()));

        REQUIRE(io::copy_file(file2, file3));
        io::backup(file3, false);
        REQUIRE(io::is_file(file3));
        REQUIRE(io::is_file(file3.string() + '~'));
        REQUIRE(io::file_size(file3.string() + '~') == static_cast<i64>(file2_content.size()));

        REQUIRE(io::copy_file(file1, file3));
        io::backup(file3, true);
        REQUIRE_FALSE(io::is_file(file3));
        REQUIRE(io::is_file(file3.string() + '~'));
        REQUIRE(io::file_size(file3.string() + '~') == static_cast<i64>(file1_content.size()));
    }

    AND_THEN("mkdir") {
        io::mkdir(test_dir / "subdir3/subdir33/subdir3");
        REQUIRE(io::is_directory(test_dir / "subdir3/subdir33/subdir3"));
        io::mkdir("");
    }
    std::filesystem::remove_all(test_dir);
}
