#include <noa/io/OS.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <catch2/catch.hpp>

using namespace ::noa;

#define CREATE_FILE(filename, string) {                                         \
std::ofstream ofstream_(filename, std::ios::out | std::ios::trunc);             \
ofstream_.write(string.data(), static_cast<std::streamsize>(string.length()));  \
ofstream_.close(); }

#define CREATE_EMPTY_FILE(filename) {                               \
std::ofstream ofstream_(filename, std::ios::out | std::ios::trunc); \
ofstream_.close(); }

TEST_CASE("OS:", "[noa][OS]") {
    // Get some data
    fs::path cwd = std::filesystem::current_path();
    fs::path test_dir = cwd / "testOS";
    std::filesystem::remove_all(test_dir);
    std::filesystem::create_directory(test_dir);

    // One empty file:
    fs::path file1 = test_dir / "file1.txt";
    std::string file1_content{};
    CREATE_FILE(file1, file1_content);

    // An another file...
    fs::path file2 = test_dir / "file2.txt";
    std::string file2_content = "Hello, this is just to create a file with a size of 60 bytes";
    CREATE_FILE(file2, file2_content);

    // And an another file...
    fs::path file3 = test_dir / "file3.txt";
    std::string file3_content = "Hello world";
    CREATE_FILE(file3, file3_content);

    // And a symlink...
    fs::path symlink1 = test_dir / "symlink1.txt";
    std::filesystem::create_symlink(file3, symlink1);

    fs::path symlink2 = test_dir / "file3_symlink.txt";
    std::filesystem::create_symlink("i_do_not_exist.txt", symlink2);

    // Subdirectories
    fs::path test_dir_subdir1 = test_dir / "subdir1";
    fs::path test_dir_subdir2 = test_dir / "subdir2";
    std::filesystem::create_directory(test_dir_subdir1);
    std::filesystem::create_directory(test_dir_subdir2);

    fs::path subdir2_file1 = test_dir_subdir2 / "file11.txt";
    fs::path subdir2_file2 = test_dir_subdir2 / "file12.txt";
    fs::path subdir2_symlink1 = test_dir_subdir2 / "symlink12.txt";
    CREATE_EMPTY_FILE(subdir2_file1);
    CREATE_EMPTY_FILE(subdir2_file2);
    fs::create_symlink(subdir2_file2, subdir2_symlink1);

    AND_THEN("existsFile") {
        REQUIRE(os::existsFile(file1));
        REQUIRE(os::existsFile(file2));
        REQUIRE(os::existsFile(file3));
        REQUIRE(os::existsFile(symlink1));
        REQUIRE_FALSE(os::existsFile("i_do_not_exist.txt"));
        REQUIRE_FALSE(os::existsFile(symlink2));
        REQUIRE_FALSE(os::existsFile(test_dir));
    }

    AND_THEN("exists") {
        REQUIRE(os::exists(file1));
        REQUIRE(os::exists(file2));
        REQUIRE(os::exists(file3));
        REQUIRE(os::exists(symlink1));
        REQUIRE_FALSE(os::exists("i_do_not_exist.txt"));
        REQUIRE_FALSE(os::exists(symlink2));
        REQUIRE(os::exists(test_dir));
    }

    AND_THEN("size") {
        REQUIRE(os::size(file1) == file1_content.size());
        REQUIRE(os::size(file2) == file2_content.size());
        REQUIRE(os::size(file3) == file3_content.size());
        REQUIRE(os::size(symlink1) == file3_content.size());
        REQUIRE_THROWS_AS(os::size(test_dir / "foo.txt"), noa::Exception);
    }

    AND_THEN("remove(All)") {
        // Remove empty dir.
        REQUIRE(os::exists(test_dir_subdir1));
        os::remove(test_dir_subdir1);
        REQUIRE_FALSE(os::exists(test_dir_subdir1));

        // Remove single file.
        REQUIRE(os::existsFile(subdir2_file1));
        os::remove(subdir2_file1);
        REQUIRE_FALSE(os::existsFile(subdir2_file1));

        // Remove symlink but not its target.
        REQUIRE(os::existsFile(subdir2_symlink1));
        os::remove(subdir2_symlink1);
        REQUIRE_FALSE(os::existsFile(subdir2_symlink1));
        REQUIRE(os::existsFile(subdir2_file2));

        // Removing non-empty directories only works with removeAll().
        REQUIRE_THROWS_AS(os::remove(test_dir_subdir2), noa::Exception);
        os::removeAll(test_dir_subdir2);
        REQUIRE_FALSE(os::exists(test_dir_subdir2));
    }

    AND_THEN("move") {
        AND_THEN("regular files") {
            REQUIRE(os::existsFile(file1));
            os::move(file1, file2);
            REQUIRE_FALSE(os::existsFile(file1));
            fs::path new_file = test_dir / "new_file.txt";
            REQUIRE_FALSE(os::existsFile(new_file));
            os::move(file2, new_file);
            REQUIRE(os::existsFile(new_file));

            // Moving file to a directory
            REQUIRE_THROWS_AS(os::move(file3, test_dir_subdir1), noa::Exception);
            os::move(file3, test_dir_subdir1 / file3.filename());
            REQUIRE_FALSE(os::existsFile(file3));
            REQUIRE(os::existsFile(test_dir_subdir1 / file3.filename()));
        }

        AND_THEN("symlinks") {
            REQUIRE(fs::is_symlink(symlink1));
            REQUIRE(os::existsFile(symlink1));
            REQUIRE_FALSE(os::existsFile(symlink2)); // invalid target

            os::move(symlink1, symlink2);
            REQUIRE_FALSE(os::existsFile(symlink1));
            REQUIRE(os::existsFile(symlink2));
            REQUIRE(os::existsFile(file3));  // symlinks are not followed
            REQUIRE(fs::is_symlink(symlink2));

            fs::path new_file = test_dir / "new_symlink.txt";
            REQUIRE_FALSE(os::exists(new_file));
            os::move(symlink2, new_file);
            REQUIRE(os::exists(new_file));
            REQUIRE(fs::is_symlink(new_file));
        }

        AND_THEN("directories") {
            fs::path new_dir = test_dir / "new_subdir";
            REQUIRE(os::exists(test_dir_subdir1));
            os::move(test_dir_subdir1, new_dir);
            REQUIRE(os::exists(new_dir));
            REQUIRE_FALSE(os::exists(test_dir_subdir1));
        }
    }

    AND_THEN("copy(File|Symlink)") {
        AND_THEN("copyFile") {
            // Copy to non-existing file.
            REQUIRE_FALSE(os::existsFile(test_dir_subdir2 / file1.filename()));
            REQUIRE(os::copyFile(file1, test_dir_subdir2 / file1.filename()));
            REQUIRE(os::existsFile(test_dir_subdir2 / file1.filename()));

            // Copy to existing file.
            REQUIRE(os::copyFile(file1, file2));
            REQUIRE_FALSE(os::copyFile(file1, file2, fs::copy_options::skip_existing));

            // Copy directory is not allowed. Use copy().
            REQUIRE_THROWS_AS(os::copyFile(test_dir_subdir1, test_dir_subdir2), noa::Exception);
            REQUIRE_THROWS_AS(os::copyFile(file1, test_dir_subdir1), noa::Exception);
            REQUIRE_THROWS_AS(os::copyFile(test_dir_subdir1, file2), noa::Exception);

            // Copy symlink is copying the target not the link.
            REQUIRE_FALSE(os::existsFile(test_dir_subdir2 / file2.filename()));
            REQUIRE(os::copyFile(symlink1, test_dir_subdir2 / file2.filename()));
            REQUIRE(os::existsFile(test_dir_subdir2 / file2.filename()));
        }

        AND_THEN("copySymlink") {
            os::copySymlink(symlink1, test_dir_subdir2 / symlink1.filename());
            REQUIRE(os::existsFile(test_dir_subdir2 / symlink1.filename()));
        }

        AND_THEN("copy") {
            // Copying files should be like copyFile
            REQUIRE_FALSE(os::existsFile(test_dir_subdir2 / file1.filename()));
            os::copy(file1, test_dir_subdir2 / file1.filename());
            REQUIRE(os::existsFile(test_dir_subdir2 / file1.filename()));

            // Copying directories, with subdirectories, etc.
            os::copy(test_dir_subdir2, test_dir_subdir1 / test_dir_subdir2.filename());

            os::mkdir(test_dir / "subdir3");
            os::mkdir(test_dir / "subdir4");
            CREATE_EMPTY_FILE(test_dir / "subdir3/file31.txt");

            os::copy(test_dir / "subdir3", test_dir / "subdir4");
            REQUIRE(os::existsFile(test_dir / "subdir4/file31.txt"));
            os::copy(test_dir / "subdir3", test_dir / "subdir4/subdir3");
            os::copy(test_dir / "subdir3", test_dir / "subdir4/subdir3"); // overwrite by default
            REQUIRE(os::existsFile(test_dir / "subdir4/subdir3/file31.txt"));
        }
    }

    AND_THEN("backup") {
        os::backup(file3, false);
        REQUIRE(os::existsFile(file3));
        REQUIRE(os::existsFile(file3.string() + '~'));
        REQUIRE(os::size(file3.string() + '~') == file3_content.size());

        REQUIRE(os::copyFile(file2, file3));
        os::backup(file3, false);
        REQUIRE(os::existsFile(file3));
        REQUIRE(os::existsFile(file3.string() + '~'));
        REQUIRE(os::size(file3.string() + '~') == file2_content.size());

        REQUIRE(os::copyFile(file1, file3));
        os::backup(file3, true);
        REQUIRE_FALSE(os::existsFile(file3));
        REQUIRE(os::existsFile(file3.string() + '~'));
        REQUIRE(os::size(file3.string() + '~') == file1_content.size());
    }

    AND_THEN("mkdir") {
        os::mkdir(test_dir / "subdir3/subdir33/subdir3");
        REQUIRE(os::exists(test_dir / "subdir3/subdir33/subdir3"));
        os::mkdir("");
    }
    std::filesystem::remove_all(test_dir);
}
