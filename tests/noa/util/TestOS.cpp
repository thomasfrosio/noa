#include <noa/util/OS.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <catch2/catch.hpp>

using namespace ::Noa;

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
        REQUIRE(OS::existsFile(file1));
        REQUIRE(OS::existsFile(file2));
        REQUIRE(OS::existsFile(file3));
        REQUIRE(OS::existsFile(symlink1));
        REQUIRE_FALSE(OS::existsFile("i_do_not_exist.txt"));
        REQUIRE_FALSE(OS::existsFile(symlink2));
        REQUIRE_FALSE(OS::existsFile(test_dir));
    }

    AND_THEN("exists") {
        REQUIRE(OS::exists(file1));
        REQUIRE(OS::exists(file2));
        REQUIRE(OS::exists(file3));
        REQUIRE(OS::exists(symlink1));
        REQUIRE_FALSE(OS::exists("i_do_not_exist.txt"));
        REQUIRE_FALSE(OS::exists(symlink2));
        REQUIRE(OS::exists(test_dir));
    }

    AND_THEN("size") {
        REQUIRE(OS::size(file1) == file1_content.size());
        REQUIRE(OS::size(file2) == file2_content.size());
        REQUIRE(OS::size(file3) == file3_content.size());
        REQUIRE(OS::size(symlink1) == file3_content.size());
        REQUIRE_THROWS_AS(OS::size(test_dir / "foo.txt"), Noa::Exception);
    }

    AND_THEN("remove(All)") {
        // Remove empty dir.
        REQUIRE(OS::exists(test_dir_subdir1));
        OS::remove(test_dir_subdir1);
        REQUIRE_FALSE(OS::exists(test_dir_subdir1));

        // Remove single file.
        REQUIRE(OS::existsFile(subdir2_file1));
        OS::remove(subdir2_file1);
        REQUIRE_FALSE(OS::existsFile(subdir2_file1));

        // Remove symlink but not its target.
        REQUIRE(OS::existsFile(subdir2_symlink1));
        OS::remove(subdir2_symlink1);
        REQUIRE_FALSE(OS::existsFile(subdir2_symlink1));
        REQUIRE(OS::existsFile(subdir2_file2));

        // Removing non-empty directories only works with removeAll().
        REQUIRE_THROWS_AS(OS::remove(test_dir_subdir2), Noa::Exception);
        OS::removeAll(test_dir_subdir2);
        REQUIRE_FALSE(OS::exists(test_dir_subdir2));
    }

    AND_THEN("move") {
        AND_THEN("regular files") {
            REQUIRE(OS::existsFile(file1));
            OS::move(file1, file2);
            REQUIRE_FALSE(OS::existsFile(file1));
            fs::path new_file = test_dir / "new_file.txt";
            REQUIRE_FALSE(OS::existsFile(new_file));
            OS::move(file2, new_file);
            REQUIRE(OS::existsFile(new_file));

            // Moving file to a directory
            REQUIRE_THROWS_AS(OS::move(file3, test_dir_subdir1), Noa::Exception);
            OS::move(file3, test_dir_subdir1 / file3.filename());
            REQUIRE_FALSE(OS::existsFile(file3));
            REQUIRE(OS::existsFile(test_dir_subdir1 / file3.filename()));
        }

        AND_THEN("symlinks") {
            REQUIRE(fs::is_symlink(symlink1));
            REQUIRE(OS::existsFile(symlink1));
            REQUIRE_FALSE(OS::existsFile(symlink2)); // invalid target

            OS::move(symlink1, symlink2);
            REQUIRE_FALSE(OS::existsFile(symlink1));
            REQUIRE(OS::existsFile(symlink2));
            REQUIRE(OS::existsFile(file3));  // symlinks are not followed
            REQUIRE(fs::is_symlink(symlink2));

            fs::path new_file = test_dir / "new_symlink.txt";
            REQUIRE_FALSE(OS::exists(new_file));
            OS::move(symlink2, new_file);
            REQUIRE(OS::exists(new_file));
            REQUIRE(fs::is_symlink(new_file));
        }

        AND_THEN("directories") {
            fs::path new_dir = test_dir / "new_subdir";
            REQUIRE(OS::exists(test_dir_subdir1));
            OS::move(test_dir_subdir1, new_dir);
            REQUIRE(OS::exists(new_dir));
            REQUIRE_FALSE(OS::exists(test_dir_subdir1));
        }
    }

    AND_THEN("copy(File|Symlink)") {
        AND_THEN("copyFile") {
            // Copy to non-existing file.
            REQUIRE_FALSE(OS::existsFile(test_dir_subdir2 / file1.filename()));
            REQUIRE(OS::copyFile(file1, test_dir_subdir2 / file1.filename()));
            REQUIRE(OS::existsFile(test_dir_subdir2 / file1.filename()));

            // Copy to existing file.
            REQUIRE(OS::copyFile(file1, file2));
            REQUIRE_FALSE(OS::copyFile(file1, file2, fs::copy_options::skip_existing));

            // Copy directory is not allowed. Use copy().
            REQUIRE_THROWS_AS(OS::copyFile(test_dir_subdir1, test_dir_subdir2), Noa::Exception);
            REQUIRE_THROWS_AS(OS::copyFile(file1, test_dir_subdir1), Noa::Exception);
            REQUIRE_THROWS_AS(OS::copyFile(test_dir_subdir1, file2), Noa::Exception);

            // Copy symlink is copying the target not the link.
            REQUIRE_FALSE(OS::existsFile(test_dir_subdir2 / file2.filename()));
            REQUIRE(OS::copyFile(symlink1, test_dir_subdir2 / file2.filename()));
            REQUIRE(OS::existsFile(test_dir_subdir2 / file2.filename()));
        }

        AND_THEN("copySymlink") {
            OS::copySymlink(symlink1, test_dir_subdir2 / symlink1.filename());
            REQUIRE(OS::existsFile(test_dir_subdir2 / symlink1.filename()));
        }

        AND_THEN("copy") {
            // Copying files should be like copyFile
            REQUIRE_FALSE(OS::existsFile(test_dir_subdir2 / file1.filename()));
            OS::copy(file1, test_dir_subdir2 / file1.filename());
            REQUIRE(OS::existsFile(test_dir_subdir2 / file1.filename()));

            // Copying directories, with subdirectories, etc.
            OS::copy(test_dir_subdir2, test_dir_subdir1 / test_dir_subdir2.filename());

            OS::mkdir(test_dir / "subdir3");
            OS::mkdir(test_dir / "subdir4");
            CREATE_EMPTY_FILE(test_dir / "subdir3/file31.txt");

            OS::copy(test_dir / "subdir3", test_dir / "subdir4");
            REQUIRE(OS::existsFile(test_dir / "subdir4/file31.txt"));
            OS::copy(test_dir / "subdir3", test_dir / "subdir4/subdir3");
            OS::copy(test_dir / "subdir3", test_dir / "subdir4/subdir3"); // overwrite by default
            REQUIRE(OS::existsFile(test_dir / "subdir4/subdir3/file31.txt"));
        }
    }

    AND_THEN("backup") {
        OS::backup(file3, false);
        REQUIRE(OS::existsFile(file3));
        REQUIRE(OS::existsFile(file3.string() + '~'));
        REQUIRE(OS::size(file3.string() + '~') == file3_content.size());

        REQUIRE(OS::copyFile(file2, file3));
        OS::backup(file3, false);
        REQUIRE(OS::existsFile(file3));
        REQUIRE(OS::existsFile(file3.string() + '~'));
        REQUIRE(OS::size(file3.string() + '~') == file2_content.size());

        REQUIRE(OS::copyFile(file1, file3));
        OS::backup(file3, true);
        REQUIRE_FALSE(OS::existsFile(file3));
        REQUIRE(OS::existsFile(file3.string() + '~'));
        REQUIRE(OS::size(file3.string() + '~') == file1_content.size());
    }

    AND_THEN("mkdir") {
        OS::mkdir(test_dir / "subdir3/subdir33/subdir3");
        REQUIRE(OS::exists(test_dir / "subdir3/subdir33/subdir3"));
        OS::mkdir("");
    }
    std::filesystem::remove_all(test_dir);
}
