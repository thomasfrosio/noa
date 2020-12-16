#include <catch2/catch.hpp>
#include "../../Helpers.h"

#include "noa/util/OS.h"

using namespace ::Noa;

#define CREATE_FILE(filename, string) {                                         \
std::ofstream ofstream_(filename, std::ios::out | std::ios::trunc);             \
ofstream_.write(string.data(), static_cast<std::streamsize>(string.length()));  \
ofstream_.close(); }

#define CREATE_EMPTY_FILE(filename) {                               \
std::ofstream ofstream_(filename, std::ios::out | std::ios::trunc); \
ofstream_.close(); }


TEST_CASE("OS:", "[noa][OS]") {
    // Get some fixtures
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

    errno_t err{Errno::good};
    AND_THEN("existsFile") {
        //@CLION-formatter:off
        REQUIRE(OS::existsFile(file1, err));                    REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::existsFile(file2, err));                    REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::existsFile(file3, err));                    REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::existsFile(symlink1, err));                 REQUIRE_ERRNO_GOOD(err);
        REQUIRE(!OS::existsFile("i_do_not_exist.txt", err));    REQUIRE_ERRNO_GOOD(err);
        REQUIRE(!OS::existsFile(symlink2, err));                REQUIRE_ERRNO_GOOD(err);
        REQUIRE(!OS::existsFile(test_dir, err));                REQUIRE_ERRNO_GOOD(err);
        //@CLION-formatter:on
    }

    AND_THEN("exists") {
        //@CLION-formatter:off
        REQUIRE(OS::exists(file1, err));                    REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::exists(file2, err));                    REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::exists(file3, err));                    REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::exists(symlink1, err));                 REQUIRE_ERRNO_GOOD(err);
        REQUIRE(!OS::exists("i_do_not_exist.txt", err));    REQUIRE_ERRNO_GOOD(err);
        REQUIRE(!OS::exists(symlink2, err));                REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::exists(test_dir, err));                 REQUIRE_ERRNO_GOOD(err);
        //@CLION-formatter:on
    }

    AND_THEN("size") {
        //@CLION-formatter:off
        REQUIRE(OS::size(file1, err) == file1_content.size());      REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::size(file2, err) == file2_content.size());      REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::size(file3, err) == file3_content.size());      REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::size(symlink1, err) == file3_content.size());   REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::size(test_dir / "foo.txt", err) == 0U);         REQUIRE(err == Errno::fail_os);
        //@CLION-formatter:on
    }

    AND_THEN("remove(All)") {
        // Remove empty dir.
        REQUIRE(OS::exists(test_dir_subdir1, err));
        REQUIRE(OS::remove(test_dir_subdir1) == Errno::good);
        REQUIRE(!OS::exists(test_dir_subdir1, err));
        REQUIRE_ERRNO_GOOD(err);

        // Remove single file.
        REQUIRE(OS::existsFile(subdir2_file1, err));
        REQUIRE(OS::remove(subdir2_file1) == Errno::good);
        REQUIRE(!OS::existsFile(subdir2_file1, err));
        REQUIRE_ERRNO_GOOD(err);

        // Remove symlink but not its target.
        REQUIRE(OS::existsFile(subdir2_symlink1, err));
        REQUIRE(OS::remove(subdir2_symlink1) == Errno::good);
        REQUIRE(!OS::existsFile(subdir2_symlink1, err));
        REQUIRE(OS::existsFile(subdir2_file2, err));
        REQUIRE_ERRNO_GOOD(err);

        // Removing non-empty directories only works with removeAll().
        REQUIRE(OS::remove(test_dir_subdir2) == Errno::fail_os);
        REQUIRE(OS::removeAll(test_dir_subdir2) == Errno::good);
        REQUIRE(!OS::exists(test_dir_subdir2, err));
        REQUIRE_ERRNO_GOOD(err);
    }

    AND_THEN("move") {
        AND_THEN("regular files") {
            REQUIRE(OS::existsFile(file1, err));
            err = OS::move(file1, file2); REQUIRE_ERRNO_GOOD(err);
            REQUIRE(!OS::existsFile(file1, err));
            fs::path new_file = test_dir / "new_file.txt";
            REQUIRE(!OS::existsFile(new_file, err));
            err = OS::move(file2, new_file); REQUIRE_ERRNO_GOOD(err);
            REQUIRE(OS::existsFile(new_file, err));

            // Moving file to a directory
            err = OS::move(file3, test_dir_subdir1);
            REQUIRE(err == Errno::fail_os);
            err = OS::move(file3, test_dir_subdir1 / file3.filename());
            REQUIRE_ERRNO_GOOD(err);
            REQUIRE(!OS::existsFile(file3, err));
            REQUIRE(OS::existsFile(test_dir_subdir1 / file3.filename(), err));
        }

        AND_THEN("symlinks") {
            REQUIRE(fs::is_symlink(symlink1));
            REQUIRE(OS::existsFile(symlink1, err));
            REQUIRE(!OS::existsFile(symlink2, err)); // invalid target
            REQUIRE_ERRNO_GOOD(err);

            err = OS::move(symlink1, symlink2); REQUIRE_ERRNO_GOOD(err);
            REQUIRE(!OS::existsFile(symlink1, err));
            REQUIRE(OS::existsFile(symlink2, err));
            REQUIRE(OS::existsFile(file3, err));  // symlinks are not followed
            REQUIRE(fs::is_symlink(symlink2));

            fs::path new_file = test_dir / "new_symlink.txt";
            REQUIRE(!OS::exists(new_file, err));
            err = OS::move(symlink2, new_file); REQUIRE_ERRNO_GOOD(err);
            REQUIRE(OS::exists(new_file, err));
            REQUIRE(fs::is_symlink(new_file));
        }

        AND_THEN("directories") {
            fs::path new_dir = test_dir / "new_subdir";
            REQUIRE(OS::exists(test_dir_subdir1, err));
            err = OS::move(test_dir_subdir1, new_dir);
            REQUIRE_ERRNO_GOOD(err);
            REQUIRE(OS::exists(new_dir, err));
            REQUIRE(!OS::exists(test_dir_subdir1, err));
        }
    }

    AND_THEN("copy(File|Symlink") {
        AND_THEN("copyFile") {
            // Copy to non-existing file.
            REQUIRE(!OS::existsFile(test_dir_subdir2 / file1.filename(), err));
            err = OS::copyFile(file1, test_dir_subdir2 / file1.filename());
            REQUIRE(OS::existsFile(test_dir_subdir2 / file1.filename(), err));
            REQUIRE_ERRNO_GOOD(err);

            // Copy to existing file.
            err = OS::copyFile(file1, file2);
            REQUIRE_ERRNO_GOOD(err);
            err = OS::copyFile(file1, file2, fs::copy_options::none);
            REQUIRE(err == Errno::fail_os);

            // Copy directory is not allowed. Use copy().
            err = OS::copyFile(test_dir_subdir1, test_dir_subdir2);
            REQUIRE(err == Errno::fail_os);
            err = OS::copyFile(file1, test_dir_subdir1);
            REQUIRE(err == Errno::fail_os);
            err = OS::copyFile(test_dir_subdir1, file2);
            REQUIRE(err == Errno::fail_os);

            // Copy symlink is copying the target not the link.
            REQUIRE(!OS::existsFile(test_dir_subdir2 / file2.filename(), err));
            err = OS::copyFile(symlink1, test_dir_subdir2 / file2.filename());
            REQUIRE(OS::existsFile(test_dir_subdir2 / file2.filename(), err));
            REQUIRE_ERRNO_GOOD(err);
        }

        AND_THEN("copySymlink") {
            OS::copySymlink(symlink1, test_dir_subdir2 / symlink1.filename());
            REQUIRE(OS::existsFile(test_dir_subdir2 / symlink1.filename(), err));
            REQUIRE_ERRNO_GOOD(err);
        }

        AND_THEN("copy") {
            // Copying files should be like copyFile
            REQUIRE(!OS::existsFile(test_dir_subdir2 / file1.filename(), err));
            err = OS::copy(file1, test_dir_subdir2 / file1.filename());
            REQUIRE(OS::existsFile(test_dir_subdir2 / file1.filename(), err));
            REQUIRE_ERRNO_GOOD(err);

            // Copying directories, with subdirectories, etc.
            err = OS::copy(test_dir_subdir2, test_dir_subdir1 / test_dir_subdir2.filename());
            REQUIRE_ERRNO_GOOD(err);

            OS::mkdir(test_dir / "subdir3");
            OS::mkdir(test_dir / "subdir4");
            CREATE_EMPTY_FILE(test_dir / "subdir3/file31.txt");

            err = OS::copy(test_dir / "subdir3", test_dir / "subdir4");
            REQUIRE(OS::existsFile(test_dir / "subdir4/file31.txt", err));
            REQUIRE_ERRNO_GOOD(err);
            err = OS::copy(test_dir / "subdir3", test_dir / "subdir4/subdir3");
            err = OS::copy(test_dir / "subdir3", test_dir / "subdir4/subdir3");
            REQUIRE(OS::existsFile(test_dir / "subdir4/subdir3/file31.txt", err));
        }
    }

    AND_THEN("backup") {
        err = OS::backup(file3, true);
        REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::existsFile(file3, err));
        REQUIRE(OS::existsFile(file3.string() + '~', err));
        REQUIRE(OS::size(file3.string() + '~', err) == file3_content.size());

        OS::copyFile(file2, file3);
        err = OS::backup(file3, true);
        REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::existsFile(file3, err));
        REQUIRE(OS::existsFile(file3.string() + '~', err));
        REQUIRE(OS::size(file3.string() + '~', err) == file2_content.size());

        OS::copyFile(file1, file3);
        err = OS::backup(file3, false);
        REQUIRE_ERRNO_GOOD(err);
        REQUIRE(!OS::existsFile(file3, err));
        REQUIRE(OS::existsFile(file3.string() + '~', err));
        REQUIRE(OS::size(file3.string() + '~', err) == file1_content.size());
    }

    AND_THEN("mkdir") {
        err = OS::mkdir(test_dir / "subdir3/subdir33/subdir3");
        REQUIRE_ERRNO_GOOD(err);
        REQUIRE(OS::exists(test_dir / "subdir3/subdir33/subdir3", err));
        err = OS::mkdir("");
        REQUIRE_ERRNO_GOOD(err);
    }
    std::filesystem::remove_all(test_dir);
}
