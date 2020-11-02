//
// Created by thomas on 11/09/2020.
//

#include "noa/Base.h"
#include "noa/files/ProjectFile.h"
#include "noa/util/Arrays.h"
#include "noa/files/ImageFile.h"


int main(int argc, const char** argv) {
    using namespace Noa;
    Log::Init("programs.log", "AKIRA", ::Noa::Log::level::verbose);

    try {
        std::string path = "/home/thomas/Projects/noa/tests/noa/fixtures/"
                           "TestProject_valid_file.noa";
        ProjectFile project_file(path, std::ios::in, true);
        project_file.load("NOA_");
        project_file.save("NOA_", "./saved_project_file.noa");

        ImageFile image_file("something.txt");

    } catch (const Noa::Error& e) {
        e.print();
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        Log::getCoreLogger()->error("Unknown error: {}", e.what());
        return EXIT_FAILURE;
    }

}
