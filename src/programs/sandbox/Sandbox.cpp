//
// Created by thomas on 06/12/2020.
//

#include <noa/Base.h>
#include <noa/util/String.h>


int main(int argc, const char** argv) {
    using namespace Noa;

    try {
        Log::init("sandbox.log", Log::Level::alert);
        std::string str = "1,2,3,4,5";
        std::array<std::string, 5> arr5{};
        errno_t err = String::parse(str, arr5);
        std::cout << err << '\n';
        for (auto& e : arr5)
            std::cout << e << '\n';
        return EXIT_SUCCESS;
    } catch (const Noa::Error& e) {
        e.print();
        return EXIT_FAILURE;
    }
}
