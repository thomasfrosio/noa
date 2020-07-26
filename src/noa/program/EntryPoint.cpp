/**
 * @file EntryPoint.cpp
 * @brief Entry point for various programs.
 * @author Thomas - ffyr2w
 * @date 19 Jul 2020
 */

#include "../include/Parser.h"
#include "../include/Log.h"

int main(int argc, char** argv) {

    Noa::Log::Init();
    Noa::Log::getCoreLogger()->trace("1");
    Noa::Log::getCoreLogger()->info("2");
    Noa::Log::getCoreLogger()->warn("3");
    Noa::Log::getCoreLogger()->error("4");
    Noa::Log::getCoreLogger()->critical("5");

    Noa::Log::getAppLogger()->warn("Initialized");


    Noa::Parser parser(argc, argv);



    // noa::Parser::get()::parse_command_line(argc, argv);
    // switch (noa::Parser::get()::program) {
    //      case "fft":
    //          noa::fft();
    //          std::vector<std::string>* input = noa::Parser::instance()::getOption("Input", "i");


    // end

    return 0;
}
