//
// Created by thomas on 12/07/2020.
//

#pragma once

#include <memory>

#include "../core/Parser.h"
#include "../core/Options.h"

//namespace Noa::fft {
//
//    // program options
//    class Options : public Noa::Options {
//    public:
//        std::string input, output;
//        bool redundant, centered;
//
//        // update(parser):
//        //      input = parser.getOption("input");
//    };
//
//    const char* opt_string(R"(
//input:i:1N:Input MRC file:image.mrc:@:
//output:o:1N:Output MRC file:image_fft.mrc:@:
//redundant:r:1B:Whether or not the transform should be redundant:true:@:
//centered:c:1B:Whether or not the transform should be centered:true:@:)");
//
//    // program
//    int fft(Noa::Parser& a_parser);
//
//    // a bunch of functions related to the fft program
//    class HelperFFT {
//
//    };
//
//}
