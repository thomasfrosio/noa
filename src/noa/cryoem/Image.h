//
// Created by thomas on 09/12/2020.
//
#pragma once


namespace Noa {


    /*
     * This class should just be a holder of the data and offers ways to access this data, plus
     * giving us some basic info about it (shape, pixel size, device, complex etc.).
     *
     * Then, other functions will be called to do things with this data.
     *
     *
     *
     * Image image;
     * image.getShape()
     * image.setShape()
     * image.getPixelSize()
     * image.setPixelSize()
     *
     * image.toHost()
     * image.toDevice()
     * image.isHost()
     * image.isDevice()
     *
     * image.isComplex()
     * image.isFT()
     * image.isHalf()
     *
     *
     * FFT::Transformer transformer(Image.isDevice());
     * FFT::C2R(image, transformer, ...);
     * FFT::R2C(image, transformer, ...);
     * FFT::conv(image, transformer, ...);
     * FFT::symmetry();
     *
     * Mask::bandpass(image);
     * Mask::ellipsoid(image);
     * Mask::cylinder(image);
     * Mask::rectangle(image);
     * Mask::shape(image);
     * Mask::taper();
     * Mask::molecular();
     * Mask::eraseEllipsoid();
     * Mask::limits();
     *
     * Transform::pad();
     * Transform::crop();
     * Transform::resize();
     * Transform::linear();
     * Transform::cubic();
     *
     * Scalar::add();
     * Scalar::subtract();
     * Scalar::multiply();
     * Scalar::divide();
     * Scalar::apply(image, predicate);
     * Scalar::apply(image, limits, predicate);
     *
     *
     */
    class Image {

    };
}
