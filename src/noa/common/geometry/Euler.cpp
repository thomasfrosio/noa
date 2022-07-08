#include <iostream>

#include "noa/common/Exception.h"
#include "noa/common/Types.h"
#include "noa/common/Math.h"
#include "noa/common/geometry/Euler.h"
#include "noa/common/geometry/Transform.h"
#include "noa/common/string/Format.h"

namespace {
    using namespace ::noa;

    bool isValid(std::string_view axes) {
        if (axes == "ZYZ" || axes == "ZXZ" || axes == "ZYX" || axes == "ZXY" ||
            axes == "XYZ" || axes == "XYX" || axes == "XZX" || axes == "XZY" ||
            axes == "YXY" || axes == "YXZ" || axes == "YZX" || axes == "YZY") {
            return true;
        }
        return false;
    }

    template<typename T>
    Float3<T> rotm2xyx(Mat33<T> rotm) {
        // Rx(k3) @ Ry(k2) @ Rx(k1) = { c1c2c3-s1s3,  s1c2c3+c1s3, -s2c3,
        //                             -c1c2s3-s1c3, -s1c2s3+c1c3,  s2s3,
        //                                     c1s2,         s1s2,    c2}
        Float3<T> euler;
        euler[1] = math::acos(rotm[2][2]);
        if (math::abs(rotm[2][0]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = math::atan2(-rotm[1][0], rotm[1][1]);
            euler[2] = 0;
        } else {
            euler[0] = math::atan2(rotm[2][1], rotm[2][0]);
            euler[2] = math::atan2(rotm[1][2], -rotm[0][2]);
        }
        return euler;
    }

    template<typename T>
    Float3<T> rotm2yzy(Mat33<T> rotm) {
        // Ry(k3) @ Rz(k2) @ Ry(k1) = { c1c2c3-s1s3, -s2c3,  s1c2c3+c1c3,
        //                                     c1s2,    c2,         s1s2,
        //                                  -c1c2s3,  s2s3, -s1c2s3+c1c3}
        Float3<T> euler;
        euler[1] = math::acos(rotm[1][1]);
        if (math::abs(rotm[1][2]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = math::atan2(-rotm[0][2], rotm[0][0]);
            euler[2] = 0;
        } else {
            euler[0] = math::atan2(rotm[1][0], rotm[1][2]);
            euler[2] = math::atan2(rotm[0][1], -rotm[2][1]);
        }
        return euler;
    }

    template<typename T>
    Float3<T> rotm2zxz(Mat33<T> rotm) {
        // Rz(k3) @ Rx(k2) @ Rz(k1) = { -s1c2s3+c1c3, -c1c2s3-s1c3,  s2s3,
        //                               s1c2c3+s1s3,  c1c2c3-s1s3, -s2c3,
        //                                      s1s2,         c1s2,    c2}
        Float3<T> euler;
        euler[1] = math::acos(rotm[0][0]);
        if (math::abs(rotm[2][0]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = math::atan2(-rotm[2][1], rotm[2][2]);
            euler[2] = 0;
        } else {
            euler[0] = math::atan2(rotm[0][2], rotm[0][1]);
            euler[2] = math::atan2(rotm[2][0], -rotm[1][0]);
        }
        return euler;
    }

    template<typename T>
    Float3<T> rotm2xzx(Mat33<T> rotm) {
        // Rx(k3) @ Rz(k2) @ Rx(k1) = {  c2,       -c1s2,         s1s2,
        //                             s2c3,   c1c2c3-s3, -s1c2c3-c1s3,
        //                             s2s3, c1c2s3+s1c3, -s1c2s3+c1c3}
        Float3<T> euler;
        euler[1] = math::acos(rotm[2][2]);
        if (math::abs(rotm[2][0]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = math::atan2(rotm[0][1], rotm[0][0]);
            euler[2] = 0;
        } else {
            euler[0] = math::atan2(rotm[2][0], -rotm[2][1]);
            euler[2] = math::atan2(rotm[0][2], rotm[1][2]);
        }
        return euler;
    }

    template<typename T>
    Float3<T> rotm2yxy(Mat33<T> rotm) {
        // Ry(k3) @ Rx(k2) @ Ry(k1) = { -s1c2s3+c1c3, s2s3, c1c2s3+s1c3,
        //                                      s1s2,   c2,       -c1s2,
        //                              -s1c2c3-c1s3, s2c3, c1c2c3-s1s3}
        Float3<T> euler;
        euler[1] = math::acos(rotm[1][1]);
        if (math::abs(rotm[2][1]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = math::atan2(rotm[2][0], rotm[2][2]);
            euler[2] = 0;
        } else {
            euler[0] = math::atan2(rotm[1][2], -rotm[1][0]);
            euler[2] = math::atan2(rotm[2][1], rotm[0][1]);
        }
        return euler;
    }

    template<typename T>
    Float3<T> rotm2zyz(Mat33<T> rotm) {
        // Rz(k3) @ Ry(k2) @ Rz(k1) = { c1c2c3-s1s3, -s1c2c3-c1s3, s2c3,
        //                              c1c2s3+s1c3, -s1c2s3+c1c3, s2s3,
        //                                    -c1s2,         s1s2,   c2}
        Float3<T> euler;
        euler[1] = math::acos(rotm[0][0]);
        if (math::abs(rotm[2][0]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = math::atan2(rotm[1][2], rotm[1][1]);
            euler[2] = 0;
        } else {
            euler[0] = math::atan2(rotm[0][1], -rotm[0][2]);
            euler[2] = math::atan2(rotm[1][0], rotm[2][0]);
        }
        return euler;
    }

    template<typename T>
    Float3<T> rotm2xyz(Mat33<T> rotm) {
        // Rz(k3) @ Ry(k2) @ Rx(k1) = { c2c3, s1s2c3-c1s3, c1s2c3+s1s3,
        //                              c2s3, s1s2s3+c1c3, c1s2s3-s1c3,
        //                               -s2,        s1c2,        c1c2}
        Float3<T> euler;
        euler[1] = -math::asin(rotm[0][2]);
        if (math::abs(rotm[2][2]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = math::atan2(-rotm[1][0], rotm[1][1]);
            euler[2] = 0;
        } else {
            euler[0] = math::atan2(rotm[0][1], rotm[0][0]);
            euler[2] = math::atan2(rotm[1][2], rotm[2][2]);
        }
        return euler;
    }

    template<typename T>
    Float3<T> rotm2yzx(Mat33<T> rotm) {
        // Rx(k3) @ Rz(k2) @ Ry(k1) = {        c1c2,  -s2,        s1c2,
        //                              c1s2c3+s1s3, c2c3, s1s2c3-c1s3,
        //                              c1s2s3-s1c3, c2s3, s1s2s3+c1c3}
        Float3<T> euler;
        euler[1] = -math::asin(rotm[2][1]);
        if (math::abs(rotm[2][2]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = math::atan2(-rotm[0][2], rotm[0][0]);
            euler[2] = 0;
        } else {
            euler[0] = math::atan2(rotm[2][0], rotm[2][2]);
            euler[2] = math::atan2(rotm[0][1], rotm[1][1]);
        }
        return euler;
    }

    template<typename T>
    Float3<T> rotm2zxy(Mat33<T> rotm) {
        // Ry(k3) @ Rx(k2) @ Rz(k1) = { s1s2s3+c1c3, c1s2s3-s1c3, c2s3,
        //                                     s1c2,        c1c2,  -s2,
        //                              s1s2c3-c1s3, c1s2c3+s1s3, c2c3}
        Float3<T> euler;
        euler[1] = -math::asin(rotm[1][0]);
        if (math::abs(rotm[1][2]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = math::atan2(-rotm[2][1], rotm[2][2]);
            euler[2] = 0;
        } else {
            euler[0] = math::atan2(rotm[1][2], rotm[1][1]);
            euler[2] = math::atan2(rotm[2][0], rotm[0][0]);
        }
        return euler;
    }

    template<typename T>
    Float3<T> rotm2xzy(Mat33<T> rotm) {
        // Ry(k3) @ Rz(k2) @ Rx(k1) = { c2c3, -c1s2c3+s1s3,  s1s2c3+c1s3,
        //                                s2,         c1c2,        -s1c2,
        //                             -c2s3,  c1s2s3+s1c3, -s1s2s3+c1c3}
        Float3<T> euler;
        euler[1] = math::asin(rotm[1][2]);
        if (math::abs(rotm[2][2]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = math::atan2(rotm[0][1], rotm[0][0]);
            euler[2] = 0;
        } else {
            euler[0] = math::atan2(-rotm[1][0], rotm[1][1]);
            euler[2] = math::atan2(-rotm[0][2], rotm[2][2]);
        }
        return euler;
    }

    template<typename T>
    Float3<T> rotm2yxz(Mat33<T> rotm) {
        // Rz(k3) @ Rx(k2) @ Ry(k1) = {-s1s2s3+c1c3, -c2s3,  c1s2s3+s1c3,
        //                              s1s2c3+c1s3,  c2c3, -c1s2c3+s1s3,
        //                                    -s1c2,    s2,         c1c2}
        Float3<T> euler;
        euler[1] = math::asin(rotm[0][1]);
        if (math::abs(rotm[1][1]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = math::atan2(rotm[2][0], rotm[2][2]);
            euler[2] = 0;
        } else {
            euler[0] = math::atan2(-rotm[0][2], rotm[0][0]);
            euler[2] = math::atan2(-rotm[2][1], rotm[1][1]);
        }
        return euler;
    }

    template<typename T>
    Float3<T> rotm2zyx(Mat33<T> rotm) {
        // Rx(k3) @ Ry(k2) @ Rz(k1) = {        c1c2,        -s1c2,    s2,
        //                              c1s2s3+s1c3, -s1s2s3+c1c3, -c2s3,
        //                             -c1s2c3+s1s3,  s1s2c3+c1s3,  c2c3}
        Float3<T> euler;
        euler[1] = math::asin(rotm[2][0]);
        if (math::abs(rotm[1][1]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = math::atan2(rotm[1][2], rotm[1][1]);
            euler[2] = 0;
        } else {
            euler[0] = math::atan2(-rotm[2][1], rotm[2][2]);
            euler[2] = math::atan2(-rotm[1][0], rotm[0][0]);
        }
        return euler;
    }
}

namespace noa::geometry {
    template<typename T>
    Mat33<T> euler2matrix(Float3<T> angles, std::string_view axes, bool intrinsic, bool right_handed) {
        const std::string lower_axes = string::upper(string::trim(axes));
        if (!isValid(lower_axes))
            NOA_THROW("Axes \"{}\" are not valid", lower_axes);

        const Float3<T> right_angles = right_handed ? angles : angles * -1;
        Mat33<T> r1, r2, r3;
        if (lower_axes == "ZYZ") { // TODO table lookup?
            r1 = rotateZ(right_angles[0]);
            r2 = rotateY(right_angles[1]);
            r3 = rotateZ(right_angles[2]);
        } else if (lower_axes == "ZXZ") {
            r1 = rotateZ(right_angles[0]);
            r2 = rotateX(right_angles[1]);
            r3 = rotateZ(right_angles[2]);
        } else if (lower_axes == "ZYX") {
            r1 = rotateZ(right_angles[0]);
            r2 = rotateY(right_angles[1]);
            r3 = rotateX(right_angles[2]);
        } else if (lower_axes == "ZXY") {
            r1 = rotateZ(right_angles[0]);
            r2 = rotateX(right_angles[1]);
            r3 = rotateY(right_angles[2]);
        } else if (lower_axes == "XYZ") {
            r1 = rotateX(right_angles[0]);
            r2 = rotateY(right_angles[1]);
            r3 = rotateZ(right_angles[2]);
        } else if (lower_axes == "XYX") {
            r1 = rotateX(right_angles[0]);
            r2 = rotateY(right_angles[1]);
            r3 = rotateX(right_angles[2]);
        } else if (lower_axes == "XZX") {
            r1 = rotateX(right_angles[0]);
            r2 = rotateZ(right_angles[1]);
            r3 = rotateX(right_angles[2]);
        } else if (lower_axes == "XZY") {
            r1 = rotateX(right_angles[0]);
            r2 = rotateZ(right_angles[1]);
            r3 = rotateY(right_angles[2]);
        } else if (lower_axes == "YXY") {
            r1 = rotateY(right_angles[0]);
            r2 = rotateX(right_angles[1]);
            r3 = rotateY(right_angles[2]);
        } else if (lower_axes == "YXZ") {
            r1 = rotateY(right_angles[0]);
            r2 = rotateX(right_angles[1]);
            r3 = rotateZ(right_angles[2]);
        } else if (lower_axes == "YZX") {
            r1 = rotateY(right_angles[0]);
            r2 = rotateZ(right_angles[1]);
            r3 = rotateX(right_angles[2]);
        } else if (lower_axes == "YZY") {
            r1 = rotateY(right_angles[0]);
            r2 = rotateZ(right_angles[1]);
            r3 = rotateY(right_angles[2]);
        }
        return intrinsic ? r1 * r2 * r3 : r3 * r2 * r1;
    }

    template Mat33<float> euler2matrix<float>(Float3<float>, std::string_view, bool, bool);
    template Mat33<double> euler2matrix<double>(Float3<double>, std::string_view, bool, bool);

    template<typename T>
    Float3<T> matrix2euler(const Mat33<T>& rotation, std::string_view axes, bool intrinsic, bool right_handed) {
        std::string lower_axes = string::upper(string::trim(axes));
        if (!isValid(lower_axes))
            NOA_THROW("Axes \"{}\" are not valid", lower_axes);

        if (intrinsic)
            lower_axes = string::reverse(std::move(lower_axes));

        Float3<T> euler;
        if (lower_axes == "ZYZ") { // TODO table lookup?
            euler = rotm2zyz(rotation);
        } else if (lower_axes == "ZXZ") {
            euler = rotm2zxz(rotation);
        } else if (lower_axes == "ZYX") {
            euler = rotm2zyx(rotation);
        } else if (lower_axes == "ZXY") {
            euler = rotm2zxy(rotation);
        } else if (lower_axes == "XYZ") {
            euler = rotm2xyz(rotation);
        } else if (lower_axes == "XYX") {
            euler = rotm2xyx(rotation);
        } else if (lower_axes == "XZX") {
            euler = rotm2xzx(rotation);
        } else if (lower_axes == "XZY") {
            euler = rotm2xzy(rotation);
        } else if (lower_axes == "YXY") {
            euler = rotm2yxy(rotation);
        } else if (lower_axes == "YXZ") {
            euler = rotm2yxz(rotation);
        } else if (lower_axes == "YZX") {
            euler = rotm2yzx(rotation);
        } else if (lower_axes == "YZY") {
            euler = rotm2yzy(rotation);
        }

        if (intrinsic)
            euler = euler.flip();
        if (!right_handed)
            euler *= -1;
        return euler;
    }

    template Float3<float> matrix2euler<float>(const Mat33<float>&, std::string_view, bool, bool);
    template Float3<double> matrix2euler<double>(const Mat33<double>&, std::string_view, bool, bool);
}
