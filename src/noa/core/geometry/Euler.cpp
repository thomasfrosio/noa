#include "noa/core/Exception.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/geometry/Euler.hpp"
#include "noa/core/geometry/Transform.hpp"
#include "noa/core/string/Format.hpp"

namespace {
    using namespace ::noa::types;

    bool is_valid(std::string_view axes) {
        if (axes == "ZYZ" or axes == "ZXZ" or axes == "ZYX" or axes == "ZXY" or
            axes == "XYZ" or axes == "XYX" or axes == "XZX" or axes == "XZY" or
            axes == "YXY" or axes == "YXZ" or axes == "YZX" or axes == "YZY") {
            return true;
        }
        return false;
    }

    template<typename T>
    Vec3<T> rotm2xyx(Mat33<T> rotm) {
        // Rx(k3) @ Ry(k2) @ Rx(k1) = { c1c2c3-s1s3,  s1c2c3+c1s3, -s2c3,
        //                             -c1c2s3-s1c3, -s1c2s3+c1c3,  s2s3,
        //                                     c1s2,         s1s2,    c2}
        Vec3<T> euler;
        euler[1] = noa::acos(rotm[2][2]);
        if (noa::abs(rotm[2][0]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = noa::atan2(-rotm[1][0], rotm[1][1]);
            euler[2] = 0;
        } else {
            euler[0] = noa::atan2(rotm[2][1], rotm[2][0]);
            euler[2] = noa::atan2(rotm[1][2], -rotm[0][2]);
        }
        return euler;
    }

    template<typename T>
    Vec3<T> rotm2yzy(Mat33<T> rotm) {
        // Ry(k3) @ Rz(k2) @ Ry(k1) = { c1c2c3-s1s3, -s2c3,  s1c2c3+c1c3,
        //                                     c1s2,    c2,         s1s2,
        //                                  -c1c2s3,  s2s3, -s1c2s3+c1c3}
        Vec3<T> euler;
        euler[1] = noa::acos(rotm[1][1]);
        if (noa::abs(rotm[1][2]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = noa::atan2(-rotm[0][2], rotm[0][0]);
            euler[2] = 0;
        } else {
            euler[0] = noa::atan2(rotm[1][0], rotm[1][2]);
            euler[2] = noa::atan2(rotm[0][1], -rotm[2][1]);
        }
        return euler;
    }

    template<typename T>
    Vec3<T> rotm2zxz(Mat33<T> rotm) {
        // Rz(k3) @ Rx(k2) @ Rz(k1) = { -s1c2s3+c1c3, -c1c2s3-s1c3,  s2s3,
        //                               s1c2c3+s1s3,  c1c2c3-s1s3, -s2c3,
        //                                      s1s2,         c1s2,    c2}
        Vec3<T> euler;
        euler[1] = noa::acos(rotm[0][0]);
        if (noa::abs(rotm[2][0]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = noa::atan2(-rotm[2][1], rotm[2][2]);
            euler[2] = 0;
        } else {
            euler[0] = noa::atan2(rotm[0][2], rotm[0][1]);
            euler[2] = noa::atan2(rotm[2][0], -rotm[1][0]);
        }
        return euler;
    }

    template<typename T>
    Vec3<T> rotm2xzx(Mat33<T> rotm) {
        // Rx(k3) @ Rz(k2) @ Rx(k1) = {  c2,       -c1s2,         s1s2,
        //                             s2c3,   c1c2c3-s3, -s1c2c3-c1s3,
        //                             s2s3, c1c2s3+s1c3, -s1c2s3+c1c3}
        Vec3<T> euler;
        euler[1] = noa::acos(rotm[2][2]);
        if (noa::abs(rotm[2][0]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = noa::atan2(rotm[0][1], rotm[0][0]);
            euler[2] = 0;
        } else {
            euler[0] = noa::atan2(rotm[2][0], -rotm[2][1]);
            euler[2] = noa::atan2(rotm[0][2], rotm[1][2]);
        }
        return euler;
    }

    template<typename T>
    Vec3<T> rotm2yxy(Mat33<T> rotm) {
        // Ry(k3) @ Rx(k2) @ Ry(k1) = { -s1c2s3+c1c3, s2s3, c1c2s3+s1c3,
        //                                      s1s2,   c2,       -c1s2,
        //                              -s1c2c3-c1s3, s2c3, c1c2c3-s1s3}
        Vec3<T> euler;
        euler[1] = noa::acos(rotm[1][1]);
        if (noa::abs(rotm[2][1]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = noa::atan2(rotm[2][0], rotm[2][2]);
            euler[2] = 0;
        } else {
            euler[0] = noa::atan2(rotm[1][2], -rotm[1][0]);
            euler[2] = noa::atan2(rotm[2][1], rotm[0][1]);
        }
        return euler;
    }

    template<typename T>
    Vec3<T> rotm2zyz(Mat33<T> rotm) {
        // Rz(k3) @ Ry(k2) @ Rz(k1) = { c1c2c3-s1s3, -s1c2c3-c1s3, s2c3,
        //                              c1c2s3+s1c3, -s1c2s3+c1c3, s2s3,
        //                                    -c1s2,         s1s2,   c2}
        Vec3<T> euler;
        euler[1] = noa::acos(rotm[0][0]);
        if (noa::abs(rotm[2][0]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = noa::atan2(rotm[1][2], rotm[1][1]);
            euler[2] = 0;
        } else {
            euler[0] = noa::atan2(rotm[0][1], -rotm[0][2]);
            euler[2] = noa::atan2(rotm[1][0], rotm[2][0]);
        }
        return euler;
    }

    template<typename T>
    Vec3<T> rotm2xyz(Mat33<T> rotm) {
        // Rz(k3) @ Ry(k2) @ Rx(k1) = { c2c3, s1s2c3-c1s3, c1s2c3+s1s3,
        //                              c2s3, s1s2s3+c1c3, c1s2s3-s1c3,
        //                               -s2,        s1c2,        c1c2}
        Vec3<T> euler;
        euler[1] = -noa::asin(rotm[0][2]);
        if (noa::abs(rotm[2][2]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = noa::atan2(-rotm[1][0], rotm[1][1]);
            euler[2] = 0;
        } else {
            euler[0] = noa::atan2(rotm[0][1], rotm[0][0]);
            euler[2] = noa::atan2(rotm[1][2], rotm[2][2]);
        }
        return euler;
    }

    template<typename T>
    Vec3<T> rotm2yzx(Mat33<T> rotm) {
        // Rx(k3) @ Rz(k2) @ Ry(k1) = {        c1c2,  -s2,        s1c2,
        //                              c1s2c3+s1s3, c2c3, s1s2c3-c1s3,
        //                              c1s2s3-s1c3, c2s3, s1s2s3+c1c3}
        Vec3<T> euler;
        euler[1] = -noa::asin(rotm[2][1]);
        if (noa::abs(rotm[2][2]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = noa::atan2(-rotm[0][2], rotm[0][0]);
            euler[2] = 0;
        } else {
            euler[0] = noa::atan2(rotm[2][0], rotm[2][2]);
            euler[2] = noa::atan2(rotm[0][1], rotm[1][1]);
        }
        return euler;
    }

    template<typename T>
    Vec3<T> rotm2zxy(Mat33<T> rotm) {
        // Ry(k3) @ Rx(k2) @ Rz(k1) = { s1s2s3+c1c3, c1s2s3-s1c3, c2s3,
        //                                     s1c2,        c1c2,  -s2,
        //                              s1s2c3-c1s3, c1s2c3+s1s3, c2c3}
        Vec3<T> euler;
        euler[1] = -noa::asin(rotm[1][0]);
        if (noa::abs(rotm[1][2]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = noa::atan2(-rotm[2][1], rotm[2][2]);
            euler[2] = 0;
        } else {
            euler[0] = noa::atan2(rotm[1][2], rotm[1][1]);
            euler[2] = noa::atan2(rotm[2][0], rotm[0][0]);
        }
        return euler;
    }

    template<typename T>
    Vec3<T> rotm2xzy(Mat33<T> rotm) {
        // Ry(k3) @ Rz(k2) @ Rx(k1) = { c2c3, -c1s2c3+s1s3,  s1s2c3+c1s3,
        //                                s2,         c1c2,        -s1c2,
        //                             -c2s3,  c1s2s3+s1c3, -s1s2s3+c1c3}
        Vec3<T> euler;
        euler[1] = noa::asin(rotm[1][2]);
        if (noa::abs(rotm[2][2]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = noa::atan2(rotm[0][1], rotm[0][0]);
            euler[2] = 0;
        } else {
            euler[0] = noa::atan2(-rotm[1][0], rotm[1][1]);
            euler[2] = noa::atan2(-rotm[0][2], rotm[2][2]);
        }
        return euler;
    }

    template<typename T>
    Vec3<T> rotm2yxz(Mat33<T> rotm) {
        // Rz(k3) @ Rx(k2) @ Ry(k1) = {-s1s2s3+c1c3, -c2s3,  c1s2s3+s1c3,
        //                              s1s2c3+c1s3,  c2c3, -c1s2c3+s1s3,
        //                                    -s1c2,    s2,         c1c2}
        Vec3<T> euler;
        euler[1] = noa::asin(rotm[0][1]);
        if (noa::abs(rotm[1][1]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = noa::atan2(rotm[2][0], rotm[2][2]);
            euler[2] = 0;
        } else {
            euler[0] = noa::atan2(-rotm[0][2], rotm[0][0]);
            euler[2] = noa::atan2(-rotm[2][1], rotm[1][1]);
        }
        return euler;
    }

    template<typename T>
    Vec3<T> rotm2zyx(Mat33<T> rotm) {
        // Rx(k3) @ Ry(k2) @ Rz(k1) = {        c1c2,        -s1c2,    s2,
        //                              c1s2s3+s1c3, -s1s2s3+c1c3, -c2s3,
        //                             -c1s2c3+s1s3,  s1s2c3+c1s3,  c2c3}
        Vec3<T> euler;
        euler[1] = noa::asin(rotm[2][0]);
        if (noa::abs(rotm[1][1]) < static_cast<T>(1e-4)) { // Gimbal lock
            euler[0] = noa::atan2(rotm[1][2], rotm[1][1]);
            euler[2] = 0;
        } else {
            euler[0] = noa::atan2(-rotm[2][1], rotm[2][2]);
            euler[2] = noa::atan2(-rotm[1][0], rotm[0][0]);
        }
        return euler;
    }
}

namespace noa::geometry {
    template<typename T>
    Mat33<T> euler2matrix(Vec3<T> angles, const EulerOptions& options) {
        const std::string lower_axes = ns::to_upper(ns::trim(options.axes));
        check(is_valid(lower_axes), "Axes \"{}\" are not valid", lower_axes);

        const auto right_angles = options.right_handed ? angles : angles * -1;
        Mat33<T> r1, r2, r3;
        if (lower_axes == "ZYZ") { // TODO table lookup?
            r1 = rotate_z(right_angles[0]);
            r2 = rotate_y(right_angles[1]);
            r3 = rotate_z(right_angles[2]);
        } else if (lower_axes == "ZXZ") {
            r1 = rotate_z(right_angles[0]);
            r2 = rotate_x(right_angles[1]);
            r3 = rotate_z(right_angles[2]);
        } else if (lower_axes == "ZYX") {
            r1 = rotate_z(right_angles[0]);
            r2 = rotate_y(right_angles[1]);
            r3 = rotate_x(right_angles[2]);
        } else if (lower_axes == "ZXY") {
            r1 = rotate_z(right_angles[0]);
            r2 = rotate_x(right_angles[1]);
            r3 = rotate_y(right_angles[2]);
        } else if (lower_axes == "XYZ") {
            r1 = rotate_x(right_angles[0]);
            r2 = rotate_y(right_angles[1]);
            r3 = rotate_z(right_angles[2]);
        } else if (lower_axes == "XYX") {
            r1 = rotate_x(right_angles[0]);
            r2 = rotate_y(right_angles[1]);
            r3 = rotate_x(right_angles[2]);
        } else if (lower_axes == "XZX") {
            r1 = rotate_x(right_angles[0]);
            r2 = rotate_z(right_angles[1]);
            r3 = rotate_x(right_angles[2]);
        } else if (lower_axes == "XZY") {
            r1 = rotate_x(right_angles[0]);
            r2 = rotate_z(right_angles[1]);
            r3 = rotate_y(right_angles[2]);
        } else if (lower_axes == "YXY") {
            r1 = rotate_y(right_angles[0]);
            r2 = rotate_x(right_angles[1]);
            r3 = rotate_y(right_angles[2]);
        } else if (lower_axes == "YXZ") {
            r1 = rotate_y(right_angles[0]);
            r2 = rotate_x(right_angles[1]);
            r3 = rotate_z(right_angles[2]);
        } else if (lower_axes == "YZX") {
            r1 = rotate_y(right_angles[0]);
            r2 = rotate_z(right_angles[1]);
            r3 = rotate_x(right_angles[2]);
        } else if (lower_axes == "YZY") {
            r1 = rotate_y(right_angles[0]);
            r2 = rotate_z(right_angles[1]);
            r3 = rotate_y(right_angles[2]);
        }
        return options.intrinsic ? r1 * r2 * r3 : r3 * r2 * r1;
    }

    template Mat33<float> euler2matrix<float>(Vec3<float>, const EulerOptions&);
    template Mat33<double> euler2matrix<double>(Vec3<double>, const EulerOptions&);

    template<typename T>
    Vec3<T> matrix2euler(const Mat33<T>& rotation, const EulerOptions& options) {
        std::string lower_axes = ns::to_upper(ns::trim(options.axes));
        check(is_valid(lower_axes), "Axes \"{}\" are not valid", lower_axes);

        if (options.intrinsic)
            lower_axes = ns::reverse(std::move(lower_axes));

        Vec3<T> euler;
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

        if (options.intrinsic)
            euler = euler.flip();
        if (not options.right_handed)
            euler *= -1;
        return euler;
    }

    template Vec3<float> matrix2euler<float>(const Mat33<float>&, const EulerOptions&);
    template Vec3<double> matrix2euler<double>(const Mat33<double>&, const EulerOptions&);
}
