#include "noa/core/Error.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/geometry/Euler.hpp"
#include "noa/core/geometry/Transform.hpp"
#include "noa/core/utils/Strings.hpp"

namespace {
    using namespace ::noa::types;

    template<typename T>
    auto rotm2xyx(Mat33<T> rotm) -> Vec3<T> {
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
    auto rotm2yzy(Mat33<T> rotm) -> Vec3<T> {
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
    auto rotm2zxz(Mat33<T> rotm) -> Vec3<T> {
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
    auto rotm2xzx(Mat33<T> rotm) -> Vec3<T> {
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
    auto rotm2yxy(Mat33<T> rotm) -> Vec3<T> {
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
    auto rotm2zyz(Mat33<T> rotm) -> Vec3<T> {
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
    auto rotm2xyz(Mat33<T> rotm) -> Vec3<T> {
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
    auto rotm2yzx(Mat33<T> rotm) -> Vec3<T> {
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
    auto rotm2zxy(Mat33<T> rotm) -> Vec3<T> {
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
    auto rotm2xzy(Mat33<T> rotm) -> Vec3<T> {
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
    auto rotm2yxz(Mat33<T> rotm) -> Vec3<T> {
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
    auto rotm2zyx(Mat33<T> rotm) -> Vec3<T> {
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
    template<nt::any_of<f32, f64> T>
    auto euler2matrix(Vec3<T> angles, const EulerOptions& options) -> Mat33<T> {
        const std::string lower_axes = ns::to_upper(ns::trim(options.axes));
        angles = options.right_handed ? angles : angles * -1;

        Mat33<T> r1, r2, r3;
        if (lower_axes == "ZYZ") {
            r1 = rotate_z(angles[0]);
            r2 = rotate_y(angles[1]);
            r3 = rotate_z(angles[2]);
        } else if (lower_axes == "ZXZ") {
            r1 = rotate_z(angles[0]);
            r2 = rotate_x(angles[1]);
            r3 = rotate_z(angles[2]);
        } else if (lower_axes == "ZYX") {
            r1 = rotate_z(angles[0]);
            r2 = rotate_y(angles[1]);
            r3 = rotate_x(angles[2]);
        } else if (lower_axes == "ZXY") {
            r1 = rotate_z(angles[0]);
            r2 = rotate_x(angles[1]);
            r3 = rotate_y(angles[2]);
        } else if (lower_axes == "XYZ") {
            r1 = rotate_x(angles[0]);
            r2 = rotate_y(angles[1]);
            r3 = rotate_z(angles[2]);
        } else if (lower_axes == "XYX") {
            r1 = rotate_x(angles[0]);
            r2 = rotate_y(angles[1]);
            r3 = rotate_x(angles[2]);
        } else if (lower_axes == "XZX") {
            r1 = rotate_x(angles[0]);
            r2 = rotate_z(angles[1]);
            r3 = rotate_x(angles[2]);
        } else if (lower_axes == "XZY") {
            r1 = rotate_x(angles[0]);
            r2 = rotate_z(angles[1]);
            r3 = rotate_y(angles[2]);
        } else if (lower_axes == "YXY") {
            r1 = rotate_y(angles[0]);
            r2 = rotate_x(angles[1]);
            r3 = rotate_y(angles[2]);
        } else if (lower_axes == "YXZ") {
            r1 = rotate_y(angles[0]);
            r2 = rotate_x(angles[1]);
            r3 = rotate_z(angles[2]);
        } else if (lower_axes == "YZX") {
            r1 = rotate_y(angles[0]);
            r2 = rotate_z(angles[1]);
            r3 = rotate_x(angles[2]);
        } else if (lower_axes == "YZY") {
            r1 = rotate_y(angles[0]);
            r2 = rotate_z(angles[1]);
            r3 = rotate_y(angles[2]);
        } else {
            panic("Axes \"{}\" are not valid", lower_axes);
        }
        return options.intrinsic ? r1 * r2 * r3 : r3 * r2 * r1;
    }

    template auto euler2matrix<f32>(Vec3<f32>, const EulerOptions&) -> Mat33<f32>;
    template auto euler2matrix<f64>(Vec3<f64>, const EulerOptions&) -> Mat33<f64>;

    template<nt::any_of<f32, f64> T>
    auto matrix2euler(const Mat33<T>& rotation, const EulerOptions& options) -> Vec3<T> {
        std::string lower_axes = ns::to_upper(ns::trim(options.axes));
        if (options.intrinsic)
            lower_axes = ns::reverse(std::move(lower_axes));

        Vec3<T> euler;
        if (lower_axes == "ZYZ") {
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
        } else {
            panic("Axes \"{}\" are not valid", lower_axes);
        }

        if (options.intrinsic)
            euler = euler.flip();
        if (not options.right_handed)
            euler *= -1;
        return euler;
    }

    template auto matrix2euler<f32>(const Mat33<f32>&, const EulerOptions&) -> Vec3<f32>;
    template auto matrix2euler<f64>(const Mat33<f64>&, const EulerOptions&) -> Vec3<f64>;
}
