#pragma once

#include <ceres/ceres.h>

namespace inslam {

class StereoDepthCost {
public:
    StereoDepthCost(double obsPt_X, double obsPt_Y, double obsPt_Z, double depth) {
        obsPtX = obsPt_X;
        obsPtY = obsPt_Y;
        obsPtZ = obsPt_Z;
        stereoDepth = depth;
    }

    template <typename T>
    bool operator()(const T *const camera2, const T *const point, T *residual) const {
        //world to cam
        T pt[3];
        CamProjection(camera2, point, pt);

        //3d Point residual in cam
        T point_norm = sqrt(pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2]);
        if (point_norm < T(1e-10)) {
            point_norm = T(1e-10);
        }
        pt[0] = pt[0] / point_norm;
        pt[1] = pt[1] / point_norm;
        pt[2] = pt[2] / point_norm;

        T dx1 = pt[0] - T(obsPtX);
        T dy1 = pt[1] - T(obsPtY);
        T dz1 = pt[2] - T(obsPtZ);
        T dx2 = -pt[0] - T(obsPtX);
        T dy2 = -pt[1] - T(obsPtY);
        T dz2 = -pt[2] - T(obsPtZ);

        T err1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;
        T err2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
        if (err1 < err2) {
            residual[0] = dx1;
            residual[1] = dy1;
            residual[2] = dz1;
        } else {
            residual[0] = dx2;
            residual[1] = dy2;
            residual[2] = dz2;
        }
        residual[3] = T(0.0);
        if(stereoDepth > 1.0 && stereoDepth < 8.0) {
            residual[3] = T(0.05) * (point_norm - stereoDepth) / stereoDepth;
        }

        return true;
    }

    template <typename T>
    static inline bool CamProjection(const T *camera, const T *point, T *pt) {
        T rotation[3] = {camera[0], camera[1], camera[2]};
        pt[0] = point[0];
        pt[1] = point[1];
        pt[2] = point[2];
        RotateByAxisAngle(rotation, pt);
        pt[0] += camera[3];
        pt[1] += camera[4];
        pt[2] += camera[5];

        return true;
    }

    template <typename T>
    static void RotateByAxisAngle(T *rotation, T *p) {
        T a = rotation[0];
        T b = rotation[1];
        T c = rotation[2];
        T angle = sqrt(a * a + b * b + c * c);
        if (angle < T(1e-10)) {
            angle = T(1e-10);
        }
        T n1 = a / angle;
        T n2 = b / angle;
        T n3 = c / angle;
        T p1 = p[0];
        T p2 = p[1];
        T p3 = p[2];
        T NXP1 = n2 * p3 - p2 * n3;
        T NXP2 = n3 * p1 - p3 * n1;
        T NXP3 = n1 * p2 - p1 * n2;
        T NdotP = n1 * p1 + n2 * p2 + n3 * p3;
        T sinAngle = sin(angle);
        T cosAngle = cos(angle);
        p[0] = p1 * cosAngle + NXP1 * sinAngle + n1 * NdotP * (T(1) - cosAngle);
        p[1] = p2 * cosAngle + NXP2 * sinAngle + n2 * NdotP * (T(1) - cosAngle);
        p[2] = p3 * cosAngle + NXP3 * sinAngle + n3 * NdotP * (T(1) - cosAngle);
    }

    static ceres::CostFunction *Create(const double obsPtX, const double obsPtY, const double obsPtZ , double depth) {
        return (new ceres::AutoDiffCostFunction<StereoDepthCost, 4, 6, 3>(
            new StereoDepthCost(obsPtX, obsPtY, obsPtZ, depth)));
    }

private:
    double obsPtX;
    double obsPtY;
    double obsPtZ;
    double stereoDepth;
};

} // namespace inslam
