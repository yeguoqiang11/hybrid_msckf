#include "Triangulation.h"

namespace hybrid_msckf {

/**
 *  A = [R*n1, -n2]
 *  b = -t
 *  d = (A.T * A).inv() * A.T * b
 * */
void ComputeDepthsOnSphere(const std::vector<cv::Point3f> &spherePts1,
                           const std::vector<cv::Point3f> &spherePts2,
                           const cv::Mat &R,
                           const cv::Mat &t,
                           std::vector<double> &depthList) {
    depthList.resize(spherePts1.size());
    double r11 = R.at<double>(0, 0);
    double r12 = R.at<double>(0, 1);
    double r13 = R.at<double>(0, 2);
    double r21 = R.at<double>(1, 0);
    double r22 = R.at<double>(1, 1);
    double r23 = R.at<double>(1, 2);
    double r31 = R.at<double>(2, 0);
    double r32 = R.at<double>(2, 1);
    double r33 = R.at<double>(2, 2);
    double tx = t.at<double>(0, 0);
    double ty = t.at<double>(1, 0);
    double tz = t.at<double>(2, 0);
    for (int i = 0; i < spherePts1.size(); ++i) {
        double x1 = spherePts1[i].x;
        double y1 = spherePts1[i].y;
        double z1 = spherePts1[i].z;
        double x2 = spherePts2[i].x;
        double y2 = spherePts2[i].y;
        double z2 = spherePts2[i].z;

        double rx1 = r11 * x1 + r12 * y1 + r13 * z1;
        double ry1 = r21 * x1 + r22 * y1 + r23 * z1;
        double rz1 = r31 * x1 + r32 * y1 + r33 * z1;

        // A = [a11, a12; a21, a22];
        double a11 = rx1 * rx1 + ry1 * ry1 + rz1 * rz1;
        double a12 = -rx1 * x2 - ry1 * y2 - rz1 * z2;
        double a21 = a12;
        double a22 = x2 * x2 + y2 * y2 + z2 * z2;

        // A.T * b = [b1; b2]
        double b1 = -rx1 * tx - ry1 * ty - rz1 * tz;
        double b2 = x2 * tx + y2 * ty + z2 * tz;

        // inv(A.T*A) * A.T * b
        double D = a11 * a22 - a21 * a12;
        double D1 = b1 * a22 - b2 * a12;
        // double D2 = a11 * b2 - a21 * b1;
        if (fabs(D) < 1e-12f) {
            if (D > 0) {
                D = 1e-12f;
            }
            else {
                D = -1e-12;
            }
        }
        depthList[i] = D1 / D;
    }
}

void ComputeDepthsOnSphere(const cv::Point3f &spherePts1,
                           const cv::Point3f &spherePts2,
                           const cv::Mat &R,
                           const cv::Mat &t,
                           double &depth) {
    double r11 = R.at<double>(0, 0);
    double r12 = R.at<double>(0, 1);
    double r13 = R.at<double>(0, 2);
    double r21 = R.at<double>(1, 0);
    double r22 = R.at<double>(1, 1);
    double r23 = R.at<double>(1, 2);
    double r31 = R.at<double>(2, 0);
    double r32 = R.at<double>(2, 1);
    double r33 = R.at<double>(2, 2);
    double tx = t.at<double>(0, 0);
    double ty = t.at<double>(1, 0);
    double tz = t.at<double>(2, 0);
    double x1 = spherePts1.x;
    double y1 = spherePts1.y;
    double z1 = spherePts1.z;
    double x2 = spherePts2.x;
    double y2 = spherePts2.y;
    double z2 = spherePts2.z;

    double rx1 = r11 * x1 + r12 * y1 + r13 * z1;
    double ry1 = r21 * x1 + r22 * y1 + r23 * z1;
    double rz1 = r31 * x1 + r32 * y1 + r33 * z1;

    // A = [a11, a12; a21, a22];
    double a11 = rx1 * rx1 + ry1 * ry1 + rz1 * rz1;
    double a12 = -rx1 * x2 - ry1 * y2 - rz1 * z2;
    double a21 = a12;
    double a22 = x2 * x2 + y2 * y2 + z2 * z2;

    // A.T * b = [b1; b2]
    double b1 = -rx1 * tx - ry1 * ty - rz1 * tz;
    double b2 = x2 * tx + y2 * ty + z2 * tz;

    // inv(A.T*A) * A.T * b
    double D = a11 * a22 - a21 * a12;
    double D1 = b1 * a22 - b2 * a12;
    // double D2 = a11 * b2 - a21 * b1;
    if (fabs(D) < 1e-12f) {
        if (D > 0) {
            D = 1e-12f;
        } else {
            D = -1e-12;
        }
    }
    depth = D1 / D;
}

// P2 = RP1 + t
void ComputeProjectErrorsOnSphere(const std::vector<cv::Point3f> &spherePts1,
                                  const std::vector<double> &depthList,
                                  const cv::Mat &R,
                                  const cv::Mat &t,
                                  const std::vector<cv::Point3f> &spherePts2,
                                  std::vector<double> &errorList) {
    errorList.resize(spherePts1.size() * 3);
    double r11 = R.at<double>(0, 0);
    double r12 = R.at<double>(0, 1);
    double r13 = R.at<double>(0, 2);
    double r21 = R.at<double>(1, 0);
    double r22 = R.at<double>(1, 1);
    double r23 = R.at<double>(1, 2);
    double r31 = R.at<double>(2, 0);
    double r32 = R.at<double>(2, 1);
    double r33 = R.at<double>(2, 2);
    double tx = t.at<double>(0, 0);
    double ty = t.at<double>(1, 0);
    double tz = t.at<double>(2, 0);
    for (int i = 0; i < spherePts1.size(); ++i) {
        double x1 = spherePts1[i].x;
        double y1 = spherePts1[i].y;
        double z1 = spherePts1[i].z;
        double x2 = spherePts2[i].x;
        double y2 = spherePts2[i].y;
        double z2 = spherePts2[i].z;
        double d1 = depthList[i];
        double Px1 = d1 * x1;
        double Py1 = d1 * y1;
        double Pz1 = d1 * z1;
        double Px2 = r11 * Px1 + r12 * Py1 + r13 * Pz1 + tx;
        double Py2 = r21 * Px1 + r22 * Py1 + r23 * Pz1 + ty;
        double Pz2 = r31 * Px1 + r32 * Py1 + r33 * Pz1 + tz;
        double norm = sqrt(Px2 * Px2 + Py2 * Py2 + Pz2 * Pz2);
        Px2 /= norm;
        Py2 /= norm;
        Pz2 /= norm;
        double dx = Px2 - x2;
        double dy = Py2 - y2;
        double dz = Pz2 - z2;
        double err = sqrt(dx * dx + dy * dy + dz * dz);
        double dx2 = -Px2 - x2;
        double dy2 = -Py2 - y2;
        double dz2 = -Pz2 - z2;
        double err2 = sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);
        if (err < err2) {
            errorList[i * 3] = dx;
            errorList[i * 3 + 1] = dy;
            errorList[i * 3 + 2] = dz;
        }
        else {
            errorList[i * 3] = dx2;
            errorList[i * 3 + 1] = dy2;
            errorList[i * 3 + 2] = dz2;
        }
    }
}

inline void SolveFunc(cv::Mat &AtA, cv::Mat &Atb, cv::Mat &result) {
    int w = AtA.cols;
    int ptCnt = AtA.rows - 3;
    double *AtAData = (double *)AtA.data;
    double CtCBData[9] = {0};  // 3x3 matrix
    double *Atb_data = (double *)Atb.data;
    int x0 = AtA.cols - 3;
    for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
            double sum = 0;
            for (int k = 0; k < ptCnt; ++k) {
                sum += AtAData[k * w + x0 + j] * AtAData[k * w + x0 + i];
            }
            CtCBData[i * 3 + j] = -sum;
            CtCBData[j * 3 + i] = -sum;
        }
    }
    CtCBData[0] += ptCnt;
    CtCBData[4] += ptCnt;
    CtCBData[8] += ptCnt;

    double sum0 = 0, sum1 = 0, sum2 = 0;
    for (int i = 0; i < ptCnt; ++i) {
        sum0 += -AtAData[i * w + x0] * Atb_data[i];
        sum1 += -AtAData[i * w + x0 + 1] * Atb_data[i];
        sum2 += -AtAData[i * w + x0 + 2] * Atb_data[i];
    }
    sum0 += Atb_data[ptCnt];
    sum1 += Atb_data[ptCnt + 1];
    sum2 += Atb_data[ptCnt + 2];
    double bb_data[3] = {0};  // 3x1
    bb_data[0] = sum0;
    bb_data[1] = sum1;
    bb_data[2] = sum2;

    double a1 = CtCBData[0];
    double b1 = CtCBData[1];
    double c1 = CtCBData[2];
    double a2 = CtCBData[3];
    double b2 = CtCBData[4];
    double c2 = CtCBData[5];
    double a3 = CtCBData[6];
    double b3 = CtCBData[7];
    double c3 = CtCBData[8];
    double sum = a1 * (b2 * c3 - c2 * b3) - a2 * (b1 * c3 - c1 * b3) + a3 * (b1 * c2 - c1 * b2);
    sum = std::max(sum, 1e-10);
    double tmp1 = (b2 * c3 - c2 * b3) * bb_data[0] + (c1 * b3 - b1 * c3) * bb_data[1] + (b1 * c2 - c1 * b2) * bb_data[2];
    double tmp2 = (c2 * a3 - a2 * c3) * bb_data[0] + (a1 * c3 - c1 * a3) * bb_data[1] + (a2 * c1 - a1 * c2) * bb_data[2];
    double tmp3 = (a2 * b3 - b2 * a3) * bb_data[0] + (b1 * a3 - a1 * b3) * bb_data[1] + (a1 * b2 - a2 * b1) * bb_data[2];
    result.create(3, 1, CV_64F);
    double *result_data = (double *)result.data;
    result_data[0] = tmp1 / sum;
    result_data[1] = tmp2 / sum;
    result_data[2] = tmp3 / sum;
}

inline void SolveFunc(double A[9], double b[3], double x[3]) {
    double a1 = A[0];
    double b1 = A[1];
    double c1 = A[2];
    double a2 = A[3];
    double b2 = A[4];
    double c2 = A[5];
    double a3 = A[6];
    double b3 = A[7];
    double c3 = A[8];
    double sum = a1 * (b2 * c3 - c2 * b3) - a2 * (b1 * c3 - c1 * b3) + a3 * (b1 * c2 - c1 * b2);
    if (fabs(sum) < 1e-10) {
        if (sum > 0) {
            sum = 1e-10;
        }
        else {
            sum = -1e-10;
        }
    }
    double tmp1 = (b2 * c3 - c2 * b3) * b[0] + (c1 * b3 - b1 * c3) * b[1] + (b1 * c2 - c1 * b2) * b[2];
    double tmp2 = (c2 * a3 - a2 * c3) * b[0] + (a1 * c3 - c1 * a3) * b[1] + (a2 * c1 - a1 * c2) * b[2];
    double tmp3 = (a2 * b3 - b2 * a3) * b[0] + (b1 * a3 - a1 * b3) * b[1] + (a1 * b2 - a2 * b1) * b[2];
    x[0] = tmp1 / sum;
    x[1] = tmp2 / sum;
    x[2] = tmp3 / sum;
}

// Pn = Rn * P0 + tn
// Solve P0
void ComputePoint3dOnSphere(const std::vector<cv::Mat> &RatationList,
                            const std::vector<cv::Mat> &translationList,
                            const std::vector<cv::Point3f> &spherePts,
                            cv::Point3f &pt3d,
                            double &avgError,
                            double &avgDisparity) {
    int w = int(spherePts.size() + 3);
    double CtC_B[9] = {0};
    double Ctb1_b2[3] = {0};

    for (int i = 0; i < RatationList.size(); ++i) {
        double x = spherePts[i].x;
        double y = spherePts[i].y;
        double z = spherePts[i].z;
        double *RData = (double *)RatationList[i].data;
        double *tData = (double *)translationList[i].data;
        double tx = tData[0];
        double ty = tData[1];
        double tz = tData[2];
        double rtx = RData[0] * x + RData[3] * y + RData[6] * z;
        double rty = RData[1] * x + RData[4] * y + RData[7] * z;
        double rtz = RData[2] * x + RData[5] * y + RData[8] * z;
        double rttx = RData[0] * tx + RData[3] * ty + RData[6] * tz;
        double rtty = RData[1] * tx + RData[4] * ty + RData[7] * tz;
        double rttz = RData[2] * tx + RData[5] * ty + RData[8] * tz;

        double c1 = -rtx;
        double c2 = -rty;
        double c3 = -rtz;
        CtC_B[0] += c1 * c1 - 1;
        CtC_B[1] += c1 * c2;
        CtC_B[2] += c1 * c3;
        CtC_B[4] += c2 * c2 - 1;
        CtC_B[5] += c2 * c3;
        CtC_B[8] += c3 * c3 - 1;

        double b = rtx * rttx + rty * rtty + rtz * rttz;
        Ctb1_b2[0] += c1 * b + rttx;
        Ctb1_b2[1] += c2 * b + rtty;
        Ctb1_b2[2] += c3 * b + rttz;
    }

    CtC_B[3] = CtC_B[1];
    CtC_B[6] = CtC_B[2];
    CtC_B[7] = CtC_B[5];
    double X[3] = {0};
    SolveFunc(CtC_B, Ctb1_b2, X);

    pt3d.x = static_cast<float>(X[0]);
    pt3d.y = static_cast<float>(X[1]);
    pt3d.z = static_cast<float>(X[2]);

    double x = pt3d.x;
    double y = pt3d.y;
    double z = pt3d.z;
    avgError = 0;
    avgDisparity = 0;
    double *tData0 = (double *)RatationList[0].data;
    for (int i = 0; i < RatationList.size(); ++i) {
        double *RData = (double *)RatationList[i].data;
        double *tData = (double *)translationList[i].data;
        double tx = tData[0];
        double ty = tData[1];
        double tz = tData[2];
        double X0 = RData[0] * x + RData[1] * y + RData[2] * z;
        double Y0 = RData[3] * x + RData[4] * y + RData[5] * z;
        double Z0 = RData[6] * x + RData[7] * y + RData[8] * z;
        double X = X0 + tx;
        double Y = Y0 + ty;
        double Z = Z0 + tz;
        X0 += tData0[0];
        Y0 += tData0[1];
        Z0 += tData0[2];

        // for reprojection error
        double normP = sqrt(X * X + Y * Y + Z * Z);
        normP = std::max(normP, 1e-10);
        X /= normP;
        Y /= normP;
        Z /= normP;

        // for disparity
        double normP0 = sqrt(X0 * X0 + Y0 * Y0 + Z0 * Z0);
        normP0 = std::max(normP0, 1e-10);
        X0 /= normP0;
        Y0 /= normP0;
        Z0 /= normP0;

        double dx = X - spherePts[i].x;
        double dy = Y - spherePts[i].y;
        double dz = Z - spherePts[i].z;
        double err = sqrt(dx * dx + dy * dy + dz * dz);
        dx = -X - spherePts[i].x;
        dy = -Y - spherePts[i].y;
        dz = -Z - spherePts[i].z;
        double err2 = sqrt(dx * dx + dy * dy + dz * dz);
        if (err < err2) {
            avgError += err;

            double dX0 = X0 - spherePts[i].x;
            double dY0 = Y0 - spherePts[i].y;
            double dZ0 = Z0 - spherePts[i].z;
            double disparity = sqrt(dX0 * dX0 + dY0 * dY0 + dZ0 * dZ0);
            avgDisparity += disparity;
        }
        else {
            avgError += err2;

            double dX0 = -X0 - spherePts[i].x;
            double dY0 = -Y0 - spherePts[i].y;
            double dZ0 = -Z0 - spherePts[i].z;
            double disparity2 = sqrt(dX0 * dX0 + dY0 * dY0 + dZ0 * dZ0);
            avgDisparity += disparity2;
        }
    }
    avgError /= RatationList.size();
    avgDisparity /= RatationList.size();
}

template void ComputePoint3dOnSphere<2, float>(float* RotationList[2],
                float* translationList[2],
                cv::Point3f spherePts[2],
                cv::Point3f &pt3d);

template void ComputePoint3dOnSphere<2, double>(double* RotationList[2],
                double* translationList[2],
                cv::Point3f spherePts[2],
                cv::Point3f &pt3d);

template <typename T>
inline void SolveFuncTempl(T A[9], T b[3], T x[3]) {
    T a1 = A[0];
    T b1 = A[1];
    T c1 = A[2];
    T a2 = A[3];
    T b2 = A[4];
    T c2 = A[5];
    T a3 = A[6];
    T b3 = A[7];
    T c3 = A[8];
    T sum = a1 * (b2 * c3 - c2 * b3) - a2 * (b1 * c3 - c1 * b3) + a3 * (b1 * c2 - c1 * b2);
    if (fabs(sum) < 1e-10) {
        if (sum > 0) {
            sum = (T)1e-10;
        }
        else {
            sum = (T)(-1e-10);
        }
    }
    T tmp1 = (b2 * c3 - c2 * b3) * b[0] + (c1 * b3 - b1 * c3) * b[1] + (b1 * c2 - c1 * b2) * b[2];
    T tmp2 = (c2 * a3 - a2 * c3) * b[0] + (a1 * c3 - c1 * a3) * b[1] + (a2 * c1 - a1 * c2) * b[2];
    T tmp3 = (a2 * b3 - b2 * a3) * b[0] + (b1 * a3 - a1 * b3) * b[1] + (a1 * b2 - a2 * b1) * b[2];
    x[0] = tmp1 / sum;
    x[1] = tmp2 / sum;
    x[2] = tmp3 / sum;
}

template <int N, typename T>
void ComputePoint3dOnSphere(T* RotationList[N],
            T* translationList[N],
            cv::Point3f spherePts[N],
            cv::Point3f &pt3d) {
    T CtC_B[9] = {0};
    T Ctb1_b2[3] = {0};

    for (int i = 0; i < N; ++i) {
        T x = spherePts[i].x;
        T y = spherePts[i].y;
        T z = spherePts[i].z;
        T *RData = (T *)RotationList[i];
        T *tData = (T *)translationList[i];

        T rttx = tData[0];
        T rtty = tData[1];
        T rttz = tData[2];

        T rtx = RData[0] * x + RData[3] * y + RData[6] * z;
        T rty = RData[1] * x + RData[4] * y + RData[7] * z;
        T rtz = RData[2] * x + RData[5] * y + RData[8] * z;

        T c1 = -rtx;
        T c2 = -rty;
        T c3 = -rtz;
        CtC_B[0] += c1 * c1 - 1;
        CtC_B[1] += c1 * c2;
        CtC_B[2] += c1 * c3;
        CtC_B[4] += c2 * c2 - 1;
        CtC_B[5] += c2 * c3;
        CtC_B[8] += c3 * c3 - 1;

        T b = rtx * rttx + rty * rtty + rtz * rttz;
        Ctb1_b2[0] += c1 * b + rttx;
        Ctb1_b2[1] += c2 * b + rtty;
        Ctb1_b2[2] += c3 * b + rttz;
    }

    CtC_B[3] = CtC_B[1];
    CtC_B[6] = CtC_B[2];
    CtC_B[7] = CtC_B[5];
    T X[3] = {0};
    SolveFuncTempl<T>(CtC_B, Ctb1_b2, X);

    pt3d.x = static_cast<float>(X[0]);
    pt3d.y = static_cast<float>(X[1]);
    pt3d.z = static_cast<float>(X[2]);
}


}
