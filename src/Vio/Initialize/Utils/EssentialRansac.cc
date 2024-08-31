#include "Vio/Initialize/Utils/EssentialRansac.h"
#include "Vio/Initialize/Utils/Triangulation.h"

namespace inslam {

    // P2 = RP1+t
int InliersCount(const std::vector<cv::Point3f> &spherePts1,
                 const std::vector<cv::Point3f> &spherePts2,
                 const cv::Mat &E,
                 double threshold,
                 int currMaxCount) {
    double *EData = (double *)E.data;
    float e11 = float(EData[0]);
    float e12 = float(EData[1]);
    float e13 = float(EData[2]);
    float e21 = float(EData[3]);
    float e22 = float(EData[4]);
    float e23 = float(EData[5]);
    float e31 = float(EData[6]);
    float e32 = float(EData[7]);
    float e33 = float(EData[8]);
    int cnt = 0;
    for (int i = 0; i < spherePts1.size(); ++i) {
        float x1 = spherePts1[i].x;
        float y1 = spherePts1[i].y;
        float z1 = spherePts1[i].z;
        float x2 = spherePts2[i].x;
        float y2 = spherePts2[i].y;
        float z2 = spherePts2[i].z;
        float a = e11 * x1 + e12 * y1 + e13 * z1;
        float b = e21 * x1 + e22 * y1 + e23 * z1;
        float c = e31 * x1 + e32 * y1 + e33 * z1;
        float err = fabs(a * x2 + b * y2 + c * z2) / sqrtf(a * a + b * b);
        if (err <= threshold) {
            ++cnt;
        }
        if (cnt + spherePts1.size() - i < currMaxCount) {
            break;
        }
    }

    return cnt;
}

float EssentialError(const cv::Point3f &spherePt1, const cv::Point3f &spherePt2, const cv::Mat &E) {
    double *EData = (double *)E.data;
    float e11 = float(EData[0]);
    float e12 = float(EData[1]);
    float e13 = float(EData[2]);
    float e21 = float(EData[3]);
    float e22 = float(EData[4]);
    float e23 = float(EData[5]);
    float e31 = float(EData[6]);
    float e32 = float(EData[7]);
    float e33 = float(EData[8]);

    float x1 = spherePt1.x;
    float y1 = spherePt1.y;
    float z1 = spherePt1.z;
    float x2 = spherePt2.x;
    float y2 = spherePt2.y;
    float z2 = spherePt2.z;
    float a = e11 * x1 + e12 * y1 + e13 * z1;
    float b = e21 * x1 + e22 * y1 + e23 * z1;
    float c = e31 * x1 + e32 * y1 + e33 * z1;
    float err = fabs(a * x2 + b * y2 + c * z2) / sqrtf(a * a + b * b);
    return err;
}

int RansacE(const std::vector<cv::Point3f> &spherePts1, const std::vector<cv::Point3f> &spherePts2, cv::Mat &bestE, float threshold) {
    if (spherePts1.size() < 8) {
        return 0;
    }
    int iter = 0;
    int idList[8] = {0};
    int maxCnt = 0;
    bestE = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat A(8, 9, CV_64F);
    cv::Mat result(8, 1, CV_64F);
    cv::Mat E(3, 3, CV_64F);
    double *AData = (double *)A.data;
    int validIterCnt = 0;
    cv::RNG rng;
    while (iter < 1000) {
        ++iter;
        std::vector<int> ids(spherePts1.size(), 0);
        for (int i = 0; i < ids.size(); ++i) {
            ids[i] = i;
        }
        for (int i = 0; i < 8; ++i) {
            // int id = rand() % (ids.size() - i);
            int id = rng.uniform(0, static_cast<int>(ids.size()) - i);
            idList[i] = ids[id];
            int endID = int(ids.size() - i - 1);
            ids[id] = ids[endID];
        }

        for (int i = 0; i < 8; ++i) {
            int id = idList[i];
            double x1 = spherePts1[id].x;
            double y1 = spherePts1[id].y;
            double z1 = spherePts1[id].z;
            double x2 = spherePts2[id].x;
            double y2 = spherePts2[id].y;
            double z2 = spherePts2[id].z;
            AData[i * 9] = x1 * x2;
            AData[i * 9 + 1] = y1 * x2;
            AData[i * 9 + 2] = z1 * x2;
            AData[i * 9 + 3] = x1 * y2;
            AData[i * 9 + 4] = y1 * y2;
            AData[i * 9 + 5] = z1 * y2;
            AData[i * 9 + 6] = z2 * x1;
            AData[i * 9 + 7] = z2 * y1;
            AData[i * 9 + 8] = z2 * z1;
        }

        bool solveFlag = cv::solve(A(cv::Rect(0, 0, 8, 8)), A(cv::Rect(8, 0, 1, 8)), result, cv::DECOMP_LU);
        if (!solveFlag) {
            continue;
        }
        memcpy(E.data, result.data, sizeof(double) * 8);
        E.at<double>(2, 2) = -1;
        E /= cv::norm(E);

        // no a good result, use svd for instead
        /*if (fabs(E.at<double>(2, 2)) < 1e-6 || fabs(E.at<double>(2, 2) + 1) < 1e-6) {
            cv::Mat W, U, VT;
            cv::SVDecomp(A, W, U, VT, cv::SVD::Flags::FULL_UV);
            memcpy(E.data, VT.data + sizeof(double) * 72, sizeof(double) * 9);
        }*/
        int cnt = InliersCount(spherePts1, spherePts2, E, threshold, maxCnt);
        if (cnt > maxCnt) {
            maxCnt = cnt;
            E.copyTo(bestE);
        }
        ++validIterCnt;
        double p = double(maxCnt) / spherePts1.size();
        double tmp = 1 - pow(p, 8);
        tmp = std::min(tmp, 1 - 1e-6);
        int enoughIterCnt = static_cast<int>(-10 / log10(tmp) + 50);
        if (validIterCnt > enoughIterCnt) {
            break;
        }
    }
    return maxCnt;
}

int GetInliers(const std::vector<cv::Point3f> &spherePts1,
               const std::vector<cv::Point3f> &spherePts2,
               const cv::Mat &E,
               double threshold,
               cv::Mat &inlierMask) {
    if (spherePts1.size() != spherePts2.size()) {
        std::cerr << "error! size of spherePts1 doesn't equal to size of spherePts2!" << std::endl;
        return -1;
    }
    inlierMask = cv::Mat::zeros(static_cast<int>(spherePts1.size()), 1, CV_8UC1);
    double *EData = (double *)E.data;
    float e11 = float(EData[0]);
    float e12 = float(EData[1]);
    float e13 = float(EData[2]);
    float e21 = float(EData[3]);
    float e22 = float(EData[4]);
    float e23 = float(EData[5]);
    float e31 = float(EData[6]);
    float e32 = float(EData[7]);
    float e33 = float(EData[8]);
    int cnt = 0;
    for (int i = 0; i < spherePts1.size(); ++i) {
        float x1 = spherePts1[i].x;
        float y1 = spherePts1[i].y;
        float z1 = spherePts1[i].z;
        float x2 = spherePts2[i].x;
        float y2 = spherePts2[i].y;
        float z2 = spherePts2[i].z;
        float a = e11 * x1 + e12 * y1 + e13 * z1;
        float b = e21 * x1 + e22 * y1 + e23 * z1;
        float c = e31 * x1 + e32 * y1 + e33 * z1;
        float err = fabs(a * x2 + b * y2 + c * z2) / sqrtf(a * a + b * b);
        if (err <= threshold) {
            ++cnt;
            inlierMask.at<uchar>(i, 0) = 1;
        }
    }
    return cnt;
}

void RTFromEssentialMatrix(const std::vector<cv::Point3f> &spherePts1,
                           const std::vector<cv::Point3f> &spherePts2,
                           const cv::Mat &E,
                           cv::Mat &R,
                           cv::Mat &_t) {
    cv::Mat R1, R2;
    cv::Mat t;
    cv::decomposeEssentialMat(E, R1, R2, t);
    int validCnt1 = FrontPointCnt(spherePts1, spherePts2, R1, t);
    int validCnt2 = FrontPointCnt(spherePts1, spherePts2, R2, t);
    int validCnt3 = FrontPointCnt(spherePts1, spherePts2, R1, -t);
    int validCnt4 = FrontPointCnt(spherePts1, spherePts2, R2, -t);
    int maxValidCnt = std::max(std::max(std::max(validCnt1, validCnt2), validCnt3), validCnt4);
    if (maxValidCnt == validCnt1) {
        R1.copyTo(R);
        t.copyTo(_t);
    }
    if (maxValidCnt == validCnt2) {
        R2.copyTo(R);
        t.copyTo(_t);
    }
    if (maxValidCnt == validCnt3) {
        R1.copyTo(R);
        cv::Mat tmp = -t.clone();
        tmp.copyTo(_t);
    }
    if (maxValidCnt == validCnt4) {
        R2.copyTo(R);
        cv::Mat tmp = -t.clone();
        tmp.copyTo(_t);
    }
}

double ReprojError(double x, double y, double z, cv::Mat& R, cv::Mat& t, double x2, double y2, double z2){
    double *RData = (double *)R.data;
    double *tData = (double *)t.data;
    double X = RData[0] * x + RData[1] * y + RData[2] * z + tData[0];
    double Y = RData[3] * x + RData[4] * y + RData[5] * z + tData[1];
    double Z = RData[6] * x + RData[7] * y + RData[8] * z + tData[2];
    double r = sqrt(X * X + Y * Y + Z * Z);
    r = std::max(r, 1e-10);
    double dx = X / r - x2;
    double dy = Y / r - y2;
    double dz = Z / r - z2;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

double DepthScale(const std::vector<double> &depths1, const std::vector<double> &depths2) {
    std::vector<double> scaleList;
    for (int i = 0; i < depths1.size(); ++i) {
        if (depths1[i] < 1e-3 || depths2[i] < 1e-3) {
            continue;
        }
        double scale = depths1[i] / depths2[i];
        scaleList.push_back(scale);
    }
    if (scaleList.size() == 0) {
        return 1;
    }
    std::sort(scaleList.begin(), scaleList.end());
    int medianID = static_cast<int>(scaleList.size()) / 2;
    return scaleList[medianID];
}

void RansacEssentialAndPnP(const std::vector<cv::Point3f> &spherePts1,
                           const std::vector<cv::Point3f> &spherePts2,
                           const std::vector<double> &depthList1,
                           cv::Mat &R,
                           cv::Mat &_t,
                           int cntThreshold,
                           float threshold) {
    if (spherePts1.size() < 8) {
        return;
    }
    int iter = 0;
    int idList[8] = {0};
    int maxCnt = 0;
    int maxEssentialCnt = 0;
    cv::Mat bestE = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat A(8, 9, CV_64F);
    cv::Mat result(8, 1, CV_64F);
    cv::Mat E(3, 3, CV_64F);
    double *AData = (double *)A.data;
    int validIterCnt = 0;
    cv::RNG rng;
    while (iter < 1000) {
        ++iter;
        std::vector<int> ids(spherePts1.size(), 0);
        for (int i = 0; i < ids.size(); ++i) {
            ids[i] = i;
        }
        for (int i = 0; i < 8; ++i) {
            int id = rng.uniform(0, static_cast<int>(ids.size()) - i);
            idList[i] = ids[id];
            int endID = int(ids.size() - i - 1);
            ids[id] = ids[endID];
        }

        for (int i = 0; i < 8; ++i) {
            int id = idList[i];
            double x1 = spherePts1[id].x;
            double y1 = spherePts1[id].y;
            double z1 = spherePts1[id].z;
            double x2 = spherePts2[id].x;
            double y2 = spherePts2[id].y;
            double z2 = spherePts2[id].z;
            AData[i * 9] = x1 * x2;
            AData[i * 9 + 1] = y1 * x2;
            AData[i * 9 + 2] = z1 * x2;
            AData[i * 9 + 3] = x1 * y2;
            AData[i * 9 + 4] = y1 * y2;
            AData[i * 9 + 5] = z1 * y2;
            AData[i * 9 + 6] = z2 * x1;
            AData[i * 9 + 7] = z2 * y1;
            AData[i * 9 + 8] = z2 * z1;
        }

        bool solveFlag = cv::solve(A(cv::Rect(0, 0, 8, 8)), A(cv::Rect(8, 0, 1, 8)), result, cv::DECOMP_LU);
        if (!solveFlag) {
            continue;
        }
        memcpy(E.data, result.data, sizeof(double) * 8);
        E.at<double>(2, 2) = -1;
        E /= cv::norm(E);

        int cntEssential = InliersCount(spherePts1, spherePts2, E, threshold, maxEssentialCnt);
        maxEssentialCnt = std::max(cntEssential, maxEssentialCnt);
        if (cntEssential < cntThreshold) {
            continue;
        }
        cv::Mat tmpR = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat tmpT = cv::Mat::zeros(3, 1, CV_64F);
        RTFromEssentialMatrix(spherePts1, spherePts2, E, tmpR, tmpT);
        std::vector<double> tmpDepths;
        ComputeDepthsOnSphere(spherePts1, spherePts2, tmpR, tmpT, tmpDepths);
        double scale = DepthScale(depthList1, tmpDepths);
        tmpT *= scale;
        int cntReproj = 0;
        for (int i = 0; i < depthList1.size(); ++i) {
            double d = depthList1[i];
            if (d < 0) {
                continue;
            }
            double x1 = d * spherePts1[i].x;
            double y1 = d * spherePts1[i].y;
            double z1 = d * spherePts1[i].z;
            double x2 = spherePts2[i].x;
            double y2 = spherePts2[i].y;
            double z2 = spherePts2[i].z;
            double err = ReprojError(x1, y1, z1, tmpR, tmpT, x2, y2, z2);
            if (err < threshold) {
                ++cntReproj;
            }
        }
        // cntReproj *= 3;
        if (maxCnt < cntEssential + cntReproj) {
            maxCnt = cntEssential + cntReproj;
            tmpR.copyTo(R);
            tmpT.copyTo(_t);
        }
    }
}

// P2 = RP1 + t
int FrontPointCnt(const std::vector<cv::Point3f> &spherePts1,
                  const std::vector<cv::Point3f> &spherePts2,
                  const cv::Mat &R,
                  const cv::Mat &t) {
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
    int validCnt = 0;
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
        double a11 = rx1 * rx1 + ry1 * ry1 + rz1 * rz1;
        double a12 = -rx1 * x2 - ry1 * y2 - rz1 * z2;
        double a21 = a12;
        double a22 = x2 * x2 + y2 * y2 + z2 * z2;
        double b1 = -rx1 * tx - ry1 * ty - rz1 * tz;
        double b2 = x2 * tx + y2 * ty + z2 * tz;
        double D = a11 * a22 - a21 * a12;
        double D1 = b1 * a22 - b2 * a12;
        double D2 = a11 * b2 - a21 * b1;
        if (fabs(D) < 1e-12f) {
            if (D > 0) {
                D = 1e-12f;
            } else {
                D = -1e-12;
            }
        }
        double d1 = D1 / D;
        double d2 = D2 / D;
        if (d1 > 0 && d2 > 0) {
            ++validCnt;
        }
    }

    return validCnt;
}

void FindRTFromSpherePairs(const std::vector<cv::Point3f> &spherePts1,
                           const std::vector<cv::Point3f> &spherePts2,
                           cv::Mat &R,
                           cv::Mat &_t,
                           float threshold) {
    cv::Mat bestE = cv::Mat::eye(3, 3, CV_64FC1);
    RansacE(spherePts1, spherePts2, bestE, threshold);
    cv::Mat R1, R2;
    cv::Mat t;
    cv::decomposeEssentialMat(bestE, R1, R2, t);
    int validCnt1 = FrontPointCnt(spherePts1, spherePts2, R1, t);
    int validCnt2 = FrontPointCnt(spherePts1, spherePts2, R2, t);
    int validCnt3 = FrontPointCnt(spherePts1, spherePts2, R1, -t);
    int validCnt4 = FrontPointCnt(spherePts1, spherePts2, R2, -t);
    int maxValidCnt = std::max(std::max(std::max(validCnt1, validCnt2), validCnt3), validCnt4);
    if (maxValidCnt == validCnt1) {
        R1.copyTo(R);
        t.copyTo(_t);
    }
    if (maxValidCnt == validCnt2) {
        R2.copyTo(R);
        t.copyTo(_t);
    }
    if (maxValidCnt == validCnt3) {
        R1.copyTo(R);
        cv::Mat tmp = -t.clone();
        tmp.copyTo(_t);
    }
    if (maxValidCnt == validCnt4) {
        R2.copyTo(R);
        cv::Mat tmp = -t.clone();
        tmp.copyTo(_t);
    }
}

void FindRTByEssentialAndPnP(const std::vector<cv::Point3f> &spherePts1,
                             const std::vector<cv::Point3f> &spherePts2,
                             const std::vector<double> &depthList1,
                             cv::Mat &R,
                             cv::Mat &_t,
                             float threshold) {
    cv::Mat bestE = cv::Mat::eye(3, 3, CV_64FC1);
    int maxCnt = RansacE(spherePts1, spherePts2, bestE, threshold);
    int cntThreshold = static_cast<int>(maxCnt * 0.7);
    RansacEssentialAndPnP(spherePts1, spherePts2, depthList1, R, _t, cntThreshold, threshold);
}

}
