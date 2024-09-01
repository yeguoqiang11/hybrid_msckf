#pragma once
#include <opencv2/opencv.hpp>

namespace hybrid_msckf {
float EssentialError(const cv::Point3f &spherePt1, const cv::Point3f &spherePt2, const cv::Mat &E);

void RansacEssentialAndPnP(const std::vector<cv::Point3f> &spherePts1,
                           const std::vector<cv::Point3f> &spherePts2,
                           const std::vector<double> &depthList1,
                           cv::Mat &R,
                           cv::Mat &_t,
                           int cntThreshold,
                           float threshold);

/*
 * find essential matrix by ransac, P2.t() * E * P1 = 0
 * @param spherePts1: sphere points1
 * @param spherePts2: sphere points2
 * @param bestE: essential matrix
 * @param threshold: threshold
 * return: inlier count
 */
int RansacE(const std::vector<cv::Point3f> &spherePts1, const std::vector<cv::Point3f> &spherePts2, cv::Mat &bestE, float threshold);

/*
 * get inliers
 * @param spherePts1: sphere points1
 * @param spherePts2: sphere points2
 * @param E: essential matrix
 * @param threshold: threshold
 * @param inlierMask: mask of inliers
 */
int GetInliers(const std::vector<cv::Point3f> &spherePts1,
               const std::vector<cv::Point3f> &spherePts2,
               const cv::Mat &E,
               double threshold,
               cv::Mat &inlierMask);

double ReprojError(double x, double y, double z, cv::Mat& R, cv::Mat& t, double x2, double y2, double z2);

// P2 = RP1 + t
int FrontPointCnt(const std::vector<cv::Point3f> &spherePts1,
                  const std::vector<cv::Point3f> &spherePts2,
                  const cv::Mat &R,
                  const cv::Mat &t);

/*
 * Compute R and T from match sphere points
 * @param spherePts1: sphere point on the first camera.
 * @param spherePts2: sphere point on the second camera.
 * @param R: R matrix
 * @param t: t vector
 * @param threshold: reprojection err threshold.
 */
void FindRTFromSpherePairs(const std::vector<cv::Point3f> &spherePts1,
                           const std::vector<cv::Point3f> &spherePts2,
                           cv::Mat &R,
                           cv::Mat &t,
                           float threshold);

void FindRTByEssentialAndPnP(const std::vector<cv::Point3f> &spherePts1,
                             const std::vector<cv::Point3f> &spherePts2,
                             const std::vector<double> &depthList1,
                             cv::Mat &R,
                             cv::Mat &_t,
                             float threshold);
}  // namespace hybrid_msckf
