#pragma once
#include <opencv2/opencv.hpp>

namespace hybrid_msckf {
/*
 * compute depth, P2=RP1 + t
 * @param spherePts1: sphere feature points on first sphere
 * @param spherePts2: sphere feature points on second sphere
 * @param R: Rotation matrix 
 * @param t: Translation vector 
 * @param depthList: output every pair sphere point depth
 */
void ComputeDepthsOnSphere(const std::vector<cv::Point3f> &spherePts1,
                           const std::vector<cv::Point3f> &spherePts2,
                           const cv::Mat &R,
                           const cv::Mat &t,
                           std::vector<double> &depthList);

void ComputeDepthsOnSphere(const cv::Point3f &spherePts1,
                           const cv::Point3f &spherePts2,
                           const cv::Mat &R,
                           const cv::Mat &t,
                           double &depth);

/*
 * compute reprojection err.
 * @param spherePts1: sphere feature points on first sphere
 * @param depthList: every pair sphere point depth
 * @param R: Rotation matrix 
 * @param t: Translation vector 
 * @param spherePts2: sphere feature points on second sphere
 * @param errorList: out err list
 */
void ComputeProjectErrorsOnSphere(const std::vector<cv::Point3f> &spherePts1,
                                  const std::vector<double> &depthList,
                                  const cv::Mat &R,
                                  const cv::Mat &t,
                                  const std::vector<cv::Point3f> &spherePts2,
                                  std::vector<double> &errorList);

/*
 * triangulate point by multiple poses, without cv::Mat.
 * @param RatationList: all rotation matrix
 * @param translationList: all tranlastions
 * @param spherePts: all camera sphere pt
 * @param pt3d: output 3d point relative to the first camera coord
 * @param avgError: avg reprojection err
 * @param avgDisparity: avg disparity
 */
void ComputePoint3dOnSphere(const std::vector<cv::Mat> &RatationList,
                            const std::vector<cv::Mat> &translationList,
                            const std::vector<cv::Point3f> &spherePts,
                            cv::Point3f &pt3d,
                            double &avgError,
                            double &avgDisparity);

template <int N, typename T>
void ComputePoint3dOnSphere(T* RotationList[N],
        T* translationList[N],
        cv::Point3f spherePts[N],
        cv::Point3f &pt3d);

template <typename T>
inline void SolveFuncTempl(T A[9], T b[3], T x[3]);

}
