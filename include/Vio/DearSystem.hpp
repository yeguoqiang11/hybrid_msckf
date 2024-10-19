#ifndef __INSLAM_VIO_DEARSYSTEM_H__
#define __INSLAM_VIO_DEARSYSTEM_H__

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "../../src/Vio/HybridSlam.h"

namespace featslam{

class Caimura;
class ImageProcessor;

class DearSystem {
public:
    DearSystem(const std::string& configFile, bool useEkf = false);
    
    void StartWith(const cv::Mat &img, double t_img, const Eigen::Matrix<double, 11, 1> &tpqv);
    
    void Track(const cv::Mat &img, double t_img, std::vector<Eigen::Matrix<double,7,1> > &vimu,
               const cv::Mat &rightImg = cv::Mat());
    
    void ResetGyroscopeBias(const Eigen::Vector3d& bg);
    
    const std::vector<Eigen::Vector3d>& GetMapPoints();
  
    std::vector<Eigen::Vector3d> mTrajectory;

    // for init VSlam
    void RunInitVSlam(const cv::Mat &img, double t_img, const cv::Mat &imgR = cv::Mat());

    void GetInitCameraPose(cv::Mat &R, cv::Mat &t);

    void SetSaveSlamPoseFlag(bool flag);

    bool RunImuCalibrationEx(const cv::Mat &img, double t_img, std::vector<Eigen::Matrix<double,7,1> > &vimu,
                             const cv::Mat &rightImg = cv::Mat());

    std::vector<Eigen::Matrix<double,8,1> > mTQPs, mhybridTQPs;
    
private:
    std::shared_ptr<Caimura> caim_;
    std::shared_ptr<Caimura> rightCam_ = nullptr;
    std::shared_ptr<ImageProcessor> imageProcessor_;
    std::shared_ptr<HybridSlam> hybridslam_ = nullptr;
    int frameIdGenerator_;
    bool useEkf_;
    bool initialSuccess_;
};

}//namespace featslam{

#endif//__INSLAM_VIO_DEARSYSTEM_H__
