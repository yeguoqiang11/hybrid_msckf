#ifndef __INSLAM_VIO_DEARSYSTEM_H__
#define __INSLAM_VIO_DEARSYSTEM_H__

#include <atomic>
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "Vio/Initialize/Initializer.hpp"
#include "Vio/SlidingWindow.hpp"

namespace inslam {

class Caimura;
class EkfVio;
class MsckfVio;
class ImageProcessor;

class DearSystem {
public:
    DearSystem(const std::string& configFile, bool useEkf = false);
    
    void StartWith(const cv::Mat &img, double t_img, const Eigen::Matrix<double, 11, 1> &tpqv);

    void InputImu(const Eigen::Matrix<double, 7, 1> &imuData);

    bool Track(const cv::Mat &img, double t_img, const cv::Mat &rightImg = cv::Mat());

    bool FastImuCallback(const Eigen::Matrix<double, 7, 1> &imuData, Eigen::Matrix<double, 11, 1> &tqpv);

    void ResetGyroscopeBias(const Eigen::Vector3d& bg);
    
    const std::vector<Eigen::Vector3d>& GetMapPoints();

    std::vector<Eigen::Matrix4d> GetSlidingWindowPoses();

    std::vector<Eigen::Vector3d> GetActivePoints();

    void GetPoseVel(Eigen::Matrix4d &Twi, Eigen::Matrix4d &Twc, Eigen::Vector3d &vel);

    // for init VSlam
    void RunInitVSlam(const cv::Mat &img, double t_img, const cv::Mat &imgR = cv::Mat());

    void GetInitCameraPose(cv::Mat &R, cv::Mat &t);

    std::vector<Eigen::Vector3d> GetInitMapPoint();

    void SetSaveSlamPoseFlag(bool flag);

    //r, t:from cam0 to cami,format: time r00 r01 r02 r10 r11 r12 r20 r21 r22 t0 t1 t2
    void SaveSlamPose(const std::string &path, Eigen::Matrix4d &T_C0_G, Eigen::Matrix4d &T_I_C);

    void SaveVioResults(const std::string &path);

    bool RunImuCalibrationEx(const cv::Mat &img, double t_img, std::vector<Eigen::Matrix<double,7,1> > &vimu,
                             const cv::Mat &rightImg = cv::Mat());

    inline int InputWidth() { return caim_->width(); }

    inline int InputHeight() { return caim_->height(); }

    // Format: t, qw, qx, qy, qz, px, py, pz
    std::vector<Eigen::Matrix<double,11,1> > resultPoseVels_;

private:
    void GetImuInterval(double t0, double t1, std::vector<Eigen::Matrix<double, 7, 1> > &imuInterval);

    bool FastIntegrate(const Eigen::Matrix<double, 7, 1> &imuData);

    void UpdateLatestImuState(const Eigen::Matrix<double, 17, 1> &state, const Eigen::Matrix<double, 7, 1> &imu);

    std::shared_ptr<Caimura> caim_;
    std::shared_ptr<Caimura> rightCam_ = nullptr;
    std::shared_ptr<ImageProcessor> imageProcessor_;
    std::shared_ptr<EkfVio> ekfVio_ = nullptr;
    std::shared_ptr<MsckfVio> msckfVio_ = nullptr;
    std::shared_ptr<SlidingWindow> optimizer_ = nullptr;
    int frameIdGenerator_;
    bool useEkf_;

    std::shared_ptr<Initializer> initializer_;
    bool initialSuccess_;

    // IMU data buffer
    std::deque<Eigen::Matrix<double, 7, 1> > imuBuffer_;

    //
    bool hasFirstImage = false;
    double lastTimestamp_ = -1;

    // t, q, p, v, bg, ba
    std::mutex fastImuMutex_;
    std::deque<Eigen::Matrix<double, 7, 1> > fastImuBuffer_;
    Eigen::Matrix<double, 7, 1> latestImuData_;
    Eigen::Matrix<double, 17, 1> latestImuState_;
    std::atomic<bool> hasLatestImuState_{false};
};

}//namespace inslam {

#endif//__INSLAM_VIO_DEARSYSTEM_H__
