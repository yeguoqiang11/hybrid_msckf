#include "Vio/DearSystem.hpp"
#include "Utils/json.hpp"
#include "Vio/EkfVio.hpp"
#include "Vio/MsckfVio.h"
#include "Vio/Caimura.hpp"
#include "Vio/ImageProcessor.h"
#include "Utils/PerformanceTest.h"
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;
using namespace Eigen;

namespace featslam{

DearSystem::DearSystem(const string& configFile, bool useEkf)
        : frameIdGenerator_(0), useEkf_(useEkf) {
    ifstream ifs(configFile);
    if (!ifs.is_open()) {
        cerr << "failed to open config file: " << configFile << endl;
        exit(-1);
    }
    nlohmann::json config = nlohmann::json::parse(ifs);
    ifs.close();

    // Camera parameters
    nlohmann::json cam_node = config["cameras"][0];
    caim_ = make_shared<Caimura>(cam_node);
    if (config["cameras"].size() > 1) {
        rightCam_ = make_shared<Caimura>(config["cameras"][1]);
    }

    // IMU
    double gyroSigma = config["gyro_noise_density"];
    double gyroRW = config["gyro_random_walk"];
    double accSigma = config["acc_noise_density"];
    double accRW = config["acc_random_walk"];
    double gyroRange = config["gyro_range"];
    double accRange = config["acc_range"];
    Vector3d gyroBias, accBias;
    for (int i = 0; i < 3; i++) {
        gyroBias(i) = config["gyro_bias"][i];
        accBias(i) = config["acc_bias"][i];
    }
    caim_->SetImuNoise(gyroSigma, gyroRW, accSigma, accRW);
    caim_->SetImuRange(gyroRange, accRange);
    caim_->SetImuBias(gyroBias, accBias);


    // Inverse depth
    double rho = config["prior_inverse_depth"];
    double rho_sigma = config["prior_inverse_depth_sigma"];
    double min_rho = config["min_inverse_depth"];
    double max_rho = config["max_inverse_depth"];
    caim_->SetScenePriors(rho, rho_sigma, min_rho, max_rho);

    caim_->PrintParams();

    // Create functional modules
    imageProcessor_ = make_shared<ImageProcessor>(config["feature_tracker"], caim_);
    hybridslam_ = make_shared<HybridSlam>(config, caim_, rightCam_);

    // Results containers
    mTrajectory.clear();
    mTrajectory.reserve(30000);
}



void DearSystem::Track(const Mat &img, double t_img, vector<Matrix<double, 7, 1> > &vimu,
                       const Mat &rightImg) {
    if(!initialSuccess_) {
         Matrix<double, 7, 1> qv;
         if(!initializer_->Run(img, t_img, vimu, qv, rightImg)) {
             return;
         }
         Matrix<double, 11, 1> init_tpqv;
         init_tpqv << t_img, 0, 0, 0, qv;
         StartWith(img, t_img, init_tpqv);
         cout << "init_tpqv:" << init_tpqv << std::endl;
         return;
     }

    Timer timer;
    imageProcessor_->Process(img, frameIdGenerator_, t_img, rightImg, rightCam_);
    auto features = imageProcessor_->GetActiveFeatures();
    double imageProcessTime = timer.Passd();

    Matrix<double, 7, 1> pose;
    Eigen::Vector3d imu_p;
    Eigen::Vector4d imu_Q;

    // hybrid slam
    hybridslam_->Run(features, t_img, frameIdGenerator_, vimu);
    imu_p = hybridslam_->ImuPose();
    imu_Q = hybridslam_->ImuOrieation();
    
    // mTrajectory.push_back(pose.tail(3));
    mTrajectory.push_back(imu_p);
    
    frameIdGenerator_++;

    // Visualize debug info
    Mat visImg;
    cvtColor(img, visImg, cv::COLOR_GRAY2BGR);
    visImg(Rect(0, 0, 300, 100)) -= Scalar(120, 120, 120);
    imageProcessor_->DrawDebugInfo(visImg);
    hybridslam_->DrawDebugInformation(visImg);

    imshow("frame", visImg);
    cv::waitKey(2);
}


void DearSystem::StartWith(const Mat &img, double t_img, const Matrix<double, 11, 1> &tpqv)
{
    imageProcessor_->Process(img, frameIdGenerator_, t_img);
    auto features = imageProcessor_->GetActiveFeatures();
    vector<Matrix<double, 7, 1>> imuDatas;

    Vector4d q = tpqv.segment(4, 4);
    Vector3d p = tpqv.segment(1, 3);
    Vector3d v = tpqv.tail(3);

    // hybridslam
    hybridslam_->Initialize(q, p, v, frameIdGenerator_);
    hybridslam_->Run(features, t_img, frameIdGenerator_, imuDatas);

    frameIdGenerator_++;
    
    mTrajectory.push_back(p);
    
    initialSuccess_ = true;
}


void DearSystem::ResetGyroscopeBias(const Eigen::Vector3d& bg)
{
	caim_->SetGyroBias(bg);
}


const vector<Vector3d>& DearSystem::GetMapPoints()
{
    return hybridslam_->AllMapPoints(); 

}


vector<Matrix4d> DearSystem::GetSlidingWindowPoses() {
    return hybridslam_->SlidingWindowPose();
}

vector<Vector3d> DearSystem::GetActivePoints() {
    return hybridslam_->MapPoints();
}


//for init debug
void DearSystem::GetInitCameraPose(cv::Mat &R, cv::Mat &t) {
    initializer_->GetCameraPose(R, t);
}


std::vector<Eigen::Vector3d>  DearSystem::GetInitMapPoint()
{
    return initializer_->GetMapPoint();
}


void DearSystem::SetSaveSlamPoseFlag(bool flag) {
    initializer_->SetSaveSlamPoseFlag(flag);
}


void DearSystem::SaveSlamPose(const std::string &path) {
    initializer_->SaveSlamPose(path);
}


void DearSystem::RunInitVSlam(const cv::Mat &img, double t_img, const Mat &imgR) {
    initializer_->RunVSlam(img, t_img, imgR);
    return;
}


bool DearSystem::RunImuCalibrationEx(const Mat &img, double t_img, vector<Matrix<double, 7, 1> > &vimu,
                                     const Mat &rightImg) {

    return initializer_->RunCalibrateImu(img, t_img, vimu, rightImg);;
}

}//namespace featslam{
