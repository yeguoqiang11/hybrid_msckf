#include "Vio/DearSystem.hpp"
#include "Utils/json.hpp"
#include "Vio/EkfVio.hpp"
#include "Vio/MsckfVio.h"
#include "Vio/Caimura.hpp"
#include "Vio/ImageProcessor.h"
#include "Utils/PerformanceTest.h"
#include "Utils/DataIOUtil.h"
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;
using namespace Eigen;

namespace inslam {

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

    // Visual inertial filter parameters
    int sliding_window_size = config["sliding_window_size"];

    // Create functional modules
    imageProcessor_ = make_shared<ImageProcessor>(config["feature_tracker"], caim_);
    if (useEkf) {
        ekfVio_ = make_shared<EkfVio>(config, caim_, rightCam_);
    } else {
        msckfVio_ = make_shared<MsckfVio>(sliding_window_size, caim_, rightCam_);
    }

    optimizer_ = make_shared<SlidingWindow>(config, caim_, rightCam_);

    initializer_ = std::make_shared<Initializer>(config["initialize"], caim_, rightCam_);
    initialSuccess_ = false;

    // IMU buffer
    imuBuffer_.clear();

    // Results containers
    resultPoseVels_.clear();

}


void DearSystem::InputImu(const Eigen::Matrix<double, 7, 1> &imuData) {
    imuBuffer_.push_back(imuData);
}


bool DearSystem::Track(const Mat &img, double t_img, const Mat &rightImg) {
    if (t_img > imuBuffer_.back()(0)) {
        cerr << "image timestamp is newer than IMU" << endl;
        return false;
    }

    if (!hasFirstImage) {
        lastTimestamp_ = t_img;
        hasFirstImage = true;
        return false;
    }

    // Get IMU interval between last frame and this frame.
    vector<Matrix<double, 7, 1> > vimu;
    GetImuInterval(lastTimestamp_, t_img, vimu);
    lastTimestamp_ = t_img;

    if(!initialSuccess_) {
         Matrix<double, 7, 1> qv;
         if(!initializer_->Run(img, t_img, vimu, qv, rightImg)) {
             return false;
         }
         Matrix<double, 11, 1> init_tpqv;
         init_tpqv << t_img, 0, 0, 0, qv;
         StartWith(img, t_img, init_tpqv);
         cout << "init_tpqv:" << init_tpqv << std::endl;
         return true;
     }

    Timer timer;
    imageProcessor_->Process(img, frameIdGenerator_, t_img, rightImg, rightCam_);
    auto features = imageProcessor_->GetActiveFeatures();
    double imageProcessTime = timer.Passd();

    Matrix<double, 10, 1> poseVel;
    Matrix<double, 17, 1> imuState;
    if (useEkf_) {
        ekfVio_->Run(features, t_img, frameIdGenerator_, vimu);
        poseVel = ekfVio_->GetPoseVel();
        imuState = ekfVio_->GetImuState();
    } else {
        msckfVio_->Run(features, t_img, frameIdGenerator_, vimu);
        poseVel = msckfVio_->GetPoseVel();
        imuState = msckfVio_->GetImuState();
    }
    double vioFilterTime = timer.Passd() - imageProcessTime;

    optimizer_->Run(features, frameIdGenerator_, imuState, vimu);
    poseVel = optimizer_->PoseVel();

    Matrix<double, 11, 1> tqp;
    tqp << t_img, poseVel;
    resultPoseVels_.push_back(tqp);

    UpdateLatestImuState(imuState, vimu.back());

    frameIdGenerator_++;

    // Visualize debug info
    Mat visImg;
    cvtColor(img, visImg, cv::COLOR_GRAY2BGR);
    visImg(Rect(0, 0, 300, 100)) -= Scalar(120, 120, 120);
    imageProcessor_->DrawDebugInfo(visImg);
    if (useEkf_) {
        ekfVio_->DrawDebugInfo(visImg);
    }

    stringstream ss;
    ss.precision(2);
    ss << "time(ms): tracking=" << imageProcessTime << " filter=" << vioFilterTime;
    putText(visImg, ss.str(), Point(5, 60), cv::FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255,255,255), 1);

    imshow("frame", visImg);
    cv::waitKey(2);

    return true;
}


void DearSystem::StartWith(const Mat &img, double t_img, const Matrix<double, 11, 1> &tpqv)
{
    imageProcessor_->Process(img, frameIdGenerator_, t_img);
    auto features = imageProcessor_->GetActiveFeatures();
    vector<Matrix<double, 7, 1>> imuDatas;

    Vector4d q = tpqv.segment(4, 4);
    Vector3d p = tpqv.segment(1, 3);
    Vector3d v = tpqv.tail(3);
    if (useEkf_) {
        ekfVio_->Initialize(q, p, v, frameIdGenerator_, t_img);
        ekfVio_->Run(features, t_img, frameIdGenerator_, imuDatas);
    } else {
        msckfVio_->Initialize(q, p, v, frameIdGenerator_, t_img);
        msckfVio_->Run(features, t_img, frameIdGenerator_, imuDatas);
    }

    frameIdGenerator_++;

    Matrix<double, 11, 1> poseVel;
    poseVel << t_img, q, p, v;
    resultPoseVels_.push_back(poseVel);

    initialSuccess_ = true;
}


bool DearSystem::FastImuCallback(const Matrix<double, 7, 1> &imuData, Matrix<double, 11, 1> &tqpv) {
    fastImuMutex_.lock();
    fastImuBuffer_.push_back(imuData);
    bool ok = FastIntegrate(imuData);
    tqpv = latestImuState_.head(11);
    fastImuMutex_.unlock();
    return ok;
}


void DearSystem::UpdateLatestImuState(const Matrix<double, 17, 1> &state,
                                      const Matrix<double, 7, 1> &imu) {
    unique_lock<mutex> lock(fastImuMutex_);
    latestImuState_ = state;
    latestImuData_ = imu;
    hasLatestImuState_ = true;

    const double ts = state(0);
    while (!fastImuBuffer_.empty() && fastImuBuffer_.front()(0) < ts + 1.0e-4) {
        fastImuBuffer_.pop_front();
    }

    for (const Matrix<double, 7, 1> &data : fastImuBuffer_) {
        FastIntegrate(data);
    }
}


bool DearSystem::FastIntegrate(const Eigen::Matrix<double, 7, 1> &imuData) {
    if (!hasLatestImuState_) {
        return false;
    }

    if (imuData(0) <= latestImuState_(0)) {
        return false;
    }

    Quaterniond q(latestImuState_(1), latestImuState_(2), latestImuState_(3), latestImuState_(4));
    Vector3d p = latestImuState_.segment(5, 3);
    Vector3d v = latestImuState_.segment(8, 3);
    Vector3d bg = latestImuState_.segment(11, 3);
    Vector3d ba = latestImuState_.segment(14, 3);
    Matrix3d R = q.matrix();

    double dt = imuData(0) - latestImuState_(0);
    Vector3d w = (imuData.segment(1, 3) + latestImuData_.segment(1, 3)) * 0.5 - bg;
    Vector3d a = R * (imuData.tail(3) - ba) - Vector3d(0, 0, 9.81);
    Vector3d pnew = p + v * dt + 0.5 * dt * dt * a;
    Vector3d vnew = v + dt * a;
    Quaterniond qnew = q * MathUtil::VecToQuat(w * dt);

    latestImuState_.head(11) << imuData(0), qnew.w(), qnew.vec(), pnew, vnew;
    latestImuData_ = imuData;
    return true;
}


void DearSystem::ResetGyroscopeBias(const Eigen::Vector3d& bg)
{
	caim_->SetGyroBias(bg);
}


const vector<Vector3d>& DearSystem::GetMapPoints()
{
    if (useEkf_) {
        return ekfVio_->mvMapPoints;
    } else {
        return msckfVio_->lostMapPoints_;
    }

}


vector<Matrix4d> DearSystem::GetSlidingWindowPoses() {
    if (useEkf_) {
        return ekfVio_->QuerySlidingWindowPoses();
    } else {
        return msckfVio_->QuerySlidingWindowPoses();
    }
}

vector<Vector3d> DearSystem::GetActivePoints() {
    if (useEkf_) {
        return ekfVio_->QueryActiveFeatures();
    }
    return vector<Vector3d>();
}


void DearSystem::GetPoseVel(Eigen::Matrix4d &Twi, Eigen::Matrix4d &Twc, Eigen::Vector3d &vel) {
    Twi.setIdentity();
    Twc.setIdentity();
    if (resultPoseVels_.empty()) {
        return;
    }
    const Matrix<double, 11, 1> tqpv = resultPoseVels_.back();
    Quaterniond qiw(tqpv(1), tqpv(2), tqpv(3), tqpv(4));
    Vector3d pwi = tqpv.segment(5, 3);
    Twi.topRows(3) << qiw.matrix(), pwi;
    Twc = Twi * caim_->Tic_;
    vel = tqpv.tail(3);
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


void DearSystem::SaveSlamPose(const std::string &path,
                              Eigen::Matrix4d &T_C0_G,
                              Eigen::Matrix4d &T_I_C) {
    initializer_->SaveSlamPose(path, T_C0_G, T_I_C);
}


void DearSystem::SaveVioResults(const std::string &path) {
    ofstream ofs(path);
    if (!ofs.is_open()) {
        cerr << "failed to open output file: " << path << endl;
        return;
    }

    for (const Eigen::Matrix<double, 11, 1> &pose : resultPoseVels_) {
        ofs << setprecision(18) << pose(0) << " " << setprecision(12)
             << pose(5) << " " << pose(6) << " " << pose(7) << " "
             << pose(4) << " " << pose(1) << " " << pose(2) << " " << pose(3) << "\n";
    }
    ofs.close();
    cout << "vio results have been saved at: " << path << endl;
}

void DearSystem::RunInitVSlam(const cv::Mat &img, double t_img, const Mat &imgR) {
    initializer_->RunVSlam(img, t_img, imgR);
    return;
}


bool DearSystem::RunImuCalibrationEx(const Mat &img, double t_img, vector<Matrix<double, 7, 1> > &vimu,
                                     const Mat &rightImg) {

    return initializer_->RunCalibrateImu(img, t_img, vimu, rightImg);;
}


void DearSystem::GetImuInterval(double t0, double t1, vector<Matrix<double, 7, 1>> &imuInterval) {
    vector<Eigen::Matrix<double, 7, 1> > tempImuVector;

    while ( (*(imuBuffer_.begin() +1 ))(0) < t0) {
        imuBuffer_.pop_front();
    }

    for (const auto &it : imuBuffer_) {
        tempImuVector.push_back(it);
        if (it(0) > t1) {
            break;
        }
    }

    DataIOUtil::GetImuInterval(tempImuVector, imuInterval, t0, t1, 0);
}

}//namespace inslam {
