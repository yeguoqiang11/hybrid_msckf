#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <deque>
#include "Vio/Caimura.hpp"
#include "Vio/Initialize/InitTypes.hpp"
#include "Vio/Initialize/Utils/PoseInitialization.h"
#include "Vio/Initialize/Imu/ImuAligment.h"
#include "Imu/ImuCalibration.h"

namespace hybrid_msckf {
enum InitTrackState { NOT_START = 0, NOT_INITIALIZED = 1, INITIALIZED = 2, TRACKING_LOST = 3 };
enum UpdateResult { RESULT_NO_KEYFRAME, RESULT_IS_KEYFRAME, RESULT_FAILURE };

class FrameHandler
{
public:
    FrameHandler() = delete;
    FrameHandler(MapPtr map, const nlohmann::json &config, std::shared_ptr<Caimura> leftCaim,
                             std::shared_ptr<Caimura> rightCam = nullptr);

    bool Run(const cv::Mat &imgL, double imgTime, const cv::Mat &imgR = cv::Mat());
    void SetImuParam(double an, double gn, double ban, double bgn);
    void SetImuExtrinsic(const Eigen::Matrix3d &Rcb, const Eigen::Vector3d &tcb);
    void SetInitBa(const Eigen::Vector3d &ba);
    void SetInitBg(const Eigen::Vector3d &bg);
    void EnableCalibrateImu();
    void Reset();

    Eigen::Matrix<double, 7, 1> GetQV() {
        return qv_;
    }

    //for debug show
    cv::Mat R_, t_;
    void GetCameraPose(cv::Mat &R, cv::Mat &t);
    std::vector<Eigen::Vector3d> GetMapPoint();
    void SetSaveSlamPoseFlag(bool flag);
    //format: r00 r01 r02 r10 r11 r12 r20 r21 r22 t0 t1 t2
    void SaveSlamPose(const std::string &path,
                      Eigen::Matrix4d &T_G_C0,
                      Eigen::Matrix4d &T_C_I);
    void DebugShow(int showType);

private:
    UpdateResult ProcessFirstFrame();
    UpdateResult ProcessSecondFrame();
    UpdateResult ProcessFrame();
    void PostProcess(UpdateResult res, const cv::Mat &imgR);
    bool TryInitialize();
    void LocalMapping(UpdateResult res);
    void OptmizeInitMappoint();
    void Track(FramePtr currFrame, FramePtr lastFrame);
    void GetNewCorners(int nNewFeature);
    void CalcStereo(const cv::Mat &Ir);
    bool CheckEpipolar(const Eigen::Vector3d &ray1, const Eigen::Vector3d &ray2,
                       const Eigen::Matrix3d &E12, double angularThresh);
    bool PnPIncrement();
    void Update3DPoints();
    void GlobalStateRecovery();

    void CheckCameraStatic(const std::vector<Feature> &currFeas, const std::vector<Feature> &lastFeas);
    double CalcStd(std::deque<Eigen::Vector3d> &dataList, Eigen::Vector3d &dataMean);

    bool TryCalibImu();

public:
    struct ImuParam {
        double acc_n = 0.1;
        double gyro_n = 0.01;
        double ba_n = 0.001;
        double bg_n = 0.0001;
        Eigen::Matrix3d Rcb;
        Eigen::Vector3d tcb, Gw;

        Eigen::Vector3d ba, bg;
        ImuParam() {
            ba.setZero();
            bg.setZero();
        }
    };
    void InsertImu(double gyro[3], double acc[3], double timestamp);
    void SetFinishedFrame(double timestamp);
    void ImuInitInsert(Frame frame, int frameId, double timestamp);

private:
    InitTrackState state_;
    MapPtr map_;
    int lastKeyframeFeatureCount_;
    double angularResolution_;
    std::shared_ptr<PoseInitialization> initilization_;
    Eigen::Matrix<double, 7, 1> qv_;

    cv::Mat camMask_;
    cv::Mat mask_;
    int maxFeature_;
    int radius_;
    int featureId_;
    int trackFailCnt_;
    std::shared_ptr<Caimura> leftCam_;
    std::shared_ptr<Caimura> rightCam_;
    FramePtr lastFrame_, currFrame_;
    bool doStereo_;
    std::vector<FramePtr> initFrameList_;
    int lastInitFeatureCount_;

    // static initialize
    int staticImgCount_;
    std::deque<Eigen::Vector3d> staticAccDatas_;
    std::deque<Eigen::Vector3d> staticGyrDatas_;

    // Imu information
    bool doImuAligment_;
    Eigen::Vector3d initBa_;
    bool newVioFlag_;
    ImuParam imuParam_;
    VisualInertialState refImustate_, lastImustate_;
    VisualInertialAligment imuInitializing_;

    // Calibrate imu extrinsic rotation and time delay
    bool doCalibrateImuEx_;
    std::shared_ptr<ImuCalibration> imuCalib_;

    // save to file
    bool saveSlamPoseFlag_;
    std::vector<FramePtr> frameDataset_;
};

} //namespace hybrid_msckf
