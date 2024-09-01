#ifndef IMUALIGMENT_H
#define IMUALIGMENT_H
#include <assert.h>
#include <unordered_map>
#include <Eigen/Core>

#include "Vio/Initialize/Imu/Imu.h"
#include "Vio/Initialize/InitTypes.hpp"

namespace hybrid_msckf {
class VisualInertialAligment {
public:
    VisualInertialAligment() {
        use_linear_alignment_ = true;
        bg_.setZero();
        ba_.setZero();
        acc_i_ << 0.0, 0.0, -9.8;
        last_gyro_.setZero();
        last_acc_.setZero();
        tcb_.setZero();
        Rcb_.setIdentity();

        gravity_mag_ = 0.98;
        acc_noise_ = 0.1;
        gyro_noise_ = 0.01;
        ba_noise_ = 0.0001;
        bg_noise_ = 0.00001;
        has_extrinsic_ = false;
        is_success_ = false;
        acc_std_thres_ = 0.25;
        start_aligment_frame_num_ = 30;
        start_calib_frame_num_ = 15;
        cost_thres_ = 0.02;
        calib_deg_thres_ = 25;  // degree
        imu_init_time_dif = 0.11;
    }

    ~VisualInertialAligment() {}

    bool IsSuccess() { return is_success_; }

    bool HasExtrinsic() { return has_extrinsic_; }

    bool ExtrinsicCalibration();

    bool Alignment(bool doStereo = false);

    bool Reset(bool force);

    bool AligmentReset(bool force);

    void SetMap(const std::shared_ptr<Map> &map) {
        map_ = map;
    }

    void SetBa(const Eigen::Vector3d &ba) {
        ba_ = ba;
    }

    void SetStartAligmentFrameNum(int n) {
        start_aligment_frame_num_ = n;
    }

    Eigen::Vector3d GetBa() { return ba_; }
    Eigen::Vector3d GetBg() { return bg_; }
    Eigen::Vector3d GetGravityVector() { return Gw_; }
    double GetScale() { return scale_; }  // scale = imu_vij / visual_vij
    Eigen::Matrix3d GetRotationExtrinsic() { return Rcb_; }
    std::vector<VisualInertialState> &GetStates() { return state_dataset_; }
    std::vector<Frame> &GetFrames() { return frame_dataset_; }
    int GetStatesNum() { return static_cast<int>(state_dataset_.size()); }

    bool GetLastFrame(Frame &frame) { 
        if (frame_dataset_.size() > 0) {
            frame = frame_dataset_.back();
            return true;
        } else {
            return false;
        }
    }

    void SetExtrinsic(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) {
        Rcb_ = R;
        tcb_ = t;
        has_extrinsic_ = true;
    }

    void InsertFrame(Frame &frame) {
        if (frame_dataset_.size() <= start_aligment_frame_num_) {
            frame_dataset_.push_back(frame);
        } else {
            frame_dataset_.push_back(frame);
            if (frame_dataset_.front().imu_state.imu0.GetMoveAccStd() < 0.08) {
                frame_dataset_.erase(frame_dataset_.begin());
            }
        }
    }

    void FrameOptimization(double angularResolution);

private:
    void GyroBiasEstimating();

    void InsertState(cv::Mat &Rwc, cv::Mat &twc, VisualInertialState &state);

    void FrameTimeBasedResize(double time);

    bool AlignmentByLinear(double &scale_out, Eigen::Vector3d &Gw_out, Eigen::Vector3d &bg_out,
                           std::vector<Eigen::Vector3d> &vel_out);

    bool AlignmentByOptimize(double &scale_out, Eigen::Vector3d &Gw_out, Eigen::Vector3d &bg_out,
                             std::vector<Eigen::Vector3d> &vel_out, bool doStereo);

    double AccStd();

    bool AvergGravityDir(Eigen::Vector3d &Gw);

private:
    bool use_linear_alignment_;
    std::vector<VisualInertialState> state_dataset_;
    std::vector<Frame> frame_dataset_;

    Eigen::Matrix3d Rcb_;
    Eigen::Vector3d tcb_, acc_i_;

    Eigen::Vector3d last_gyro_, last_acc_;
    double acc_noise_, gyro_noise_, ba_noise_, bg_noise_;
    double scale_, gravity_mag_, last_time_;
    double acc_std_thres_, calib_deg_thres_, cost_thres_;

    bool has_extrinsic_, is_success_;

    Eigen::Vector3d Gw_, ba_, bg_;
    int start_aligment_frame_num_, start_calib_frame_num_;

    std::shared_ptr<Map> map_;
    double imu_init_time_dif;
};

}  // namespace hybrid_msckf
#endif
