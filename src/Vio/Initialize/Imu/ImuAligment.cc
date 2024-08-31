#include "Vio/Initialize/Imu/ImuAligment.h"
#include "Vio/Initialize/Imu/CamCost.h"
#include "Vio/Initialize/Imu/ImuCostFunction.h"
#include "Vio/Initialize/Optimization.hpp"
#include "Vio/Initialize/Imu/LinearAlignment.h"

#include <unordered_map>
#include <ceres/ceres.h>

namespace inslam {
inline Eigen::Matrix3d VectorSkew(Eigen::Vector3d rvec) {
    Eigen::Matrix3d skew;
    skew << 0.0, -rvec[2], rvec[1], rvec[2], 0.0, -rvec[0], -rvec[1], rvec[0], 0.0;
    return skew;
}

inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d rvec) {
    double theta = rvec.norm();
    if (fabs(theta) < 1.0e-08) {
        if (theta < 0) {
            theta = -1.0e-08;
        } else {
            theta = 1.0e-08;
        }
    }
    Eigen::Vector3d nvec = rvec / theta;
    Eigen::Matrix3d n_hat;
    n_hat << 0.0, -nvec[2], nvec[1], nvec[2], 0.0, -nvec[0], -nvec[1], nvec[0], 0.0;
    Eigen::Matrix3d R = cos(theta) * Eigen::Matrix3d::Identity() + (1.0 - cos(theta)) * nvec * nvec.transpose() + sin(theta) * n_hat;
    return R;
}

inline Eigen::Vector3d RotationMatrix2Vector3d(const Eigen::Matrix3d R) {
    double theta = acos((R.trace() - 1.0) * 0.5);

    Eigen::Matrix3d Right = (R - R.transpose()) * 0.5;
    Eigen::Vector3d rvec;
    rvec[0] = (Right(2, 1) - Right(1, 2)) * 0.5;
    rvec[1] = (Right(0, 2) - Right(2, 0)) * 0.5;
    rvec[2] = (Right(1, 0) - Right(0, 1)) * 0.5;
    double rv_norm = rvec.norm();
    if (rv_norm < 1.0e-07) {
        return rvec;
    }

    double cos_theta = cos(theta);
    double sin_theta = sin(theta);
    if (fabs(sin_theta) < 1.0e-08 || fabs(cos_theta) > 1) {
        return rvec;
    }
    rvec /= sin_theta;

    return rvec * theta;
}

bool VisualInertialAligment::ExtrinsicCalibration() {
    if (static_cast<int>(frame_dataset_.size()) < start_calib_frame_num_) {
        return false;
    }
    state_dataset_.clear();
    for (int i = 0; i < static_cast<int>(frame_dataset_.size()); i++) {
        Frame &frame = frame_dataset_[i];
        cv::Mat Rwc, twc;
        frame.GetPose(Rwc, twc);
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> eig_R((double *)Rwc.data);
        Eigen::Map<Eigen::Vector3d> eig_t((double *)twc.data);
        VisualInertialState state;
        state.Rwc = eig_R;
        state.twc = eig_t;
        state.imu0 = frame.imu_state.imu0;
        state_dataset_.push_back(state);
    }
    double total_angle = 0;
    for (size_t i = 0; i < state_dataset_.size(); i++) {
        Eigen::Vector3d rvec = RotationMatrix2Vector3d(state_dataset_[i].imu0.Rji_);
        total_angle += rvec.norm();
    }
    double thres = calib_deg_thres_ * CV_PI / 180.0;
    std::cout << "estimating extrinsic rotation angle: " << total_angle << "," << thres << std::endl;
    if (total_angle < thres) {
        std::cout << "estimating rotation fail, not enough rotation!!!" << std::endl;
        return false;
    }

    std::vector<Eigen::Matrix3d> inertial_R, visual_R;

    Eigen::Matrix3d tmpA;
    Eigen::Vector3d Pbji, Pcji, tmpb;
    int pose_num = static_cast<int>(state_dataset_.size());
    int cols = 3 * (pose_num - 1);
    Eigen::MatrixXd A(cols, 3), b(cols, 1);
    int start_id = 0;
    for (size_t i = 0; i < static_cast<size_t>(pose_num) - 1; i++) {
        VisualInertialState &state0 = state_dataset_[i];
        VisualInertialState &state1 = state_dataset_[i + 1];
        Eigen::Matrix3d Rbjbi = state1.imu0.Rji_;
        Eigen::Matrix3d Rcjci = state0.Rwc * state1.Rwc.transpose();
        Eigen::Vector3d rvec_bjbi = RotationMatrix2Vector3d(Rbjbi);
        Eigen::Vector3d rvec_cjci = RotationMatrix2Vector3d(Rcjci);
        if (rvec_cjci.norm() < 0.01) {
            continue;
        }

        inertial_R.push_back(Rbjbi);
        visual_R.push_back(Rcjci);

        double thetab = rvec_bjbi.norm();
        double thetac = rvec_cjci.norm();

        rvec_cjci /= thetac;
        rvec_bjbi /= thetab;

        Pbji = 2.0 * sin(thetab / 2.0) * rvec_bjbi;
        Pcji = 2.0 * sin(thetac / 2.0) * rvec_cjci;

        tmpA = VectorSkew(Pbji + Pcji);
        tmpb = Pcji - Pbji;
        A.block<3, 3>(3 * start_id, 0) = tmpA;
        b.block<3, 1>(3 * start_id, 0) = tmpb;
        start_id++;
    }

    if (start_id < 0.5 * start_calib_frame_num_) {
        std::cout << "estimating rotation fail, not eough rotation frame!!!" << std::endl;
        cv::waitKey(0);
        return false;
    }
    A = A.topRows(start_id * 3);
    b = b.topRows(start_id * 3);

    Eigen::Vector3d sol = A.colPivHouseholderQr().solve(b);
    double sol_norm = sol.norm();
    Eigen::Vector3d Pcb = 2.0 * sol / sqrt(1.0 + sol_norm * sol_norm);
    double Pcb_norm = Pcb.norm();
    Rcb_ = (1.0 - Pcb_norm * Pcb_norm / 2.0) * Eigen::Matrix3d::Identity()
          + 0.5 * (Pcb * Pcb.transpose() + sqrt(4.0 - Pcb_norm * Pcb_norm) * VectorSkew(Pcb));

    double error = 0;
    for (size_t i = 0; i < visual_R.size(); i++) {
        Eigen::Matrix3d &Rcjci = visual_R[i];
        Eigen::Matrix3d &Rbjbi = inertial_R[i];

        Eigen::Matrix3d R_hat = Rcb_ * Rcjci * Rcb_.transpose();
        Eigen::Vector3d rbvec = RotationMatrix2Vector3d(Rbjbi);
        Eigen::Vector3d r_hat = RotationMatrix2Vector3d(R_hat);
        Eigen::Vector3d err = r_hat - rbvec;
        error += err.norm();
    }
    error /= visual_R.size();
    std::cout << "********Rcb: " << Rcb_ << std::endl;
    std::cout << "*******estimating extrinsic error: " << error << std::endl;
    cv::waitKey(0);
    if (error < 0.05) {
        has_extrinsic_ = true;
        return true;
    }

    return false;
}

bool VisualInertialAligment::AligmentReset(bool force) {
    if (static_cast<int>(state_dataset_.size()) >= start_aligment_frame_num_ || force) {
        ba_.setZero();
        bg_.setZero();

        is_success_ = false;
        state_dataset_.clear();
        frame_dataset_.clear();
        return true;
    }
    return false;
}

void VisualInertialAligment::GyroBiasEstimating() {
    ceres::Problem problem;
    double bg[3] = {0, 0, 0};
    for (size_t i = 0; i < state_dataset_.size() - 1; i++) {
        VisualInertialState &statei = state_dataset_[i];
        VisualInertialState &statej = state_dataset_[i + 1];
        Eigen::Matrix3d Rbjbi = Rcb_ * statei.Rwc * statej.Rwc.transpose() * Rcb_.transpose();

        Eigen::Matrix3d sqrt_info = statej.imu0.GetCov().block<3, 3>(0, 0).inverse();
        sqrt_info = Eigen::LLT<Eigen::Matrix3d>(sqrt_info).matrixL().transpose();
        ceres::CostFunction *cost = new GyroBiasErr(sqrt_info, statej.imu0.Rji_, Rbjbi, statej.imu0.dRg_);
        problem.AddResidualBlock(cost, NULL, bg);
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 5;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;

    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Vector3d deltaBg;
    deltaBg << bg[0], bg[1], bg[2];
    bg_ = state_dataset_.back().imu0.bg_ + deltaBg;
    for (auto &state : state_dataset_)
    {
        state.imu0.bg_ = bg_;
        state.imu0.RePreintegration();
    }
}


bool VisualInertialAligment::Alignment(bool do_stereo) {
    if (static_cast<int>(state_dataset_.size()) < start_aligment_frame_num_) {
        return false;
    }

    double acc_std = AccStd();
    if (acc_std < acc_std_thres_) {
        return false;
    }

    bool alignment_flag = false;
    double scale;
    Eigen::Vector3d Gw, bg;
    // vel_ref: express under reference camera frame
    std::vector<Eigen::Vector3d> vel_ref;
    if(use_linear_alignment_) {
        std::cout << "do linear alignment..." << std::endl;
        alignment_flag = AlignmentByLinear(scale, Gw, bg, vel_ref);
    } else {
        std::cout << "do optimize alignment..." << std::endl;
        alignment_flag = AlignmentByOptimize(scale, Gw, bg, vel_ref, do_stereo);
    }

    if (alignment_flag) {
        scale_ = scale;
        bg_ = bg;
        Gw_ = Gw;
        for (size_t i = 0; i < state_dataset_.size(); i++) {
            double dt = frame_dataset_[i].Time() - state_dataset_[i].timestamp;
            if (dt > 0.0001) {
                std::cout << "imu init state recovery error!!!" << std::endl;
                assert(false);
            }
            frame_dataset_[i].SetTraslationScale(scale_);
            state_dataset_[i].Vj = vel_ref[i];
            state_dataset_[i].twc *= scale_;
            frame_dataset_[i].imu_state = state_dataset_[i];
        }

        is_success_ = true;
        std::cout << " Init scale: " << scale_ << std::endl;
        std::cout << " Init bg: " << bg_.transpose() << std::endl;
        std::cout << " Init Gw: " << Gw_.transpose() << std::endl;
        std::cout << "----------------" << std::endl;
    }
    return true;
}

bool VisualInertialAligment::AlignmentByLinear(double &scale_out, Eigen::Vector3d &Gw_out, Eigen::Vector3d &bg_out,
                                               std::vector<Eigen::Vector3d> &vel_out) {
    VectorXd x;
    bool result = IMUAlignmentByLinear(state_dataset_, bg_out, Gw_out, x, Rcb_, tcb_);

    // update velocity: transfor from current body frame(bk) to reference camera frame(c0)
    int frameCnt = static_cast<int>(state_dataset_.size());
    vel_out.resize(frameCnt);
    for (int i = 0; i < frameCnt; i++) {
        VisualInertialState &state = state_dataset_[i];
        vel_out[i] = state.Rwc.transpose() * Rcb_.transpose() *x.segment<3>(i * 3);
    }
    scale_out = (x.tail<1>())(0);

    return result;
}

bool VisualInertialAligment::AlignmentByOptimize(double &scale_out, Eigen::Vector3d &Gw_out, Eigen::Vector3d &bg_out,
                                                 std::vector<Eigen::Vector3d> &vel_out, bool doStereo) {
    AvergGravityDir(Gw_out);
    Gw_out *= acc_i_.norm() / Gw_out.norm();

    std::vector<Eigen::Vector3d> vel_set;
    for (size_t i = 0; i < state_dataset_.size() - 1; i++) {
        VisualInertialState &statei = state_dataset_[i];
        VisualInertialState &statej = state_dataset_[i + 1];
        Eigen::Vector3d vel = -statej.Rwc.transpose() * statej.twc - (-statei.Rwc.transpose() * statei.twc);
        vel /= statej.imu0.dT_;
        if (i == 0) {
            vel_set.push_back(vel);
            continue;
        }

        vel_set.push_back((vel + vel_set.back()) * 0.5);
        if (i == (state_dataset_.size() - 2)) {
            vel_set.push_back(vel);
        }
    }

    GyroBiasEstimating();

    double scale_cand[7] = {1.0, 0.05, 0.1, 0.5, 2, 5, 10};
    Eigen::Vector3d good_ba, good_bg, good_rgw, good_rcb;
    double good_scale, min_cost = 1e8;
    std::vector<Eigen::Vector3d> good_vel;
    good_vel.resize(state_dataset_.size());

    Eigen::Vector3d baj, bgj;
    baj.setZero();
    bgj = bg_;

    int scaleSize = doStereo ? 1 : 7;
    for (size_t i = 0; i < static_cast<size_t>(scaleSize); i++) {
        double init_scale = scale_cand[i];

        double ba[3] = {ba_(0), ba_(1), ba_(2)};
        double bg[3] = {bg_(0), bg_(1), bg_(2)};
        std::unordered_map<int, double *> vel_array;
        for (int j = 0; j < static_cast<int>(vel_set.size()); j++) {
            vel_array[j] = new double[3]{vel_set[j](0), vel_set[j](1), vel_set[j](2)};
        }

        double rvec_gw[3] = {0, 0, 0};
        double scale = init_scale;
        ceres::Problem problem;
        for (int n = 0; n < static_cast<int>(state_dataset_.size()) - 1; n++) {
            VisualInertialState &state_i = state_dataset_[n];
            VisualInertialState &state_j = state_dataset_[n + 1];
            if (state_j.imu0.dT_ < 0.001) {
                continue;
            }
            Eigen::Matrix3d Rbiw = state_i.Rwc.transpose() * Rcb_.transpose();
            Eigen::Matrix3d Rbjw = state_j.Rwc.transpose() * Rcb_.transpose();
            Eigen::Vector3d Pj = (-state_j.Rwc.transpose() * state_j.twc);
            Eigen::Vector3d Pi = (-state_i.Rwc.transpose() * state_i.twc);

            Eigen::Matrix<double, 9, 9> sqrt_info = state_j.imu0.cov_.block<9, 9>(0, 0).inverse();
            sqrt_info = Eigen::LLT<Eigen::Matrix<double, 9, 9>>(sqrt_info).matrixL().transpose();
            ceres::CostFunction *init_cost = new ImuInitCost(sqrt_info, Pi, Pj, Gw_out, tcb_, bgj, Rcb_, Rbiw, Rbjw, state_j.imu0);
            problem.AddResidualBlock(init_cost, NULL, vel_array[n], vel_array[n + 1], ba, rvec_gw, &scale);
        }
        problem.SetParameterBlockConstant(ba);
        if(doStereo) {
            problem.SetParameterBlockConstant(&scale);
        }
        // problem.SetParameterBlockConstant(rvec_gw);

        ceres::Solver::Options options;
        options.max_num_iterations = 5;
        options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        double cost;
        ceres::Problem::EvaluateOptions eval_opts;
        problem.Evaluate(eval_opts, &cost, NULL, NULL, NULL);

        if (cost < min_cost) {
            min_cost = cost;
            good_scale = scale;
            good_ba << ba[0], ba[1], ba[2];
            good_bg << bg[0], bg[1], bg[2];
            good_rgw << rvec_gw[0], rvec_gw[1], rvec_gw[2];
            for (int m = 0; m < static_cast<int>(good_vel.size()); m++) {
                good_vel[m] = Eigen::Vector3d(vel_array[m][0], vel_array[m][1], vel_array[m][2]);
                delete[] vel_array[m];
            }
        }
    }
    if (true) {
        scale_out = good_scale;
        bg_out = good_bg;
        Eigen::Matrix3d Rgw = RotationVector2Matrix3d(good_rgw);
        Gw_out = Rgw * Gw_out;
        vel_out.resize(state_dataset_.size());
        for (int i = 0; i < static_cast<int>(vel_out.size()); i++) {
            vel_out[i] = good_vel[i] * scale_out;
        }
        std::cout << "min cost: " << min_cost << std::endl;
    }

    if (min_cost < 8e5) {
        return true;
    }
    std::cout << "optimize cost too large: " << min_cost << std::endl;
    return false;
}


double VisualInertialAligment::AccStd() {
    if (frame_dataset_.size() < 3) {
        return 0.0;
    }

    Eigen::Vector3d acc_mean(0, 0, 0);

    for (size_t i = 1; i < frame_dataset_.size(); i++) {
        VisualInertialState &state = frame_dataset_[i].imu_state;
        acc_mean += state.imu0.Vij_ / state.imu0.dT_;
    }
    acc_mean /= static_cast<double>(frame_dataset_.size()) - 1.0;

    double acc_std = 0;
    for (size_t i = 1; i < frame_dataset_.size(); i++) {
        VisualInertialState &state = frame_dataset_[i].imu_state;
        Eigen::Vector3d acc_std0 = state.imu0.Vij_ / state.imu0.dT_ - acc_mean;
        acc_std += acc_std0.transpose() * acc_std0;
    }
    acc_std = sqrt(acc_std / (frame_dataset_.size() - 1));

    return acc_std;
}

bool VisualInertialAligment::AvergGravityDir(Eigen::Vector3d &Gw) {
    if (state_dataset_.size() < 5) {
        std::cout << "Not enough visual inertial state to Initilizing gravity direction!!!" << std::endl;
        return false;
    }

    Eigen::Vector3d acc_w;
    // VisualInertialState &state0 = state_dataset_[0];
    // acc_w = state0.Rwc.transpose() * Rcb.transpose() * state0.imu0.GetAvgAcc();
    acc_w.setZero();

    for (size_t i = 1; i < state_dataset_.size(); i++) {
        VisualInertialState &state = state_dataset_[i];
        acc_w += state.Rwc.transpose() * Rcb_.transpose() * state.imu0.GetAvgAcc();
        // Eigen::Vector3d acc = state.Rwc.transpose() * Rcb.transpose() * state.imu0.GetAvgAcc();
    }

    Gw = acc_w / (state_dataset_.size() - 1);
    return true;
}

bool VisualInertialAligment::Reset(bool force) {
    if (state_dataset_.size() >= start_calib_frame_num_ || force) {
        has_extrinsic_ = false;
        is_success_ = false;
        ba_.setZero();
        bg_.setZero();
        Rcb_.setIdentity();
        tcb_.setZero();
        state_dataset_.clear();
        frame_dataset_.clear();
        return true;
    }
    return false;
}

void VisualInertialAligment::InsertState(cv::Mat &Rwc, cv::Mat &twc, VisualInertialState &state) {
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> eig_R((double *)Rwc.data);
    Eigen::Map<Eigen::Vector3d> eig_t((double *)twc.data);
    state.Rwc = eig_R;
    state.twc = eig_t;
    state_dataset_.push_back(state);
}

void VisualInertialAligment::FrameOptimization(double angularResolution) {
    FrameTimeBasedResize(imu_init_time_dif);
    state_dataset_.clear();
    if (frame_dataset_.size() < start_aligment_frame_num_) {
        return;
    }

    bool begin_to_save = false;
    std::vector<Frame> tmp_frames;
    std::vector<MapPointPtr> &mapPoints = map_->mapPoints_;
    Eigen::Matrix3d sqrt_info;
    sqrt_info.setIdentity();
    double last_pose[6] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < frame_dataset_.size(); i++) {
        Frame &frame = frame_dataset_[i];
        if (frame.FrameId() != -2) {
            cv::Mat Rwc, twc;
            frame.GetPose(Rwc, twc);
            cv::Mat rvec;
            cv::Rodrigues(Rwc, rvec);
            double *rdata = (double*)rvec.data;
            double *tdata = (double*)twc.data;
            last_pose[0] = rdata[0]; last_pose[1] = rdata[1]; last_pose[2] = rdata[2];
            last_pose[3] = tdata[0]; last_pose[4] = tdata[1]; last_pose[5] = tdata[2];
            begin_to_save = true;
            if (tmp_frames.size() != 0) {
                tmp_frames.push_back(frame);
            }
            continue;
        }
        ceres::Problem problem;
        ceres::LossFunction *loss_function = new ceres::HuberLoss(2.0 * angularResolution);
        const std::vector<Feature> &features = frame.Features();
        int feature_num = 0;
        for (int j = 0; j < features.size(); j++) {
            const Feature &feat = features[j];
            MapPointPtr &map_pt = mapPoints[feat.featureID];
            if (map_pt->pt3dFlag) {
                Eigen::Vector3d obs(feat.spherePt1.x, feat.spherePt1.y, feat.spherePt1.z);
                Eigen::Vector3d ptw(map_pt->pt3d.x, map_pt->pt3d.y, map_pt->pt3d.z);
                ceres::CostFunction *cost = new CamProjPoseError(sqrt_info, obs, ptw);
                problem.AddResidualBlock(cost, loss_function, last_pose);
                feature_num++;
            }
        }

        //double t0 = cv::getTickCount();
        ceres::Solver::Options options;
        options.max_num_iterations = 5;
        options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
        // options.linear_solver_type = ceres::LinearSolverType::DENSE_SCHUR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        // options.num_threads = 4;
        options.minimizer_progress_to_stdout = false;
        options.function_tolerance = 5.0e-02;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        //double t1 = cv::getTickCount();
        //double dt = (t1 - t0) * 1000.0 / cv::getTickFrequency();
        // std::cout << "***each vio optimization cost time: " << dt << ", feat num: " << feature_num << std::endl;

        cv::Vec6d pose0;
        pose0(0) = last_pose[0];
        pose0(1) = last_pose[1];
        pose0(2) = last_pose[2];
        pose0(3) = last_pose[3];
        pose0(4) = last_pose[4];
        pose0(5) = last_pose[5];
        frame.SetPose(pose0);
        frame.SetFrameId(-1);
        cv::Mat Rwc, twc;
        frame.GetPose(Rwc, twc);
        cv::Mat tcw = Rwc.t() * twc;

        // Eigen::Map<Eigen::Matrix<double, 3, 3,Eigen::RowMajor>> eig_Rwc((double*)Rwc.data);
        // Eigen::Matrix3d Rbw = (Rcb * eig_Rwc).transpose();
        // if (frame.imu_state.imu0.dT_ > 0.001) {
        //     Eigen::Vector3d accw = Rbw * frame.imu_state.imu0.raw_data_.back().acc_;
        //     std::cout << "accw: " << accw.transpose() << std::endl;
        // }
        // std::cout << "pose: " << pose0.t() << std::endl;

        double len = cv::norm(tcw);
        if (len > 0.05) {
            begin_to_save = true;
        }
        if (begin_to_save) {
            tmp_frames.push_back(frame);
        }
        // std::cout.precision(13);
        // std::cout << frame.Time() << "th pose: " << tcw.t() << std::endl;
        // std::cout << std::setprecision(13) << frame.GetTime() << "th pose: " << pose0.t() << std::endl;
    }
    // cv::waitKey(0);
    if (tmp_frames.size() != 0) {
        frame_dataset_.swap(tmp_frames);
    }

    double acc_std = AccStd();
    std::cout << "***acc_std: " << acc_std << std::endl;
    if (acc_std < acc_std_thres_) {
        // AligmentReset(true);
        return;
    }

    FrameLocalOptimization(frame_dataset_, map_, angularResolution);

    for (int i = 0; i < frame_dataset_.size(); i++) {
        Frame &frame = frame_dataset_[i];
        cv::Mat Rwc, twc;
        frame.GetPose(Rwc, twc);
        frame.imu_state.timestamp = frame.Time();
        InsertState(Rwc, twc, frame.imu_state);
    }
    for (int i = 0; i < frame_dataset_.size() - 1; i++) {
        Frame &frame = frame_dataset_[i];
        Frame &framej = frame_dataset_[i + 1];
        cv::Mat Rwc, twc;
        framej.GetPose(Rwc, twc);
        cv::Mat tcw = Rwc.t() * twc;
        // std::cout << std::setprecision(13) << framej.Time() << "th tcw: " << tcw.t() << "," << framej.imu_state.imu0.GetMoveAccStd()
        //           << "," << cv::norm(tcw) << std::endl;
        if (i == (frame_dataset_.size() - 1))
            break;

        double t = framej.Time() - frame.Time();
        double err_t = t - framej.imu_state.imu0.dT_;
        if (err_t > 0.0001) {
            std::cout.precision(13);
            std::cout << "err time: " << framej.Time() << "," << frame.Time() << "dt: " << framej.imu_state.imu0.dT_ << std::endl;
            // std::cout << "err id: " << framej.FrameId() << "th time: "<< std::setprecision(13) << framej.Time() << std::endl;
            std::cout << "time error_____" << std::endl;
            exit(10);
        }
    }
    std::cout << "frame num: " << frame_dataset_.size() << std::endl;
}

void VisualInertialAligment::FrameTimeBasedResize(double time) {
    if (frame_dataset_.size() < 5) {
        return;
    }

    std::vector<Frame> used_frame;
    used_frame.push_back(frame_dataset_.front());
    used_frame.push_back(frame_dataset_[1]);
    bool changed = false;
    for (int i = 2; i < frame_dataset_.size(); i++) {
        Frame &frame = frame_dataset_[i];
        Frame &ref_frame = used_frame.back();
        if (i == (frame_dataset_.size() - 1)) {
            if (changed) {
                ref_frame.imu_state.imu0.RePreintegration();
                used_frame.push_back(frame);
            } else {
                if (ref_frame.imu_state.imu0.dT_ > time) {
                    used_frame.push_back(frame);
                } else {
                    ref_frame.imu_state.imu0.InsertNewImu(frame.imu_state.imu0);
                    frame.imu_state.imu0 = ref_frame.imu_state.imu0;
                    ref_frame = frame;
                    ref_frame.imu_state.imu0.RePreintegration();
                }
            }
            break;
        }
        if (ref_frame.imu_state.imu0.dT_ < time) {
            ref_frame.imu_state.imu0.InsertNewImu(frame.imu_state.imu0);
            ref_frame.imu_state.imu0.dT_ += frame.imu_state.imu0.dT_;
            frame.imu_state.imu0 = ref_frame.imu_state.imu0;
            ref_frame = frame;
            changed = true;
            continue;
        }
        if (changed) {
            ref_frame.imu_state.imu0.RePreintegration();
            changed = false;
        }
        used_frame.push_back(frame);
    }
    frame_dataset_.swap(used_frame);
}

}  // namespace inslam
