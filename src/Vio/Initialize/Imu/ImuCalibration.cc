#include "Vio/Initialize/Imu/ImuCalibration.h"
#include "Vio/Initialize/Optimization.hpp"

#include <unordered_map>
#include <ceres/ceres.h>

namespace hybrid_msckf {
static inline Eigen::Matrix3d VectorSkew(Eigen::Vector3d rvec) {
    Eigen::Matrix3d skew;
    skew << 0.0, -rvec[2], rvec[1], rvec[2], 0.0, -rvec[0], -rvec[1], rvec[0], 0.0;
    return skew;
}

static inline Eigen::Matrix3d RotationVector2Matrix3d(const Eigen::Vector3d rvec) {
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

static inline Eigen::Vector3d RotationMatrix2Vector3d(const Eigen::Matrix3d R) {
    double theta = acos((R.trace() - 1.0) * 0.5);

    Eigen::Matrix3d Right = (R - R.transpose()) * 0.5;
    Eigen::Vector3d rvec;
    rvec[0] = (Right(2, 1) - Right(1, 2)) * 0.5;
    rvec[1] = (Right(0, 2) - Right(2, 0)) * 0.5;
    rvec[2] = (Right(1, 0) - Right(0, 1)) * 0.5;
    double rvNorm = rvec.norm();
    if (rvNorm < 1.0e-07) {
        return rvec;
    }

    double cosTheta = cos(theta);
    double sinTheta = sin(theta);
    if (fabs(sinTheta) < 1.0e-08 || fabs(cosTheta) > 1) {
        return rvec;
    }
    rvec /= sinTheta;

    return rvec * theta;
}

static inline Eigen::Matrix<double, 4, 4> Omega(Eigen::Matrix<double, 3, 1> w) {
    Eigen::Matrix<double, 4, 4> mat;
    mat.block(0, 0, 3, 3) = -VectorSkew(w);
    mat.block(3, 0, 1, 3) = -w.transpose();
    mat.block(0, 3, 3, 1) = w;
    mat(3, 3) = 0;
    return mat;
}

static inline Eigen::Matrix<double, 4, 1> Quatnorm(Eigen::Matrix<double, 4, 1> qt) {
    if (qt(3, 0) < 0) {
        qt *= -1;
    }
    return qt / qt.norm();
}

bool ImuCalibration::ExtrinsicCalibration(std::vector<Eigen::Matrix3d> &RcList, std::vector<Eigen::Matrix3d> &RimuList,
                                          Eigen::Matrix3d &calibRcbResult, double &error) {
    error = 1e8;
    std::vector<Eigen::Matrix3d> inertial_R, visual_R;

    Eigen::Matrix3d tmpA;
    Eigen::Vector3d Pbji, Pcji, tmpb;
    int pose_num = static_cast<int>(RcList.size());
    int cols = 3 * (pose_num - 1);
    Eigen::MatrixXd A(cols, 3), b(cols, 1);
    int start_id = 0;
    for (size_t i = 0; i < static_cast<size_t>(pose_num) - 1; i++) {
        Eigen::Matrix3d Rbjbi = RimuList[i];
        Eigen::Matrix3d Rcjci = RcList[i];
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

    if (start_id < 0.2 * startCalibFrameNum_) {
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

    error = 0;
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
    calibRcbResult = Rcb_;
    std::cout << "********Rcb: " << Rcb_ << std::endl;
    std::cout << "*******estimating extrinsic error: " << error << std::endl;

    return true;
}

bool ImuCalibration::Calibrate() {
    if(static_cast<int>(frameDataset_.size()) < startCalibFrameNum_) {
        return false;
    }

    LocalOptimization(frameDataset_, map_, angularResolution_);

    std::vector<Eigen::Matrix3d> RcjiList;
    std::vector<double> frameTimeList;
    for(int i = 0; i < static_cast<int>(frameDataset_.size()) - 1; i++) {
        FramePtr frameI = frameDataset_[i];
        FramePtr frameJ = frameDataset_[i + 1];
        frameTimeList.push_back(frameI->Time());
        cv::Mat rji = frameI->R() * frameJ->R().t();
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R_ji((double*)rji.data);
        RcjiList.push_back(R_ji);
    }
    frameTimeList.push_back(frameDataset_.back()->Time());

    int imuCnt = static_cast<int>(imuDataList_.size());
    std::vector<Eigen::Matrix3d> RImuList(imuCnt);
    Eigen::Vector3d gyro0, gyro1;
    Eigen::Matrix3d RImu = Eigen::Matrix3d::Identity();
    RImuList[0] = RImu;
    for(int i = 1; i < imuCnt; i++) {
        // Runge-Kutta integrate
        gyro0 = imuDataList_[i -1].segment(1, 3);
        gyro1 = imuDataList_[i].segment(1, 3);
        double dt = imuDataList_[i](0) - imuDataList_[i - 1](0);
        Eigen::Vector4d dqVec;
        IntegrateGyroRK4(dt, gyro0, gyro1, dqVec);
        Eigen::Quaterniond dq(dqVec);
        Eigen::Matrix3d dR = dq.toRotationMatrix();
        RImu = RImu * dR;

        // median integral
        //Eigen::Vector3d gyro = (imuDataList_[i].segment(1, 3) + imuDataList_[i + 1].segment(1, 3)) / 2;
        //double dt = imuDataList_[i+1](0) - imuDataList_[i](0);
        //Eigen::Vector3d rvec = gyro * dt;
        //Eigen::Matrix3d dR = RotationVector2Matrix3d(rvec);
        //RImu = RImu * dR;

        RImuList[i] = RImu;
    }

    double calibDeltaTResult = CalibrateRawTime(RcjiList, frameTimeList, RImuList);
    std::cout << "calib  RAW deltaT: " << calibDeltaTResult << std::endl;
    CalibrateRotAndTime(calibDeltaTResult, RcjiList, frameTimeList, RImuList);
    return true;
}


void ImuCalibration::CalibrateRotAndTime(double rawDeltaT,
                                         std::vector<Eigen::Matrix3d> &RcjiList,
                                         std::vector<double> &frameTimeList,
                                         std::vector<Eigen::Matrix3d> &RImuList) {
    double timeScope = 0.005;
    double minErr = 1e8;
    Eigen::Matrix3d calibRcbResult;
    double calibDeltaTResult = 0;
    std::vector<Eigen::Matrix3d> RcList, RgList;
    for(double dT = rawDeltaT - timeScope; dT < rawDeltaT + timeScope; dT += 0.0005) {
        InterpolationImuPose(dT, RcjiList, frameTimeList, RImuList, RcList, RgList);
        if(!RcList.empty()) {
            double aveErr = 0;
            Eigen::Matrix3d tmpRcb;
            CalibrationExRotation(RcList, RgList, tmpRcb, aveErr);
            //ExtrinsicCalibration(RcList, RgList, tmpRcb, aveErr);
            if(aveErr < minErr) {
                calibRcbResult = tmpRcb;
                calibDeltaTResult = dT;
                minErr = aveErr;
            }
        }
    }

    double degreeErr = 180 * minErr / CV_PI;
    std::cout << "calib average error(degree): " << degreeErr << std::endl;
    if(degreeErr > 0.6) {
        std::cout << "Warning: calibrete average error is too large: " << degreeErr << std::endl;
    }
    std::cout << "calib Rc->b: " << calibRcbResult << std::endl;
    std::cout << "calib img deltaT: " << calibDeltaTResult << std::endl;
    std::cout << "Calibration completed" << std::endl;
}


double ImuCalibration::CalibrateRawTime(std::vector<Eigen::Matrix3d> &RcjiList,
                                     std::vector<double> &frameTimeList,
                                     std::vector<Eigen::Matrix3d> &RImuList) {
    double minErr = 1e8;
    double calibDeltaTResult = 1e6;
    std::vector<Eigen::Matrix3d> RcList, RgList;
    for(double dT = minDeltaT_; dT < maxDeltaT_; dT += alignTimeStep_) {
        InterpolationImuPose(dT, RcjiList, frameTimeList, RImuList, RcList, RgList);
        if(!RcList.empty()) {
            double aveErr = 0;
            double sumErr = 0;
            for (size_t i = 1; i < RcList.size(); i++) {
                Eigen::Matrix3d &Rcjci = RcList[i];
                Eigen::Matrix3d &Rbjbi = RgList[i];

                Eigen::Vector3d rbvec = RotationMatrix2Vector3d(Rbjbi);
                Eigen::Vector3d rcvec = RotationMatrix2Vector3d(Rcjci);

                double err = fabs(rcvec.norm() - rbvec.norm()) * (rcvec.norm() + rbvec.norm());
                sumErr += err;
            }
            aveErr = sumErr / RcList.size();

            if(aveErr < minErr) {
                calibDeltaTResult = dT;
                minErr = aveErr;
            }
        }
    }

    // save to file for check data alignment
#if 0
    std::ofstream ofs("./deltaR.csv");
    if (ofs.is_open()) {
        for (size_t i = 0; i < RcListTmp.size(); i++) {
            Eigen::Matrix3d &Rcjci = RcListTmp[i];
            Eigen::Matrix3d &Rbjbi = RgListTmp[i];

            Eigen::Vector3d rbvec = RotationMatrix2Vector3d(Rbjbi);
            Eigen::Vector3d rcvec = RotationMatrix2Vector3d(Rcjci);
            double err = fabs(rcvec.norm() - rbvec.norm());
            ofs << i << "," << rcvec.norm() << "," << rbvec.norm() << "," << err << "\n";
        }
        ofs.close();
    }
#endif

    return  calibDeltaTResult;
}


void ImuCalibration::InterpolationImuPose(double dT,
                                          std::vector<Eigen::Matrix3d> &RcjiList,
                                          std::vector<double> &frameTimeList,
                                          std::vector<Eigen::Matrix3d> &RImuList,
                                          std::vector<Eigen::Matrix3d> &RcList,
                                          std::vector<Eigen::Matrix3d> &RgList) {
    int imuCnt = static_cast<int>(imuDataList_.size());
    RcList.clear();
    RgList.clear();
    int iImu = 1;
    for(int i = 0; i < static_cast<int>(frameTimeList.size() - 1); i++) {
        double t0 = frameTimeList[i] + dT;
        double t1 = frameTimeList[i + 1] + dT;
        if(imuDataList_[0](0) > t0 || imuDataList_.back()(0) < t1) {
            continue;
        }

        bool bfirst = true;
        double ratio;
        Eigen::Matrix3d deltaR0, deltaR1;
        Eigen::Matrix3d imuR0, imuR1;
        for(; iImu < imuCnt; iImu++) {
            const Eigen::Matrix<double, 7, 1> &imu = imuDataList_[iImu];
            const Eigen::Matrix<double, 7, 1> &imua = imuDataList_[iImu - 1];
            double tImu = imu(0);
            if (tImu < t0) {
                continue;
            } else if (bfirst) {
                ratio = (tImu - t0) / (tImu - imua(0));
                Eigen::Matrix<double, 7, 1> imuInterp = ratio * imua + (1.0 - ratio) * imu;
                Eigen::Vector3d gyroMid = (imuInterp.segment(1, 3) + imu.segment(1, 3)) / 2;
                deltaR0 = RotationVector2Matrix3d((tImu - t0) * gyroMid);
                imuR0 = RImuList[iImu];
                bfirst = false;
            } else if (tImu > t1) {
                ratio = (tImu - t1) / (tImu - imua(0));
                Eigen::Matrix<double, 7, 1> imuInterp = ratio * imua + (1.0 - ratio) * imu;
                Eigen::Vector3d gyroMid = (imuInterp.segment(1, 3) + imua.segment(1, 3)) / 2;
                deltaR1 = RotationVector2Matrix3d((t1 - imua(0)) * gyroMid);
                imuR1 = RImuList[iImu - 1];
                break;
            }
        }
        iImu -= 2;
        RcList.push_back(RcjiList[i]); //R_ck+1->ck
        RgList.push_back(deltaR0 * imuR0.transpose() * imuR1 * deltaR1); //R_bk+1->bk
    }
}


bool ImuCalibration::CalibrationExRotation(std::vector<Eigen::Matrix3d> &RcList, std::vector<Eigen::Matrix3d> &RimuList,
                                                   Eigen::Matrix3d &calibRcbResult, double &error)
{
    error = 1e8;
    int frame_count = static_cast<int>(RcList.size());
    Eigen::MatrixXd A(frame_count * 4, 4);
    A.setZero();
    int startID = 0;
    for (int i = 1; i <= frame_count; i++)
    {
        Eigen::Matrix3d Rc_g = Rcb_.transpose() * RimuList[i] * Rcb_;
        Eigen::Quaterniond r1(RcList[i]);
        Eigen::Quaterniond r2(Rc_g);

        Eigen::Vector3d rvec_cjci = RotationMatrix2Vector3d(RcList[i]);
        if(rvec_cjci.norm() < CV_PI / 180) {
            continue;
        }

        double angular_distance = 180 / CV_PI * r1.angularDistance(r2);

        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        Eigen::Matrix4d L, R;

        double w = Eigen::Quaterniond(RcList[i]).w();
        Eigen::Vector3d q = Eigen::Quaterniond(RcList[i]).vec();
        L.block<3, 3>(0, 0) = w * Eigen::Matrix3d::Identity() + VectorSkew(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Eigen::Quaterniond R_ij(RimuList[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Eigen::Matrix3d::Identity() - VectorSkew(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        A.block<4, 4>((startID) * 4, 0) = huber * (L - R);
        startID++;
    }

    if(startID < frame_count * 0.2) {
        std::cout << "estimating rotation fail, not eough rotation frame!!!" << std::endl;
        return false;
    }

    A = A.topRows(startID * 4);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 4, 1> x = svd.matrixV().col(3);
    Eigen::Quaterniond estimated_R(x);
    Rcb_ = estimated_R.toRotationMatrix().inverse();
    //std::cout << "singularValues " << svd.singularValues().transpose() << std::endl;

    Eigen::Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>();
    if (ric_cov(1) > 0.25)
    {
        error = 0;
        for (size_t i = 0; i < RcList.size(); i++) {
            Eigen::Matrix3d &Rcjci = RcList[i];
            Eigen::Matrix3d &Rbjbi = RimuList[i];

            Eigen::Matrix3d R_hat = Rcb_ * Rcjci * Rcb_.transpose();
            Eigen::Vector3d rbvec = RotationMatrix2Vector3d(Rbjbi);
            Eigen::Vector3d r_hat = RotationMatrix2Vector3d(R_hat);
            Eigen::Vector3d err = r_hat - rbvec;
            error += err.norm();
        }
        error /= RcList.size();
        calibRcbResult = Rcb_;
        return true;
    } else {
        std::cout << "Calibrate fail, need more camera rotation" << std::endl;
        return false;
    }
}


bool ImuCalibration::Reset() {
    frameDataset_.clear();
    imuDataList_.clear();
    return false;
}


void ImuCalibration::IntegrateGyroRK4(double dt, const Eigen::Vector3d &w0,
                                    const Eigen::Vector3d &w1, Eigen::Vector4d &dq) {
    // Pre-compute things
    Eigen::Vector3d wHat = w0;
    Eigen::Vector3d wAlpha = (w1-w0)/dt;

    // k1 ================
    Eigen::Vector4d dq0 = {0,0,0,1};
    Eigen::Vector4d q0Dot = 0.5*Omega(wHat)*dq0;
    Eigen::Vector4d k1Q = q0Dot*dt;

    // k2 ================
    wHat += 0.5*wAlpha*dt;
    Eigen::Vector4d dq1 = Quatnorm(dq0+0.5*k1Q);
    Eigen::Vector4d q1Dot = 0.5*Omega(wHat)*dq1;
    Eigen::Vector4d k2Q = q1Dot*dt;

    // k3 ================
    Eigen::Vector4d dq2 = Quatnorm(dq0+0.5*k2Q);
    Eigen::Vector4d q2Dot = 0.5*Omega(wHat)*dq2;
    Eigen::Vector4d k3Q = q2Dot*dt;

    // k4 ================
    wHat += 0.5*wAlpha*dt;
    Eigen::Vector4d dq3 = Quatnorm(dq0+k3Q);
    Eigen::Vector4d q3Dot = 0.5*Omega(wHat)*dq3;
    Eigen::Vector4d k4Q = q3Dot*dt;

    // y+dt ================
    dq = Quatnorm(dq0+(1.0/6.0)*k1Q+(1.0/3.0)*k2Q+(1.0/3.0)*k3Q+(1.0/6.0)*k4Q);
    //new_q = quat_multiply(dq, q_0);
}


}  // namespace hybrid_msckf
