#include "HybridSlam.h"

const double MAHALA95_TABLE[] = {
        3.8415,    5.9915,    7.8147,    9.4877,   11.0705,   12.5916,   14.0671,   15.5073,   16.9190,   18.3070,
        19.6751,   21.0261,   22.3620,   23.6848,   24.9958,   26.2962,   27.5871,   28.8693,   30.1435,   31.4104,
        32.6706,  33.9244,   35.1725,   36.4150,   37.6525,   38.8851,   40.1133,   41.3371,   42.5570,   43.7730,
        44.9853,   46.1943,   47.3999,   48.6024,   49.8018,   50.9985,   52.1923,   53.3835,   54.5722,   55.7585,
        56.9424,   58.1240,   59.3035,   60.4809,   61.6562,   62.8296,   64.0011,   65.1708,   66.3386,   67.5048,
        68.6693,   69.8322,   70.9935,   72.1532,   73.3115,   74.4683,   75.6237,   76.7778,   77.9305,   79.0819,
        80.2321,   81.3810,   82.5287,   83.6753,   84.8206,   85.9649,   87.1081,   88.2502,   89.3912,   90.5312,
        91.6702,   92.8083,   93.9453,   95.0815,   96.2167,   97.3510,   98.4844,   99.6169,  100.7486,  101.8795,
        103.0095,  104.1387,  105.2672,  106.3948,  107.5217,  108.6479,  109.7733,  110.8980,  112.0220,  113.1453,
        114.2679,  115.3898,  116.5110,  117.6317,  118.7516,  119.8709,  120.9896,  122.1077,  123.2252,  124.3421,
        125.4584,  126.5741,  127.6893,  128.8039,  129.9180,  131.0315,  132.1444,  133.2569,  134.3688,  135.4802,
        136.5911,  137.7015,  138.8114,  139.9208,  141.0297,  142.1382,  143.2461,  144.3537,  145.4607,  146.5674,
        147.6735,  148.7793,  149.8846,  150.9894,  152.0939,  153.1979,  154.3015,  155.4047,  156.5075,  157.6099,
        158.7119,  159.8135,  160.9148,  162.0156,  163.1161,  164.2162,  165.3159,  166.4153,  167.5143,  168.6130,
        169.7113,  170.8092,  171.9068,  173.0041,  174.1010,  175.1976,  176.2938,  177.3897,  178.4854,  179.5806,
        180.6756,  181.7702,  182.8646,  183.9586,  185.0523,  186.1458,  187.2389,  188.3317,  189.4242,  190.5165,
        191.6084,  192.7001,  193.7914,  194.8825,  195.9734,  197.0639,  198.1542,  199.2442,  200.3339,  201.4234,
        202.5126,  203.6015,  204.6902,  205.7786,  206.8668,  207.9547,  209.0424,  210.1298,  211.2170,  212.3039,
        213.3906,  214.4771,  215.5633,  216.6492,  217.7350,  218.8205,  219.9058,  220.9908,  222.0756,  223.1602,
        224.2446,  225.3288,  226.4127,  227.4964,  228.5799,  229.6632,  230.7463,  231.8292,  232.9118,  233.9943
};
namespace featslam{

inline void MatrixXdReduce(Eigen::MatrixXd &m, int idx, int len) {
    int idx1 = idx + len;
    int cols = m.cols();
    m.block(idx, 0, cols - idx1, idx) = m.block(idx1, 0, cols - idx1, idx);
    m.block(0, idx, idx, cols - idx1) = m.block(0, idx1, idx, cols - idx1);
    m.block(idx, idx, cols - idx1, cols - idx1) = m.block(idx1, idx1, cols - idx1, cols - idx1);
    m.conservativeResize(cols - len, cols - len);
}

inline Eigen::Matrix<double, 2, 3> TangentPlaneCalc(Eigen::Vector3d &vec) {
    Eigen::Vector3d tmp(0., 0., 1.);
    double cos_theta = tmp.transpose() * vec;
    if (fabs(cos_theta) > 0.9) {
        tmp << 1., 0., 0.;
    }
    Eigen::Vector3d b1 = (tmp - vec.transpose() * tmp * vec); b1 /= b1.norm();
    Eigen::Vector3d b2 = vec.cross(b1); b2 /= b2.norm();
    Eigen::Matrix<double, 2, 3> out;
    out.row(0) = b1.transpose();
    out.row(1) = b2.transpose();
    return out;
}

HybridSlam::HybridSlam(const nlohmann::json &config, std::shared_ptr<Caimura> cam0, std::shared_ptr<Caimura> cam1)
    : cam0_(std::move(cam0)), cam1_(std::move(cam1)) {
    slidingwindow_num_ = 10; // config["sliding_window_size"];
    state_features_num_ = config["num_instate_features"];
    use_unitsphere_error_ = config["use_unit_sphere_error"].get<bool>();
    use_extrinsic_calib_ = false;
    use_rolling_shutter_calib_ = true;
    use_td_calib_ = false;
    use_FEJ_ = false;
    imu_state_dim_ = 15;

    if ((cam0_->IsEquirectangular() || cam0_->IsEquidistant()) && !use_unitsphere_error_) {
        std::cerr << "pano camera or equidistant camera only support "
                "unit sphere reprojection errors for now!" << std::endl;
        use_unitsphere_error_ = false;
    }

    gravity_ << 0., 0, -9.81;
    features_noise_ = 1;
    features_noise_ *= std::pow(cam0_->GetAngularResolution(), 2);
    std::cout << "feature noise: " << features_noise_ << std::endl;

    img_width0_ = cam0_->width();
    img_height0_ = cam0_->height();
    cx0_ = cam0_->cx();
    cy0_ = cam0_->cy();
    // Stereo pose
    if (cam1_ != nullptr) {
        img_width1_ = cam1_->width();
        img_height1_ = cam1_->height();
        cx1_ = cam1_->cx();
        cy1_ = cam1_->cy();
        Rclcr_ = cam1_->mR_C_s * cam0_->mR_S_c;
        tclcr_ = cam1_->mp_C_s + cam1_->mR_C_s * cam0_->mp_S_c;
        // Rclcr_ << 0.999598,   0.0259439,  0.0114278,
        //           -0.0261918, 0.999413,   0.0221044,
        //           -0.0108476, -0.0223948, 0.99969;
        // Rclcr_ = Rclcr_.transpose();
        // tclcr_ << -0.0053, -0.103039, -0.0010;
        // tclcr_ = -Rclcr_ * tclcr_;
        // evo
        // Rclcr_ <<  0.9993668017304382, -0.027485117698636007, -0.022595661179908296,
        //            0.02743886189037998, 0.9996207102737596, -0.002354665567541322,
        //            0.022651809138030058, 0.0017331753709412423, 0.9997419125183749;
        // tclcr_ << -0.07541198697931975, 0.00012983918976516615, -0.001993342084771335;
        // simulation
        // Rclcr_ <<  0.99959, -0.026191, -0.0108476,
        //            0.0259439, 0.999413, -0.0223948,
        //            0.0114278, 0.0221044, 0.99969;
        // tclcr_ << 0.00258824, 0.103094, 0.00333787;
    }

    // imu noise [ng, na, nwg, nwa]
    imu_noise_.resize(12, 12);
    imu_noise_.setIdentity();
    imu_noise_.block<3, 3>(0, 0) *= cam0_->msigma_gyr * cam0_->msigma_gyr;
    imu_noise_.block<3, 3>(3, 3) *= cam0_->msigma_acc * cam0_->msigma_acc;
    imu_noise_.block<3, 3>(6, 6) *= cam0_->mrando_gyr * cam0_->mrando_gyr;
    imu_noise_.block<3, 3>(9, 9) *= cam0_->mrando_acc * cam0_->mrando_acc;

    // keyframe threshold
    keyframe_rot_thres_ = 10.0 * 3.14 / 180.0;
    keyframe_translation_thres_ = 0.1;

    // from imu to camera extrinsic
    Ric_ = cam0_->mR_C_s;
    tci_ = cam0_->mp_S_c;

    // feature observation num threshold to add to state
    features_obs_num_thres_ = 5;
    triangulation_max_iteration_ = 5;
    least_landmark_num_thres_ = 30;
    max_landmark_num_thres_ = 150;

    // one pixel for huber function threshold
    triangulation_huber_thres_ = 1.5;

    // reprojection error to outlier
    reproj_err_thres_ = 2.5;

    // max and min depth inverse
    max_depth_inv_ = 10;
    min_depth_inv_ = 0.02;

    // open Ric and tci extrinsic calibration
    // Ric = (I - dtheta_ic) * Ric_hat
    // tci = tci_hat + dtci
    if (use_extrinsic_calib_) {
        extrinsic_cov_id_ = imu_state_dim_;
        imu_state_dim_ += 6;
    }

    // open time delay calibration
    // td = td_hat + dtd // td_hat is image input time
    // image_t = captured_t + td // captured_t is image input time
    if (use_td_calib_) {
        td_cov_id_ = imu_state_dim_;
        imu_state_dim_ += 1;
    }

    // rolling shutter time calibration
    if (use_rolling_shutter_calib_) {
        rs_cov_id_ = imu_state_dim_;
        imu_state_dim_ += 1;
    }
}

void HybridSlam::Initialize(Eigen::Vector4d& q, Eigen::Vector3d& p, Eigen::Vector3d &v, int frame_id) {
    std::cout << "-----hybrid VIO initialized-----" << std::endl;
    frame_states_.clear();
    feature_states_.clear();
    map_.clear();

    // QIG, PGI, VGI, bg, ba
    t_imu_.ba = cam0_->mba; //+ Eigen::Vector3d(-0.033, -0.312, 0.044);
    t_imu_.bg = cam0_->mbg;
    t_imu_.Vw = v;
    t_imu_.tiw = p;
    Eigen::Quaterniond q_Ij_W(q(0), q(1), q(2), q(3));
    t_imu_.Rwi = q_Ij_W.matrix().transpose();
    Eigen::Vector4d JPLQwi = MathUtil::JPLMatrix2Quat(t_imu_.Rwi);
    t_imu_.Qwi = JPLQwi;

    // old state
    t_imu_.Qwi_null = t_imu_.Qwi;
    t_imu_.Rwi_null = t_imu_.Rwi;
    t_imu_.tiw_null = t_imu_.tiw;
    t_imu_.Vw_null = t_imu_.Vw;
    t_imu_.gyro.setZero();
 
    // QIG = QIG_hat * dtheta, PGI = PGI_hat + dp, VGI = VGI_hat + dv;
    // bg = bg_hat + dbg, ba = ba_hat + dba
    P_.resize(imu_state_dim_, imu_state_dim_);
    P_.setZero();

    int i = 0;
    for (i = 0; i < 3; i++) P_(i, i) = 1.0e-07;
    for (i = 3; i < 6; i++) P_(i, i) = 1.0e-07;
    for (i = 6; i < 9; i++) P_(i, i) = 1.0e-07;
    for (i = 9; i < 12; i++) P_(i, i) = cam0_->mrando_gyr * cam0_->mrando_gyr;
    for (i = 12; i < 15; i++) P_(i, i) = cam0_->mrando_acc * cam0_->mrando_acc;
    if (use_extrinsic_calib_) {
        for (i = 0; i < 6; i++) P_(extrinsic_cov_id_ + i, extrinsic_cov_id_ + i) = 1.0e-05;
    }
    if (use_td_calib_) {
        P_(td_cov_id_, td_cov_id_) = 1.0e-06;
    }
    if (use_rolling_shutter_calib_) {
        P_(rs_cov_id_, rs_cov_id_) = 1.0e-06;
    }
}

void HybridSlam::Run(const std::vector<FeatureObservation> &featureMsgs,
                     double t_img,
                     int frame_id,
                     const std::vector<Eigen::Matrix<double,7,1> > &vimu) {
    std::cout << "********Hybrid Slam**********" << std::endl;
    std::cout.precision(13);
    std::cout << "timestamp: " << t_img << std::endl;
    // Propagate(vimu);
    // RK4Propagate(vimu);
    if (use_FEJ_) {
        FEJMedianPropagate(vimu);
    } else {
        MedianPropagate(vimu);
    }

    // OCMedianPropagate(vimu);

    t_imu_.frame_id = frame_id;
    MapUpdate(featureMsgs);

    FeatureAugment();

    NewObsFeatureUpdate();

    LostFeaturesUpdate();

    FrameStateAugment();

    DrawBias();

    static Eigen::Matrix3d Rwi0 = t_imu_.Rwi;
    std::cout << "imu Pw: " << t_imu_.tiw.transpose() << std::endl;
    std::cout << "feature state num: " << feature_states_.size() << std::endl;
    // std::cout << "imu position cov: " << P_.block<3, 3>(0, 0) << std::endl;
    std::cout << "tci: " << tci_.transpose() << std::endl;
    // std::cout << "Ric: " << Ric_ << std::endl;
    std::cout << "Rci: " << Ric_.transpose() << std::endl;
    // Eigen::Matrix3d Rcrci = Ric_.transpose() * Rclcr_.transpose();
    // Eigen::Vector3d tcrci = tci_ - Rcrci * tclcr_;
    // std::cout << "Rcrci: " << Rcrci << std::endl;
    // std::cout << "tcrci: " << tcrci.transpose() << std::endl;
    std::cout << "ba: " << t_imu_.ba.transpose() << std::endl;
    std::cout << "bg: " << t_imu_.bg.transpose() << std::endl;
    std::cout << "**********End Hybrid Slam******" << std::endl;
    if (frame_id > 1) {
        // cv::waitKey(0);
    }
}

// Qwi(k+1) = Qwi(k) + 0.5 * Omega(gyro * dt) * Qwi(k), Qwi = [x, y, z, w] is a JPL quaternion
// (I - dtetha(k+1)) * Rwi(k+1) = exp((-w - dbg) * dt) * (I - dtheta(k)) * Rwi(k)
// dtheta(k+1) = Rk(k+1) * dtheta(k) - Rk(k+1)*Jr(-w * dt) * dt * (dbg + ng)
// dp(k+1) = dp(k) + dv(k) * dt - 0.5 * Riw(k) * skew(acc) * dtheta(k) - 0.5 * Riw(k) * dt2 * dba - 0.5 * Riw(k) * dt2 * na
// dv(k+1) = dv(k) - Riw(k) * dt * dtheta(k) - Riw(k) * dt * dba - Riw(k) * dt * na
// dstate = [dtheta, dp, dv, dbg, dba]
void HybridSlam::Propagate(const std::vector<Eigen::Matrix<double, 7, 1> > &vimu) {
    for (size_t i = 0; i < vimu.size(); i++) {
        const Eigen::Matrix<double, 7, 1> &data = vimu[i];
        if (t_imu_.timestamp < 0 && i == 0) {
            t_imu_.timestamp = data(0);
            continue;
        }
        // std::cout << "Qwi: " << t_imu_.Qwi.transpose() << std::endl;
        Eigen::Vector3d w = data.segment(1, 3) - t_imu_.bg;
        Eigen::Vector3d acc = data.tail(3) - t_imu_.ba;
        double dt = data(0) - t_imu_.timestamp;
        double dt2 = dt * dt;

        // state integration
        Eigen::Matrix4d omega; omega.setZero();
        omega.topLeftCorner(3, 3) = -MathUtil::VecToSkew(w);
        omega.topRightCorner(3, 1) = w;
        omega.bottomLeftCorner(1, 3) = -w.transpose();
        omega *= 0.5;
        Eigen::Vector4d Qwi = t_imu_.Qwi + omega * t_imu_.Qwi * dt;
        Qwi /= Qwi.norm();

        t_imu_.tiw = t_imu_.tiw + t_imu_.Vw * dt + 0.5 * gravity_ * dt2 + 0.5 * t_imu_.Rwi.transpose() *
                     acc * dt2;
        t_imu_.Vw = t_imu_.Vw + gravity_ * dt + t_imu_.Rwi.transpose() * acc * dt;

        // error covariace propagation
        Eigen::Matrix<double, 15, 15> F;
        F.setIdentity();
        Eigen::Matrix3d dJr = MathUtil::Jr(-w * dt);
        Eigen::Matrix3d dR = MathUtil::Vec2RotationMatrix(-w * dt);
        Eigen::Matrix3d Riw = t_imu_.Rwi.transpose();

        F.block<3, 3>(0, 0) = dR;
        F.block<3, 3>(0, 9) = -dR * dJr * dt;
        F.block<3, 3>(3, 0) = -0.5 * Riw * MathUtil::VecToSkew(acc) * dt2;
        F.block<3, 3>(3, 6) = dt * Eigen::Matrix3d::Identity();
        F.block<3, 3>(3, 12) = -0.5 * Riw * dt2;
        F.block<3, 3>(6, 0) = -Riw * MathUtil::VecToSkew(acc) * dt;
        F.block<3, 3>(6, 12) = -Riw * dt;

        Eigen::Matrix<double, 15, 12> G;
        G.setZero();
        G.block<3, 3>(0, 0) = -dR * dJr * dt;
        G.block<3, 3>(3, 3) = -0.5 * Riw * dt2;
        G.block<3, 3>(6, 3) = -Riw * dt;
        G.block<3, 3>(9, 6).setIdentity();
        G.block<3, 3>(12, 9).setIdentity();

        // dx = J * di;
        // di_hat = phi * di + nr;
        // cov(dx, di_hat) = E(dx * di_hat^t) = E(dx * (phi * di + nr)^t)
        // cov(dx, di_hat) = E(dx * di^t * phi^t) = E(dx * di^t) *phi ^t
        P_.block<15, 15>(0, 0) = F * P_.block<15, 15>(0, 0) * F.transpose() + G * imu_noise_ * G.transpose();
        if (P_.cols() > 15) {
            int cols = P_.cols();
            P_.block(0, 15, 15, cols - 15) = F * P_.block(0, 15, 15, cols - 15);
            P_.block(15, 0, cols - 15, 15) = P_.block(0, 15, 15, cols - 15).transpose();
        }
        P_ = 0.5 * (P_ + P_.transpose());
        
        // state pass
        Eigen::Matrix3d Rwi = MathUtil::JPLQuat2Matrix(Qwi);
        t_imu_.Rwi_null = t_imu_.Rwi;
        t_imu_.Rwi = Rwi;
        t_imu_.Qwi = Qwi;
        t_imu_.timestamp = data(0);
    }
}

// Ric = (I - dtheta_ic) * Ric_hat; _hat is estimated state, the other is true state;
// tci = tci_hat + dtci;
void HybridSlam::MedianPropagate(const std::vector<Eigen::Matrix<double, 7, 1>> &vimu) {
    Eigen::Vector3d mean_acc(0., 0., 0.), mean_gyro(0., 0., 0.);
    for (size_t i = 1; i < vimu.size(); i++) {
        double dt = vimu[i][0] - vimu[i - 1][0];
        double dt2 = dt * dt;
        Eigen::Vector3d w = 0.5 * (vimu[i].segment(1, 3) + vimu[i - 1].segment(1, 3)) - t_imu_.bg;
        Eigen::Vector3d acc = 0.5 * (vimu[i].tail(3) + vimu[i - 1].tail(3)) - t_imu_.ba;
        mean_acc += acc;
        mean_gyro += w;

        // state integration
        Eigen::Matrix4d omega; omega.setZero();
        omega.topLeftCorner(3, 3) = -MathUtil::VecToSkew(w);
        omega.topRightCorner(3, 1) = w;
        omega.bottomLeftCorner(1, 3) = -w.transpose();
        omega *= 0.5;
        Eigen::Vector4d Qwi = t_imu_.Qwi + omega * t_imu_.Qwi * dt;
        Qwi /= Qwi.norm();

        t_imu_.tiw = t_imu_.tiw + t_imu_.Vw * dt + 0.5 * gravity_ * dt2 + 0.5 * t_imu_.Rwi.transpose() *
                     acc * dt2;
        t_imu_.Vw = t_imu_.Vw + gravity_ * dt + t_imu_.Rwi.transpose() * acc * dt;

        // error covariace propagation
        Eigen::MatrixXd F;
        F.conservativeResize(imu_state_dim_, imu_state_dim_);
        F.setIdentity();
        Eigen::Matrix3d dJr = MathUtil::Jr(-w * dt);
        Eigen::Matrix3d dR = MathUtil::Vec2RotationMatrix(-w * dt);
        Eigen::Matrix3d Riw = t_imu_.Rwi.transpose();

        F.block<3, 3>(0, 0) = dR;  // dtheta / dtheta
        F.block<3, 3>(0, 9) = -dR * dJr * dt; // dtheta / dbg
        F.block<3, 3>(3, 0) = -0.5 * Riw * MathUtil::VecToSkew(acc) * dt2; // dp / dtheta
        F.block<3, 3>(3, 6) = dt * Eigen::Matrix3d::Identity(); // dp / dv
        F.block<3, 3>(3, 12) = -0.5 * Riw * dt2;  // dp / dba
        F.block<3, 3>(6, 0) = -Riw * MathUtil::VecToSkew(acc) * dt; // dv / dtheta
        F.block<3, 3>(6, 12) = -Riw * dt; // dv / dba

        Eigen::MatrixXd G;
        G.conservativeResize(imu_state_dim_, 12);
        G.setZero();
        G.block<3, 3>(0, 0) = -dR * dJr * dt;
        G.block<3, 3>(3, 3) = -0.5 * Riw * dt2;
        G.block<3, 3>(6, 3) = -Riw * dt;
        G.block<3, 3>(9, 6).setIdentity();
        G.block<3, 3>(12, 9).setIdentity();

        // dx = J * di;
        // di_hat = phi * di + nr;
        // cov(dx, di_hat) = E(dx * di_hat^t) = E(dx * (phi * di + nr)^t)
        // cov(dx, di_hat) = E(dx * di^t * phi^t) = E(dx * di^t) *phi ^t
        Eigen::MatrixXd noise = imu_noise_;
        noise.block<6, 6>(0, 0) /= dt;
        noise.block<6, 6>(6, 6) *= dt;
        if (fabs(dt) < 1.0e-06) {
            continue;
        }

        P_.block(0, 0, imu_state_dim_, imu_state_dim_) = F * P_.block(0, 0, imu_state_dim_, imu_state_dim_) * F.transpose()
            + G * noise * G.transpose();
        if (P_.cols() > imu_state_dim_) {
            int cols = P_.cols();
            P_.block(0, imu_state_dim_, imu_state_dim_, cols - imu_state_dim_) = F * P_.block(0, imu_state_dim_, imu_state_dim_, cols - imu_state_dim_);
            P_.block(imu_state_dim_, 0, cols - imu_state_dim_, imu_state_dim_) = P_.block(0, imu_state_dim_, imu_state_dim_, cols - imu_state_dim_).transpose();
        }
        P_ = 0.5 * (P_ + P_.transpose());
        
        // state pass
        Eigen::Matrix3d Rwi = MathUtil::JPLQuat2Matrix(Qwi);
        t_imu_.Rwi_null = t_imu_.Rwi;
        t_imu_.Rwi = Rwi;
        t_imu_.Qwi = Qwi;
        t_imu_.gyro = w;
        t_imu_.timestamp = vimu[i](0);
    }
    if (vimu.size() > 2) {
        mean_acc /= vimu.size() - 1;
        mean_acc += t_imu_.Rwi * gravity_;
        std::cout << "mean pure motion acc: " << mean_acc.transpose() << std::endl;
        mean_gyro /= vimu.size() - 1;
        std::cout << "mean pure motion gyr: " << mean_gyro.transpose() << std::endl;
        std::cout << "acc: " << vimu.back().tail(3).transpose() << std::endl;
    }
}

// observability constraint propagation
void HybridSlam::OCMedianPropagate(const std::vector<Eigen::Matrix<double, 7, 1>> &vimu) {
    for (size_t i = 1; i < vimu.size(); i++) {
        double dt = vimu[i][0] - vimu[i - 1][0];
        double dt2 = dt * dt;
        Eigen::Vector3d w = 0.5 * (vimu[i].segment(1, 3) + vimu[i - 1].segment(1, 3)) - t_imu_.bg;
        Eigen::Vector3d acc = 0.5 * (vimu[i].tail(3) + vimu[i - 1].tail(3)) - t_imu_.ba;

        // state integration
        Eigen::Matrix4d omega; omega.setZero();
        omega.topLeftCorner(3, 3) = -MathUtil::VecToSkew(w);
        omega.topRightCorner(3, 1) = w;
        omega.bottomLeftCorner(1, 3) = -w.transpose();
        omega *= 0.5;
        Eigen::Vector4d Qwi = t_imu_.Qwi + omega * t_imu_.Qwi * dt;
        Qwi /= Qwi.norm();
        Eigen::Matrix3d Rwi = MathUtil::JPLQuat2Matrix(Qwi);

        t_imu_.tiw = t_imu_.tiw + t_imu_.Vw * dt + 0.5 * gravity_ * dt2 + 0.5 * t_imu_.Rwi.transpose() *
                     acc * dt2;
        t_imu_.Vw = t_imu_.Vw + gravity_ * dt + t_imu_.Rwi.transpose() * acc * dt;

        // error covariace propagation
        Eigen::Matrix<double, 15, 15> F;
        F.setIdentity();
        Eigen::Matrix3d dJr = MathUtil::Jr(-w * dt);
        Eigen::Matrix3d dR = MathUtil::Vec2RotationMatrix(-w * dt);
        Eigen::Matrix3d Riw = t_imu_.Rwi.transpose();

        F.block<3, 3>(0, 0) = dR;  // dtheta / dtheta
        F.block<3, 3>(0, 9) = -dR * dJr * dt; // dtheta / dbg
        F.block<3, 3>(3, 0) = -0.5 * Riw * MathUtil::VecToSkew(acc) * dt2; // dp / dtheta
        F.block<3, 3>(3, 6) = dt * Eigen::Matrix3d::Identity(); // dp / dv
        F.block<3, 3>(3, 12) = -0.5 * Riw * dt2;  // dp / dba
        F.block<3, 3>(6, 0) = -Riw * MathUtil::VecToSkew(acc) * dt; // dv / dtheta
        F.block<3, 3>(6, 12) = -Riw * dt; // dv / dba

        // Observability-constraint modification of F
        Eigen::Matrix3d phi_qq = Rwi * t_imu_.Rwi_null.transpose();
        Eigen::Matrix3d phi_vq = F.block<3, 3>(6, 0);
        Eigen::Vector3d u = t_imu_.Rwi_null * gravity_;
        Eigen::Vector3d v = MathUtil::VecToSkew(t_imu_.Vw_null - t_imu_.Vw) * gravity_;
        phi_vq = phi_vq - (phi_vq * u - v) * u.transpose() / (u.transpose() * u);
        Eigen::Matrix3d phi_pq = F.block<3, 3>(3, 0);
        v = MathUtil::VecToSkew(dt * t_imu_.Vw_null + t_imu_.tiw_null - t_imu_.tiw) * gravity_;
        phi_pq = phi_pq - (phi_pq * u - v) * u.transpose() / (u.transpose() * u);
        F.block<3, 3>(0, 0) = phi_qq;
        F.block<3, 3>(6, 0) = phi_vq;
        F.block<3, 3>(3, 0) = phi_pq;

        Eigen::Matrix<double, 15, 12> G;
        G.setZero();
        G.block<3, 3>(0, 0) = -dR * dJr * dt;
        G.block<3, 3>(3, 3) = -0.5 * Riw * dt2;
        G.block<3, 3>(6, 3) = -Riw * dt;
        G.block<3, 3>(9, 6).setIdentity();
        G.block<3, 3>(12, 9).setIdentity();

        // dx = J * di;
        // di_hat = phi * di + nr;
        // cov(dx, di_hat) = E(dx * di_hat^t) = E(dx * (phi * di + nr)^t)
        // cov(dx, di_hat) = E(dx * di^t * phi^t) = E(dx * di^t) *phi ^t
        P_.block<15, 15>(0, 0) = F * P_.block<15, 15>(0, 0) * F.transpose() + G * imu_noise_ * G.transpose();
        if (P_.cols() > 15) {
            int cols = P_.cols();
            P_.block(0, 15, 15, cols - 15) = F * P_.block(0, 15, 15, cols - 15);
            P_.block(15, 0, cols - 15, 15) = P_.block(0, 15, 15, cols - 15).transpose();
        }
        P_ = 0.5 * (P_ + P_.transpose());
        
        // state pass
        t_imu_.Rwi = Rwi;
        t_imu_.Qwi = Qwi;
        t_imu_.timestamp = vimu[i](0);
        t_imu_.Qwi_null = t_imu_.Qwi;
        t_imu_.tiw_null = t_imu_.tiw;
        t_imu_.Vw_null = t_imu_.Vw;
        t_imu_.Rwi_null = t_imu_.Rwi;
    }
}


void HybridSlam::FEJMedianPropagate(const std::vector<Eigen::Matrix<double, 7, 1>> &vimu) {
    if (!use_FEJ_) {
        t_imu_.Qwi_null = t_imu_.Qwi;
        t_imu_.Rwi_null = t_imu_.Rwi;
        t_imu_.tiw_null = t_imu_.tiw;
        t_imu_.Vw_null = t_imu_.Vw;
    }
    for (size_t i = 1; i < vimu.size(); i++) {
        double dt = vimu[i](0) - vimu[i - 1](0);
        double dt2 = dt * dt;
        Eigen::Vector3d w = 0.5 * (vimu[i].segment(1, 3) + vimu[i - 1].segment(1, 3)) - t_imu_.bg;
        Eigen::Vector3d acc = 0.5 * (vimu[i].tail(3) + vimu[i - 1].tail(3)) - t_imu_.ba;

        // state integration
        Eigen::Matrix4d omega; omega.setZero();
        omega.topLeftCorner(3, 3) = -MathUtil::VecToSkew(w);
        omega.topRightCorner(3, 1) = w;
        omega.bottomLeftCorner(1, 3) = -w.transpose();
        omega *= 0.5;
        t_imu_.Qwi = t_imu_.Qwi + omega * t_imu_.Qwi * dt;
        t_imu_.Qwi /= t_imu_.Qwi.norm();
        Eigen::Matrix3d Rwi = MathUtil::JPLQuat2Matrix(t_imu_.Qwi);
        t_imu_.Rwi = Rwi;

        t_imu_.tiw = t_imu_.tiw + t_imu_.Vw * dt + 0.5 * gravity_ * dt2 + 0.5 * t_imu_.Rwi.transpose() *
                     acc * dt2;
        t_imu_.Vw = t_imu_.Vw + gravity_ * dt + t_imu_.Rwi.transpose() * acc * dt;

        Eigen::Matrix3d dR = t_imu_.Rwi * t_imu_.Rwi_null.transpose();

        Eigen::Vector3d dp = t_imu_.tiw - t_imu_.tiw_null - t_imu_.Vw_null * dt - 0.5 * gravity_ * dt2;
        Eigen::Vector3d dv = t_imu_.Vw - t_imu_.Vw_null - acc * dt;
        Eigen::Matrix3d phi_pq = -MathUtil::VecToSkew(dp) * t_imu_.Rwi_null.transpose();
        Eigen::Matrix3d phi_vq = -MathUtil::VecToSkew(dv) * t_imu_.Rwi_null.transpose();
        Eigen::Matrix3d Jr = MathUtil::Jr(-w * dt);

        Eigen::MatrixXd phi;
        phi.conservativeResize(imu_state_dim_, imu_state_dim_);
        phi.setIdentity();
        phi.block<3, 3>(0, 0) = dR;
        phi.block<3, 3>(0, 9) = -dR * Jr * dt;
        phi.block<3, 3>(3, 0) = phi_pq;
        phi.block<3, 3>(3, 6) = dt * Eigen::Matrix3d::Identity();
        phi.block<3, 3>(3, 12) = -0.5 * t_imu_.Rwi_null.transpose() * dt2;
        phi.block<3, 3>(6, 0) = phi_vq;
        phi.block<3, 3>(6, 12) = -t_imu_.Rwi_null.transpose() * dt;

        Eigen::MatrixXd G;
        G.conservativeResize(imu_state_dim_, 12);
        G.setZero();
        G.block<3, 3>(0, 0) = -dR * Jr * dt;
        G.block<3, 3>(3, 3) = -0.5 * t_imu_.Rwi_null.transpose() * dt2;
        G.block<3, 3>(6, 3) = -t_imu_.Rwi_null.transpose() * dt;
        G.block<3, 3>(9, 6).setIdentity();
        G.block<3, 3>(12, 9).setIdentity();

        Eigen::MatrixXd noise = imu_noise_;
        noise.block<6, 6>(0, 0) /= dt;
        noise.block<6, 6>(6, 6) *= dt;
        P_.block(0, 0, imu_state_dim_, imu_state_dim_) = phi * P_.block(0, 0, imu_state_dim_, imu_state_dim_) * phi.transpose()
            + G * noise * G.transpose();
        if (P_.cols() > imu_state_dim_) {
            int cols = P_.cols();
            P_.block(0, imu_state_dim_, imu_state_dim_, cols - imu_state_dim_) = phi * P_.block(0, imu_state_dim_, imu_state_dim_, cols - imu_state_dim_);
            P_.block(imu_state_dim_, 0, cols - imu_state_dim_, imu_state_dim_) = P_.block(0, imu_state_dim_, imu_state_dim_, cols - imu_state_dim_).transpose();
        }
        P_ = 0.5 * (P_ + P_.transpose());

        t_imu_.Rwi_null = t_imu_.Rwi;
        t_imu_.Qwi_null = t_imu_.Qwi;
        t_imu_.tiw_null = t_imu_.tiw;
        t_imu_.Vw_null = t_imu_.Vw;
        t_imu_.timestamp = vimu[i](0);
    }
}


void HybridSlam::PredictNewState(Eigen::Vector3d &acc, Eigen::Vector3d &w, double dt) {
    Eigen::Matrix4d omega;
    omega.setZero();
    omega.topLeftCorner(3, 3) = -MathUtil::VecToSkew(w);
    omega.topRightCorner(3, 1) = w;
    omega.bottomLeftCorner(1, 3) = -w.transpose();

    // dq_dt  = 0.5 * omega * q
    double gyro_norm = w.norm();
    Eigen::Vector4d dq_dt, dq_dt2;
    if (gyro_norm > 1.e-05) {
        dq_dt = (cos(gyro_norm * dt * 0.5) * Eigen::Matrix4d::Identity() + 1. / gyro_norm
                 * sin(gyro_norm * dt * 0.5) * omega) * t_imu_.Qwi;
        dq_dt2 = (cos(gyro_norm * dt * 0.25) * Eigen::Matrix4d::Identity() + 1. / gyro_norm
                  * sin(gyro_norm * dt * 0.25) * omega) * t_imu_.Qwi;
    } else {
        dq_dt = (Eigen::Matrix4d::Identity() + 0.5 * dt * omega) * cos(gyro_norm * dt * 0.5) * t_imu_.Qwi;
        dq_dt2 = (Eigen::Matrix4d::Identity() + 0.25 * dt * omega) * cos(gyro_norm * dt * 0.25) * t_imu_.Qwi;
    }

    Eigen::Matrix3d Rwi = MathUtil::JPLQuat2Matrix(dq_dt);
    Eigen::Matrix3d Rwi2 = MathUtil::JPLQuat2Matrix(dq_dt2);

    // k1 = f(tn, un);
    Eigen::Vector3d k1_dv = t_imu_.Rwi.transpose() * acc + gravity_;
    Eigen::Vector3d k1_dp = t_imu_.Vw;

    // k2 = f(tn + dt / 2, yn + k1 * dt / 2)
    Eigen::Vector3d k2_dv = Rwi2.transpose() * acc + gravity_;
    Eigen::Vector3d k2_dp = t_imu_.Vw + k1_dv * dt / 2.;

    // k3 = f(tn + dt / 2, yn + k2 * dt / 2)
    Eigen::Vector3d k3_dv = Rwi2.transpose() * acc + gravity_;
    Eigen::Vector3d k3_dp = t_imu_.Vw + k2_dv * dt / 2.;

    // k4 = f(tn + dt, yn + k3 * dt)
    Eigen::Vector3d k4_dv = Rwi.transpose() * acc + gravity_;
    Eigen::Vector3d k4_dp = t_imu_.Vw + k3_dv * dt;

    t_imu_.Qwi = dq_dt;
    t_imu_.Rwi = Rwi;
    t_imu_.Vw = t_imu_.Vw + (k1_dv + 2. * k2_dv + 2. * k3_dv + k4_dv) * dt / 6.;
    t_imu_.tiw = t_imu_.tiw + (k1_dp + 2. * k2_dp + 2. * k3_dp + k4_dp) * dt / 6.;
}



void HybridSlam::RK4Propagate(const std::vector<Eigen::Matrix<double, 7, 1> > &vimu) {
    for (size_t i = 1; i < vimu.size(); i++) {
        Eigen::Vector3d acc = vimu[i].tail(3) - t_imu_.ba;
        Eigen::Vector3d w = vimu[i].segment(1, 3) - t_imu_.bg;
        double dt = vimu[i](0) - vimu[i - 1](0);
        double dt2 = dt * dt;
        Eigen::Vector3d Vw_old = t_imu_.Vw;
        Eigen::Vector3d tiw_old = t_imu_.tiw;
        Eigen::Matrix3d Rwi_old = t_imu_.Rwi;
        PredictNewState(acc, w, dt);

        // pw(k) = pw + vw * dt + 0.5 * gw * dt2 + Riw * s(acc);
        // s(acc) = Rwi * (Pw(k) - pw - vw * dt - 0.5 * gw * dt2)
        // phi_pq = -Riw * [s(acc)X] = -Riw * [s(acc)X] * Rwi * Riw
        // phi_pq = -[Riw * s(acc)X] * Riw = -[(Pw(k) - pw - vw * dt - 0.5 * gw * dt2)X] * Riw
        // vw(k) = vw + gw * dt + Riw * y(acc)
        // y(acc) = Rwi * (vw(k) - vw - gw * dt)
        // phi_vq = -Riw * [y(acc)X] = -[Riw * y(acc)X] * Riw
        // phi_vq = -[(vw(k) - vw - gw * dt)X] * Riw
        Eigen::Vector3d dp = t_imu_.tiw - tiw_old - Vw_old * dt - 0.5 * gravity_ * dt2;
        Eigen::Matrix3d phi_pq = -MathUtil::VecToSkew(dp) * Rwi_old.transpose();
        Eigen::Vector3d dv = t_imu_.Vw - Vw_old - gravity_ * dt;
        Eigen::Matrix3d phi_vq = -MathUtil::VecToSkew(dv) * Rwi_old.transpose();

        Eigen::Matrix3d dJr = MathUtil::Jr(-w * dt);
        Eigen::Matrix3d dR = t_imu_.Rwi * Rwi_old.transpose();

        Eigen::Matrix<double, 15, 15> Fdt;
        Fdt.setIdentity();
        Fdt.block<3, 3>(0, 0) = dR;
        Fdt.block<3, 3>(0, 9) = -dR * dJr * dt;
        Fdt.block<3, 3>(3, 0) = phi_pq;
        Fdt.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;
        Fdt.block<3, 3>(3, 12) = -0.5 * Rwi_old.transpose() * dt2;
        Fdt.block<3, 3>(6, 0) = phi_vq;
        Fdt.block<3, 3>(6, 12) = -Rwi_old.transpose() * dt;

        Eigen::Matrix<double, 15, 12> Gdt;
        Gdt.setZero();
        Gdt.block<3, 3>(0, 0) = -dR * dJr * dt;
        Gdt.block<3, 3>(3, 3) = -0.5 * Rwi_old.transpose() * dt2;
        Gdt.block<3, 3>(6, 3) = -Rwi_old.transpose() * dt;
        Gdt.block<3, 3>(9, 6).setIdentity();
        Gdt.block<3, 3>(12, 9).setIdentity();

        // dnew = Fdt * di + Gdt * nq;
        // E(dnew * dnew^t) = E((Fdt * di + Gdt * nq) * (Fdt * di + Gdt * nq)^t)
        // = Fdt * cov(di) * Fdt^t + Gdt * cov(nq) * Gdt^t
        // = Fdt * P * Fdt^t + Gdt * Q * Gdt^t
        P_.block<15, 15>(0, 0) = Fdt * P_.block<15, 15>(0, 0) * Fdt.transpose() + Gdt * imu_noise_ * Gdt.transpose();
        if (frame_states_.size() > 0) {
            int dim = P_.cols() - 15;
            // dx = J * di;
            // di_hat = phi * di + nr;
            // cov(dx, di_hat) = E(dx * di_hat^t) = E(dx * (phi * di + nr)^t)
            // cov(dx, di_hat) = E(dx * di^t * phi^t) = E(dx * di^t) *phi ^t
            P_.block(0, 15, 15, dim) = Fdt * P_.block(0, 15, 15, dim);
            P_.block(15, 0, dim, 15) = P_.block(0, 15, 15, dim).transpose();
        }
        P_ = 0.5 * (P_ + P_.transpose());
    }
}


void HybridSlam::MapUpdate(const std::vector<FeatureObservation> &featureMsgs) {
    t_imu_.obs0.clear();
    t_imu_.obs1.clear();
    t_imu_.pt0s.clear();
    for (size_t i = 0; i < featureMsgs.size(); i++) {
        const FeatureObservation & feat = featureMsgs[i];
        if (map_.find(feat.id) == map_.end()) {
            map_.insert(std::make_pair(feat.id, VioFeature(feat.id)));
        }
        map_.at(feat.id).AddObservation(feat);
        t_imu_.obs0[feat.id] = feat.ray0;
        t_imu_.pt0s[feat.id] = Eigen::Vector2d(feat.pt0.x, feat.pt0.y);
        
        if (feat.isStereo) {
            t_imu_.obs1[feat.id] = feat.ray1;
            t_imu_.pt1s[feat.id] = Eigen::Vector2d(feat.pt1.x, feat.pt1.y);   
        }
    }
}

void HybridSlam::CovarianceReduceUpdate(int idx, int len) {
    MatrixXdReduce(P_, idx, len);

    // move feature covariance id
    for (auto &feat : feature_states_) {
        if (feat.second.cov_id > idx) {
            feat.second.cov_id -= len;
        }
    }

    // move frame covariance id
    for (auto &frame : frame_states_) {
        if (frame.cov_id > idx) {
            frame.cov_id -= len;
        }
    }
}

void HybridSlam::RemoveOutlier(size_t pt_id) {
    FeatureType feat = feature_states_[pt_id];
    CovarianceReduceUpdate(feat.cov_id, 1);
    feature_states_.erase(pt_id);
}

// feature augmentation
void HybridSlam::FeatureAugment() {
    std::vector<size_t> rm_ids;
    for (auto &feat : feature_states_) {
        if (t_imu_.obs0.find(feat.first) == t_imu_.obs0.end()) {
            CovarianceReduceUpdate(feat.second.cov_id, 1);
            rm_ids.push_back(feat.first);
        }        
    }
    for (size_t i = 0; i < rm_ids.size(); i++) {
        feature_states_.erase(rm_ids[i]);
    }

    if (feature_states_.size() > max_landmark_num_thres_) {
        return;
    }

    int good_count = 0;
    int f_num = frame_states_.size();
    int last_id = frame_states_.size() - 1;
    std::vector<std::vector<int>> pf_seens;
    std::vector<size_t> ptids;
    int obs_count = 0;
    int try_trian = 0;

    for (auto iti : t_imu_.obs0) {
        if (feature_states_.find(iti.first) == feature_states_.end()) {
            int count = 0;
            std::vector<int> seen_idxs;
            for (int i = 0; i < f_num; i++) {
                int idx = f_num - i - 1;
                const SensorState &frame = frame_states_[idx];
                if (frame.obs0.find(iti.first) != frame.obs0.end()) {
                    count++;
                    seen_idxs.push_back(idx);
                    if (frame.obs1.find(iti.first) != frame.obs1.end()) {
                        count++;
                    }                    
                } else {
                    break;
                }
            }

            if (count > features_obs_num_thres_) {
                double rho;
                bool flag = TryTriangulation(seen_idxs, iti.first, last_id, rho);
                try_trian++;
                if (flag) {
                    FeatureStateAugment(seen_idxs, iti.first, last_id, rho);
                    pf_seens.push_back(seen_idxs);
                    ptids.push_back(iti.first);
                    obs_count += seen_idxs.size() * 2;        
                    good_count++;
                }
            }
        }
    }

    std::cout << "try triangulation num: " << try_trian << std::endl;
    if (good_count > 0) {
        // exit(0);
    }

    // new feature update
    Eigen::MatrixXd H(2 * obs_count, P_.cols());
    Eigen::VectorXd r(2 * obs_count);
    int ridx = 0;
    for (size_t i = 0; i < ptids.size(); i++) {
        Eigen::MatrixXd Hx;
        Eigen::VectorXd rr;
        NewFeatureJacobians(ptids[i], pf_seens[i], rr, Hx);
        if (rr.rows() > 0) {
            if (true) {
                r.segment(ridx, rr.rows()) = rr;
                H.block(ridx, 0, rr.rows(), P_.cols()) = Hx;
                ridx += rr.rows();
            }
        }
    }

    std::cout << "new feature addition:" << std::endl;
    if (r.rows() > 0) {
        r.conservativeResize(ridx);
        std::cout << "before rr: " << r.norm() << "," << r.rows() << std::endl;
        H.conservativeResize(ridx, P_.cols());
        MeasurementUpdate(H, r);
    }

    ridx = 0;
    r.conservativeResize(2 * obs_count);
    for (size_t i = 0; i < ptids.size(); i++) {
        Eigen::MatrixXd Hx;
        Eigen::VectorXd rr;
        NewFeatureJacobians(ptids[i], pf_seens[i], rr, Hx);
        if (rr.rows() > 0) {
            if (true) {
                r.segment(ridx, rr.rows()) = rr;
                ridx += rr.rows();
            }
        }
    }
    if (r.rows() > 0) {
        std::cout << "after new triangulation update rr: " << r.norm() << "," << r.rows() << std::endl;
    }

    std::cout << "**adding new features: " << good_count << std::endl;
    std::cout << "Pw: " << t_imu_.tiw.transpose() << std::endl;
    if (good_count >= 1) {
        // cv::waitKey(0);
    }
}

void HybridSlam::MeasurementUpdate(const Eigen::MatrixXd &H, const Eigen::VectorXd &r) {
    int oldest_id = frame_states_.front().cov_id;
    int next_id = frame_states_[1].cov_id;
    // Hx.block(0, oldest_id, Hx.rows(), 6).setZero();
    // P_.block(0, oldest_id, P_.rows(), 6).setZero();
    // P_.block(oldest_id, 0, 6, P_.cols()).setZero();
    // P_.block(0, next_id, P_.rows(), 6).setZero();
    // P_.block(next_id, 0, 6, P_.cols()).setZero();

    if (H.rows() < 1) return;
    // r = H * dx + n;
    // r = H * (dx - 0) + n
    // r = 0 - (-H * dx)
    // r = r0 + H * dx + n;
    // H^t * H * dx = -H^t * r0;
    // H_thin = -H

    Eigen::MatrixXd H_thin;
    Eigen::VectorXd r_thin;
    if (H.rows() > H.cols()) {
        Eigen::HouseholderQR<Eigen::MatrixXd> qr_H(H);
        Eigen::MatrixXd Q = qr_H.householderQ();
        Eigen::MatrixXd Q1 = Q.leftCols(H.cols());
        H_thin = -Q1.transpose() * H;
        r_thin = Q1.transpose() * r;
    } else {
        H_thin = -H;
        r_thin = r;
    }
    // K = P * H^t * (H * P * H^t + R)_inv;
    // K^t = (H * P * H^t + R)_inv * H * P;
    Eigen::MatrixXd S = H_thin * P_ * H_thin.transpose() + features_noise_ *
                        Eigen::MatrixXd::Identity(H_thin.rows(), H_thin.rows());
    Eigen::MatrixXd Kt = S.ldlt().solve(H_thin * P_);
    Eigen::MatrixXd K = Kt.transpose();
    Eigen::VectorXd dx = K * r_thin;
    if (dx.norm() > 20 || dx.norm() == std::numeric_limits<double>::infinity() ||
        dx.norm() == std::numeric_limits<double>::signaling_NaN() || 
        dx.norm() == std::numeric_limits<double>::quiet_NaN()) {
        std::cout << "too big update in line: " << __LINE__  << "with dx norm: " << dx.norm() << std::endl;
        exit(10);
    }
    if (dx.hasNaN()) {
        std::cout << "has NaN in dx in line: " << __LINE__ << std::endl;
        exit(10);
    }
    
    Eigen::Vector4d dQ = MathUtil::SmallAngle2Quat(dx.segment<3>(0));
    t_imu_.Qwi = MathUtil::JPLQuatMultiply(dQ, t_imu_.Qwi);
    t_imu_.Rwi = MathUtil::JPLQuat2Matrix(t_imu_.Qwi);
    t_imu_.tiw += dx.segment<3>(3);
    t_imu_.Vw += dx.segment<3>(6);
    t_imu_.bg += dx.segment<3>(9);
    t_imu_.ba += dx.segment<3>(12);

    if (use_extrinsic_calib_) {
        Eigen::Matrix3d dRic = MathUtil::Vec2RotationMatrix(-dx.segment<3>(extrinsic_cov_id_));
        Ric_ = dRic * Ric_;
        tci_ += dx.segment<3>(extrinsic_cov_id_ + 3);
    }
    if (use_td_calib_) {
        t_imu_.td += dx(td_cov_id_);
    }

    if (use_rolling_shutter_calib_) {
        t_imu_.tr += dx(rs_cov_id_);
    }

    // Rwi = (I - [dthetax]) * Rwi
    for (size_t i = 0; i < frame_states_.size(); i++) {
        SensorState &frame = frame_states_[i];
        Eigen::Matrix3d dR = MathUtil::Vec2RotationMatrix(-dx.segment<3>(frame.cov_id));
        frame.Rwi = dR * frame.Rwi;
        frame.tiw += dx.segment<3>(frame.cov_id + 3);
    }

    for (auto &feat : feature_states_) {
        feat.second.depth_inv += dx(feat.second.cov_id);
    }

    // P = P - Pxz * Pzz_inv * Pzx;
    // P = P - P * H^t * (H * P * H^t + R)_inv * H * P
    // P = P - K * H * P
    // P = (I - K * H) * P
    Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(K.rows(), H_thin.cols()) - K * H_thin;
    P_ = I_KH * P_;
    P_ = 0.5 * (P_ + P_.transpose());
}


// observation jacobian calculation
void HybridSlam::FeatureJacobian(const Eigen::Vector3d &ray0, int host_id, double rho0, Eigen::Vector3d &pfg,
                                 Eigen::Matrix<double, 3, 6> &dpfg_dx1, Eigen::Matrix<double, 3, 6> &dpfg_dx0,
                                 Eigen::Matrix3d &drayg_dpfg, Eigen::Vector3d &dpfg_drho0, int guest_id, bool is_stereo) {
    SensorState &frame0 = frame_states_[host_id];
    SensorState &frame1 = frame_states_[guest_id];
    Eigen::Matrix3d &Rwi0 = frame0.Rwi;
    Eigen::Vector3d &tiw0 = frame0.tiw;
    Eigen::Matrix3d &Rwi1 = frame1.Rwi;
    Eigen::Vector3d &tiw1 = frame1.tiw;

    Eigen::Vector3d pf0 = ray0 / rho0;
    Eigen::Vector3d pfw = Rwi0.transpose() * Ric_.transpose() * pf0 + (tiw0 + Rwi0.transpose() * tci_);
    Eigen::Vector3d pf1 = Ric_ * Rwi1 * pfw - Ric_ * Rwi1 * tiw1 - Ric_ * tci_;

    Eigen::Matrix<double, 3, 6> dpf1_dx1;
    dpf1_dx1.leftCols(3) = Ric_ * MathUtil::VecToSkew(Rwi1 * (pfw - tiw1));
    dpf1_dx1.rightCols(3) = -Ric_ * Rwi1;
    Eigen::Matrix3d dpf1_dpfw = Ric_ * Rwi1;
    Eigen::Matrix<double, 3, 6> dpfw_dx0;
    dpfw_dx0.leftCols(3) = -Rwi0.transpose() * MathUtil::VecToSkew(Ric_.transpose() * pf0 + tci_);
    dpfw_dx0.rightCols(3).setIdentity();
    Eigen::Matrix3d dpfw_dpf0 = Rwi0.transpose() * Ric_.transpose();
    Eigen::Vector3d dpf0_drho0 = -ray0 / rho0 / rho0;
    if (!is_stereo) {
        // pf0 = ray0 / rho
        // pfw = Rwi0^t * Ric^t * pf0 + (tiw0 + Rwi0^t * tci)
        // pf1 = Ric * Rwi1 * pfw - Ric * Rwi1 * (tiw1  + Rwi1^t * tci)
        // pf1 = Ric * Rwi1 * pfw - Ric * Rwi1 * tiw1 - Ric * tci
        // dray1 / dpf1 = (rho * I33 - rho^3 * [pf1(0) * pf^t; pf(1) * pf^t; pf(2) * pf^t])
        // dpf1 / dx1 = [Ric * skew(Rwi1 * pfw - Rwi1 * tiw1), -Ric * Rwi1]
        // dpf1 / dpfw = Ric * Rwi1
        // dpfw / dx0 = [-Rwi0^t * skew(Ric^t * pf0 + tci), I]
        // dpfw / dpf0 = Rwi0^t * Ric^t
        // dpf0 / drho = -ray0 / rho / rho
        double rho1 = 1. / pf1.norm();
        double rho3 = rho1 * rho1 * rho1;
        drayg_dpfg.setIdentity();
        drayg_dpfg *= rho1;
        drayg_dpfg.row(0) -= rho3 * pf1(0) * pf1.transpose();
        drayg_dpfg.row(1) -= rho3 * pf1(1) * pf1.transpose();
        drayg_dpfg.row(2) -= rho3 * pf1(2) * pf1.transpose();

        dpfg_dx1 = dpf1_dx1;
        dpfg_dx0 = dpf1_dpfw * dpfw_dx0;
        dpfg_drho0 = dpf1_dpfw * dpfw_dpf0 * dpf0_drho0;
        pfg = pf1;
    } else {
        // pf0 = ray0 / rho0;
        // pfw = Rwi0^t * Ric^t * pf0 + (tiw0 + Rwi0^t * tci)
        // pf1 = Ric * Rwi1 * pfw - Ric * Rwi1 * (tiw1  + Rwi1^t * tci)
        // pf1 = Ric * Rwi1 * pfw - Ric * Rwi1 * tiw1 - Ric * tci
        // pfg = Rclcr * pf1 + tclcr
        // rayg = pfg / pfg.norm()
        Eigen::Vector3d pfg = Rclcr_ * pf1 + tclcr_;
        double rho1 = 1. / pfg.norm();
        double rho3 = rho1 * rho1 * rho1;
        Eigen::Matrix3d dpfg_dpf1 = Rclcr_;
        drayg_dpfg.setIdentity();
        drayg_dpfg *= rho1;
        drayg_dpfg.row(0) -= rho3 * pfg(0) * pfg.transpose();
        drayg_dpfg.row(1) -= rho3 * pfg(1) * pfg.transpose();
        drayg_dpfg.row(2) -= rho3 * pfg(2) * pfg.transpose();

        dpfg_dx1 = dpfg_dpf1 * dpf1_dx1;
        dpfg_dx0 = dpfg_dpf1 * dpf1_dpfw * dpfw_dx0;
        dpfg_drho0 = dpfg_dpf1 * dpf1_dpfw * dpfw_dpf0 * dpf0_drho0;        
    }
}

// feature observation update
void HybridSlam::NewFeatureJacobians(size_t pt_id, const std::vector<int> &seen_ids, Eigen::VectorXd &r, Eigen::MatrixXd &H) {
    if (feature_states_.find(pt_id) == feature_states_.end()) {
        std::cout << "Cannot find new features!!!in line:" << __LINE__ << std::endl;
        exit(10);
    }

    FeatureType &feat = feature_states_[pt_id];
    SensorState &host = frame_states_[feat.host_id];
    int host_id = feat.host_id;
    if (host.frame_id != feat.frame_id) {
        std::cout << "host id conflict with frame id in line: " << __LINE__ << std::endl;
        exit(10);
    }
    // r = H * dx + nr;
    H.resize(seen_ids.size() * 4, P_.cols());
    r.resize(seen_ids.size() * 4);
    H.setZero();
    r.setZero();
    int ridx = 0;
    const Eigen::Vector3d &ray0 = feat.ray;
    const Eigen::Matrix3d &Rwi0 = host.Rwi;
    const Eigen::Vector3d &tiw0 = host.tiw;
    const double &rho0 = feat.depth_inv;
    int cov_idx0 = host.cov_id;
    int feat_cov_idx = feat.cov_id;
    const Eigen::Matrix<double, 2, 3> &tangent = feat.tangentplane;

    for (size_t i = 0; i < seen_ids.size(); i++) {
        const int &idx = seen_ids[i];
        if (idx == feat.host_id) continue;
        SensorState &frame1 = frame_states_[idx];
        Eigen::Vector3d ray1 = frame1.obs0[pt_id];
        const Eigen::Matrix3d &Rwi1 = frame1.Rwi;
        const Eigen::Vector3d &tiw1 = frame1.tiw;

        // r = (ray1 - pf / pf.norm())
        // r = Hx0 * dx0 + Hx1 * dx1 + Hf * drho0 + nr
        Eigen::MatrixXd Hx0, Hx1, Hf, Hic, Htd, Htr;
        Eigen::VectorXd rr;
        if (use_extrinsic_calib_ || use_td_calib_ || use_rolling_shutter_calib_) {
            // TwoFrameJacobian(ray0, rho0, ray1, Rwi0, tiw0, Rwi1, tiw1, Hx0, Hx1, Hf, Hic, Htd, rr, false);
            TwoFrameJacobian(host_id, idx, pt_id, ray0, rho0, Hx0, Hx1, Hf, Hic, Htd, Htr, rr, false);
        } else {
            TwoFrameJacobian(ray0, rho0, ray1, Rwi0, tiw0, Rwi1, tiw1, Hx0, Hx1, Hf, rr, false);
        }
        double proj_err = rr.norm() / cam0_->GetAngularResolution();
        // std::cout << "proj err: " << proj_err << std::endl;
        if (proj_err > reproj_err_thres_) continue;

        r.segment<2>(ridx * 2) = tangent * rr;
        H.block<2, 6>(ridx * 2, cov_idx0) = tangent * Hx0;
        H.block<2, 6>(ridx * 2, frame1.cov_id) = tangent * Hx1;
        H.block<2, 1>(ridx * 2, feat_cov_idx) = tangent * Hf;
        if (use_extrinsic_calib_) {
            H.block<2, 6>(ridx * 2, extrinsic_cov_id_) = tangent * Hic;
        }
        if (use_td_calib_) {
            H.block<2, 1>(ridx * 2, td_cov_id_) = tangent * Htd;
        }
        if (use_rolling_shutter_calib_) {
            H.block<2, 1>(ridx * 2, rs_cov_id_) = tangent * Htr;
        }
        ridx++;
        if (frame1.obs1.find(pt_id) != frame1.obs1.end()) {
            ray1 = frame1.obs1[pt_id];
            if (use_extrinsic_calib_ || use_td_calib_ || use_rolling_shutter_calib_) {
                // TwoFrameJacobian(ray0, rho0, ray1, Rwi0, tiw0, Rwi1, tiw1, Hx0, Hx1, Hf, Hic, Htd, rr, true);
                TwoFrameJacobian(host_id, idx, pt_id, ray0, rho0, Hx0, Hx1, Hf, Hic, Htd, Htr, rr, true);
            } else {
                TwoFrameJacobian(ray0, rho0, ray1, Rwi0, tiw0, Rwi1, tiw1, Hx0, Hx1, Hf, rr, true);
            }
            proj_err = rr.norm() / cam0_->GetAngularResolution();
            // std::cout << "r proj_err: " << proj_err << std::endl;
            if (proj_err > reproj_err_thres_) continue;

            r.segment<2>(ridx * 2) = tangent * rr;
            H.block<2, 6>(ridx * 2, cov_idx0) = tangent * Hx0;
            H.block<2, 6>(ridx * 2, frame1.cov_id) = tangent * Hx1;
            H.block<2, 1>(ridx * 2, feat_cov_idx) = tangent * Hf;
            if (use_extrinsic_calib_) {
                H.block<2, 6>(ridx * 2, extrinsic_cov_id_) = tangent * Hic;
            }
            if (use_td_calib_) {
                H.block<2, 1>(ridx * 2, td_cov_id_) = tangent * Htd;
            }
            if (use_rolling_shutter_calib_) {
                H.block<2, 1>(ridx * 2, rs_cov_id_) = tangent * Htr;
            }
            ridx++;
        }
    }

    if (host.obs1.find(pt_id) != host.obs1.end()) {
        Eigen::Vector3d ray1 = host.obs1[pt_id];
        Eigen::MatrixXd Hf;
        Eigen::VectorXd rr;
        StereoJacobian(ray0, rho0, ray1, Hf, rr);
        double proj_err = rr.norm() / cam0_->GetAngularResolution();
        if (proj_err < reproj_err_thres_) {
            r.segment<2>(ridx * 2) = tangent * rr;
            H.block<2, 1>(ridx * 2, feat_cov_idx) = tangent * Hf;
            ridx++;
        }
    }
    r.conservativeResize(ridx * 2);
    H.conservativeResize(ridx * 2, P_.cols());
}

// Jacobian and residual in stereo camera
void HybridSlam::StereoJacobian(const Eigen::Vector3d &ray0, const double &rho0, const Eigen::Vector3d &ray1,
                                Eigen::MatrixXd &Hf, Eigen::VectorXd &r) {
    // pfr = Rclcr * pf0 + tclcr
    // r = ray1 - pfr
    Eigen::Vector3d pf0 = ray0 / rho0;
    Eigen::Vector3d pfr = Rclcr_ * pf0 + tclcr_;
    double rho1 = 1. / pfr.norm();
    double rho3 = rho1 * rho1 * rho1;
    r = ray1 - pfr * rho1;
    Eigen::Vector3d dpf0_drh0 = -pf0 / rho0;
    Eigen::Matrix3d dpfr_dpf0 = Rclcr_;
    Eigen::Matrix3d dray1_dpfr;
    dray1_dpfr.setIdentity();
    dray1_dpfr *= rho1;
    dray1_dpfr.row(0) -= rho3 * pfr(0) * pfr.transpose();
    dray1_dpfr.row(1) -= rho3 * pfr(1) * pfr.transpose();
    dray1_dpfr.row(2) -= rho3 * pfr(2) * pfr.transpose();
    Hf = -dray1_dpfr * dpfr_dpf0 * dpf0_drh0;
}

// Jacobian and residual between two frame
void HybridSlam::TwoFrameJacobian(const Eigen::Vector3d &ray0, const double &rho0, const Eigen::Vector3d &ray1,
                                  const Eigen::Matrix3d &Rwi0, const Eigen::Vector3d &tiw0, const Eigen::Matrix3d &Rwi1,
                                  const Eigen::Vector3d &tiw1, Eigen::MatrixXd &Hx0, Eigen::MatrixXd &Hx1, Eigen::MatrixXd &Hf,
                                  Eigen::VectorXd &r, bool is_stereo) {
    // r = ray1 - pf1 / pf1.norm()
    // r = Hx0 * dx0 + Hx1 * dx1 + Hf * drho0 + nr;
    // pf0 = ray0 / rho0;
    // pfw = Rwi0^t * Ric^t * pf0 + (tiw0 + Rwi0^t * tci)
    // pf1 = Ric * Rwi1 * pfw - Ric * Rwi1 * (tiw1 + Rwi1^t * tci)
    // pf1 = Ric * Rwi1 * pfw - Ric * Rwi1 * tiw1 - Ric * tci
    Eigen::Vector3d pf0 = ray0 / rho0;
    Eigen::Vector3d pfw = Rwi0.transpose() * Ric_.transpose() * pf0 + tiw0 + Rwi0.transpose() * tci_;
    Eigen::Vector3d pf1 = Ric_ * Rwi1 * pfw - Ric_ * Rwi1 * tiw1 - Ric_ * tci_;
    Eigen::Vector3d dpf0_drho0 = -pf0 / rho0;
    Eigen::Matrix3d dpfw_dpf0 = Rwi0.transpose() * Ric_.transpose();
    Eigen::Matrix<double, 3, 6> dpfw_dx0;
    dpfw_dx0.leftCols(3) = -Rwi0.transpose() * MathUtil::VecToSkew(Ric_.transpose() * pf0 + tci_);
    dpfw_dx0.rightCols(3).setIdentity();
    Eigen::Matrix3d dpf1_dpfw = Ric_ * Rwi1;
    Eigen::Matrix<double, 3, 6> dpf1_dx1;
    dpf1_dx1.leftCols(3) = Ric_ * MathUtil::VecToSkew(Rwi1 * (pfw - tiw1));
    dpf1_dx1.rightCols(3) = -Ric_ * Rwi1;
    if (!is_stereo) {
        // r = (ray0 - pf1 / pf1.norm())
        double rho1 = 1. / pf1.norm();
        double rho3 = rho1 * rho1 * rho1;
        Eigen::Matrix3d dray1_dpf1;
        dray1_dpf1.setIdentity();
        dray1_dpf1 *= rho1;
        dray1_dpf1.row(0) -= rho3 * pf1(0) * pf1.transpose();
        dray1_dpf1.row(1) -= rho3 * pf1(1) * pf1.transpose();
        dray1_dpf1.row(2) -= rho3 * pf1(2) * pf1.transpose();

        r = ray1 - pf1 * rho1;
        Hf = -dray1_dpf1 * dpf1_dpfw * dpfw_dpf0 * dpf0_drho0;
        Hx0 = -dray1_dpf1 * dpf1_dpfw * dpfw_dx0;
        Hx1 = -dray1_dpf1 * dpf1_dx1;
    } else {
        // pfr = Rclcr * pf1 + tclcr
        // r = ray1 - pfr / pfr.norm();
        Eigen::Vector3d pfr = Rclcr_ * pf1 + tclcr_;
        double rhor = 1. / pfr.norm();
        double rho3 = rhor * rhor * rhor;
        Eigen::Matrix3d dpfr_dpf1 = Rclcr_;
        Eigen::Matrix3d dray1_dpfr;
        dray1_dpfr.setIdentity();
        dray1_dpfr *= rhor;
        dray1_dpfr.row(0) -= rho3 * pfr(0) * pfr.transpose();
        dray1_dpfr.row(1) -= rho3 * pfr(1) * pfr.transpose();
        dray1_dpfr.row(2) -= rho3 * pfr(2) * pfr.transpose();

        r = ray1 - pfr * rhor;
        Hf = -dray1_dpfr * dpfr_dpf1 * dpf1_dpfw * dpfw_dpf0 * dpf0_drho0;
        Hx0 = -dray1_dpfr * dpfr_dpf1 * dpf1_dpfw * dpfw_dx0;
        Hx1 = -dray1_dpfr * dpfr_dpf1 * dpf1_dx1;
    }
}

// Ric = (I - dtheta_ic) * Ric_hat;
// tci = tci_hat + dtci;
void HybridSlam::TwoFrameJacobian(const Eigen::Vector3d &ray0, const double &rho0, const Eigen::Vector3d &ray1,
                                  const Eigen::Matrix3d &Rwi0, const Eigen::Vector3d &tiw0, const Eigen::Matrix3d &Rwi1,
                                  const Eigen::Vector3d &tiw1, Eigen::MatrixXd &Hx0, Eigen::MatrixXd &Hx1, Eigen::MatrixXd &Hf,
                                  Eigen::MatrixXd &Hic, Eigen::MatrixXd &Htd, Eigen::VectorXd &r, bool is_stereo) {
    // r = ray1 - pf1 / pf1.norm()
    // r = Hx0 * dx0 + Hx1 * dx1 + Hf * drho0 + Hic * dic + nr;
    // pf0 = ray0 / rho0;
    // pfw = Rwi0^t * Ric^t * pf0 + (tiw0 + Rwi0^t * tci)
    // pf1 = Ric * Rwi1 * pfw - Ric * Rwi1 * (tiw1 + Rwi1^t * tci)
    // pf1 = Ric * Rwi1 * pfw - Ric * Rwi1 * tiw1 - Ric * tci
    Eigen::Vector3d pf0 = ray0 / rho0;
    Eigen::Vector3d pfw = Rwi0.transpose() * Ric_.transpose() * pf0 + tiw0 + Rwi0.transpose() * tci_;
    Eigen::Vector3d pf1 = Ric_ * Rwi1 * pfw - Ric_ * Rwi1 * tiw1 - Ric_ * tci_;
    Eigen::Vector3d dpf0_drho0 = -pf0 / rho0;
    Eigen::Matrix3d dpfw_dpf0 = Rwi0.transpose() * Ric_.transpose();
    Eigen::Matrix<double, 3, 6> dpfw_dx0;
    dpfw_dx0.leftCols(3) = -Rwi0.transpose() * MathUtil::VecToSkew(Ric_.transpose() * pf0 + tci_);
    dpfw_dx0.rightCols(3).setIdentity();
    Eigen::Matrix3d dpf1_dpfw = Ric_ * Rwi1;
    Eigen::Matrix<double, 3, 6> dpf1_dx1;
    dpf1_dx1.leftCols(3) = Ric_ * MathUtil::VecToSkew(Rwi1 * (pfw - tiw1));
    dpf1_dx1.rightCols(3) = -Ric_ * Rwi1;
    Eigen::Matrix<double, 3, 6> dpf1_dic;
    dpf1_dic.leftCols(3) = MathUtil::VecToSkew(pf1);
    dpf1_dic.rightCols(3) = -Ric_;
    Eigen::Matrix<double, 3, 6> dpfw_dic;
    dpfw_dic.leftCols(3) = -Rwi0.transpose()  * Ric_.transpose() * MathUtil::VecToSkew(pf0);
    dpfw_dic.rightCols(3) = Rwi0.transpose();
    dpf1_dic += dpf1_dpfw * dpfw_dic;

    if (!is_stereo) {
        // r = (ray0 - pf1 / pf1.norm())
        double rho1 = 1. / pf1.norm();
        double rho3 = rho1 * rho1 * rho1;
        Eigen::Matrix3d dray1_dpf1;
        dray1_dpf1.setIdentity();
        dray1_dpf1 *= rho1;
        dray1_dpf1.row(0) -= rho3 * pf1(0) * pf1.transpose();
        dray1_dpf1.row(1) -= rho3 * pf1(1) * pf1.transpose();
        dray1_dpf1.row(2) -= rho3 * pf1(2) * pf1.transpose();

        r = ray1 - pf1 * rho1;
        Hf = -dray1_dpf1 * dpf1_dpfw * dpfw_dpf0 * dpf0_drho0;
        Hx0 = -dray1_dpf1 * dpf1_dpfw * dpfw_dx0;
        Hx1 = -dray1_dpf1 * dpf1_dx1;
        Hic = -dray1_dpf1 * dpf1_dic;
    } else {
        // pfr = Rclcr * pf1 + tclcr
        // r = ray1 - pfr / pfr.norm();
        Eigen::Vector3d pfr = Rclcr_ * pf1 + tclcr_;
        double rhor = 1. / pfr.norm();
        double rho3 = rhor * rhor * rhor;
        Eigen::Matrix3d dpfr_dpf1 = Rclcr_;
        Eigen::Matrix3d dray1_dpfr;
        dray1_dpfr.setIdentity();
        dray1_dpfr *= rhor;
        dray1_dpfr.row(0) -= rho3 * pfr(0) * pfr.transpose();
        dray1_dpfr.row(1) -= rho3 * pfr(1) * pfr.transpose();
        dray1_dpfr.row(2) -= rho3 * pfr(2) * pfr.transpose();

        r = ray1 - pfr * rhor;
        Hf = -dray1_dpfr * dpfr_dpf1 * dpf1_dpfw * dpfw_dpf0 * dpf0_drho0;
        Hx0 = -dray1_dpfr * dpfr_dpf1 * dpf1_dpfw * dpfw_dx0;
        Hx1 = -dray1_dpfr * dpfr_dpf1 * dpf1_dx1;
        Hic = -dray1_dpfr * dpfr_dpf1 * dpf1_dic;
    }
}

void HybridSlam::TwoFrameJacobian(int frame_id0, int frame_id1, size_t pt_id, const Eigen::Vector3d &ray0, const double &rho0,
                                  Eigen::MatrixXd &Hx0, Eigen::MatrixXd &Hx1, Eigen::MatrixXd &Hf, Eigen::MatrixXd &Hic,
                                  Eigen::MatrixXd &Htd, Eigen::MatrixXd &Htr, Eigen::VectorXd &r, bool is_stereo) {
    SensorState &frame0 = frame_states_[frame_id0];
    Eigen::Matrix3d Rwi0 = frame0.Rwi;
    Eigen::Vector3d tiw0 = frame0.tiw;
    Eigen::Vector3d gyro0 = frame0.gyro;
    Eigen::Vector3d Vw0 = frame0.Vw;
    const Eigen::Matrix3d &Rwi0_null = frame0.Rwi_null;
    const Eigen::Vector3d &tiw0_null = frame0.tiw_null;
    const Eigen::Vector3d &Vw0_null = frame0.Vw_null;
    Eigen::Matrix3d Rwi1, Rwi1_null;
    Eigen::Vector3d tiw1, ray1, tiw1_null;
    Eigen::Vector3d gyro1, Vw1, Vw1_null;
    double rc;
    if (frame_id1 < 0) {
        Rwi1 = t_imu_.Rwi;
        tiw1 = t_imu_.tiw;
        gyro1 = t_imu_.gyro;
        Vw1 = t_imu_.Vw;
        Rwi1_null = t_imu_.Rwi_null;
        tiw1_null = t_imu_.tiw_null;
        Vw1_null = t_imu_.Vw_null;
        if (!is_stereo) {
            ray1 = t_imu_.obs0[pt_id];
            rc = (t_imu_.pt0s[pt_id](1) - cy0_) / img_height0_;
        } 
        else {
            ray1 = t_imu_.obs1[pt_id];
            rc = (t_imu_.pt1s[pt_id](1) - cy1_) / img_height1_;
        }
    } else {
        SensorState &frame1 = frame_states_[frame_id1];
        Rwi1 = frame1.Rwi;
        tiw1 = frame1.tiw;
        gyro1 = frame1.gyro;
        Vw1 = frame1.Vw;
        Rwi1_null = frame1.Rwi_null;
        tiw1_null = frame1.tiw_null;
        Vw1_null = frame1.Vw_null;
        if (!is_stereo) {
            ray1 = frame1.obs0[pt_id];
            rc = (frame1.pt0s[pt_id](1) - cy0_) / img_height0_;
        } else {
            ray1 = frame1.obs1[pt_id];
            rc = (frame1.pt1s[pt_id](1) - cy1_) / img_height1_;
        }
    }

    if (use_rolling_shutter_calib_) {
        Rwi1 = MathUtil::Vec2RotationMatrix(-gyro1 * rc * t_imu_.tr) * Rwi1;
        tiw1 += Vw1 * rc * t_imu_.tr;
    }
    // r = ray1 - pf1 / pf1.norm()
    // r = Hx0 * dx0 + Hx1 * dx1 + Hf * drho0 + Hic * dic + nr;
    // pf0 = ray0 / rho0;
    // pfw = Rwi0^t * Ric^t * pf0 + (tiw0 + Rwi0^t * tci)
    // pf1 = Ric * Rwi1 * pfw - Ric * Rwi1 * (tiw1 + Rwi1^t * tci)
    // pf1 = Ric * Rwi1 * pfw - Ric * Rwi1 * tiw1 - Ric * tci
    Eigen::Vector3d pf0 = ray0 / rho0;
    Eigen::Vector3d pfw = Rwi0.transpose() * Ric_.transpose() * pf0 + tiw0 + Rwi0.transpose() * tci_;
    Eigen::Vector3d pf1_t = Ric_ * Rwi1 * pfw - Ric_ * Rwi1 * tiw1 - Ric_ * tci_;
    Eigen::Vector3d pf1 = pf1_t;
    if (use_FEJ_) {
        Rwi0 = Rwi0_null;
        tiw0 = tiw0_null;
        Vw0 = Vw0_null;
        Rwi1 = Rwi1_null;
        tiw1 = tiw1_null;
        Vw1 = Vw1_null;
        pfw = Rwi0.transpose() * Ric_.transpose() * pf0 + tiw0 + Rwi0.transpose() * tci_;
        pf1 = Ric_ * Rwi1 * pfw - Ric_ * Rwi1 * tiw1 - Ric_ * tci_;    
    }
    Eigen::Vector3d dpf0_drho0 = -pf0 / rho0;
    Eigen::Matrix3d dpfw_dpf0 = Rwi0.transpose() * Ric_.transpose();
    Eigen::Matrix<double, 3, 6> dpfw_dx0;
    dpfw_dx0.leftCols(3) = -Rwi0.transpose() * MathUtil::VecToSkew(Ric_.transpose() * pf0 + tci_);
    dpfw_dx0.rightCols(3).setIdentity();
    Eigen::Matrix3d dpf1_dpfw = Ric_ * Rwi1;
    Eigen::Matrix<double, 3, 6> dpf1_dx1;
    dpf1_dx1.leftCols(3) = Ric_ * MathUtil::VecToSkew(Rwi1 * (pfw - tiw1));
    dpf1_dx1.rightCols(3) = -Ric_ * Rwi1;
    Eigen::Matrix<double, 3, 6> dpf1_dic, dpfw_dic;
    if (use_extrinsic_calib_) {
        dpf1_dic.leftCols(3) = MathUtil::VecToSkew(pf1);
        dpf1_dic.rightCols(3) = -Ric_;
        dpfw_dic.leftCols(3) = -Rwi0.transpose()  * Ric_.transpose() * MathUtil::VecToSkew(pf0);
        dpfw_dic.rightCols(3) = Rwi0.transpose();
        dpf1_dic += dpf1_dpfw * dpfw_dic;
    }

    Eigen::Matrix<double, 3, 1> dpf1_dt, dpfw_dt;
    if (use_td_calib_) {
        dpf1_dt = Ric_ * MathUtil::VecToSkew(Rwi1 * (pfw - tiw1)) * gyro1 - Ric_ * Rwi1 * Vw1;
        dpfw_dt = -Rwi0.transpose() * MathUtil::VecToSkew(Ric_.transpose() * pf0 + tci_) * gyro0 + Vw0;
        dpf1_dt += dpf1_dpfw * dpfw_dt;
    }

    Eigen::Matrix<double, 3, 1> dpf1_dtr;
    if (use_rolling_shutter_calib_) {
        dpf1_dtr = rc * (Ric_ * MathUtil::VecToSkew(Rwi1 * (pfw - tiw1)) * gyro1 - Ric_ * Rwi1 * Vw1);
    }

    if (!is_stereo) {
        // r = (ray0 - pf1 / pf1.norm())
        double rho1 = 1. / pf1.norm();
        double rho3 = rho1 * rho1 * rho1;
        Eigen::Matrix3d dray1_dpf1;
        dray1_dpf1.setIdentity();
        dray1_dpf1 *= rho1;
        dray1_dpf1.row(0) -= rho3 * pf1(0) * pf1.transpose();
        dray1_dpf1.row(1) -= rho3 * pf1(1) * pf1.transpose();
        dray1_dpf1.row(2) -= rho3 * pf1(2) * pf1.transpose();

        r = ray1 - pf1_t / pf1_t.norm();
        Hf = -dray1_dpf1 * dpf1_dpfw * dpfw_dpf0 * dpf0_drho0;
        Hx0 = -dray1_dpf1 * dpf1_dpfw * dpfw_dx0;
        Hx1 = -dray1_dpf1 * dpf1_dx1;
        if (use_extrinsic_calib_) {
            Hic = -dray1_dpf1 * dpf1_dic;
        }
        if (use_td_calib_) {
            Htd = -dray1_dpf1 * dpf1_dt;
        }
        if (use_rolling_shutter_calib_) {
            Htr = -dray1_dpf1 * dpf1_dtr;
        }
    } else {
        // pfr = Rclcr * pf1 + tclcr
        // r = ray1 - pfr / pfr.norm();
        Eigen::Vector3d pfr = Rclcr_ * pf1 + tclcr_;
        double rhor = 1. / pfr.norm();
        double rho3 = rhor * rhor * rhor;
        Eigen::Matrix3d dpfr_dpf1 = Rclcr_;
        Eigen::Matrix3d dray1_dpfr;
        dray1_dpfr.setIdentity();
        dray1_dpfr *= rhor;
        dray1_dpfr.row(0) -= rho3 * pfr(0) * pfr.transpose();
        dray1_dpfr.row(1) -= rho3 * pfr(1) * pfr.transpose();
        dray1_dpfr.row(2) -= rho3 * pfr(2) * pfr.transpose();

        Eigen::Vector3d pfr_t = Rclcr_ * pf1_t + tclcr_;
        r = ray1 - pfr_t / pfr_t.norm();
        Hf = -dray1_dpfr * dpfr_dpf1 * dpf1_dpfw * dpfw_dpf0 * dpf0_drho0;
        Hx0 = -dray1_dpfr * dpfr_dpf1 * dpf1_dpfw * dpfw_dx0;
        Hx1 = -dray1_dpfr * dpfr_dpf1 * dpf1_dx1;
        if (use_extrinsic_calib_) {
            Hic = -dray1_dpfr * dpfr_dpf1 * dpf1_dic;
        }
        if (use_td_calib_) {
            Htd = -dray1_dpfr * dpfr_dpf1 * dpf1_dt;
        }
        if (use_rolling_shutter_calib_) {
            Htr = -dray1_dpfr * dpfr_dpf1 * dpf1_dtr;
        }
    }    
}

void HybridSlam::OCTwoFrameJacobian(const Eigen::Vector3d &ray0, const double &rho0, const Eigen::Vector3d &ray1,
                                    const Eigen::Matrix3d &Rwi0, const Eigen::Vector3d &tiw0, const Eigen::Matrix3d &Rwi1,
                                    const Eigen::Vector3d &tiw1, Eigen::MatrixXd &Hx0, Eigen::MatrixXd &Hx1, Eigen::MatrixXd &Hf,
                                    Eigen::VectorXd &r, bool is_stereo) {

}



// feature state augmentation: covariance augmentation, feature state addition
void HybridSlam::FeatureStateAugment(const std::vector<int> &seen_ids, size_t pt_id, int host_id, const double &rho) {
    SensorState &frame = frame_states_[host_id];
    // covariance augmentation
    // pf0 = ray0 / rho
    // r = tangent * (ray - pf / pf.norm()) + nr
    // r = Jx * dx + Jf * df + nr;
    // df = Jf_inv * (r - Jx * dx - nr)
    // Pff = E((df - E(df)) * (df - E(df))^t), E(df) = Jf_inv * r;
    // Pff = E((-Jx * dx - nr) ^ 2)
    // Pff = Jf_inv * (Jx * Pxx * Jx^t + Rr) * Jf_inv^t
    // Pxf = E((dx) * (Jf_inv * (-Jx * dx - nr))^t)
    // Pxf = -Pxx * Jx^t * Jf_inv^t

    Eigen::Vector3d ray0 = frame.obs0[pt_id];
    Eigen::Matrix<double, 2, 3> tangent = TangentPlaneCalc(ray0);
    Eigen::MatrixXd Jx(4 * seen_ids.size(), P_.cols());
    Eigen::VectorXd Jf(4 * seen_ids.size());
    Jx.setZero();
    Jf.setZero();
    int Jidx = 0;
    for (size_t i = 0; i < seen_ids.size(); i++) {
        int idx = seen_ids[i];
        if (idx == host_id) continue;
        SensorState &frame1 = frame_states_[idx];
        Eigen::MatrixXd Hx0, Hx1, Hf, Hic, Htd, Htr;
        Eigen::VectorXd r;
        if (use_extrinsic_calib_ || use_td_calib_ || use_rolling_shutter_calib_) {
            // TwoFrameJacobian(ray0, rho, frame1.obs0[pt_id], frame.Rwi, frame.tiw, frame1.Rwi, frame1.tiw,
            //         Hx0, Hx1, Hf, Hic, Htd, r, false);
            TwoFrameJacobian(host_id, idx, pt_id, ray0, rho, Hx0, Hx1, Hf, Hic, Htd, Htr, r, false);
        } else {
            TwoFrameJacobian(ray0, rho, frame1.obs0[pt_id], frame.Rwi, frame.tiw, frame1.Rwi, frame1.tiw,
                            Hx0, Hx1, Hf, r, false);
        }


        Eigen::Matrix<double, 2, 6> dr_dx1 = tangent * Hx1;
        Eigen::Matrix<double, 2, 6> dr_dx0 = tangent * Hx0;
        Eigen::Matrix<double, 2, 1> dr_drho = tangent * Hf;
        Jx.block(2 * Jidx, frame.cov_id, 2, 6) = dr_dx0;
        Jx.block(2 * Jidx, frame1.cov_id, 2, 6) = dr_dx1;
        if (use_extrinsic_calib_) {
            Jx.block<2, 6>(2 * Jidx, extrinsic_cov_id_) = tangent * Hic;
        }
        if (use_td_calib_) {
            Jx.block<2, 1>(2 * Jidx, td_cov_id_) = tangent * Htd;
        }
        if (use_rolling_shutter_calib_) {
            Jx.block<2, 1>(2 * Jidx, rs_cov_id_) = tangent * Htr;
        }
        Jf.segment<2>(2 * Jidx) = dr_drho;
        Jidx++;
        if (frame1.obs1.find(pt_id) != frame1.obs1.end()) {
            StereoJacobian(ray0, rho, frame1.obs1[pt_id], Hf, r);
            Jf.segment<2>(2 * Jidx) = tangent * Hf;
            Jidx++;
        }
    }
    Jx.conservativeResize(Jidx * 2, P_.cols());
    Jf.conservativeResize(Jidx * 2);

    // Pff = E((-Jx * dx - nr) ^ 2)
    // Pff = Jf_inv * (Jx * Pxx * Jx^t + Rr) * Jf_inv^t
    // Pxf = E((dx) * (Jf_inv * (-Jx * dx - nr))^t)
    // Pxf = -Pxx * Jx^t * Jf_inv^t
    // r = Jx * dx + Jf * drho + nr
    // Jf^t * r = Jf^t * Jx * dx + Jf^t * Jf * drho + Jf^t * nr; 
    // Jf_inv = Jf^t / (Jf^t * Jf)
    // Jf_inv * r = Jf_inv * Jx * dx + drho + Jf_inv * nr;
    // drho = Jf_inv (r - Jx * dx - nr);

    Eigen::MatrixXd Jf_inv = Jf.transpose() / (Jf.transpose() * Jf);
    Eigen::MatrixXd Pff = Jf_inv * (Jx * P_ * Jx.transpose() +
                          features_noise_ * Eigen::MatrixXd::Identity(Jx.rows(), Jx.rows())) * Jf_inv.transpose();
    Eigen::MatrixXd Pxf = -P_ * Jx.transpose() * Jf_inv.transpose();

    int cols = P_.cols();
    P_.conservativeResize(cols + 1, cols + 1);
    P_.block(0, cols, cols, 1) = Pxf;
    P_.block(cols, 0, 1, cols) = Pxf.transpose();
    P_.block(cols, cols, 1, 1) = Pff;

    // state adding
    FeatureType feat;
    feat.frame_id = frame.frame_id;
    feat.host_id = host_id;
    feat.depth_inv = rho;
    feat.cov_id = cols;
    feat.tangentplane = tangent;
    feat.ray = ray0;
    feature_states_[pt_id] = feat;
}

// imu state clone
void HybridSlam::FrameStateAugment() {
    if (!IsKeyFrame()) {
        return;
    }
    int cols = P_.cols();
    std::cout << "cov cols and state size: " << cols << "," << frame_states_.size() << std::endl;
    
    // clone state and state augmentation
    SensorState new_state0;
    new_state0.Rwi = t_imu_.Rwi;
    new_state0.frame_id = t_imu_.frame_id;
    new_state0.tiw = t_imu_.tiw;
    new_state0.Vw = t_imu_.Vw;
    new_state0.gyro = t_imu_.gyro;
    new_state0.cov_id = cols;
    if (use_FEJ_) {
        new_state0.tiw_null = t_imu_.tiw_null;
        new_state0.Vw_null = t_imu_.Vw_null;
        new_state0.Rwi_null = t_imu_.Rwi_null;
    }
    for (auto &it : t_imu_.obs0) {
        new_state0.obs_depth_inv[it.first] = -1.0;
        new_state0.obs0[it.first] = it.second;
        if (use_rolling_shutter_calib_) {
            if (t_imu_.pt0s.find(it.first) != t_imu_.pt0s.end()) {
                new_state0.pt0s[it.first] = t_imu_.pt0s[it.first];
            }
        }
        if (t_imu_.obs1.find(it.first) != t_imu_.obs1.end()) {
            new_state0.obs1[it.first] = t_imu_.obs1[it.first];
            if (use_rolling_shutter_calib_) {
                if (t_imu_.pt1s.find(it.first) != t_imu_.pt1s.end()) {
                    new_state0.pt1s[it.first] = t_imu_.pt1s[it.first];
                }
            }
        }
    }

    // covariance augmentation
    // dnew = J * dx = [I6X6, 0] * dx
    // E(dnew * dnew^t) = J * E(dx * dx^t) * J^t = J * P * J^t
    // E(dnew * dx) = J * E(dx * dx^t) = J * P
    Eigen::MatrixXd J;
    J.resize(6, cols);
    J.setZero();
    J.block<6, 6>(0, 0).setIdentity();

    Eigen::MatrixXd P21 = J * P_;
    Eigen::MatrixXd P22 = J * P_ * J.transpose();
    P_.conservativeResize(cols + 6, cols + 6);
    P_.block(cols, 0, 6, cols) = P21;
    P_.block(0, cols, cols, 6) = P21.transpose();
    P_.block(cols, cols, 6, 6) = P22;
    P_ = 0.5 * (P_ + P_.transpose());
    frame_states_.push_back(new_state0);
    std::cout << "adding frame id and frame id, cov id: " << frame_states_.size() - 1 << "," << new_state0.frame_id << ","
              << new_state0.cov_id << std::endl;

    if (frame_states_.size() <= slidingwindow_num_) {
        return;
    }

    SensorState &new_state = frame_states_.back();
    int new_id = frame_states_.size() - 1;
    SensorState &anchor_frame = frame_states_[1];
    int anchor_id = 1;
    SensorState old_state = frame_states_.front();
    // feature host propagation
    // pf_new = Ric * (Rwin * ((Riw * Rci) * pf_old / rho + (tiw + Riw * tci)) + twin) + tic
    // pf_new = Ric * Rwin * Riw * Rci * pf_old / rho + Ric * Rwin * tiw + Ric * Rwin * Riw * tci + Ric * twin + tic
    // pf_new = Ric * Rwin * Rwi^t * Rci * pf_old / rho + Ric * Rwin * tiw + Ric * Rwin * Rwi^t * tci + Ric * (-Rwin * tiwn) + tic
    for (auto &feat : feature_states_) {
        if (feat.second.frame_id == old_state.frame_id) {
            if (new_state.obs0.find(feat.first) != new_state.obs0.end()) {
                // pf0 = ray0 / rho
                // pfw = Rwi0^t * Ric^t * pf0 + (tiw0 + Rwi0^t * tci)
                // pf1 = Ric * Rwi1 * pfw - Ric * Rwi1 * (tiw1  + Rwi1^t * tci)
                // pf1 = Ric * Rwi1 * pf2 - Ric * Rwi1 * tiw1 - Ric * tci
                Eigen::Vector3d pf1, dpf1_drho0;
                Eigen::Matrix3d dray1_dpf1;
                Eigen::Matrix<double, 3, 6> dpf1_dx1, dpf1_dx0;
                FeatureJacobian(feat.second.ray, 0, feat.second.depth_inv, pf1, dpf1_dx1, dpf1_dx0,
                                dray1_dpf1, dpf1_drho0, new_id, false);

                double rho1 = 1.0 / pf1.norm(); // rho1 = 1 / depth
                Eigen::Vector3d ray = pf1 * rho1;
                const double &rho0 = feat.second.depth_inv;
                cols = P_.cols();
                // jacobian calculate
                Eigen::MatrixXd Jf(1, cols);
                Jf.setZero();
                Eigen::MatrixXd drho1_dptf1 = -0.5 * rho1 * rho1 * rho1 * 2.0 * pf1.transpose();
                
                Jf.block<1, 6>(0, new_state.cov_id) = drho1_dptf1 * dpf1_dx1;
                Jf.block<1, 6>(0, old_state.cov_id) = drho1_dptf1 * dpf1_dx0;
                Jf.block<1, 1>(0, feat.second.cov_id) = drho1_dptf1 * dpf1_drho0;
                // feature covariance update
                Eigen::MatrixXd Pfx = Jf * P_;
                Eigen::MatrixXd Pff = Jf * P_ * Jf.transpose();
                P_.conservativeResize(cols + 1, cols + 1);
                P_.block(cols, 0, 1, cols) = Pfx;
                P_.block(0, cols, cols, 1) = Pfx.transpose();
                P_.block(cols, cols, 1, 1) = Pff;

                // prune old feature covariance
                CovarianceReduceUpdate(feat.second.cov_id, 1);
                
                Eigen::MatrixXd tangent = TangentPlaneCalc(ray);
                feat.second.cov_id = cols - 1;
                feat.second.frame_id = new_state.frame_id;
                feat.second.host_id = new_id;
                feat.second.depth_inv = rho1;
                feat.second.ray = ray;
                feat.second.tangentplane = tangent;
            }

        }
    }
    
    std::cout << "new cov cols and state size: " << P_.cols() << "," << frame_states_.size() << std::endl;
    // remove old frame state covariance
    int rm_id = old_state.cov_id;
    std::cout << "rm cov id: " << rm_id << std::endl;
    CovarianceReduceUpdate(rm_id, 6);

    // remove old frame
    frame_states_.erase(frame_states_.begin());
    for (auto &feat : feature_states_) {
        if (feat.second.host_id > 0) {
            feat.second.host_id -= 1;
        }
    }
    std::cout << "erasing frame, oldest frame frame id, cov id: " << frame_states_.front().frame_id << ","
              << frame_states_.front().cov_id << std::endl;
    // exit(10);
}

bool HybridSlam::IsKeyFrame() {
    if (frame_states_.size() < slidingwindow_num_ || feature_states_.size() < least_landmark_num_thres_) {
        return true;
    }

    const SensorState &keyframe = frame_states_.back();
    double t_len = (keyframe.tiw - t_imu_.tiw).norm();
    Eigen::Matrix3d dR = keyframe.Rwi * t_imu_.Rwi.transpose();
    double theta = acos((dR.trace() - 1) * 0.5);
    if (fabs(theta) > keyframe_rot_thres_ || t_len > keyframe_translation_thres_) {
        std::cout << "keyframe---------------------" << std::endl;
        return true;
    }
    return false;
}

bool HybridSlam::TryTriangulation(const std::vector<int> &seen_ids, size_t pt_id, int host_id, double &rho) {

    int idx0 = seen_ids[0];
    for (size_t i = 0; i < seen_ids.size(); i++) {
        double t_len = (frame_states_[idx0].tiw - frame_states_[seen_ids[i]].tiw).norm();
        if (t_len > 0.05 || frame_states_[seen_ids[i]].obs1.find(pt_id) != frame_states_[seen_ids[i]].obs1.end()) {
            bool flag = Triangulation(seen_ids, pt_id, host_id, rho);
            return flag;         
        }
    }
    return false;
}
bool HybridSlam::Triangulation(const std::vector<int> &seen_ids, size_t pt_id, int host_id, double &rho_out) {
    // pfk = uk / rho
    // ptw = Rwi(k)^t * Ric^t * pfi + (tiw(k) + Rwi(k)^t * tci)
    // pfj = Ric * Rwi(j) * ptw - Ric * Rwi(j) * (tiw(j) + Rwi(j)^t * tci)
    // pfj = Ric * Rwi(j) * ptw - Ric * Rwi(j) * tiw(j) - Ric * tci
    // uj x pfj = 0 =>
    double H = 0., b = 0.;
    SensorState &host = frame_states_[host_id];
    Eigen::Matrix3d Rcw_h = host.Rwi.transpose() * Ric_.transpose();
    Eigen::Vector3d tcw_h = host.tiw + host.Rwi.transpose() * tci_;
    Eigen::Vector3d ray0 = host.obs0[pt_id];
    std::vector<Eigen::Matrix3d> vRs;
    std::vector<Eigen::Vector3d> vts, vrays;
    for (size_t i = 0; i < seen_ids.size(); i++) {
        int idx = seen_ids[i];
        if (idx != host_id) {
            SensorState &guest = frame_states_[idx];
            Eigen::Matrix3d Rwc = Ric_ * guest.Rwi;
            Eigen::Vector3d twc = -Rwc * guest.tiw - Ric_ * tci_;
            // pfj = Rwc * (Rcwh * uk / rho + tcwh) + twc
            // pfj = Rwc * Rcwh * uk / rho + Rwc * tcwh + twc
            // [ujx] * Rwc * Rcwh * uk / rho = -[ujx] * (Rwc * tcwh + twc)
            // let N = [ujx] * Rwc * Rcwh * uk
            // then N^t * N * 1.0 / rho = -N * [ujx] * (Rwc * tcwh + twc)
            Eigen::Vector3d N = MathUtil::VecToSkew(guest.obs0[pt_id]) * Rwc * Rcw_h * ray0;
            Eigen::Vector3d y = -MathUtil::VecToSkew(guest.obs0[pt_id]) * (Rwc * tcw_h + twc);
            H += N.transpose() * N;
            b += N.transpose() * y;
            vRs.push_back(Rwc * Rcw_h);
            vts.push_back(Rwc * tcw_h + twc);
            vrays.push_back(guest.obs0[pt_id]);
            if (guest.obs1.find(pt_id) != guest.obs1.end()) {
                Rwc = Rclcr_ * Rwc;
                twc = Rclcr_ * twc + tclcr_;
                N = MathUtil::VecToSkew(guest.obs1[pt_id]) * Rwc * Rcw_h * ray0;
                y = -MathUtil::VecToSkew(guest.obs1[pt_id]) * (Rwc * tcw_h + twc);
                H += N.transpose() * N;
                b += N.transpose() * y;
                vRs.push_back(Rwc * Rcw_h);
                vts.push_back(Rwc * tcw_h + twc);
                vrays.push_back(guest.obs1[pt_id]);
            }
        }
    }

    if (host.obs1.find(pt_id) != host.obs1.end()) {
        // pfr = Rclcr * uk / rho + tclcr
        Eigen::Vector3d N = MathUtil::VecToSkew(host.obs1[pt_id]) * Rclcr_ * ray0;
        Eigen::Vector3d y = -MathUtil::VecToSkew(host.obs1[pt_id]) * tclcr_;
        H += N.transpose() * N;
        b += N.transpose() * y;
        vRs.push_back(Rclcr_);
        vts.push_back(tclcr_);
        vrays.push_back(host.obs1[pt_id]);
    }
    double depth = b / H;
    double rho = 1.0 / depth;
    // inverse depth optimization
    double huber_thres = cam0_->GetAngularResolution() * triangulation_huber_thres_;
    double cost = 0;
    std::vector<Eigen::Matrix<double, 2, 3>> vtangentbases;
    for(size_t i = 0; i < vRs.size(); i++) {
        Eigen::Vector3d pf = vRs[i] * ray0 / rho + vts[i];
        Eigen::Matrix<double, 2, 3> tangentbase = TangentPlaneCalc(vrays[i]);
        Eigen::Vector2d err = tangentbase * (vrays[i] - pf / pf.norm());
        // std::cout << "err: " << err.norm() / cam0_->GetAngularResolution() << std::endl;
        double res = err.norm(), w = 1.0;
        if (res > huber_thres) {
            w = 2.0 * huber_thres / res;
        }
        cost += err.squaredNorm();
        vtangentbases.push_back(tangentbase);
    }
    double lambda = 1.0e-05, A, r;

    bool is_stoped = false;
    for (size_t i = 0; i < triangulation_max_iteration_; i++) {
        for (size_t j = 0; j < 3; j++) {
            A = 0.;
            r = 0.;
            for (size_t n = 0; n < vRs.size(); n++) {
                Eigen::Vector3d pf = vRs[n] * ray0 / rho + vts[n];
                Eigen::Vector2d err = vtangentbases[n] * (vrays[n] - pf / pf.norm());
                
                double rho1 = 1.0 / pf.norm();
                double rho3 = rho1 * rho1 * rho1;
                Eigen::Matrix3d dray1_dpf;
                // d(x / sqrt(x^2 + y^2 + z^2)) = dx / sqrt(x^2 + y^2 + z^2) + x * d(1 / sqrt(x^2 + y^2 + z^2))
                // d(1 / sqrt(x^2 + y^2 + z^2)) = -0.5 * d(x^2 + y^2 + z^2) / norm^3 = -0.5 * rho^3 * d(x^2 + y^2 + z^2)
                dray1_dpf.setIdentity();
                dray1_dpf *= rho1;
                dray1_dpf.row(0) += -pf(0) * rho3 * pf.transpose();
                dray1_dpf.row(1) += -pf(1) * rho3 * pf.transpose();
                dray1_dpf.row(2) += -pf(2) * rho3 * pf.transpose();
                
                // f(x) = (r - J * dx) * (r - J * dx)
                // f(x) = r * r - 2 * r * J * dx + dx^t * J ^ t * J * dx
                // d(fx) / dx = 0 => -J^t * r + J^t * J * dx = 0
                // J^t * J * dx = J^t * r; A * dx = r;
                Eigen::Vector3d dpf_drho = -vRs[n] * ray0 / (rho * rho);
                Eigen::Vector2d J = vtangentbases[n] * dray1_dpf * dpf_drho;
                double res = err.norm();
                double w = 1.0;
                if (res > huber_thres) {
                    w = 2.0 * huber_thres / res;
                }
                A += w * w * J.transpose() * J;
                r += w * w * J.transpose() * err;
            }
            // Ax = r
            double drho = r / (A + lambda);
            rho += drho;
            
            double new_cost = 0.;
            for(size_t n = 0; n < vRs.size(); n++) {
                Eigen::Vector3d pf = vRs[n] * ray0 / rho + vts[n];
                Eigen::Vector2d err = vtangentbases[n] * (vrays[n] - pf / pf.norm());

                double res = err.norm(), w = 1.0;
                if (res > huber_thres) {
                    w = 2.0 * huber_thres / res;
                }
                new_cost += err.squaredNorm();
            }
            if (fabs(drho) < 0.01) { is_stoped = true; }
            if (new_cost < cost) {
                cost = new_cost;
                lambda = lambda <= 1.0e-8? lambda : lambda * 0.1;
                break;
            } else {
                rho -= drho;
                lambda = lambda >= 1.0e2? 1.0e2 : lambda * 10;
            }
            if (is_stoped) break;
        }
        if (is_stoped) {
            break;
        }
    }

    depth = 1. / rho;
    double each_cost = sqrt(cost / vRs.size()) / cam0_->GetAngularResolution();
    if (rho < max_depth_inv_ && rho > min_depth_inv_ && each_cost < 2.) {
        rho_out = rho;
        return true;
    } else {
        return false;
    }
}

void HybridSlam::UpdateTest(Eigen::MatrixXd &H, Eigen::VectorXd &r) {
    Eigen::MatrixXd J = H.transpose() * H;
    Eigen::VectorXd b = -H.transpose() * r;
    Eigen::VectorXd dx = J.ldlt().solve(b);

    Eigen::Vector4d dQ = MathUtil::SmallAngle2Quat(dx.segment<3>(0));
    t_imu_.Qwi = MathUtil::JPLQuatMultiply(dQ, t_imu_.Qwi);
    t_imu_.Rwi = MathUtil::JPLQuat2Matrix(t_imu_.Qwi);
    t_imu_.tiw += dx.segment<3>(3);
    t_imu_.Vw += dx.segment<3>(6);
    t_imu_.bg += dx.segment<3>(9);
    t_imu_.ba += dx.segment<3>(12);

    // Rwi = (I - [dthetax]) * Rwi
    for (size_t i = 0; i < frame_states_.size(); i++) {
        SensorState &frame = frame_states_[i];
        Eigen::Matrix3d dR = MathUtil::Vec2RotationMatrix(-dx.segment<3>(frame.cov_id));
        frame.Rwi = dR * frame.Rwi;
        frame.tiw += dx.segment<3>(frame.cov_id + 3);
    }

    for (auto &feat : feature_states_) {
        feat.second.depth_inv += dx(feat.second.cov_id);
    }
}

void HybridSlam::JacobiansTest(const Eigen::MatrixXd &H, const Eigen::VectorXd &r) {
    // r = r0 + H * dx + nr
    Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(H.cols(), H.cols()) * 0.005;
    Eigen::VectorXd dx = (H.transpose() * H + eye).ldlt().solve(-H.transpose() * r);
    for (auto &frame : frame_states_) {
        Eigen::VectorXd dxi = dx.segment<6>(frame.cov_id);
        std::cout << frame.frame_id << "th dt: " << dxi.bottomRows(3).transpose() << std::endl;
        frame.tiw += dxi.bottomRows(3);
        frame.Rwi = MathUtil::Vec2RotationMatrix(-dx.topRows(3)) * frame.Rwi;
    }
    for (auto &feat : feature_states_) {
        feat.second.depth_inv = dx(feat.second.cov_id);
    }
}

// Tracking feature in state observation update
void HybridSlam::NewObsFeatureUpdate() {
    std::vector<size_t> obs_ids;
    for (const auto &feat : t_imu_.obs0) {
        if (feature_states_.find(feat.first) != feature_states_.end()) {
            obs_ids.push_back(feat.first);
        }
    }

    std::unordered_map<size_t, double> reproj_err;
    if (obs_ids.size() > 0) {
        std::cout << "new observation update in imu: " << std::endl;
        Eigen::MatrixXd J(obs_ids.size() * 4, P_.cols());
        Eigen::VectorXd r(obs_ids.size() * 4);
        J.setZero();
        r.setZero();
        int Jidx = 0;
        int cols = P_.cols();
        int good_pt = 0;
        std::vector<size_t> outlier_ids;
        for (size_t i = 0; i < obs_ids.size(); i++) {
            const size_t &idx = obs_ids[i];
            Eigen::MatrixXd Jx;
            Eigen::VectorXd rr;
            NewObsFeatureJacobian(idx, Jx, rr);
            double proj_err = sqrt(rr.squaredNorm() * 2 / rr.rows()) / cam0_->GetAngularResolution();
            if (proj_err < reproj_err_thres_) {
                // std::cout << "proj err: " << proj_err << std::endl;
                reproj_err[idx] = proj_err;
                J.block(Jidx, 0, Jx.rows(), cols) = Jx;
                r.segment(Jidx, rr.rows()) = rr;
                Jidx += rr.rows();
                good_pt++;
            } else if (proj_err > reproj_err_thres_) {
                outlier_ids.push_back(idx);
            }
        }
        J.conservativeResize(Jidx, cols);
        r.conservativeResize(Jidx);
        std::cout << "imu obs before r: " << r.norm() << "," << r.rows() << "," << good_pt << std::endl;
        std::cout << "before imu pw: " << t_imu_.tiw.transpose() << std::endl;
        MeasurementUpdate(J, r);
        // UpdateTest(J, r);
        std::cout << "imu obs updated feature num: " << obs_ids.size() << std::endl;
        std::cout << "updated imu pw: " << t_imu_.tiw.transpose() << std::endl;

        Jidx = 0;
        J.conservativeResize(obs_ids.size() * 4, cols);
        r.conservativeResize(obs_ids.size() * 4);
        for (size_t i = 0; i < obs_ids.size(); i++) {
            const size_t &idx = obs_ids[i];
            Eigen::MatrixXd Jx;
            Eigen::VectorXd rr;
            NewObsFeatureJacobian(idx, Jx, rr);
            double proj_err = sqrt(rr.squaredNorm() * 2 / rr.rows()) / cam0_->GetAngularResolution();
            if (proj_err < reproj_err_thres_) {
                reproj_err[idx] = proj_err;
                J.block(Jidx, 0, Jx.rows(), cols) = Jx;
                r.segment(Jidx, rr.rows()) = rr;
                Jidx += rr.rows();
                good_pt++;
            } 
        }
        std::cout << "updated r: " << r.norm() << "," << r.rows() << std::endl;
        for (size_t i = 0; i < outlier_ids.size(); i++) {
            RemoveOutlier(outlier_ids[i]);
        }
        // exit(0);
    }
}

// Calculate jacobian and residual for current imu feature observation
void HybridSlam::NewObsFeatureJacobian(const size_t &pt_id, Eigen::MatrixXd &J, Eigen::VectorXd &r) {
    FeatureType &feat = feature_states_[pt_id];
    const Eigen::Vector3d &ray0 = feat.ray;
    const Eigen::MatrixXd &tangent = feat.tangentplane;
    const double &rho0 = feat.depth_inv;

    SensorState &host = frame_states_[feat.host_id];
    int host_id = feat.host_id;
    if (host.frame_id != feat.frame_id) {
        std::cout <<"host id conflict with frame id in line: " << __LINE__ << std::endl;
        exit(10);
    }
    const Eigen::Matrix3d &Rwi0 = host.Rwi;
    const Eigen::Vector3d &tiw0 = host.tiw;

    const Eigen::Matrix3d &Rwi1 = t_imu_.Rwi;
    const Eigen::Vector3d &tiw1 = t_imu_.tiw;
    Eigen::Vector3d ray1 = t_imu_.obs0[pt_id];

    // rr = r0 + Jx * dx + Ji * dxi + Jf * drho0
    Eigen::MatrixXd Jx, Ji, Jf, Jic, Jtd, Jtr;
    Eigen::VectorXd rr;
    if (use_extrinsic_calib_ || use_td_calib_ || use_rolling_shutter_calib_) {
        // TwoFrameJacobian(ray0, rho0, ray1, Rwi0, tiw0, Rwi1, tiw1, Jx, Ji, Jf, Jic, Jtd, rr, false);
        TwoFrameJacobian(host_id, -1, pt_id, ray0, rho0, Jx, Ji, Jf, Jic, Jtd, Jtr, rr, false);
    } else {
        TwoFrameJacobian(ray0, rho0, ray1, Rwi0, tiw0, Rwi1, tiw1, Jx, Ji, Jf, rr, false);
    }
    
    // Jx.setZero();
    // Ji.setZero();
    if (t_imu_.obs1.find(pt_id) != t_imu_.obs1.end()) {
        // fill in jacobians and residuals
        J.conservativeResize(4, P_.cols());
        r.conservativeResize(4);
        J.setZero();
        r.setZero();

        J.block<2, 6>(0, host.cov_id) = tangent * Jx;
        J.block<2, 6>(0, 0) = tangent * Ji;
        J.block<2, 1>(0, feat.cov_id) = tangent * Jf;
        if (use_extrinsic_calib_) {
            J.block<2, 6>(0, extrinsic_cov_id_) = tangent * Jic;
        }
        if (use_td_calib_) {
            J.block<2, 1>(0, td_cov_id_) = tangent * Jtd;
        }
        if (use_rolling_shutter_calib_) {
            J.block<2, 1>(0, rs_cov_id_) = tangent * Jtr;
        }
        r.segment<2>(0) = tangent * rr;
        // stereo jacobians and residuals
        ray1 = t_imu_.obs1[pt_id];
        if (use_extrinsic_calib_ || use_td_calib_ || use_rolling_shutter_calib_) {
            // TwoFrameJacobian(ray0, rho0, ray1, Rwi0, tiw0, Rwi1, tiw1, Jx, Ji, Jf, Jic, Jtd, rr, true);
            TwoFrameJacobian(host_id, -1, pt_id, ray0, rho0, Jx, Ji, Jf, Jic, Jtd, Jtr, rr, true);
        } else {
            TwoFrameJacobian(ray0, rho0, ray1, Rwi0, tiw0, Rwi1, tiw1, Jx, Ji, Jf, rr, true);
        }
        // Jx.setZero();
        // Ji.setZero();
        J.block<2, 6>(2, host.cov_id) = tangent * Jx;
        J.block<2, 6>(2, 0) = tangent * Ji;
        J.block<2, 1>(2, feat.cov_id) = tangent * Jf;
        if (use_extrinsic_calib_) {
            J.block<2, 6>(2, extrinsic_cov_id_) = tangent * Jic;
        }
        if (use_td_calib_) {
            J.block<2, 1>(2, td_cov_id_) = tangent * Jtd;
        }
        if (use_rolling_shutter_calib_) {
            J.block<2, 1>(2, rs_cov_id_) = tangent * Jtr;
        }
        r.segment<2>(2) = tangent * rr;
    } else {
        J.conservativeResize(2, P_.cols());
        r.conservativeResize(2);
        J.setZero();
        J.block<2, 6>(0, host.cov_id) = tangent * Jx;
        J.block<2, 6>(0, 0) = tangent * Ji;
        J.block<2, 1>(0, feat.cov_id) = tangent * Jf;
        if (use_extrinsic_calib_) {
            J.block<2, 6>(0, extrinsic_cov_id_) = tangent * Jic;
        }
        if (use_td_calib_) {
            J.block<2, 1>(0, td_cov_id_) = tangent * Jtd;
        }
        if (use_rolling_shutter_calib_) {
            J.block<2, 1>(0, rs_cov_id_) = tangent * Jtr;
        }
        r.segment<2>(0) = tangent * rr;
    }
}
void HybridSlam::LostFeaturesUpdate() {
    if (frame_states_.size() < features_obs_num_thres_) {
        return;
    }

    SensorState &keyframe = frame_states_.back();
    int last_id = frame_states_.size() - 1;
    std::vector<std::vector<int>> seens_ids;
    std::vector<size_t> pt_ids;
    int total_seens = 0;
    for(auto it : keyframe.obs0) {
        if (feature_states_.find(it.first) == feature_states_.end()) {
            if (t_imu_.obs0.find(it.first) == t_imu_.obs0.end()) {
                int seen_count = 0;
                std::vector<int> seen_ids;
                for (size_t i = 0; i < frame_states_.size() - 1; i++) {
                    int idx = last_id - i - 1;
                    SensorState &frame = frame_states_[idx];
                    if (frame.obs0.find(it.first) != frame.obs0.end()) {
                        seen_ids.push_back(idx);
                        seen_count++;
                        if (frame.obs1.find(it.first) != frame.obs1.end()) {
                            seen_count++;
                        }
                    }
                }
                if (seen_count > 2) {
                    total_seens += seen_count;
                    pt_ids.push_back(it.first);
                    seens_ids.push_back(seen_ids);
                }
            }
        }
    }

    std::unordered_map<size_t, double> reproj_err;
    Eigen::MatrixXd H(total_seens * 2, P_.cols());
    Eigen::VectorXd r(total_seens * 2);
    H.setZero();
    r.setZero();
    int good_idx = 0;
    for (size_t i = 0; i < pt_ids.size(); i++) {
        Eigen::MatrixXd J;
        Eigen::VectorXd rr;
        LostFeatureJacobian(seens_ids[i], pt_ids[i], J, rr);

        if (rr.rows() > 0) {
            int rows = rr.rows();
            if (true) {
                H.block(good_idx, 0, rows, P_.cols()) = J;
                r.segment(good_idx, rows) = rr;
                reproj_err[pt_ids[i]] = rr.norm() / cam0_->GetAngularResolution();
                good_idx += rows;
            }
        }
    }
    if (good_idx > 0) {
        H.conservativeResize(good_idx, P_.cols());
        r.conservativeResize(good_idx);
        std::cout << "lost update r norm, rows: " << r.norm() << "," << r.rows() << std::endl;

        MeasurementUpdate(H, r);
    }

    std::cout << "lost feature updated: " << good_idx << "," << pt_ids.size() << std::endl;
    if (good_idx > 0) {
        // cv::waitKey(0);
    }
}

void HybridSlam::LostFeatureJacobian(const std::vector<int> &seenids, size_t pt_id, Eigen::MatrixXd &J, Eigen::VectorXd &r) {
    if (seenids.size() < 2) {
        return;
    }
    int host_id = seenids.front();
    SensorState &host = frame_states_[host_id];
    const int &cov_id0 = host.cov_id;
    double rho0 = 1.;
    if (!TryTriangulation(seenids, pt_id, host_id, rho0)) {
        return;
    }
    Eigen::Vector3d ray0 = host.obs0[pt_id];
    const Eigen::Matrix3d &Rwi0 = host.Rwi;
    const Eigen::Vector3d &tiw0 = host.tiw;
    Eigen::Matrix<double, 2, 3> tangent = TangentPlaneCalc(ray0);

    // b = b0 + Hx * x + Hf * drho0 + nb;
    Eigen::MatrixXd Hx(seenids.size() * 4, P_.cols()), Hf(seenids.size() * 4, 1);
    Eigen::VectorXd b(seenids.size() * 4);
    Hx.setZero(); b.setZero(); Hf.setZero();
    int good_idx = 0;
    for (size_t i = 0; i < seenids.size(); i++) {
        int idx = seenids[i];
        if (idx == host_id) continue;
        SensorState &guest = frame_states_[idx];
        const int &cov_id1 = guest.cov_id;
        Eigen::Vector3d ray1 = guest.obs0[pt_id];
        const Eigen::Matrix3d &Rwi1 = guest.Rwi;
        const Eigen::Vector3d &tiw1 = guest.tiw;
        Eigen::MatrixXd Hx0, Hx1, Hfx, Hic, Htd, Htr;
        Eigen::VectorXd rr;
        if (use_extrinsic_calib_ || use_td_calib_ || use_rolling_shutter_calib_) {
            // TwoFrameJacobian(ray0, rho0, ray1, Rwi0, tiw0, Rwi1, tiw1, Hx0, Hx1, Hfx, Hic, Htd, rr, false);
            TwoFrameJacobian(host_id, idx, pt_id, ray0, rho0, Hx0, Hx1, Hfx, Hic, Htd, Htr, rr, false);
        } else {
            TwoFrameJacobian(ray0, rho0, ray1, Rwi0, tiw0, Rwi1, tiw1, Hx0, Hx1, Hfx, rr, false);
        }
        double proj_err = rr.norm() / cam0_->GetAngularResolution();
        if (proj_err > reproj_err_thres_) continue;
        // tangent * b = tangent * b0 + tangent * Hx * dx + tangent * Hf * drho0 + tangent * nb;
        Hx.block<2, 6>(good_idx * 2, cov_id0) = tangent * Hx0;
        Hx.block<2, 6>(good_idx * 2, cov_id1) = tangent * Hx1;
        if (use_extrinsic_calib_) {
            Hx.block<2, 6>(good_idx * 2, extrinsic_cov_id_) = tangent * Hic;
        }
        if (use_td_calib_) {
            Hx.block<2, 1>(good_idx * 2, td_cov_id_) = tangent * Htd;
        }
        if (use_rolling_shutter_calib_) {
            Hx.block<2, 1>(good_idx * 2, rs_cov_id_) = tangent * Htr;
        }
        Hf.block<2, 1>(good_idx * 2, 0) = tangent * Hfx;
        b.segment<2>(good_idx * 2) = tangent * rr;
        good_idx++;

        if (guest.obs1.find(pt_id) != guest.obs1.end()) {            
            ray1 = guest.obs1[pt_id];
            if (use_extrinsic_calib_ || use_td_calib_ || use_rolling_shutter_calib_) {
                // TwoFrameJacobian(ray0, rho0, ray1, Rwi0, tiw0, Rwi1, tiw1, Hx0, Hx1, Hfx, Hic, Htd, rr, true);
                TwoFrameJacobian(host_id, idx, pt_id, ray0, rho0, Hx0, Hx1, Hfx, Hic, Htd, Htr, rr, true);
            } else {
                TwoFrameJacobian(ray0, rho0, ray1, Rwi0, tiw0, Rwi1, tiw1, Hx0, Hx1, Hfx, rr, true);
            }
            double proj_err = rr.norm() / cam0_->GetAngularResolution();
            if (proj_err > reproj_err_thres_) continue;
            Hx.block<2, 6>(good_idx * 2, cov_id0) = tangent * Hx0;
            Hx.block<2, 6>(good_idx * 2, cov_id1) = tangent * Hx1;
            if (use_extrinsic_calib_) {
                Hx.block<2, 6>(good_idx * 2, extrinsic_cov_id_) = tangent * Hic;
            }
            if (use_td_calib_) {
                Hx.block<2, 1>(good_idx * 2, td_cov_id_) = tangent * Htd;
            }
            if (use_rolling_shutter_calib_) {
                Hx.block<2, 1>(good_idx * 2, rs_cov_id_) = tangent * Htr;
            }
            Hf.block<2, 1>(good_idx * 2, 0) = tangent * Hfx;
            b.segment<2>(good_idx * 2) = tangent * rr;
            good_idx++;
        }
    }

    if (good_idx > 1) {
        Hx.conservativeResize(good_idx * 2, P_.cols());
        Hf.conservativeResize(good_idx * 2, 1);
        b.conservativeResize(good_idx * 2);
        // b = b0 + Hx * dx + Hf * drho0 + nb;
        // Hf = U * sigma * vt, sigma = diag(x, 0, 0, ...)

        Eigen::JacobiSVD<Eigen::MatrixXd> Hf_svd(Hf, Eigen::ComputeFullU | Eigen::ComputeThinV);
        Eigen::MatrixXd A = Hf_svd.matrixU().rightCols(b.rows() - 1);

        J = A.transpose() * Hx;
        r = A.transpose() * b;
    }
}

std::vector<Eigen::Matrix4d> HybridSlam::SlidingWindowPose() {
    std::vector<Eigen::Matrix4d> out_pose;
    for (const auto &frame : frame_states_) {
        Eigen::Matrix4d pose;
        pose.setIdentity();
        pose.topLeftCorner(3, 3) = frame.Rwi.transpose() * Ric_.transpose();
        pose.topRightCorner(3, 1) = frame.tiw;
        out_pose.push_back(pose);
    }
    return out_pose;
}

void HybridSlam::DrawDebugInformation(cv::Mat &vimg) {
    float r = 3;
    for (auto &it : t_imu_.pt0s) {
        if (feature_states_.find(it.first) != feature_states_.end()) {
            std::string ss = std::to_string(it.first);
            Eigen::Vector2d &pt = it.second;
            cv::rectangle(vimg, cv::Point2f(pt(0)-r, pt(1)-r), cv::Point2f(pt(0)+r, pt(1)+r), cv::Scalar(0, 255, 0), 1);
            putText(vimg, ss, cv::Point2f(pt(0), pt(1)), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1);
        }
    }
}

void HybridSlam::DrawBias() {
    cv::Mat img = cv::Mat::zeros(800, 800, CV_8UC3);
    acc_bias_.push_back(tci_);
    gyro_bias_.push_back(t_imu_.bg);
    td_vec_.push_back(t_imu_.td);
    tr_vec_.push_back(t_imu_.tr);
    if (acc_bias_.size() > 700) {
        acc_bias_.erase(acc_bias_.begin());
        gyro_bias_.erase(gyro_bias_.begin());
        td_vec_.erase(td_vec_.begin());
        tr_vec_.erase(tr_vec_.begin());
    }
    
    double acc_scale = 200;
    double gyro_scale = 1000;
    if (acc_bias_.size() < 2) return;
    for (size_t i = 1; i < acc_bias_.size(); i++) {
        // acc bias show
        float x = float(400 - acc_bias_[i - 1](0) * acc_scale);
        float y = float(400 - acc_bias_[i - 1](1) * acc_scale);
        float z = float(400 - acc_bias_[i - 1](2) * acc_scale);
        float x1 = float(400 - acc_bias_[i](0) * acc_scale);
        float y1 = float(400 - acc_bias_[i](1) * acc_scale);
        float z1 = float(400 - acc_bias_[i](2) * acc_scale);
        cv::Point2f pt((i - 1) * 1, x);
        cv::Point2f pt1(i * 1, x1);
        cv::Scalar color(0, 0, 255);
        // cv::circle(img, pt, 3, color);
        // cv::circle(img, pt1, 3, color);
        cv::line(img, pt, pt1, color, 1);
        pt.y = y; pt1.y = y1;
        // cv::circle(img, pt, 3, color);
        // cv::circle(img, pt1, 3, color);
        cv::line(img, pt, pt1, color, 1);
        pt.y = z; pt1.y = z1;
        // cv::circle(img, pt, 3, color);
        // cv::circle(img, pt1, 3, color);
        cv::line(img, pt, pt1, color, 1);
        if (i == acc_bias_.size() - 1) {
            std::string ss;
            // acc bias show
            ss = "ax: " + std::to_string(acc_bias_[i](0));
            pt1.y = x1;
            cv::putText(img, ss, pt1, 1, 0.9, color, 1);
            ss = "ay: " + std::to_string(acc_bias_[i](1));
            pt1.y = y1;
            cv::putText(img, ss, pt1, 1, 0.9, color, 1);
            ss = "az: " + std::to_string(acc_bias_[i](2));
            pt1.y = z1;
            cv::putText(img, ss, pt1, 1, 0.9, color, 1);
        }

        // gyro bias show
        x = float(600 - gyro_bias_[i - 1](0) * gyro_scale);
        y = float(600 - gyro_bias_[i - 1](1) * gyro_scale);
        z = float(600 - gyro_bias_[i - 1](2) * gyro_scale);
        x1 = float(600 - gyro_bias_[i](0) * gyro_scale);
        y1 = float(600 - gyro_bias_[i](1) * gyro_scale);
        z1 = float(600 - gyro_bias_[i](2) * gyro_scale);
        pt.y = x, pt1.y = x1;
        color = cv::Scalar(255, 0, 255);
        // cv::circle(img, pt, 3, color);
        // cv::circle(img, pt1, 3, color);
        cv::line(img, pt, pt1, color, 1);
        pt.y = y; pt1.y = y1;
        // cv::circle(img, pt, 3, color);
        // cv::circle(img, pt1, 3, color);
        cv::line(img, pt, pt1, color, 1);
        pt.y = z; pt1.y = z1;
        // cv::circle(img, pt, 3, color);
        // cv::circle(img, pt1, 3, color);
        cv::line(img, pt, pt1, color, 1);

        if (i == acc_bias_.size() - 1) {
            char buffer[200];
            // sprintf(buffer, "gyro bias: %4f, %4f, %4f\n", gyro_bias_[i](0), gyro_bias_[i](1), gyro_bias_[i](2));
            std::string ss;
            // cv::putText(img, ss, cv::Point2f(10, 10), 1, 0.9, color, 1);
            // gyro bias show
            ss = "wx: " + std::to_string(gyro_bias_[i](0));
            pt1.y = x1;
            cv::putText(img, ss, pt1, 1, 0.9, color, 1);
            ss = "wy: " + std::to_string(gyro_bias_[i](1));
            pt1.y = y1;
            cv::putText(img, ss, pt1, 1, 0.9, color, 1);
            ss = "wz: " + std::to_string(gyro_bias_[i](2));
            pt1.y = z1;
            cv::putText(img, ss, pt1, 1, 0.9, color, 1);
        }

        // td show
        x = float(200 - td_vec_[i - 1] * 500);
        x1 = float(200 - td_vec_[i] * 500);
        pt.y = x, pt1.y = x1;
        color = cv::Scalar(0, 255, 255);
        cv::line(img, pt, pt1, color, 1);
        if (i == td_vec_.size() - 1) {
            char buffer[200];
            std::string ss;
            ss = "td: " + std::to_string(td_vec_[i]);
            cv::putText(img, ss, pt1, 1, 0.9, color, 1);
        }

        // tr show
        x = float(100 - tr_vec_[i - 1] * 500);
        x1 = float(100 - tr_vec_[i] * 500);
        pt.y = x, pt1.y = x1;
        color = cv::Scalar(255, 255, 0);
        cv::line(img, pt, pt1, color, 1);
        if (i == tr_vec_.size() - 1) {
            char buffer[200];
            std::string ss;
            ss = "tr: " + std::to_string(tr_vec_[i]);
            cv::putText(img, ss, pt1, 1, 0.9, color, 1);
        }
    }
    cv::line(img, cv::Point2f(0, 400), cv::Point2f(800, 400), cv::Scalar(255, 255, 255), 1);
    cv::line(img, cv::Point2f(0, 600), cv::Point2f(800, 600), cv::Scalar(255, 255, 255), 1);
    cv::line(img, cv::Point2f(0, 200), cv::Point2f(800, 200), cv::Scalar(255, 255, 255), 1);
    cv::line(img, cv::Point2f(0, 100), cv::Point2f(800, 100), cv::Scalar(255, 255, 255), 1);


    char buffer[200];
    sprintf(buffer, "%4f th pos: %4f, %4f, %4f", t_imu_.timestamp, t_imu_.tiw(0), t_imu_.tiw(1), t_imu_.tiw(2));
    std::string ss(buffer);
    cv::putText(img, ss, cv::Point2f(10, 30), 1, 0.9, cv::Scalar(255, 255, 255), 1);
    char buffer1[200];
    sprintf(buffer1, "orient: %4f, %4f, %4f, %4f", t_imu_.Qwi(0), t_imu_.Qwi(1), t_imu_.Qwi(2), t_imu_.Qwi(3));
    ss = buffer1;
    cv::putText(img, ss, cv::Point2f(10, 50), 1, 0.9, cv::Scalar(255, 255, 255), 1);
    static std::ofstream out("/home/insta360/0data/bias.txt");
    out << acc_bias_.back()(0) << "," << acc_bias_.back()(1) << "," << acc_bias_.back()(2) << "," <<
           gyro_bias_.back()(0) << "," << gyro_bias_.back()(1) << "," << gyro_bias_.back()(2) << std::endl;
    cv::imshow("bias", img);
}


bool HybridSlam::GatingTest(Eigen::MatrixXd &H, Eigen::VectorXd &r, int cam_num) {
    // chi_square(x) = x^t * x / cov(x);
    Eigen::MatrixXd P = H * P_ * H.transpose() + features_noise_ * Eigen::MatrixXd::Identity(H.rows(), H.rows());
    double gamma = r.transpose() * P.ldlt().solve(r);

    if (gamma < MAHALA95_TABLE[cam_num]) {
        return true;
    } else {
        return false;
    }
}

std::vector<Eigen::Vector3d> HybridSlam::MapPoints() {
    std::vector<Eigen::Vector3d> out_pts;
    for (auto pt : feature_states_) {
        Eigen::Vector3d ptc = pt.second.ray / pt.second.depth_inv;
        SensorState &f = frame_states_[pt.second.host_id];
        Eigen::Vector3d ptw = f.Rwi.transpose() * Ric_.transpose() * ptc + f.tiw + f.Rwi.transpose() * tci_;
        // Eigen::Vector3d pf1 = Ric_ * t_imu_.Rwi * ptw - Ric_ * t_imu_.Rwi * t_imu_.tiw - Ric_ * tci_;
        // std::cout << pt.first << "th ptc: " << pf1.transpose() << std::endl;


        out_pts.push_back(ptw);
        totalMapPoints_[pt.first] = ptw;
    }
    return out_pts;
}

std::vector<Eigen::Vector3d>& HybridSlam::AllMapPoints() {
    return allMapPoints_;
}



void HybridSlam::GridPoints() {

}




} // namespace inslam