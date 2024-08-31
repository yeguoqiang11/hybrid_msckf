#include "SlidingWindow.hpp"
namespace inslam {
double EssentialError(const Eigen::Vector3d &pt0, const Eigen::Vector3d &pt1, const Eigen::Matrix3d &E) {
    Eigen::Vector3d tmp = E * pt0;
    tmp /= tmp.norm();
    double sin_a = pt1.transpose() * tmp;
    return fabs(sin_a / pt1.norm());
}
// calculate tangent plane from point of unit sphere
inline Eigen::Matrix<double, 2, 3> TangentPlane(const Eigen::Vector3d &point) {
    Eigen::Vector3d tmp(0, 0, 1);
    Eigen::Vector3d pt = point / point.norm();
    double cos_theta = tmp.transpose() * pt;
    if (fabs(cos_theta) > 0.9) {
        tmp = Eigen::Vector3d(1, 0, 0);
    }
    tmp = tmp - tmp.transpose() * pt * pt;
    tmp /= tmp.norm();
    Eigen::Vector3d y_axis = tmp.cross(pt);
    Eigen::Matrix<double, 2, 3> plane;
    plane.row(0) = tmp.transpose();
    plane.row(1) = y_axis.transpose();
    return plane;
}

ImuObs::ImuObs(double an, double gn, double arw, double grw, double g)
: Na(an), Ng(gn), Nba(arw), Nbg(grw), gravity(g) {
    ba.setZero();
    bg.setZero();
}

void ImuObs::SetParam(double an, double gn, double arw, double grw, double g) {
    Na = an;
    Ng = gn;
    Nba = arw;
    Nbg = grw;
    gravity = g;
    ba.setZero();
    bg.setZero();
}

void ImuObs::Initialize() {
    imu = Preintegrated(ba, bg, Na, Ng, Nba, Nbg, gravity);
    t0 = -1;
    t1 = -1;
}

void ImuObs::InitializeNext(const Eigen::Vector3d &t_ba, const Eigen::Vector3d &t_bg) {
    imu = Preintegrated(ba, bg, Na, Ng, Nba, Nbg, gravity);
    t0 = t1;
    frameId0 = frameId1;
    ba = t_ba;
    bg = t_bg;
}


void ImuObs::IntegrateImu(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyro, double time) {
    if (t1 < 0) {
        std::cout << "cannot integrate imu with negative t0!!" << std::endl;
        exit(0);
    }
    double dt = time - t1;
    imu.IntegrateNewImu(acc, gyro, dt);
    t1 = time;
    // frameId1 = frameId0;
}

void ImuObs::MergeOldImu(ImuObs imu0) {
    if (imu0.t0 == t0) {
        return;
    }
    imu0.imu.InsertNewImu(imu);
    imu = imu0.imu;
    imu.RePreintegration();
    if (imu0.t1 != t0) {
        std::cout << "merge imu error at line: " << __LINE__ << std::endl;
        exit(0);
    }
    t0 = imu0.t0;
    if (frameId0 != imu0.frameId1) {
        std::cout << "merge imu error at line: " << __LINE__ << std::endl;
        exit(0);
    }
    frameId0 = imu0.frameId0;
}


void FrameType::Initialize(double time, int frameId) {
    obs0.clear();
    obs1.clear();
    pt0s.clear();
    pt1s.clear();
    timestamp = time;
    frame_id = frameId;
}

SlidingWindow::SlidingWindow(const nlohmann::json &config, std::shared_ptr<Caimura> cam0, std::shared_ptr<Caimura> cam1)
: cam0_(std::move(cam0)), cam1_(std::move(cam1)) {
    // acc_m = acc_t + ba_t + Na + Gi;
    // dba = ba + Nba;
    // gyro_m = gyro_t + bg_t + Ng;
    // dbg = bg + Nbg;
    params_.imu.Na = cam0_->accSigma_;
    params_.imu.Ng = cam0_->gyroSigma_;
    params_.imu.Nba = cam0_->accRandomWalk_;
    params_.imu.Nbg = cam0_->gyroRandomWalk_;

    // extrinsic setting
    params_.imu.Ric0 = cam0_->Rci_;
    params_.imu.tc0i = cam0_->pic_;
    params_.imu.Rc0i = params_.imu.Ric0.transpose();
    params_.imu.tic0 = -params_.imu.Ric0 * params_.imu.tc0i;
    std::cout << "Ric: " << params_.imu.Ric0 << std::endl;    

    // noise setting
    params_.imu.noise_.conservativeResize(12, 12);
    params_.imu.noise_.setIdentity();
    params_.imu.noise_.block<3, 3>(0, 0) *= params_.imu.Ng * params_.imu.Ng;
    params_.imu.noise_.block<3, 3>(3, 3) *= params_.imu.Nbg * params_.imu.Nbg;
    params_.imu.noise_.block<3, 3>(6, 6) *= params_.imu.Na * params_.imu.Na;
    params_.imu.noise_.block<3, 3>(9, 9) *= params_.imu.Nba * params_.imu.Nba;
    params_.imu.gravity_ << 0, 0, -9.81;

    // [x, y] = [(u - cx) / fx, (v - cy) / fy]
    // r = sqrt(x * x + y * y)
    // f(theta) = r => theta; phi = atan(y, x)
    // [x_hat, y_hat, z_hat] = [cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)]
    // lambda * [x_hat, y_hat, z_hat] = world point
    // [x_hat, y_hat, z_hat] = world_point / lambda
    // [u, v] = [u_t + Nc, v_t + Nc]
    // cam0 parameters
    if (cam0_ != nullptr) {
        Camera pcam0;
        pcam0.cam_id = 0;
        pcam0.feature_noise_ = cam0_->GetAngularResolution();
        pcam0.width = cam0_->width();
        pcam0.height = cam0_->height();
        params_.cams.push_back(pcam0);
    }

    // cam1 parameters
    if (cam1_ != nullptr) {
        Camera pcam1;
        pcam1.cam_id = 1;
        pcam1.feature_noise_ = cam0_->GetAngularResolution();
        pcam1.width = cam1_->width();
        pcam1.height = cam1_->height();
        pcam1.Rc0cj = cam1_->Rci_ * cam0_->Ric_;
        pcam1.tc0cj = cam1_->pci_ + cam1_->Rci_ * cam0_->pic_;
        params_.cams.push_back(pcam1);
    }

    // set parameters for imu preintegration
    imu_.SetParam(params_.imu.Na, params_.imu.Ng, params_.imu.Nba, params_.imu.Nbg, 9.8);

    // keyframe threshold
    key_translation_thres_ = 0.1; // 0.2 m
    key_rotation_thres_ = 10 * 3.14 / 180; // 10 degree
    key_landmark_num_ = 60; // landmark num = 60
    keyframe_num_ = 10;

    // triangulation thres
    landmark_move_thres_ = 0.05;
    landmark_obs_num_ = 4;
    landmark_huber_thres_ = 3.0 * cam0_->GetAngularResolution();
    landmark_iter_num_ = 5;
    landmark_cost_thres_ = 3.0 * cam0_->GetAngularResolution();

    // debug
    tracked_num_ = 0;
}

void SlidingWindow::Run(const std::vector<FeatureObservation> &featureMsgs, int frame_id,
                        const Eigen::Matrix<double, 17, 1> &frame_pose,
                        const std::vector<Eigen::Matrix<double,7,1> > &vimu) {
    // TestDemo();
    EssentialTest();

    Preintegration(vimu, frame_id);
    FrameUpdate(featureMsgs, frame_id, frame_pose);
    std::cout << t_frame_.frame_id << "th input pose: " << t_frame_.tiw.transpose() << std::endl;
    InitPoseOptimization();
    WindowUpdate();

    // save pose
    Tiw_last_.leftCols(3) = t_frame_.Riw;
    Tiw_last_.rightCols(1) = t_frame_.tiw;

    // print log
    std::cout << window_frames_.back().frame_id << "th update pose: " << window_frames_.back().tiw.transpose() << std::endl;
    std::cout << "window frame num: " << window_frames_.size() << std::endl;
    std::cout << "----------------------" << std::endl;
    if (tracked_num_ > 0) {
        cv::waitKey(0);
    }
}

void SlidingWindow::Preintegration(const std::vector<Eigen::Matrix<double, 7, 1> > &vimu, int frame_id) {
    for (size_t i = 0; i < vimu.size(); i++) {
        const Eigen::Matrix<double, 7, 1> &data = vimu[i];
        if (i == 0 && imu_.t1 < 0 && imu_.t0 < 0) {
            imu_.t1 = data(0);
            imu_.t0 = data(0);
            continue;
        }
        imu_.IntegrateImu(data.tail(3), data.segment(1, 3), data(0));
    }
    imu_.frameId1 = frame_id;
}

void SlidingWindow::StateIntegration(const std::vector<Eigen::Matrix<double, 7, 1> > &vimu) {
    for (size_t i = 0; i < vimu.size(); i++) {

    }
}

// framePose = [timestamp, Qiw, tiw, Vw, bg, ba]
void SlidingWindow::FrameUpdate(const std::vector<FeatureObservation> &featureMsgs, int frame_id,
                     const Eigen::Matrix<double, 17, 1> &framePose) {
    // frame update
    t_frame_.Initialize(framePose(0), frame_id);
    for (size_t i = 0; i < featureMsgs.size(); i++) {
        const FeatureObservation &feat = featureMsgs[i];
        t_frame_.obs0[feat.id] = feat.ray0;
        t_frame_.pt0s[feat.id] = feat.pt0;
        if (feat.isStereo) {
            t_frame_.obs1[feat.id] = feat.ray1;
            t_frame_.pt1s[feat.id] = feat.pt1;
        }

        // map update
        if (map_.find(feat.id) == map_.end()) {
            map_[feat.id] = LandMark();
            map_[feat.id].pt_id = feat.id;
            map_[feat.id].frame_ids.push_back(frame_id);
            continue;
        }
        map_[feat.id].frame_ids.push_back(frame_id);
    }

    Eigen::Vector4d Q = framePose.segment(1, 4);
    Eigen::Vector4d Q1;
    Q1 << Q;
    Eigen::Quaterniond Qwi(Q(0), Q(1), Q(2), Q(3));
    t_frame_.Riw = Qwi.matrix();
    t_frame_.tiw = framePose.segment(5, 3);
    t_frame_.Vw = framePose.segment(8, 3);
    t_frame_.bg = framePose.segment(11, 3);
    t_frame_.ba = framePose.segment(14, 3);
    Eigen::Matrix<double, 3, 4> Tiw;
    Tiw << t_frame_.Riw, t_frame_.tiw;
    // std::cout << "sliding window Tiw:\n" << Tiw << std::endl;
}

// use current frame to update window frame
void SlidingWindow::WindowUpdate() {
    if (IsKeyFrame()) {
        CreateKeyFrame();
    } else {
        // if (window_frames_.size() >= 2) {
        //     imu_.MergeOldImu(window_frames_.back().imu);
        //     t_frame_.imu = imu_;

        //     window_frames_.erase(window_frames_.end());
        //     window_frames_.push_back(t_frame_);
        // }
    }
}

// determine whether it is key frame
bool SlidingWindow::IsKeyFrame() {
    if (window_frames_.size() < 1) return true;
    const FrameType &last_frame = window_frames_.back();
    double last_dist = (t_frame_.tiw - last_frame.tiw).norm();
    Eigen::Matrix3d dR = (t_frame_.Riw * last_frame.Riw.transpose());
    if (last_dist > key_translation_thres_) return true;
    return false;
}

// create landmark for sliding window
bool SlidingWindow::CreateLandMark(int pt_id, int host_id) {
    int seen_num = 0;
    LandMark &landmark = map_[pt_id];
    Eigen::Vector3d pos0;
    double max_t = 0;
    std::vector<int> frameIds;
    for (size_t i = 0; i < landmark.frame_ids.size(); i++) {
        int &tmp_id = landmark.frame_ids[i];
        int wind_idx = FrameIdRetrieval(tmp_id);
        if (wind_idx >= 0) {
            FrameType &frame = window_frames_[wind_idx];
            if (frame.frame_id != tmp_id) {
                std::cout << "frame retrieval wrong at line: " << __LINE__ << std::endl;
                exit(0);
            }
            if (seen_num == 0) {
                pos0 = frame.tiw;
            }
            seen_num++;
            if (frame.obs1.find(pt_id) != frame.obs1.end()) {
                seen_num++;
            }
            double tmp_t = (frame.tiw - pos0).norm();
            if (tmp_t > max_t) {
                max_t = tmp_t;
            }
            frameIds.push_back(wind_idx);
        }
    }

    if (seen_num < landmark_obs_num_ || max_t < landmark_move_thres_) return false;

    double idepth, avg_err;
    bool landmark_flag = Triangulation(frameIds, pt_id, host_id, idepth, avg_err);
    if (!landmark_flag) {
        std::cout << "trangulation failed with err: " << avg_err / cam0_->GetAngularResolution() << std::endl;
        return false;
    }
    landmark.depth_inv = idepth;
    landmark.host_id = window_frames_[host_id].frame_id;
    landmark.is_good = true;
    window_frames_[host_id].host_ptIds.push_back(pt_id);
    return true;
}

// binary search
int SlidingWindow::FrameIdRetrieval(int frame_id) const {
    if (window_frames_.empty()) return -1;
    int low = 0, high = window_frames_.size() - 1, mid;
    mid = (low + high) / 2;
    while (low <= high) {
        const int &tmp_id = window_frames_[mid].frame_id;
        if (tmp_id == frame_id) return mid;
        if (tmp_id > frame_id) high = mid - 1;
        if (tmp_id < frame_id) low = mid + 1;
        mid = (low + high) / 2;
    }
    return -1;
}

// if mono
// Pw = Riw_h * (Rci * (Uh / rhoh) + tci) + tiw_h = Riw_h * Rci * Uh/rhoh + Riw_h * tci + tiw_h
// Pj = Uj / rhoj = Ric * (Riw_j^t * Pw + twi_j) + tic = Ric * Rwi_j * Pw + Ric * twi_j + tic
// Uj X Pj = 0 => Uj X (Ric * Rwi_j * Pw + Ric * twi_j + tic) = 0;
// [UjX] * (Ric * Rwi_j * Riw_h * Rci * Uh/rhoh + Ric * Rwi_j * (Riw_h * tci + tiw_h) + Ric * twi_J + tic) = 0;
// R = Ric * Rwi_j * Riw_h * Rci, t = Ric * Rwi_j * (Riw_h * tci + tiw_h) + Ric * twi_j + tic
// => Uj X (R * Uh / rhoh + t) = 0, R and t can be got from above
// [UjX] * R * Uh/ rhoh = -[UjX] * t
// A * depth = b, A = [UjX] * R * Uh, b = -[UjX] * t;
// A^t * A * depth = A^t * b
// sum(Ai^2) * depth = sum(Ai * bi)
// if stereo
// Pr = Rclcr * Uh/rhoh + tclcr
// Ur/rhor = Rclcr * Uh/rhoh + tclcr
// Ur X (Rclcr * Uh/rhoh + tclcr) = 0
// [UrX] * Rclcr * Uh * depth = -[UrX] * tclcr
// A * depth = b, A = [UrX] * Rclcr * Uh, b = -[UrX] * tclcr
bool SlidingWindow::Triangulation(const std::vector<int> &frame_ids, int pt_id, int host_id, double &idepth, double &avg_err) {
    FrameType &host = window_frames_[host_id];
    Eigen::Matrix3d &Riw_h = host.Riw;
    Eigen::Vector3d &tiw_h = host.tiw;
    Eigen::Vector3d &ray_h = host.obs0[pt_id];

    bool is_stereo = false;
    double A = 0, b = 0;
    std::vector<Eigen::Matrix3d> Rvs;
    std::vector<Eigen::Vector3d> tvs, rayvs;
    for (size_t i = 0; i < frame_ids.size(); i++) {
        if (frame_ids[i] == host_id) continue;
        FrameType &guest = window_frames_[frame_ids[i]];
        Eigen::Vector3d &ray_g = guest.obs0[pt_id];
        Eigen::Matrix3d &Riw_g = guest.Riw;
        Eigen::Vector3d &tiw_g = guest.tiw;
        Eigen::Matrix3d Rchcg = params_.imu.Ric0 * Riw_g.transpose() * Riw_h * params_.imu.Rc0i;
        Eigen::Vector3d tchcg = params_.imu.Ric0 * Riw_g.transpose() * (Riw_h * params_.imu.tc0i
                                + tiw_h) + params_.imu.Ric0 * (-Riw_g.transpose() * tiw_g) + params_.imu.tic0;
        Eigen::Matrix3d ray_g_hat = MathUtil::VecToSkew(ray_g);
        Eigen::Vector3d H = ray_g_hat * Rchcg * ray_h;
        Eigen::Vector3d r = -ray_g_hat * tchcg;
        A += H.transpose() * H;
        b += H.transpose() * r;
        Rvs.push_back(Rchcg);
        tvs.push_back(tchcg);
        rayvs.push_back(ray_g);

        if (guest.obs1.find(pt_id) != guest.obs1.end()) {
            Eigen::Vector3d &ray_r = guest.obs1[pt_id];
            Eigen::Matrix3d ray_r_hat = MathUtil::VecToSkew(ray_r);
            // Pj = R * Uh * depth + t
            // Pj_r = Rclcr * Pj + tclcr = Rclcr * (R * Uh * depth + t) + tclcr
            // Pj_r = Ur / rhor = Rclcr * R * Uh * depth + Rclcr * t + tclcr
            // [UrX] * Rclcr * R * Uh * depth = -[UrX] * (Rclcr * t + tclcr)
            H = ray_r_hat * params_.cams[1].Rc0cj * Rchcg * ray_h;
            r = -ray_r_hat * (params_.cams[1].Rc0cj * tchcg + params_.cams[1].tc0cj);
            A += H.transpose() * H;
            b += H.transpose() * r;
            Rvs.push_back(params_.cams[1].Rc0cj * Rchcg);
            tvs.push_back(params_.cams[1].Rc0cj * tchcg + params_.cams[1].tc0cj);
            rayvs.push_back(ray_r);
            is_stereo = true;
        }
    }
    if (host.obs1.find(pt_id) != host.obs1.end()) {
        // Pj = Ur / rhor = Rclcr * Uh * depth + t
        // [UrX] * Rclcr * Uh * depth = -[UrX] * t
        Eigen::Vector3d &ray_r = host.obs1[pt_id];
        Eigen::Matrix3d rayr_hat = MathUtil::VecToSkew(ray_r);
        Eigen::Vector3d H = rayr_hat * params_.cams[1].Rc0cj * ray_h;
        Eigen::Vector3d r = -rayr_hat * params_.cams[1].tc0cj;
        A += H.transpose() * H;
        b += H.transpose() * r;
        Rvs.push_back(params_.cams[1].Rc0cj);
        tvs.push_back(params_.cams[1].tc0cj);
        rayvs.push_back(ray_r);
        is_stereo = true;
    }

    // A * depth = b
    // A/idepth = b => (A^t * A) / idepth = (A^t * b) => idepth = (A^t * A) / (A^t * b);
    double depth = b / A;
    double rho = 1.0 / depth;

    // landmark optimization
    // Uj/rhoj = R * Uh/rhoh + t
    // error = tangentplane * (Uj - (R * Uh / rho + t)/||R * Uh/rho + t||)
    std::vector<Eigen::Matrix<double, 2, 3>> tangentplanes;
    double cost = 0;
    for (size_t i = 0; i < Rvs.size(); i++) {
        Eigen::Matrix<double, 2, 3> proj_plane = TangentPlane(rayvs[i]);
        Eigen::Vector3d est_ray = Rvs[i] * ray_h / rho + tvs[i];
        est_ray /= est_ray.norm();
        Eigen::Vector2d e = proj_plane * (rayvs[i] - est_ray);
        double w = 1;
        if (e.norm() > landmark_huber_thres_) {
            w =2 * landmark_huber_thres_ / e.norm();
        }
        cost += w * w * e.squaredNorm();
        tangentplanes.push_back(proj_plane);
    }
    // std::cout << "++++++++++++" << std::endl;

    double lambda = 1.0e-03;
    // LM iteration
    for (int i = 0; i < landmark_iter_num_; i++) {
        // error = tangentplane * (Uj - (R * Uh/rho + t)/||R * Uh/rho + t||) = tangentplane * eu;
        // eu = Uj - Pj/||Pj||, Pj = R * Uh/rho + t
        // deu1/dPj = -d(x/||Pj||) = -dx/dPj /||Pj|| - x * d(1/||Pj||)/dPj
        // =-[1, 0, 0]/||Pj|| - (-0.5) * x * [2*x, 2*y, 2*z]/||Pj||^3
        // =-[1, 0, 0]/||Pj|| + x * [x, y, z]/||Pj||^3
        // deu2/dPj = -[0, 1, 0]/||Pj|| + y * [x, y, z] / ||Pj||^3
        // deu3/dPj = -[0, 0, 1]/||Pj|| + z * [x, y, z] / ||Pj||^3
        // deu/dPj = - I33/||Pj|| + [x, y, z]^t * [x, y, z] / ||Pj||^3
        // dPj/drho = -R * Uh/rho^2
        double J = 0, r = 0, w = 1;
        for (size_t i = 0; i < Rvs.size(); i++) {
            Eigen::Vector3d Pj = Rvs[i] * ray_h / rho + tvs[i];
            double rhoj = 1.0 / Pj.norm();
            Eigen::Matrix3d de_dPj; de_dPj.setIdentity();
            de_dPj *= -rhoj;
            de_dPj += Pj * Pj.transpose() * (rhoj * rhoj * rhoj);
            Eigen::Vector3d dPj_drho = -Rvs[i] * ray_h / (rho * rho);
            Eigen::Vector2d A = tangentplanes[i] * de_dPj * dPj_drho;
            Eigen::Vector2d b = tangentplanes[i] * (rayvs[i] - Pj * rhoj);
            w = 1;
            if (b.norm() > landmark_huber_thres_) {
                w = 2.0 * landmark_huber_thres_ / b.norm();
            }

            // r(x) = [x if x < huber, 2.0 * hu if x >= huber]
            // 
            J += w * w * A.transpose() * A;
            r += w * w * A.transpose() * b;
        }
        
        // std::cout << "J r: " << J << "," << r << std::endl;
        //error = e0 + J * x;
        //error^2 = e0 * e0 + 2 * e0 * J * x + x^t * J^t * J * x
        //d(error^2) / dx = 0 => J^t * e0 + J^t * J * x = 0
        //J^t * J * x = -J^t * e0
        double dx = -r / (J + lambda);
        double rho0 = rho + dx;
        double cost0 = 0;
        // std::cout << "host ray: " << ray_h.transpose() << std::endl;
        for (size_t i = 0; i < Rvs.size(); i++) {
            Eigen::Vector3d est_ray = Rvs[i] * ray_h / rho0 + tvs[i];
            est_ray /= est_ray.norm();
            Eigen::Vector2d e = tangentplanes[i] * (rayvs[i] - est_ray);
            double w = 1;
            if (e.norm() > landmark_huber_thres_) {
                w = 2 * landmark_huber_thres_ / e.norm();
            }
            cost0 += w * w * e.squaredNorm();
        }

        // inverse depth update
        if (cost0 < cost) {
            rho = rho0;
            cost = cost0;
            lambda = lambda < 1.0e-07? lambda : lambda * 0.1;
        } else {
            lambda = lambda > 1.0e3? lambda : lambda * 10;
        }
        if (fabs(dx) < 0.01) {
            break;
        }
    }
    avg_err = sqrt(cost/Rvs.size());
    if (avg_err < landmark_cost_thres_) {
        idepth = rho;
        return true;
    } else {
        return false;
    }
}

void SlidingWindow::CreateKeyFrame() {
    std::cout << "****Inserting keyframe****" << std::endl;
    t_frame_.imu = imu_;
    imu_.InitializeNext(t_frame_.ba, t_frame_.bg);
    if (window_frames_.size() > keyframe_num_) {
        window_frames_.push_back(t_frame_);
        WindowOptimization();

        int rm_landmark_num = 0;
        for (auto it : window_frames_.front().host_ptIds) {
            if (map_[it].is_good) {
                rm_landmark_num++;
                map_[it].is_good = false;
            }
        }
        std::cout << "rm landmark num: " << rm_landmark_num << std::endl;
        window_frames_.erase(window_frames_.begin());
    } else {
        window_frames_.push_back(t_frame_);
        WindowOptimization();
    }

    // Create landmark for new keyframe
    int create_landmark_num = 0;
    int track_landmark_num = 0;
    for (auto it : t_frame_.obs0) {
        if (map_[it.first].is_good) {
            track_landmark_num++;
            continue;
        }
        bool is_success = CreateLandMark(it.first, window_frames_.size() - 1);
        if (is_success) create_landmark_num++;
    }
    if (track_landmark_num == 0 && window_frames_.size() > 10) {
        std::cout << "ERROR: non tracking landmark!!!" << std::endl;
        exit(0);
    }
    tracked_num_ = track_landmark_num;
    std::cout << "create and track landmark num: " << create_landmark_num << "," << track_landmark_num << std::endl;
}

void SlidingWindow::WindowOptimization() {
    if (window_frames_.size() < 5) {
        return;
    }

    Eigen::Matrix2d proj_sqrt_info = Eigen::Matrix2d::Identity();

    Eigen::Matrix<double, 3, 4> Tci, Tc0c1;
    Tci.leftCols(3) = params_.imu.Rc0i;
    Tci.rightCols(1) = params_.imu.tc0i;

    if (params_.cams.size() > 1) {
        Tc0c1.leftCols(3) = params_.cams[1].Rc0cj;
        Tc0c1.rightCols(1) = params_.cams[1].tc0cj;
    }

    // window_frames_.back().tiw = window_frames_[window_frames_.size()-2].tiw;
    // window_frames_.back().Riw = window_frames_[window_frames_.size()-2].Riw;

    ceres::Problem problem;
    ceres::LossFunction *proj_loss = new ceres::HuberLoss(2.0 * cam0_->GetAngularResolution());
    ceres::LossFunction *imu_loss = new ceres::HuberLoss(10);
    double keyposes[window_frames_.size()][6], key_vels[window_frames_.size()][9];
    for (size_t i = 0; i < window_frames_.size(); i++) {
        FrameType &frame = window_frames_[i];
        Eigen::Map<Eigen::Matrix<double, 6, 1>> ref_pose(keyposes[i]);
        Eigen::Vector3d rvec = RotationMatrix2Vector3d(frame.Riw);
        // std::cout << "rvec: " << rvec.transpose() << std::endl;
        ref_pose << rvec, frame.tiw;
        ceres::LocalParameterization *pose_param = new PerturbationPoseParameterization();
        problem.AddParameterBlock(keyposes[i], 6, pose_param);

        Eigen::Map<Eigen::Matrix<double, 9, 1>> ref_vel(key_vels[i]);
        ref_vel << frame.Vw, frame.bg, frame.ba;
    }
    Eigen::Map<Eigen::Matrix<double, 6, 1>> Po(keyposes[window_frames_.size() - 1]);
    std::cout << window_frames_.back().frame_id << "th init rvec: " << Po.topRows(3).transpose() << std::endl;
    std::cout << window_frames_.back().frame_id << "th init tiw: " << Po.bottomRows(3).transpose() << std::endl;

    int cnt = 0;
    int residuals_count = 0;
    std::vector<ceres::ResidualBlockId> mono_res_ids, stereo_res_ids, imu_res_ids;
    std::unordered_map<int, double*> landmarks_map;
    std::vector<double> imu_err;
    std::vector<int> err_ids;

    // log saved
    std::unordered_map<int, std::vector<ceres::ResidualBlockId>> pt_res_ids;
    for (size_t i = 0; i < window_frames_.size(); i++) {
        FrameType &frame = window_frames_[i];
        int obs_count = 0;
        for (auto it : frame.obs0) {
            if (map_[it.first].is_good) {
                LandMark &lm = map_[it.first];
                int idx = FrameIdRetrieval(lm.host_id);
                if (idx != i && idx >= 0) {
                    if (landmarks_map.find(it.first) == landmarks_map.end()) {
                        landmarks_map[it.first] = new double(lm.depth_inv);
                    }
                    FrameType &host = window_frames_[idx];
                    if (host.frame_id != lm.host_id) {
                        std::cout << "frame id retrieval wrong at " << __LINE__ << std::endl;
                        exit(0);
                    }
                    if (frame.obs1.find(it.first) != frame.obs1.end()) {
                        std::vector<Eigen::Vector3d> rays;
                        rays.push_back(it.second);
                        rays.push_back(frame.obs1[it.first]);
                        
                        std::vector<Eigen::Matrix<double, 2, 3>> tangentplanes;
                        tangentplanes.push_back(TangentPlane(rays[0]));
                        tangentplanes.push_back(TangentPlane(rays[1]));

                        ceres::CostFunction *stereo_cost = new WindowStereoProjCost(proj_sqrt_info, Tci, rays,
                            host.obs0[it.first], tangentplanes, Tc0c1);
                        ceres::ResidualBlockId proj_id = problem.AddResidualBlock(stereo_cost, proj_loss, keyposes[i], keyposes[idx], landmarks_map[it.first]);
                        if (i == window_frames_.size() - 1) {
                            stereo_res_ids.push_back(proj_id);
                        }
                    } else {
                        Eigen::Matrix<double, 2, 3> tangentplane = TangentPlane(it.second);
                        ceres::CostFunction *mono_cost = new WindowProjCost(proj_sqrt_info, Tci, it.second, host.obs0[it.first],
                             tangentplane);
                        ceres::ResidualBlockId proj_id = problem.AddResidualBlock(mono_cost, proj_loss, keyposes[i], keyposes[idx], landmarks_map[it.first]);
                        if (i == window_frames_.size() - 1) {
                            mono_res_ids.push_back(proj_id);
                        }

                        if (pt_res_ids.find(it.first) == pt_res_ids.end()) {
                            pt_res_ids[it.first] = std::vector<ceres::ResidualBlockId>();
                        }
                        pt_res_ids[it.first].push_back(proj_id);
                    }
                    residuals_count++;
                    obs_count++;
                }
            }
        }

        if (i > 0) {
            FrameType &framei = window_frames_[i - 1];
            double error = TwoFrameImuError(frame.imu, framei.frame_id, frame.frame_id);
            imu_err.push_back(error);
            err_ids.push_back(i);

            // Eigen::MatrixXd imu_sqt_info = Eigen::LLT<Eigen::MatrixXd>(frame.imu.imu.cov_.inverse()).matrixL().transpose();
            // ceres::CostFunction *imu_cost = new ImuCost(imu_sqt_info, -params_.imu.gravity_, Tci.leftCols(3),
            //     Tci.rightCols(1), frame.imu.imu);
            // ceres::ResidualBlockId imu_id = problem.AddResidualBlock(imu_cost, NULL, keyposes[i-1], key_vels[i-1],
            //     keyposes[i], key_vels[i]);
            // imu_res_ids.push_back(imu_id);
        }
        // std::cout << frame.frame_id << "th obs count: " << obs_count << std::endl;
    }

    if (residuals_count <= 0) {
        return;
    }

    double proj_init_cost;
    ceres::Problem::EvaluateOptions proj_eval_opts;
    // for (size_t i = 0; i < 15 && i < imu_res_ids.size(); i++) {
    //     proj_eval_opts.residual_blocks.clear();
    //     proj_eval_opts.residual_blocks.push_back(imu_res_ids[i]);
    //     proj_eval_opts.apply_loss_function = false;
    //     std::cout << "============" << std::endl;
    //     problem.Evaluate(proj_eval_opts, &proj_init_cost, NULL, NULL, NULL);
    //     std::cout << window_frames_[i + 1].frame_id << "th imu init cost: " << sqrt(proj_init_cost)  << "," <<
    //         imu_err[i] << std::endl;
    //     double diff = imu_err[i] - sqrt(proj_init_cost);
    //     if (fabs(diff) > 0.1) {
    //         FrameType &framej = window_frames_[err_ids[i]];
    //         FrameType &framei = window_frames_[err_ids[i - 1]];
    //         double err = TwoFrameImuError(framej.imu, framei.frame_id, framej.frame_id, true);
    //         exit(0);
    //     }
    // }

    for (auto &it : pt_res_ids) {
        for (auto jt : it.second) {
            proj_eval_opts.residual_blocks.clear();
            proj_eval_opts.residual_blocks.push_back(jt);
            problem.Evaluate(proj_eval_opts, &proj_init_cost, NULL, NULL, NULL);
            std::cout << it.first << "th proj init cost: " << sqrt(proj_init_cost * 2) / cam0_->GetAngularResolution() << std::endl;
        }
        break;
    }

    problem.SetParameterBlockConstant(keyposes[0]);
    if (window_frames_.size() > 5) {
        // problem.SetParameterBlockConstant(keyposes[1]);
        // problem.SetParameterBlockConstant(keyposes[2]);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 5;
    options.linear_solver_type = ceres::LinearSolverType::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.function_tolerance = 2.0e-02;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    imu_err.clear();
    // for (size_t i = 0; i < 15 && i < imu_res_ids.size(); i++) {
    //     proj_eval_opts.residual_blocks.clear();
    //     proj_eval_opts.residual_blocks.push_back(imu_res_ids[i]);
    //     proj_eval_opts.apply_loss_function = false;
    //     std::cout << "=====" << std::endl;
    //     problem.Evaluate(proj_eval_opts, &proj_init_cost, NULL, NULL, NULL);
    //     // std::cout << window_frames_[i + 1].frame_id << "th imu final cost: " << sqrt(proj_init_cost) << std::endl;
    //     imu_err.push_back(sqrt(proj_init_cost));
    // }

    for (auto &it : pt_res_ids) {
        for (auto jt : it.second) {
            proj_eval_opts.residual_blocks.clear();
            proj_eval_opts.residual_blocks.push_back(jt);
            problem.Evaluate(proj_eval_opts, &proj_init_cost, NULL, NULL, NULL);
            std::cout << it.first << "th proj final cost: " << sqrt(proj_init_cost * 2) / cam0_->GetAngularResolution() << std::endl;
        }
        break;
    }

    std::cout << "proj num: " << residuals_count << ", stereo: " << stereo_res_ids.size() << std::endl;    
    std::cout << window_frames_.back().timestamp << "th final rvec: " << Po.topRows(3).transpose() << std::endl;
    std::cout << window_frames_.back().frame_id << "th final tiw: " << Po.bottomRows(3).transpose() << std::endl;
    RecoveryPose(keyposes);
    RecoveryVelBias(key_vels);
    // for (size_t i = 0; i < window_frames_.size(); i++) {
    //     if (i > 0) {
    //         FrameType &framei = window_frames_[i - 1];
    //         FrameType &framej = window_frames_[i];
    //         double err = TwoFrameImuError(framej.imu, framei.frame_id, framej.frame_id);
    //         // std::cout << framej.frame_id << "th final cost: " << imu_err[i - 1] << "," << err << std::endl;
    //     }
    // }
}

void SlidingWindow::RecoveryPose(double poses[][6]) {
    for (size_t i = 0; i < window_frames_.size(); i++) {
        FrameType &frame = window_frames_[i];
        Eigen::Map<Eigen::Vector3d> rvec(poses[i]);
        frame.Riw = RotationVector2Matrix3d(rvec);
        frame.tiw = Eigen::Map<Eigen::Vector3d>(poses[i] + 3);
    }
}

void SlidingWindow::RecoveryVelBias(double vel_bias[][9]) {
    for (size_t i = 0; i < window_frames_.size(); i++) {
        FrameType &frame = window_frames_[i];
        frame.Vw = Eigen::Map<Eigen::Vector3d>(vel_bias[i]);
        frame.bg = Eigen::Map<Eigen::Vector3d>(vel_bias[i] + 3);
        frame.ba = Eigen::Map<Eigen::Vector3d>(vel_bias[i] + 6);
    }
}

double SlidingWindow::TwoFrameReprojectionError(Eigen::Matrix3d Riwg, Eigen::Matrix3d Riwh, Eigen::Vector3d tiwg, Eigen::Vector3d tiwh,
                                         Eigen::Vector3d obsg, Eigen::Vector3d  obsh, double rho, bool is_stereo) {
    Eigen::Vector3d Pch = params_.imu.Rc0i * obsh / rho + params_.imu.tc0i;
    Eigen::Vector3d Pw = Riwh * Pch + tiwh;
    Eigen::Vector3d Pcg = params_.imu.Ric0 * (Riwg.transpose() * Pw - Riwg.transpose() * tiwg) + params_.imu.tic0;
    double e;
    if (!is_stereo) {
        Eigen::Vector3d error = obsg - Pcg / Pcg.norm();
        e = error.norm() / cam0_->GetAngularResolution();
    } else {
        Eigen::Vector3d Pcg1 = params_.cams[1].Rc0cj * Pcg + params_.cams[1].tc0cj;
        Eigen::Vector3d error = obsg - Pcg1 / Pcg1.norm();
        e = error.norm() / cam0_->GetAngularResolution();
        std::cout << "error: " << error.norm() << std::endl;
    }
    return e;
}

void SlidingWindow::LandMarkChangeHostFrame(int pt_id, int rm_frameId) {
    LandMark &lm = map_[pt_id];
}

Eigen::Matrix<double, 10, 1> SlidingWindow::PoseVel() {
    Eigen::Matrix<double, 10, 1> out;
    if (window_frames_.back().frame_id == t_frame_.frame_id) {
        Eigen::Quaterniond Qiw(window_frames_.back().Riw);
        out << Qiw.coeffs(), window_frames_.back().tiw, window_frames_.back().Vw;
    } else {
        Eigen::Quaterniond Qiw(t_frame_.Riw);
        out << Qiw.coeffs(), t_frame_.tiw, t_frame_.Vw;
    }
    return out;
}

double SlidingWindow::TwoFrameImuError(ImuObs &imu, int frame_id0, int frame_id1, bool show) const {
    const Preintegrated &preinteg = imu.imu;
    int idx0 = FrameIdRetrieval(frame_id0);
    int idx1 = FrameIdRetrieval(frame_id1);
    const FrameType &frame0 = window_frames_[idx0];
    const FrameType &frame1 = window_frames_[idx1];
    double dt = frame1.timestamp - frame0.timestamp;

    Eigen::Vector3d dbg = frame0.bg - preinteg.bg_;
    // std::cout << "frame0 ba: " << frame0.ba.transpose() << std::endl;
    // std::cout << "imu ba: " << preinteg.ba_.transpose() << std::endl;
    Eigen::Vector3d dba = frame0.ba - preinteg.ba_;

    Eigen::Matrix3d corrected_Rji = preinteg.Rji_ * RotationVector2Matrix3d(preinteg.dRg_ * dbg);
    Eigen::Matrix3d Rji_hat = frame0.Riw.transpose() * frame1.Riw;
    Eigen::Matrix3d err_R = corrected_Rji.transpose() * Rji_hat;

    Eigen::Vector3d corrected_Pij = preinteg.Pij_ + preinteg.dPa_ * dba + preinteg.dPg_ * dbg;
    Eigen::Vector3d corrected_Vij = preinteg.Vij_ + preinteg.dVa_ * dba + preinteg.dVg_ * dbg;
    Eigen::Vector3d dv = frame0.Riw.transpose() * (frame1.Vw - frame0.Vw - params_.imu.gravity_ * dt);
    Eigen::Vector3d dp = frame0.Riw.transpose() * (frame1.tiw - frame0.tiw - frame0.Vw * dt - 0.5 *
        params_.imu.gravity_ * dt * dt);

    Eigen::Vector3d err_v = dv - corrected_Vij;
    Eigen::Vector3d err_p = dp - corrected_Pij;
    Eigen::Vector3d err_r = RotationMatrix2Vector3d(err_R);
    Eigen::Matrix<double, 15, 1> error;

    Eigen::Vector3d err_bg = frame1.bg - frame0.bg;
    Eigen::Vector3d err_ba = frame1.ba - frame0.ba;
    error << err_r, err_v, err_p, err_bg, err_ba;
    // std::cout << "pj: " << frame1.tiw.transpose() << std::endl;
    // std::cout << "pi: " << frame0.tiw.transpose() << std::endl;
    // std::cout << "dp: " << dp.transpose() << std::endl;
    // std::cout << "cPij: " << corrected_Pij.transpose() << std::endl;
    Eigen::Vector3d rv0 = RotationMatrix2Vector3d(frame0.Riw);
    Eigen::Vector3d rv1 = RotationMatrix2Vector3d(frame1.Riw);
    Eigen::Matrix3d R0 = RotationVector2Matrix3d(rv0);
    Eigen::Matrix3d R1 = RotationVector2Matrix3d(rv1);

    Eigen::AngleAxisd rvx0(frame0.Riw);
    Eigen::Matrix3d eig; eig.setIdentity(); eig(1, 1) *= -1; eig(2, 2) *= -1;
    Eigen::AngleAxisd rvx1(frame1.Riw);

    Eigen::Matrix3d Rj = frame1.Riw.transpose() * R1;
    Eigen::Matrix3d Rjx = frame1.Riw.transpose() * RotationVector2Matrix3d(rvx1.angle() * rvx1.axis());
    Eigen::AngleAxisd rvxj(Rj);
    rv1 = rvx1.angle() * rvx1.axis();
    // std::cout << "error: " << error.segment(3, 3).transpose() << std::endl;

    // std::cout << "Rjy: " << (rvxj.angle() * rvxj.axis()).transpose() << std::endl;
    // std::cout << "Rjx: " << RotationMatrix2Vector3d(Rjx).transpose() << std::endl;
    // std::cout << "Rj: " << RotationMatrix2Vector3d(Rj).transpose() << std::endl;
    // std::cout << "Ri: " << RotationMatrix2Vector3d(frame0.Riw.transpose() * R0).transpose() << std::endl;
    // std::cout << "imu Rji: " << RotationMatrix2Vector3d(corrected_Rji).transpose() << std::endl;
    // std::cout << "cRji: " << RotationMatrix2Vector3d(frame0.Riw.transpose() * frame1.Riw).transpose() << std::endl;
    // std::cout << "Rji: " << RotationMatrix2Vector3d(R0.transpose() * R1).transpose() << std::endl;
    // std::cout << "rvj: " << RotationMatrix2Vector3d(frame1.Riw).transpose() << std::endl;
    if (show) {
        std::cout << "++++++" << std::endl;
        Eigen::Vector3d dv1 = frame1.Vw - frame0.Vw - params_.imu.gravity_ * dt;
        std::cout << "Rbiw: " << frame0.Riw << std::endl;
        std::cout << "dv1: " << dv1.transpose() << std::endl;
        std::cout << "dv: " << dv.transpose() << std::endl;
        std::cout << "cvij: " << corrected_Vij.transpose() << std::endl;
        std::cout << "error: " << error.segment(0, 9).transpose() << std::endl;
    }

    Eigen::MatrixXd sqrt_info = Eigen::LLT<Eigen::MatrixXd>(preinteg.cov_.inverse()).matrixL().transpose();
    error = sqrt_info * error;
    if (fabs(dt - preinteg.dT_) > 0.002) {
        std::cout << "imu id0~id1: " << imu.frameId0 << "," << imu.frameId1 << std::endl;
        std::cout << "input id0-id1: " << frame0.frame_id << "," << frame1.frame_id << std::endl;
        std::cout << "frame id0 id1: " << frame_id0 << "," << frame_id1 << std::endl;
        std::cout << "imu t0~t1: " << imu.t0 << "," << imu.t1 << std::endl;
        std::cout << "dt: " << dt << "," << preinteg.dT_ << std::endl;
        std::cout << "imu preintegration error at line: " << __LINE__ << std::endl;
        exit(0);
    }
    if (imu.frameId0 != frame_id0 || imu.frameId1 != frame_id1) {
        std::cout << "imu preintegration error at line: " << __LINE__ << std::endl;
        exit(0);
    }
    // std::cout << frame_id1 << "Pij: " << imu.imu.Pij_.transpose() << std::endl;
    if (show) {
        std::cout << frame_id1 << "error: " << error.norm() / sqrt(2) << std::endl;
    }
    // std::cout <<"between " << frame_id0 << "," << frame_id1 << " imu error: " << error.norm() << std::endl;

    return error.norm() / sqrt(2);
}

double SlidingWindow::InitPoseOptimization() {
    ceres::Problem problem;
    ceres::LossFunction *proj_loss = new ceres::HuberLoss(cam0_->GetAngularResolution() * 5);
    double pose[6];
    Eigen::Map<Eigen::Matrix<double, 6, 1>> posev(pose);
    if (window_frames_.size() < 10) {
        posev << RotationMatrix2Vector3d(t_frame_.Riw), t_frame_.tiw;
    } else {
        posev << RotationMatrix2Vector3d(Tiw_last_.leftCols(3)), Tiw_last_.rightCols(1);
    }

    Eigen::Matrix<double, 3, 4> Tci;
    Tci.leftCols(3) = params_.imu.Rc0i;
    Tci.rightCols(1) = params_.imu.tc0i;

    std::vector<ceres::ResidualBlockId> res_ids;
    for (auto &it : t_frame_.obs0) {
        if (map_[it.first].is_good) {
            LandMark &lm = map_[it.first];
            int idx = FrameIdRetrieval(lm.host_id);
            FrameType &host = window_frames_[idx];
            Eigen::Matrix<double, 3, 4> Tiwh;
            Tiwh.leftCols(3) = host.Riw;
            Tiwh.rightCols(1) = host.tiw;

            Eigen::Matrix<double, 2, 3> tangentplane = TangentPlane(it.second);
            ceres::CostFunction *proj_cost = new ProjCost(Tci, it.second, Tiwh, host.obs0[it.first],
                tangentplane, lm.depth_inv);
            ceres::ResidualBlockId res_id = problem.AddResidualBlock(proj_cost, proj_loss, pose);
            res_ids.push_back(res_id);
        }
    }

    double eval_cost;
    ceres::Problem::EvaluateOptions eval_opt;
    eval_opt.apply_loss_function = false;
    for (size_t i = 0; i < 10 && i < res_ids.size(); i++) {
        eval_opt.residual_blocks.clear();
        eval_opt.residual_blocks.push_back(res_ids[i]);
        problem.Evaluate(eval_opt, &eval_cost, NULL, NULL, NULL);
        std::cout << "init pose cost: " << sqrt(eval_cost * 2) / cam0_->GetAngularResolution() 
            << " residual pt num: " << res_ids.size() << std::endl;
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 8;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    // options.function_tolerance = 2.0e-02;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    for (size_t i = 0; i < 10 && i < res_ids.size(); i++) {
        eval_opt.residual_blocks.clear();
        eval_opt.residual_blocks.push_back(res_ids[i]);
        problem.Evaluate(eval_opt, &eval_cost, NULL, NULL, NULL);
        std::cout << "final pose cost: " << sqrt(eval_cost * 2) / cam0_->GetAngularResolution() 
            << " residual pt num: " << res_ids.size() << std::endl;
    }

    t_frame_.Riw = RotationVector2Matrix3d(posev.topRows(3));
    t_frame_.tiw = posev.bottomRows(3);
    // cv::waitKey(0);
}

Eigen::Matrix<double, 3, 4> SlidingWindow::EPNP(const std::vector<Eigen::Vector3d> &ptcs, const std::vector<Eigen::Vector3d> &ptws) {
    if (ptws.size() < 5) {
        std::cout << "not enough pts in pnp with pt num: " << ptws.size() << std::endl;
        exit(0);
    }
    Eigen::Vector3d c1(0, 0, 0);
    for (size_t i = 0; i < ptws.size(); i++) {
        c1 += ptws[i];
    }
    c1 /= ptws.size();

    Eigen::Matrix3d A; A.setZero();
    for (size_t i = 0; i < ptws.size(); i++) {
        Eigen::Vector3d pt = ptws[i] - c1;
        A += pt * pt.transpose();
    }

    int pt_num = ptws.size();
    // A = U * S * Vt
    // A * V = U * S = [u1, u2,..., un] * diag(s1, s2, ..., sn)
    // A * [v1, v2,..., vn] = [s1 * u1, s2 * u2,..., sn * un]
    // A * vi = si * ui
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeThinV);
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Vector3d eig = svd.singularValues();
    Eigen::Vector3d c2 = c1 + sqrt(eig(0) / pt_num) * V.col(0);
    Eigen::Vector3d c3 = c1 + sqrt(eig(1) / pt_num) * V.col(1);
    Eigen::Vector3d c4 = c1 + sqrt(eig(2) / pt_num) * V.col(2);

    // lambda * [p1, p2, p3] = sum{alphai * [p1i, p2i, p3i]}
    // if (pj != 0) lambda = sum{alphai * pji} / pj

}

Eigen::Matrix<double, 3, 4> SlidingWindow::DLTPNP(const std::vector<Eigen::Vector3d> &ptcs,
                                                  const std::vector<Eigen::Vector3d> &ptws) {

}

bool SlidingWindow::Initialization() {
    if (last_frame_.obs0.size() == 0) {
        last_frame_ = t_frame_;
        return false;
    }

    std::vector<double> disps;
    for (auto &it : t_frame_.obs0) {
        if (last_frame_.obs0.find(it.first) != last_frame_.obs0.end()) {
            double disp = (last_frame_.obs0[it.first] - it.second).norm();
            disps.push_back(disp);
        }
    }

    if (disps.size() < 50) {
        last_frame_ = t_frame_;
        return false;
    }
}

Eigen::Matrix3d SlidingWindow::NisterEssentialMatrixSolver(const std::vector<Eigen::Vector3d> &ptcs,
                                                           const std::vector<Eigen::Vector3d> &ptws) {

}

// p1^t * E * p0 = 0
Eigen::Matrix3d SlidingWindow::DLTEssentialMatrixSolver(const std::vector<Eigen::Vector3d> &pt0s,
                                                    const std::vector<Eigen::Vector3d> &pt1s, const std::vector<int> &ids) {
    // pt0^t * E * pt1 = 0
    // [x0, y0, z0] * E * [x1, y1, z1]^t = 0
    // [x0, y0, z0] * [e00 * x1 + e01 * y1 + e02 * z1, e10 * x1 + e11 * y1 + e12 * z1,
    // e20 * x1 + e21 * y1 + e22 * z1]^t = 0
    // x0 * (e00 * x1 + e01 * y1 + e02 * z1) + y0 * (e10 * x1 + e11 * y1 + e12 * z1) +
    // z0 * (e20 * x1 + e21 * y1 + e22 * z1) = 0
    // x0 * x1 * e00 + x0 * y1 * e01 + x0 * z1 * e02 + y0 * x1 * e10 + y0 * y1 * e11 +
    // y0 * z1 * e12 + z0 * x1 * e20 + z0 * y1 * e21 + z0 * z1 * e22 = 0

    int num = ids.size();
    Eigen::MatrixXd A(num, 9);
    for (size_t i = 0; i < ids.size(); i++) {
        const Eigen::Vector3d &pt0 = pt0s[ids[i]];
        const Eigen::Vector3d &pt1 = pt1s[ids[i]];
        A.row(i) << pt1[0] * pt0.transpose(), pt1[1] * pt0.transpose(), pt1[2] * pt0.transpose();
    }

    // A * e = 0
    // U * S * Vt * e = 0
    // S * Vt * e = 0
    // if e = v.last_col() => S * Vt * e = S * [0, 0,..., 1]^t = 0
    // then e = v.last_col()
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd v = svd.matrixV();
    Eigen::VectorXd v1 = v.rightCols(1);
    Eigen::Matrix3d E;
    E.row(0) = v1.topRows(3).transpose();
    E.row(1) = v1.segment<3>(3).transpose();
    E.row(2) = v1.bottomRows(3);
    E /= E(2, 2);
    return E;
}

// p1^t * E * p0 = 0
std::vector<int> SlidingWindow::EssentialInlierIndexs(const Eigen::Matrix3d &E, const std::vector<Eigen::Vector3d> &pt0s,
                                        const std::vector<Eigen::Vector3d> &pt1s, double inlier_thres) {
    // pt0^t * E * pt1 = 0
    // x * a + y * b + z * c = 0
    std::vector<int> inlier_ids;
    for (size_t i = 0; i < pt0s.size(); i++) {
        Eigen::Vector3d pt1 = pt1s[i];
        Eigen::Vector3d tmp_pt = E * pt0s[i];
        tmp_pt /= tmp_pt.norm();
        double err = pt1.dot(tmp_pt) / pt1.norm();
        if (fabs(err) < inlier_thres) {
            inlier_ids.push_back(i);
        }
    }
    return inlier_ids;
}

std::vector<int> SlidingWindow::EssentialOutlierIndexs(const Eigen::Matrix3d &E, const std::vector<Eigen::Vector3d> &pt0s,
                             const std::vector<Eigen::Vector3d> &pt1s, double inlier_thres) {
    // pt1^t * E * pt0 = 0
    // x * a + y * b + z * c = 0
    std::vector<int> outlier_ids;
    for (size_t i = 0; i < pt0s.size(); i++) {
        Eigen::Vector3d pt1 = pt1s[i];
        Eigen::Vector3d tmp_pt = E * pt0s[i];
        tmp_pt /= tmp_pt.norm();
        double err = pt1.dot(tmp_pt) / pt1.norm();
        if (fabs(err) >= inlier_thres) {
            // std::cout << i << "th err: " << err << std::endl;
            outlier_ids.push_back(i);
        }
    }
    return outlier_ids;
}

Eigen::Matrix3d SlidingWindow::RansacEssentialMatrix(const std::vector<Eigen::Vector3d> &pt0s, const std::vector<Eigen::Vector3d> &pt1s,
                                                     double confidence, double thres) {
    int MaxIteration = 100;    
    double ln_c = std::log(1 - confidence);
    double dn = 1.0 / pt0s.size();
    srand((int)time(0));

    int best_count = 0;
    Eigen::Matrix3d En;
    double outlier_thres = thres;
    for (size_t i = 0; i < MaxIteration; i++) {
        int idx = random() % pt0s.size();
        std::vector<int> idxs;
        idxs.push_back(idx);
        while(idxs.size() < 8) {
            idx = random() % pt0s.size();
            for (auto &it : idxs) {
                if (idx == it) {
                    idx = -1;
                    break;
                }
            }
            if (idx != -1) idxs.push_back(idx);
        }

        Eigen::Matrix3d E = DLTEssentialMatrixSolver(pt0s, pt1s, idxs);
        std::vector<int> inlier_ids = EssentialInlierIndexs(E, pt0s, pt1s, outlier_thres);
        if (inlier_ids.size() > best_count) {
            best_count = inlier_ids.size();
            En = DLTEssentialMatrixSolver(pt0s, pt1s, inlier_ids);

            // p(success) = p(inlier)^8
            // p(fail) = 1 - p(success) = 1 - p(inlier)^8
            // p(all_fail) = p(fail)^n = (1 - p(inlier)^8)^n
            // confidence = 1 - p(all_fail) = 1 - (1-p(inlier)^8)^n
            // 1 - confidence = (1 - P(inlier)^8)^n
            // log(1 - confidence) = n * log(1 - P(inlier)^8)
            // n = log(1 - confidence) / log(1 - p(inlier)^8)
            double p_inlier = best_count * dn;
            double p_success = std::pow(p_inlier, 8);
            if (p_success < 0.000001) p_success = 0.000001;
            MaxIteration = ln_c / std::log(1 - p_success);
            if (p_inlier > 0.99) MaxIteration = 1.0;
            if (MaxIteration < 10) MaxIteration = 10;
            std::cout << "pinlier and maxiter: " << p_inlier << "," << MaxIteration << std::endl;
        }
    }

    return En;
}

void SlidingWindow::Test() {
    Eigen::Matrix3d Rcw; Rcw.setIdentity();
    Eigen::Vector3d tcw; tcw << 0, 0, 0.5;

    std::vector<Eigen::Vector3d> obses, Pws;
    for (size_t i = 1; i < 6; i++)
        for (size_t j = 1; j < 6; j++) 
            for (size_t k = 1; k < 6; k++) {
                Eigen::Vector3d pt;
                pt << i * 5, j * 5, k * 5;
                Eigen::Vector3d pw = Rcw * pt + tcw;
                Pws.push_back(pw);
                pt /= pt.norm();
                obses.push_back(pt);
            }

    // create gaussian noise
    srand((int)time(0));
    for (size_t i = 0; i < obses.size(); i++) {
        int irandom = rand()%10;
        double noise = irandom * 1.0 * 0.001 / 10;
        int idx = rand()%3;
        Eigen::Vector3d &pt = obses[i];
        pt[idx] += noise;
        pt /= pt.norm();
    }

    std::cout << "point num: " << obses.size() << std::endl;
    // create outliers
    for (size_t i = 0; i < 1; i++) {
        int irandom = rand()%10;
        double noise = irandom * 1.0 / 20.0;
        int idx = rand()%3;
        Eigen::Vector3d &pt = obses[i];
        pt[idx] += noise;
        pt /= pt.norm();
    }

    Eigen::Matrix3d Rwc = Rcw.transpose();
    Eigen::Vector3d twc = -Rcw * tcw;
    std::cout << "gt Rwc: " << Rwc << std::endl;
    std::cout << "gt twc: " << twc.transpose() << std::endl;
    twc += Eigen::Vector3d(0, 0, 0.5);
    Rwc = Rwc * RotationVector2Matrix3d(Eigen::Vector3d(0.15, 0.001, 0.005));
    std::cout << "init Rwc: " << Rwc << std::endl;
    std::cout << "init twc: " << twc.transpose() << std::endl;

    // GN iteration
    double huber_thres = 0.006;
    double cost = 0;
    std::vector<Eigen::Matrix<double, 2, 3>> tangentplane;
    for (size_t i = 0; i < Pws.size(); i++) {
        Eigen::Vector3d pc = Rwc * Pws[i] + twc;
        pc /= pc.norm();
        Eigen::Vector3d err = pc - obses[i];
        Eigen::Matrix<double, 2, 3> tp = TangentPlane(obses[i]);
        Eigen::Vector2d e = tp * err;
        double w = 1.0;
        if (e.norm() > huber_thres) {
            w = 2.0 * huber_thres / e.norm();
        }
        cost += w * w * e.squaredNorm();
        tangentplane.push_back(tp);
    }

    std::vector<Eigen::Matrix<double, 2, 6>> Js;
    std::vector<Eigen::Vector2d> es;
    Eigen::VectorXd dx(6);
    for (size_t i = 0; i < 20; i++) {
        Eigen::MatrixXd H(6, 6); H.setZero();
        Eigen::VectorXd b(6); b.setZero();
        for (size_t j = 0; j < Pws.size(); j++) {
            Eigen::Vector3d pc = Rwc * Pws[j] + twc;
            double rho = 1.0/pc.norm();
            Eigen::Vector3d err = pc * rho - obses[j];
            Eigen::Vector2d e = tangentplane[j] * err;

            // de/dpc = d(pc/pc.norm)/dpc = d(1.0/pc.norm)*pc/dpc + dpc/dpc/pc.norm
            // = -[x*[x, y, z],y*[x, y, z],z*[x, y, z]/pc.norm^3 + I / pc.norm
            // = -[x, y, z]^t * [x, y, z] / pc.norm^3 + I / pc.norm
            Eigen::Matrix3d de_dpc = rho * Eigen::Matrix3d::Identity();
            de_dpc -= pc * pc.transpose() * rho * rho * rho;
            Eigen::Matrix<double, 3, 6> dpc_dx;
            // pc + dpc = Rwc * exp(dtheta) * Pw + twc + dt
            // pc + dpc = Rwc * (I + dtheta^) * Pw + twc + dt
            // dpc = Rwc * dtheta^ * Pw + dt
            // dpc = -Rwc * Pw^ * dtheta + dt
            dpc_dx.leftCols(3) = -Rwc * MathUtil::VecToSkew(Pws[j]);
            dpc_dx.rightCols(3).setIdentity();
            Eigen::Matrix<double, 2, 6> J;
            J = tangentplane[j] * de_dpc * dpc_dx;
            if (Js.size() < Pws.size()) {                
                Js.push_back(J);
                es.push_back(e);
            } else {
                // fix linearization point of Jacobian and residual
                if (j < 0.1 * Pws.size()) {
                    e = es[j] + Js[j] * dx;
                    J = Js[j];
                }
            }
            double w = 1.0;
            if (e.norm() > huber_thres) {
                w = huber_thres * 2.0 / e.norm();
            }
            H += w * w * J.transpose() * J;
            b += w * w * J.transpose() * e;
        }

        // e = e0 + J * dx; de^2/dx=0;
        // e^2 = e0t * e0 + 2 * e0 * J * dx + dxt * Jt * J * dx; d(xt * y) = y
        // 2 * Jt * e0 + 2 * Jt * J * dx = 0 => Jt * J * dx = -Jt * e0;
        double t0 = static_cast<double>(cv::getTickCount());
        dx = H.ldlt().solve(-b);
        double t1 = static_cast<double>(cv::getTickCount());
        Eigen::VectorXd dx1 = GaussianSolveNXNLinearEquations(H, -b);
        double t2 = static_cast<double>(cv::getTickCount());
        std::cout << "t0: " << (t1 - t0)/cv::getTickFrequency() << " t1: " << (t2 - t1)/cv::getTickFrequency() << std::endl;
        // dx = SolveNXNLinearEquations(H, -b);
        // std::cout << "dx: " << dx.transpose() << std::endl;
        // std::cout << "dx1: " << dx1.transpose() << std::endl;
        Eigen::Matrix3d Rnew = Rwc * RotationVector2Matrix3d(dx.segment<3>(0));
        Eigen::Vector3d tnew = twc + dx.bottomRows(3);

        double cost_new = 0;
        for (size_t j = 0; j < Pws.size(); j++) {
            Eigen::Vector3d pc = Rnew * Pws[j] + tnew;
            Eigen::Vector3d error = pc / pc.norm() - obses[j];
            Eigen::Vector2d e = tangentplane[j] * error;
            double w = 1.0;
            if ( e.norm() > huber_thres) {
                w = 2.0 * huber_thres / e.norm();
            }
            cost_new += w * w * e.squaredNorm();
        }
        // std::cout << "cost and new: " << cost << "," << cost_new << std::endl;
        if (cost_new < cost) {
            cost = cost_new;
            Rwc = Rnew;
            twc = tnew;
            // std::cout << "cost decrease!!!" << std::endl;
        }
        // if (dx.norm() < 0.01) break;
    }

    std::cout << "final Rwc: " << Rwc << std::endl;
    std::cout << "final twc: " << twc.transpose() << std::endl;
    exit(0);
}

Eigen::VectorXd SlidingWindow::GaussianSolveNXNLinearEquations(const Eigen::MatrixXd &H, const Eigen::VectorXd &b) {
    int dim = b.rows();
    Eigen::MatrixXd Hb(dim, dim + 1);
    Hb.leftCols(dim) = H;
    Hb.rightCols(1) = b;

    double infin = 1.0e-100;
    // Gaussian elimination
    for (int i = 0; i < dim - 1; i++) {
        double max = fabs(Hb(i, i));
        int maxid = i;
        for (int j = i + 1; j < dim; j++) {
            // std::cout << i << "," << j << "th val: " << Hb(j, i) << ",max: " << max << std::endl;
            if (fabs(Hb(j, i)) > max) {
                maxid = j;
                max = fabs(Hb(j, i));
            }
        }
        if (maxid != i) {
            auto tmp = Hb.row(i);
            Hb.row(i) = Hb.row(maxid);
            Hb.row(maxid) = tmp;
        }
        // std::cout << "---------------------" << std::endl;
        if (max < infin) {
            continue;
        }

        for (int j = i + 1; j < dim; j++) {
            double ratio = -Hb(j, i) / Hb(i, i);
            Hb.row(j) = Hb.row(i) * ratio + Hb.row(j);
        }
        // std::cout << "Hb(i, i): " << Hb(i, i) << std::endl;
    }
    if (Hb.hasNaN()) {
        exit(0);
    }
    // back-substitution
    Eigen::VectorXd sol(dim);
    for (int i = dim - 1; i >= 0; i--) {
        if (fabs(Hb(i, i)) < infin) {
            // std::cout << i << "th too small Hb: " << Hb(i, i) << std::endl;
            sol(i) = 0.0;
            continue;
        }
        
        sol(i) = Hb(i, dim);
        for (int j = dim - 1; j > i; j--) {
            sol(i) -= Hb(i, j) * sol(j);
        }
        // std::cout << i << "th sol i: " << sol(i) << "," << Hb(i, i) << std::endl;
        sol(i) /= Hb(i, i);

        // if (fabs(sol(i)) > 3) {
        //     std::cout << "wrong sol i: " << sol(i) << std::endl;
        //     exit(0);
        // }
        if (sol.hasNaN()) {
            // std::cout << "sol: " << sol(i) << "," << Hb(i, i) << std::endl;
            exit(0);
        }
    }
    return sol;
}

int SlidingWindow::CreateData(std::vector<Eigen::Vector3d> &Pws, std::vector<std::vector<Eigen::Vector3d>> &obs_v,
                    std::vector<Eigen::Matrix<double, 3, 4>> &Twcs) {
    int frame_num = 10;
    Eigen::Matrix3d dR = RotationVector2Matrix3d(Eigen::Vector3d(0.05, 0.06, 0.05));
    Eigen::Vector3d dt(0.1, 0.2, 0.15);

    // create pws
    for (size_t i = 1; i < 6; i++)
        for (size_t j = 1; j < 6; j++) 
            for (size_t k = 1; k < 6; k++) {
                Eigen::Vector3d pt;
                pt << i * 5, j * 5, k * 5;
                Pws.push_back(pt);
            }

    Eigen::Matrix3d Rcw = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tcw(0, 0, 0);
    for (int i = 0; i < frame_num; i++) {
        std::vector<Eigen::Vector3d> obses;
        Eigen::Matrix<double, 3, 4> pose;
        Eigen::Matrix3d Rwc = Rcw.transpose();
        Eigen::Vector3d twc = -Rwc * tcw;
        pose.leftCols(3) = Rwc;
        pose.rightCols(1) = twc;
        for (size_t j = 0; j < Pws.size(); j++) {
            Eigen::Vector3d Pc = Rwc * Pws[j] + twc;
            Pc /= Pc.norm();
            obses.push_back(Pc);
        }
        obs_v.push_back(obses);
        Twcs.push_back(pose);

        tcw += Rwc * dt;
        Rcw = dR * Rcw;
    }
    return frame_num;
}

bool SlidingWindow::PnPOptimization(const std::vector<Eigen::Vector3d> &Pws, std::vector<Eigen::Vector3d> &obses,
                                    Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc) {
    Eigen::Matrix3d Rt = Rwc;
    Eigen::Vector3d tt = twc;
    twc += Eigen::Vector3d(0, 0, 0.5);
    Rwc = Rwc * RotationVector2Matrix3d(Eigen::Vector3d(0.15, 0.001, 0.005));

    // GN iteration
    double huber_thres = 0.006;
    double cost = 0;
    std::vector<Eigen::Matrix<double, 2, 3>> tangentplane;
    for (size_t i = 0; i < Pws.size(); i++) {
        Eigen::Vector3d pc = Rwc * Pws[i] + twc;
        pc /= pc.norm();
        Eigen::Vector3d err = pc - obses[i];
        Eigen::Matrix<double, 2, 3> tp = TangentPlane(obses[i]);
        Eigen::Vector2d e = tp * err;
        double w = 1.0;
        if (e.norm() > huber_thres) {
            w = 2.0 * huber_thres / e.norm();
        }
        cost += w * w * e.squaredNorm();
        tangentplane.push_back(tp);
    }

    std::vector<Eigen::Matrix<double, 2, 6>> Js;
    std::vector<Eigen::Vector2d> es;
    Eigen::VectorXd dx(6);
    for (size_t i = 0; i < 8; i++) {
        Eigen::MatrixXd H(6, 6); H.setZero();
        Eigen::VectorXd b(6); b.setZero();
        for (size_t j = 0; j < Pws.size(); j++) {
            Eigen::Vector3d pc = Rwc * Pws[j] + twc;
            double rho = 1.0/pc.norm();
            Eigen::Vector3d err = pc * rho - obses[j];
            Eigen::Vector2d e = tangentplane[j] * err;

            // de/dpc = d(pc/pc.norm)/dpc = d(1.0/pc.norm)*pc/dpc + dpc/dpc/pc.norm
            // = -[x*[x, y, z],y*[x, y, z],z*[x, y, z]/pc.norm^3 + I / pc.norm
            // = -[x, y, z]^t * [x, y, z] / pc.norm^3 + I / pc.norm
            Eigen::Matrix3d de_dpc = rho * Eigen::Matrix3d::Identity();
            de_dpc -= pc * pc.transpose() * rho * rho * rho;
            Eigen::Matrix<double, 3, 6> dpc_dx;
            // pc + dpc = Rwc * exp(dtheta) * Pw + twc + dt
            // pc + dpc = Rwc * (I + dtheta^) * Pw + twc + dt
            // dpc = Rwc * dtheta^ * Pw + dt
            // dpc = -Rwc * Pw^ * dtheta + dt
            dpc_dx.leftCols(3) = -Rwc * MathUtil::VecToSkew(Pws[j]);
            dpc_dx.rightCols(3).setIdentity();
            Eigen::Matrix<double, 2, 6> J;
            J = tangentplane[j] * de_dpc * dpc_dx;
            if (Js.size() < Pws.size()) {                
                Js.push_back(J);
                es.push_back(e);
            } else {
                // fix linearization point of Jacobian and residual
                // if (j < 0.1 * Pws.size()) {
                //     e = es[j] + Js[j] * dx;
                //     J = Js[j];
                // }
            }
            double w = 1.0;
            if (e.norm() > huber_thres) {
                w = huber_thres * 2.0 / e.norm();
            }
            H += w * w * J.transpose() * J;
            b += w * w * J.transpose() * e;
        }

        // e = e0 + J * dx; de^2/dx=0;
        // e^2 = e0t * e0 + 2 * e0 * J * dx + dxt * Jt * J * dx; d(xt * y) = y
        // 2 * Jt * e0 + 2 * Jt * J * dx = 0 => Jt * J * dx = -Jt * e0;
        // double t0 = static_cast<double>(cv::getTickCount());
        // dx = H.ldlt().solve(-b);
        // double t1 = static_cast<double>(cv::getTickCount());
        // Eigen::VectorXd dx1 = GaussianSolveNXNLinearEquations(H, -b);
        // double t2 = static_cast<double>(cv::getTickCount());
        // Eigen::VectorXd ex = dx - dx1;
        // std::cout << "t0: " << (t1 - t0)/cv::getTickFrequency() << " t1: " << (t2 - t1)/cv::getTickFrequency() <<
        //     " ex: " << ex.norm() << std::endl;

        dx = GaussianSolveNXNLinearEquations(H, -b);
        Eigen::Matrix3d Rnew = Rwc * RotationVector2Matrix3d(dx.segment<3>(0));
        Eigen::Vector3d tnew = twc + dx.bottomRows(3);

        double cost_new = 0;
        for (size_t j = 0; j < Pws.size(); j++) {
            Eigen::Vector3d pc = Rnew * Pws[j] + tnew;
            Eigen::Vector3d error = pc / pc.norm() - obses[j];
            Eigen::Vector2d e = tangentplane[j] * error;
            double w = 1.0;
            if ( e.norm() > huber_thres) {
                w = 2.0 * huber_thres / e.norm();
            }
            cost_new += w * w * e.squaredNorm();
        }
        if (cost_new < cost) {
            cost = cost_new;
            Rwc = Rnew;
            twc = tnew;
            std::cout << "cost decrease!!!" << std::endl;
        }
        if (dx.norm() < 0.001) break;
    }

    Eigen::Vector3d error = tt - twc;
    Eigen::Vector3d rvec = RotationMatrix2Vector3d(Rwc * Rt.transpose());

    std::cout << "t error: " << error.transpose() << std::endl;
    std::cout << "r error: " << rvec.transpose() << std::endl;
    return true;
}

void SlidingWindow::TestDemo() {
    std::cout << "************test demo**************" << std::endl;
    std::vector<Eigen::Vector3d> Pws;
    std::vector<Eigen::Matrix<double, 3, 4>> Twcs;
    std::vector<std::vector<Eigen::Vector3d>> obs_v;
    int frame_num = CreateData(Pws, obs_v, Twcs);

    std::cout << "frame num: " << frame_num << " ,landmark num: " << Pws.size() << std::endl;
    for (int i = 0; i < frame_num; i++) {
        std::cout << "--------------------" << std::endl;
        Eigen::Matrix3d Rwc = Twcs[i].leftCols(3);
        Eigen::Vector3d twc = Twcs[i].rightCols(1);
        AddNoiseAndOutliers(obs_v[i]);
        // bool is_success = PnPOptimization(Pws, obs_v[i], Rwc, twc);
    }
    LocalOptimization(Pws, obs_v, Twcs);
    exit(0);
}

void SlidingWindow::AddNoiseAndOutliers(std::vector<Eigen::Vector3d> &obses) {
    srand((int)time(0));
    // add gaussian noise
    for (size_t i = 0; i < obses.size(); i++) {
        int irandom = rand()%10;
        double noise = irandom * 1.0 * 0.005 / 10;
        int idx = rand()%3;
        Eigen::Vector3d &pt = obses[i];
        pt[idx] += noise;
        pt /= pt.norm();
    }

    // add outliers
    for (size_t i = 0; i < 0; i++) {
        int irandom = rand()%10;
        double noise = irandom * 1.0 / 20.0;
        int idx = rand()%3;
        Eigen::Vector3d &pt = obses[i];
        pt[idx] += noise;
        pt /= pt.norm();
    }
}

void SlidingWindow::LocalOptimization(std::vector<Eigen::Vector3d> &Pws, std::vector<std::vector<Eigen::Vector3d>> &obs_v,
                                      std::vector<Eigen::Matrix<double, 3, 4>> &Twcs) {
    
    std::vector<Eigen::Matrix3d> Rwcs;
    std::vector<Eigen::Vector3d> twcs;
    for (int i = 0; i < Twcs.size(); i++) {
        Eigen::Matrix3d Rwc = Twcs[i].leftCols(3);
        Eigen::Vector3d twc = Twcs[i].rightCols(1);
        twc += Eigen::Vector3d(0, 0.02 * i, 0.02 * i);
        Rwc = Rwc * RotationVector2Matrix3d(Eigen::Vector3d(0.003 * i, 0.005 * i, 0.003 * i));
        Rwcs.push_back(Rwc);
        twcs.push_back(twc);
    }

    double huber_thres = 0.05;
    double cost = 0;
    std::vector<std::vector<Eigen::Matrix<double, 2, 3>>> tangentplanes;
    for (int i = 0; i < Rwcs.size(); i++) {
        const std::vector<Eigen::Vector3d> &obses = obs_v[i];
        Eigen::Matrix3d Rwc = Rwcs[i];
        Eigen::Vector3d twc = twcs[i];
        std::vector<Eigen::Matrix<double, 2, 3>> planes;
        cost += CalcCost(Rwc, twc, Pws, obses, planes, huber_thres, true);
        tangentplanes.push_back(planes);
        // std::cout << i << "th cost: " << cost << std::endl;
        // std::cout << "planes num: " << planes.size() << std::endl;
    }

    int dim = Pws.size() * 3 + Twcs.size() * 6;
    int pt_id0 = Twcs.size() * 6;
    int obs_dim = Twcs.size() * Pws.size() * 2;
    int res_idx = 0;
    double lambda = 1.0e-03;
    for (int it = 0; it < 15; it++) {
        res_idx = 0;
        Eigen::MatrixXd H(dim, dim), J(obs_dim, dim); H.setZero(); J.setZero();
        Eigen::VectorXd b(dim), r(obs_dim); b.setZero(); r.setZero();

        for (int i = 0; i < Rwcs.size(); i++) {
            Eigen::Matrix3d Rwc = Rwcs[i];
            Eigen::Vector3d twc = twcs[i];
            int pose_id = i * 6;
            HessianAndResidualCalc(Rwc, twc, Pws, obs_v[i], tangentplanes[i], H, b, pose_id, pt_id0, huber_thres);
            JacobAndResidualCalc(Rwc, twc, Pws, obs_v[i], tangentplanes[i], J, r, pose_id, pt_id0, res_idx, huber_thres);
        }
        
        J.leftCols(6).setZero();
        H.leftCols(6).setZero();
        H.topRows(6).setZero();
        b.topRows(6).setZero();
        // H.topLeftCorner(6, 6) = 1000 * Eigen::MatrixXd::Identity(6, 6);
        // std::cout << "r: " << r.norm() << " b: " << b.norm() << std::endl;
        // Hx = -b
        H = H + lambda * Eigen::MatrixXd::Identity(H.rows(), H.cols());
        Eigen::MatrixXd He = J.transpose() * J + lambda * Eigen::MatrixXd::Identity(H.rows(), H.cols());
        Eigen::VectorXd be = J.transpose() * r;
        Eigen::VectorXd dx = (H).ldlt().solve(-b);
        Eigen::VectorXd dx1 = (He).lu().solve((-be));
        Eigen::VectorXd dx2 = GaussianSolveNXNLinearEquations(H, -b);
        std::cout << "dx0, 1, 2: " << dx.norm() << "," << dx1.norm() << "," << dx2.norm() << std::endl;

        // Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver(H);
        
        // // const Eigen::VectorXcd &eigs = eigen_solver.eigenvalues();
        // // std::cout << "eigs: " << eigs.transpose() << std::endl;
        // Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        // const Eigen::VectorXd &eig = svd.singularValues();

        // std::cout << "H eigs: " << svd.singularValues().transpose() << std::endl;
        // std::cout << "dim: " << eig.rows() << "," << dim << std::endl;
        // exit(0);
        // dx = dx2;
        std::vector<Eigen::Matrix3d> Rwc_new;
        std::vector<Eigen::Vector3d> twc_new;
        for (int i = 0; i < Rwcs.size(); i++) {
            Eigen::Matrix3d Rwc_n = Rwcs[i] * RotationVector2Matrix3d(dx.segment<3>(i * 6));
            Eigen::Vector3d twc_n = twcs[i] + dx.segment<3>(i * 6 + 3);
            Rwc_new.push_back(Rwc_n);
            twc_new.push_back(twc_n);
        }
        std::vector<Eigen::Vector3d> Pws_new;
        for (int i = 0; i < Pws.size(); i++) {
            Eigen::Vector3d Pw = Pws[i] + dx.segment<3>(pt_id0 + i * 3);
            Pws_new.push_back(Pw);
        }

        double cost_new = 0;
        for (int i = 0; i < Rwcs.size(); i++) {
            const std::vector<Eigen::Vector3d> &obses = obs_v[i];
            Eigen::Matrix3d Rwc = Rwc_new[i];
            Eigen::Vector3d twc = twc_new[i];
            std::vector<Eigen::Matrix<double, 2, 3>> &planes = tangentplanes[i];
            cost_new += CalcCost(Rwc, twc, Pws_new, obses, planes, huber_thres);
        }

        std::cout << "cost 0 and new: " << cost << "," << cost_new << std::endl;
        if (cost_new < cost) {
            Rwcs = Rwc_new;
            twcs = twc_new;
            Pws = Pws_new;
            cost = cost_new;
            std::cout << "local optimization decrease!!!" << std::endl;
            lambda = lambda < 1.0e-08? lambda : lambda * 0.1;
        } else {
            lambda = lambda > 1.0e01? lambda : lambda * 10;
        }
    }

    for (int i = 0; i < Rwcs.size(); i++) {
        Eigen::Matrix3d Rwct = Twcs[i].leftCols(3);
        Eigen::Vector3d twct = Twcs[i].rightCols(1);
        Eigen::Matrix3d dR = Rwct.transpose() * Rwcs[i];
        Eigen::Vector3d dt = twct - twcs[i];
        Eigen::VectorXd de(6);
        de << RotationMatrix2Vector3d(dR), dt;
        std::cout << i << "th pose error: " << de.transpose() << std::endl;
    }
}

double SlidingWindow::CalcCost(const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &twc, const std::vector<Eigen::Vector3d> &Pws,
                               const std::vector<Eigen::Vector3d> &obses, std::vector<Eigen::Matrix<double, 2, 3>> &planes,
                               double huber_thres, bool calc_plane) {
    double cost = 0;
    double thres2 = huber_thres * huber_thres;
    for (int i = 0; i < Pws.size(); i++) {
        Eigen::Vector3d Pc = Rwc * Pws[i] + twc;
        Eigen::Vector3d error = Pc / Pc.norm() - obses[i];
        Eigen::Matrix<double, 2, 3> tangentplane;
        if (calc_plane) {
            tangentplane = TangentPlane(obses[i]);
            planes.push_back(tangentplane);
        } else {
            tangentplane = planes[i];
        }
        Eigen::Vector2d e = tangentplane * error;
        double w = 1.0;
        if (e.squaredNorm() > thres2) {
            w = 4.0 * thres2 / e.squaredNorm();
        }
        cost += w * e.squaredNorm();
    }
    return cost;
}

void SlidingWindow::HessianAndResidualCalc(const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &twc, const std::vector<Eigen::Vector3d> &Pws,
                                           const std::vector<Eigen::Vector3d> &obses, std::vector<Eigen::Matrix<double, 2, 3>> &planes,
                                           Eigen::MatrixXd &H, Eigen::VectorXd &r, int pose_id, int pt_id0, double huber_thres) {
    double thres2 = huber_thres * huber_thres;
    for (int i = 0; i < Pws.size(); i++) {
        Eigen::Vector3d Pc = Rwc * Pws[i] + twc;
        double rho = 1.0 / Pc.norm();
        Eigen::Vector3d error = Pc * rho - obses[i];
        Eigen::Vector2d res = planes[i] * error;
        
        // e = Pc/Pc.norm() - obs
        // de_dpc = d(Pc/Pc.norm)/dPc = dPc * rho/dPc + Pc * d(rho)/dPc
        // = I * rho - Pc * Pc^t * (rho * rho * rho)
        Eigen::Matrix3d de_dpc = rho * Eigen::Matrix3d::Identity();
        de_dpc -= Pc * Pc.transpose() * rho * rho * rho;
        Eigen::Matrix<double, 3, 6> dpc_dx;

        // pc + dpc = Rwc * exp(dtheta) * Pw + twc + dt
        // pc + dpc = Rwc * (I + dtheta^) * Pw + twc + dt
        // dpc = Rwc * dtheta^ * Pw + dt
        // dpc = -Rwc * Pw^ * dtheta + dt
        dpc_dx.leftCols(3) = -Rwc * MathUtil::VecToSkew(Pws[i]);
        dpc_dx.rightCols(3).setIdentity();
        
        // pc + dpc = Rwc * (Pw + dPw) + twc
        // dpc = Rwc * dPw
        Eigen::Matrix3d dpc_dpw = Rwc;
        double w = 1.0;
        if (res.squaredNorm() > thres2) {
            w = 4 * thres2 / res.squaredNorm();
        }
        Eigen::Matrix<double, 2, 6> dr_dx = w * planes[i] * de_dpc * dpc_dx;
        Eigen::Matrix<double, 2, 3> dr_dpw = w * planes[i] * de_dpc * dpc_dpw;

        int pt_id = pt_id0 + i * 3;
        H.block<6, 6>(pose_id, pose_id) += dr_dx.transpose() * dr_dx;
        H.block<6, 3>(pose_id, pt_id) += dr_dx.transpose() * dr_dpw;
        H.block<3, 3>(pt_id, pt_id) += dr_dpw.transpose() * dr_dpw;
        H.block<3, 6>(pt_id, pose_id) = H.block<6, 3>(pose_id, pt_id).transpose();
        r.segment<6>(pose_id) += dr_dx.transpose() * res;
        r.segment<3>(pt_id) += dr_dpw.transpose() * res;
    }
}

void SlidingWindow::JacobAndResidualCalc(const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &twc, const std::vector<Eigen::Vector3d> &Pws,
                                const std::vector<Eigen::Vector3d> &obses, std::vector<Eigen::Matrix<double, 2, 3>> &planes,
                                Eigen::MatrixXd &J, Eigen::VectorXd &r, int pose_id, int pt_id0, int &idx, 
                                double huber_thres) {
    double thres2 = huber_thres * huber_thres;
    for (int i = 0; i < Pws.size(); i++) {
        Eigen::Vector3d Pc = Rwc * Pws[i] + twc;
        double rho = 1.0 / Pc.norm();
        Eigen::Vector3d error = Pc * rho - obses[i];
        Eigen::Vector2d res = planes[i] * error;
        
        // e = Pc/Pc.norm() - obs
        // de_dpc = d(Pc/Pc.norm)/dPc = dPc * rho/dPc + Pc * d(rho)/dPc
        // = I * rho - Pc * Pc^t * (rho * rho * rho)
        Eigen::Matrix3d de_dpc = rho * Eigen::Matrix3d::Identity();
        de_dpc -= Pc * Pc.transpose() * rho * rho * rho;
        Eigen::Matrix<double, 3, 6> dpc_dx;
        // pc + dpc = Rwc * exp(dtheta) * Pw + twc + dt
        // pc + dpc = Rwc * (I + dtheta^) * Pw + twc + dt
        // dpc = Rwc * dtheta^ * Pw + dt
        // dpc = -Rwc * Pw^ * dtheta + dt
        dpc_dx.leftCols(3) = -Rwc * MathUtil::VecToSkew(Pws[i]);
        dpc_dx.rightCols(3).setIdentity();
        // pc + dpc = Rwc * (Pw + dPw) + twc
        // dpc = Rwc * dPw
        Eigen::Matrix3d dpc_dpw = Rwc;
        double w = 1.0;
        if (res.squaredNorm() > thres2) {
            w = 4 * thres2 / res.squaredNorm();
        }
        Eigen::Matrix<double, 2, 6> dr_dx = w * planes[i] * de_dpc * dpc_dx;
        Eigen::Matrix<double, 2, 3> dr_dpw = w * planes[i] * de_dpc * dpc_dpw;

        int pt_id = pt_id0 + i * 3;
        J.block<2, 6>(idx, pose_id) = dr_dx;
        J.block<2, 3>(idx, pt_id) = dr_dpw;
        r.segment<2>(idx) = res;
        idx += 2;
    }
}

Eigen::Vector3d GaussNoise(double cov) {
    int x = rand() % 500;
    int y = rand() % 500;
    int z = rand() % 500;
    double ratio = 2.0 / 499;
    double x_n = x * ratio - 1;
    double y_n = y * ratio - 1;
    double z_n = z * ratio - 1;
    return Eigen::Vector3d(x_n, y_n, z_n) * cov;
}

std::vector<int> PositiveDepthValidation(const std::vector<Eigen::Vector3d> &pt0s, const std::vector<Eigen::Vector3d> &pt1s,
                                         const Eigen::Matrix3d &R01, const Eigen::Vector3d &t01) {
    std::vector<int> inliers;
    for (int i = 0; i < pt0s.size(); i++) {
        // d0 * pt0 = pw;
        // d1 * pt1 = R01 * pw + t01
        // d1 * pt1 = R01 * (d0 * pt0) + t01
        // d1 * pt1 - d0 * R01 * pt0 = t01
        // [-R01 * pt0, pt1] * [d0, d1]^t = t01
        Eigen::Matrix<double, 3, 2> A;
        A.col(0) = -R01 * pt0s[i];
        A.col(1) = pt1s[i];
        Eigen::Vector2d depths = (A.transpose() * A).lu().solve(A.transpose() * t01);
        if (depths(0) > 0 && depths(1) > 0) {
            inliers.push_back(i);
        }
    }
    return inliers;
}

void SlidingWindow::EssentialTest() {
    Eigen::Matrix3d R01 = RotationVector2Matrix3d(Eigen::Vector3d(0.1, 0.3, 0.6));
    Eigen::Vector3d t01(1, 2, 3);

    std::vector<Eigen::Vector3d> Pws, Pt0s, Pt1s;
    for (int i = -5; i < 5; i++) {
        for (int j = -5; j < 5; j++) {
            for (int k = -5; k < 5; k++) {
                Eigen::Vector3d Pw(i * 5, j * 5, k * 5);
                if (Pw.norm() < 1) continue;
                Pws.push_back(Pw);
                Pt0s.push_back(Pw / Pw.norm());
                Eigen::Vector3d pt1 = R01 * Pw + t01;
                Pt1s.push_back(pt1 / pt1.norm());
            }
        }
    }
    std::cout << "true R01: " << R01 << std::endl;
    std::cout << "true t01: " << t01.transpose() << std::endl;
    Eigen::Matrix3d E = MathUtil::VecToSkew(t01) * R01;
    E /= E(2, 2);
    std::cout << "true E: " << E << std::endl;

    srand((int)time(0));
    // add gauss noise
    double gauss_cov = 0.001;
    for (int i = 0; i < Pt0s.size(); i++) {
        Pt0s[i] += GaussNoise(gauss_cov); Pt0s[i] /= Pt0s[i].norm();
        Pt1s[i] += GaussNoise(gauss_cov); Pt1s[i] /= Pt1s[i].norm();
    }

    // add outliers
    int outlier_num = 0.35 * Pt0s.size();
    for (int i = 0; i < outlier_num; i++) {
        int idx = rand() % 3;
        int noise = rand() % 100;
        double n = (noise / 49.5 - 1);
        Pt0s[i][idx] += n;
        Pt0s[i] /= Pt0s[i].norm();
    }

    std::vector<int> idxs;
    for (int i = 0; i < Pt0s.size(); i++) idxs.push_back(i);
    Eigen::Matrix3d E1 = RansacEssentialMatrix(Pt0s, Pt1s, 0.99, 0.005);
    for (int i = 0; i < outlier_num + 5; i++) {
        double err = EssentialError(Pt0s[i], Pt1s[i], E1);
        // std::cout << "err: " << err << std::endl;
    }
    std::cout << "E1: " << E1 << std::endl;

    Eigen::Matrix3d R1, R2;
    Eigen::Vector3d t;
    EssentialDecompose(E1, R1, R2, t);

    std::vector<int> idx0 = PositiveDepthValidation(Pt0s, Pt1s, R1, t);
    std::vector<int> idx1 = PositiveDepthValidation(Pt0s, Pt1s, R1, -t);
    std::vector<int> idx2 = PositiveDepthValidation(Pt0s, Pt1s, R2, t);
    std::vector<int> idx3 = PositiveDepthValidation(Pt0s, Pt1s, R2, -t);

    Eigen::Matrix3d est_R01;
    Eigen::Vector3d est_t01;
    int num0 = idx0.size();
    int num1 = idx1.size();
    int num2 = idx2.size();
    int num3 = idx3.size();
    int max_num = num0 > num1? num0 : num1;
    max_num = max_num > num2? max_num : num2;
    max_num = max_num > num3? max_num : num3;
    if (max_num == num0) {
        est_R01 = R1;
        est_t01 = t;
    } else if (max_num == num1) {
        est_R01 = R1;
        est_t01 = -t;
    } else if (max_num == num2) {
        est_R01 = R2;
        est_t01 = t;
    } else if (max_num == num3) {
        est_R01 = R2;
        est_t01 = -t;
    }
    double ratio = t01.norm() / est_t01.norm();
    est_t01 *= ratio;
    std::cout << "num: " << num0 << "," << num1 << "," << num2 << "," << num3 << std::endl;
    std::cout << "estimated R01: " << est_R01 << std::endl;
    std::cout << "estimated t01: " << est_t01.transpose() << std::endl;
    Eigen::Matrix3d dR = R01 * est_R01.transpose();
    Eigen::Vector3d e_rv = RotationMatrix2Vector3d(dR);
    Eigen::Vector3d e_t = t01 - est_t01;
    std::cout << "err R: " << e_rv.transpose() << std::endl;
    std::cout << "err t: " << e_t.transpose() << std::endl;
    
    std::cout << "end at essential test." << __LINE__ << std::endl;
    exit(0);
}

void SlidingWindow::EssentialDecompose(const Eigen::Matrix3d &E, Eigen::Matrix3d &R1, Eigen::Matrix3d &R2, Eigen::Vector3d &t) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // E = t^ * R
    // t^ = k * U * Z * Ut, Z = diag(1, 1, 0) * W, W = [0, -1, 0; 1, 0, 0; 0, 0 1]
    // E = t^ * R = k * U * Z * Ut * R = U * (k * Z) * Ut * R = U * diag(k, k, 0) * W * Ut * R
    // W * Ut * R = Vt => R = U * Wt * Vt
    // if k < 0 => R = U * (-Wt) * Vt = R * W * Vt
    Eigen::Matrix3d W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    Eigen::Matrix3d u = svd.matrixU();
    Eigen::Matrix3d v = svd.matrixV();

    // det(R) = 1, Rt * R = I
    if (u.determinant() < 0) u *= -1;
    if (v.determinant() < 0) v *= -1;
    R1 = u * W * v.transpose();
    R2 = u * W.transpose() * v.transpose();
    t = u.col(2);
}

void SlidingWindow::VoxelHashMapTest() {

}



} // namespace inslam