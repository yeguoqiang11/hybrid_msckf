
#include "Vio/MsckfVio.h"
#include "Utils/MathUtil.h"
#include "Utils/PerformanceTest.h"

using namespace std;
using namespace cv;
using namespace Eigen;

#define USE_UNIT_SPHERE_ERROR 1

namespace hybrid_msckf {

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


MsckfVio::MsckfVio(int maxSlidingWindowSize, shared_ptr<Caimura> cam0, shared_ptr<Caimura> cam1)
    : maxSlidingWindowSize_(maxSlidingWindowSize), caim_(move(cam0)), cam1_(move(cam1))
{
    // Make sure that the max track length > least observations num
    if (maxTrackLength_ < leastObservationNum_) {
        cerr << "Error: max track length : " << maxTrackLength_ <<
            " is smaller than least observations num:" << leastObservationNum_ << endl;
        maxTrackLength_ = leastObservationNum_;
    }

    // Make sure that the sliding window size is larger than 4
    if (maxSlidingWindowSize_ < 6) {
        cerr << "Error: siding window size must be larger than 6: " << maxSlidingWindowSize_ << endl;
        maxSlidingWindowSize_ = 6;
    }

    // Set constant parameters
    gravity_ << 0, 0, -9.81;

    noiseNc_.resize(errHeadDim_, errHeadDim_);
    noiseNc_.setZero();

    int i;
    for (i=0; i<3; i++)
        noiseNc_(i,i) = pow(caim_->gyroSigma_, 2);
    for (i=6; i<9; i++)
        noiseNc_(i,i) = pow(caim_->accSigma_, 2);
    for (i=9; i<12; i++)
        noiseNc_(i,i) = pow(caim_->gyroRandomWalk_, 2);
    for (i=12; i<15; i++)
        noiseNc_(i,i) = pow(caim_->accRandomWalk_, 2);

    // parameters
    leastObservationNum_ = 3;
    maxTrackLength_ = 6;

    imageFeatureNoise_ = 1.0;
#if USE_UNIT_SPHERE_ERROR
    imageFeatureNoise_ = imageFeatureNoise_ * std::pow(caim_->GetAngularResolution(), 2);
#endif

    translationThresh_ = 0.4; // (meter)
    rotationThresh_ = 0.261799;  // (rad), 15 degree
    trackingRateThresh_ = 0.5;

    if (cam1_ != nullptr) {
        Rrl_ = cam1_->Rci_ * caim_->Ric_;
        prl_ = cam1_->pci_ + cam1_->Rci_ * caim_->pic_;
    }
}


void MsckfVio::Initialize(Vector4d &q, Vector3d &p, Vector3d &v, int frameId, double timestamp) {
    currFrameId_ = frameId;
    currTimestamp_ = timestamp;

    // Set state vector
    state_.resize(headDim_);
    state_ << q, p, v, caim_->bg_, caim_->ba_;

    // Set covariance
    errCov_.resize(errHeadDim_, errHeadDim_);
    errCov_.setZero();

    int i;
    for (i=0; i<3; i++)     errCov_(i,i) = 3e-3;   // initial position covariance
    for (i=3; i<6; i++)     errCov_(i,i) = 1e-8;   // initial orientation covariance
    for (i=6; i<9; i++)     errCov_(i,i) = 4e-4;   // initial velocity covariance
    for (i=9; i<12; i++)    errCov_(i,i) = 1e-6;   // initial gyro bias covariance
    for (i=12; i<15; i++)   errCov_(i,i) = 1e-3;   // initial acc bias covariance

    // Reset sliding window pose info
    framePoseInfos_.clear();

}


void MsckfVio::Run(const vector<FeatureObservation> &featureMsgs,
                   double t_img, int frameId, const vector<Matrix<double, 7, 1>> &imuDatas) {
    currFrameId_ = frameId;
    currTimestamp_ = t_img;
    
    // Predict current state with IMU data.
    Propagate(imuDatas);

    // Add current frame into sliding window PoseInfo map
    framePoseInfos_[currFrameId_] = PoseInfo(frameId);
    UpdatePoseInfos(true);
    
    // Add feature tracks into feature server
    UpdateFeatureServer(featureMsgs);

    // // Remove lost features and run msckf update
    RemoveLostFeatures();

    // // Prune redundant frames and run msckf update
    PruneImuStates();

    // Augment state
    AugmentState();
    
}



void MsckfVio::Propagate(const vector<Matrix<double, 7, 1>> &imuDatas) {
    if (imuDatas.empty()) {
        return;
    }

    double dt, halfdt;
    Vector3d w1_halfdt,w2_halfdt,wtemp,  a1,a2,a_Il_middle,a_Il_end,a_G_begin,a_G_end, sl,yl,sl_G,yl_G;
    Quaterniond q_middle_Il, q_end_middle, q_end_Il;
    Matrix3d R_Il_middle, R_Il_end, R_G_end, R_G_Il, eye3;
    eye3.setIdentity();

    Quaterniond q(state_(0), state_(1), state_(2), state_(3)), qnew;
    Vector3d p = state_.segment(4, 3), pnew;
    Vector3d v = state_.segment(7, 3), vnew;
    Vector3d bw = state_.segment(10, 3);
    Vector3d ba = state_.segment(13, 3);
    int SIGMA_N = (int)errCov_.cols();
    MatrixXd SIGMA_ii = errCov_.topLeftCorner(errHeadDim_, errHeadDim_);
    MatrixXd SIGMA_ic = errCov_.topRightCorner(errHeadDim_, SIGMA_N-errHeadDim_);

    Matrix3d PHI_q_bg, PHI_v_bg;
    MatrixXd PHI = MatrixXd::Identity(errHeadDim_, errHeadDim_), Qd;

    Matrix<double,7,1> data0 = imuDatas[0], data1;

    for (int k=0; k<(int)imuDatas.size()-1; k++)
    {
        data1 = imuDatas[k+1];
        dt = data1(0) - data0(0);
        halfdt = 0.5 * dt;

        w1_halfdt = (data0.segment(1, 3) - bw) * halfdt;
        w2_halfdt = (data1.segment(1, 3) - bw) * halfdt;
        a1 = data0.tail(3) - ba;
        a2 = data1.tail(3) - ba;

        data0 = data1;

        // ----- State propagation -----
        // q_Itao_Il
        wtemp = 0.75*w1_halfdt + 0.25*w2_halfdt;
        q_middle_Il = MathUtil::VecToQuat(wtemp);
        wtemp = 0.25*w1_halfdt + 0.75*w2_halfdt;
        q_end_middle = MathUtil::VecToQuat(wtemp);
        q_end_Il = q_middle_Il * q_end_middle;

        // R_Il_Itao
        R_Il_middle = q_middle_Il.matrix();
        R_Il_end = q_end_Il.matrix();

        // sl
        a_Il_middle = R_Il_middle * (a1 + a2) * 0.5;
        a_Il_end = R_Il_end * a2;
        sl = (dt/6) * (a1 + 4*a_Il_middle + a_Il_end);

        // yl
        yl = sl * halfdt;

        // sl_G & yl_G
        R_G_Il = q.matrix();
        sl_G = R_G_Il * sl;
        yl_G = R_G_Il * yl;

        // new q, p, v
        qnew = q * q_end_Il;
        pnew = p + v*dt + yl_G + halfdt*dt*gravity_;
        vnew = v + sl_G + dt * gravity_;

        // ----- Error state covariance propagation -----
        R_G_end = qnew.matrix();
        a_G_begin = R_G_Il * a1;
        a_G_end = R_G_end * a2;

        // PHI
        PHI_q_bg = -R_G_Il * (eye3 + 4*R_Il_middle + R_Il_end) * (dt/6);
        PHI_v_bg = halfdt*halfdt * (MathUtil::VecToSkew(a_G_begin) * R_G_Il + MathUtil::VecToSkew(a_G_end)*R_G_end);
        PHI.block(0, 9, 3, 3) = PHI_q_bg;
        PHI.block(3, 0, 3, 15) << -MathUtil::VecToSkew(yl_G), eye3, eye3*dt, halfdt*PHI_v_bg, halfdt*PHI_q_bg;
        PHI.block(6, 0, 3, 3) = -MathUtil::VecToSkew(sl_G);
        PHI.block(6, 9, 3, 6) << PHI_v_bg, PHI_q_bg;

        // Noise
        Qd = halfdt * (noiseNc_ + PHI * noiseNc_ * PHI.transpose());

        // ----- Refresh -----
        q = qnew;
        p = pnew;
        v = vnew;

        SIGMA_ii = PHI * SIGMA_ii * PHI.transpose() + Qd;
        SIGMA_ic = PHI * SIGMA_ic;
    }

    state_.head(10) << q.w(), q.vec(), p, v;
    errCov_.topLeftCorner(errHeadDim_, errHeadDim_) = (SIGMA_ii+SIGMA_ii.transpose())/2;
    errCov_.topRightCorner(errHeadDim_, SIGMA_N-errHeadDim_) = SIGMA_ic;
    errCov_.bottomLeftCorner(SIGMA_N-errHeadDim_, errHeadDim_) = SIGMA_ic.transpose();
}


void MsckfVio::UpdateFeatureServer(const vector<FeatureObservation> &featureMsgs) {
    const int numFeatures = static_cast<int>(featureServer_.size());
    int numTrackedFeatures = 0;
    for (const auto &it : featureMsgs) {
        const auto featureId = it.id;
        if (featureServer_.find(featureId) == featureServer_.end()) {
            featureServer_.insert(make_pair(featureId, VioFeature(featureId)));
        } else {
            numTrackedFeatures++;
        }
        featureServer_.at(featureId).AddObservation(it);
    }
    trackingRate_ = static_cast<double>(numTrackedFeatures) / static_cast<double>(numFeatures + 1);
}


void MsckfVio::RemoveLostFeatures() {
    /* To become a msckf feature, all following conditions shall be satisfied:
     *      1. lost or tracked length >= max track length
     *      2. has enough observations
     *      3. already initialized or can be initialized (has enough motion and been successfully triangulated)
     */
    vector<size_t> invalid_lost_feature_ids;
    vector<size_t> msckf_feature_ids;

    map<size_t, Matrix<double, 3, 4> > cameraPoses;
    CollectCameraPoses(cameraPoses);

    for (auto &it : featureServer_) {
        bool isLost = it.second.latestFrameId_ != currFrameId_;
        if (isLost) {
            // Must have enough observations for MSCKF update
            if (it.second.observations_.size() < leastObservationNum_) {
                invalid_lost_feature_ids.push_back(it.first);
                continue;
            }

            // Try to initialize it
            if (!it.second.initialized_) {
                if (it.second.Triangulate(cameraPoses, caim_, cam1_)) {
                    it.second.Refine(cameraPoses, caim_, cam1_);
                }
            }

            if (it.second.initialized_) {
                msckf_feature_ids.push_back(it.first);
            } else {
                invalid_lost_feature_ids.push_back(it.first);
            }

        } else if (it.second.observations_.size() >= maxTrackLength_ ) {
            if (!it.second.initialized_) {
                if (it.second.Triangulate(cameraPoses, caim_, cam1_)) {
                    it.second.Refine(cameraPoses, caim_, cam1_);
                }
            }
            if (it.second.initialized_) {
                msckf_feature_ids.push_back(it.first);
            }
        }
    }

    // Msckf update
    set<size_t> involved_frame_ids;
    for (const auto &it : framePoseInfos_) {
        involved_frame_ids.insert(it.first);
    }
    MsckfUpdate(msckf_feature_ids, involved_frame_ids);

    // Remove all lost features and msckf involved features from FeatureServer
    for (const auto featureId : invalid_lost_feature_ids) {
        featureServer_.erase(featureId);
    }

    for (const auto featureId : msckf_feature_ids) {
        featureServer_.erase(featureId);
    }

}


void MsckfVio::PruneImuStates() {
    if (framePoseInfos_.size() < maxSlidingWindowSize_) {
        return;
    }

    map<size_t, Matrix<double, 3, 4> > camera_poses;
    CollectCameraPoses(camera_poses);
    Matrix<double, 3, 4> poseRl = Matrix<double, 3, 4>::Identity();
    if (cam1_ != nullptr) {
        poseRl << Rrl_, prl_;
    }

    // Find two redundant camera states to remove
    vector<size_t> redundant_frame_ids;
    FindRedundantFrames(redundant_frame_ids);

    // Collect related features that have been observed by these two frames.
    vector<size_t> msckf_feature_ids;
    for (auto &it : featureServer_) {
        auto &feature = it.second;

        vector<size_t> related_frame_ids;
        for (const size_t frameId : redundant_frame_ids) {
            if (feature.observations_.find(frameId) != feature.observations_.end()) {
                related_frame_ids.push_back(frameId);
            }
        }

        if (related_frame_ids.size() < 2) {
            continue;
        }

        if (!feature.initialized_) {
            if (feature.Triangulate(camera_poses, caim_, cam1_)) {
                feature.Refine(camera_poses, caim_, cam1_);
            }
        }

        if (feature.initialized_) {
            msckf_feature_ids.push_back(it.first);
        }
    }

    // Msckf update
    set<size_t> involved_frame_ids;
    for (const auto frameId : redundant_frame_ids) {
        involved_frame_ids.insert(frameId);
    }
    MsckfUpdate(msckf_feature_ids, involved_frame_ids);

    // erase related observations
    for (auto &it : featureServer_) {
        for (const auto frameId : redundant_frame_ids) {
            if (it.second.observations_.count(frameId) > 0) {
                it.second.observations_.erase(frameId);
            }
        }
    }

    // Erase redundant frames from state vector & error state covariance matrix & frame pose info map
    VectorXi stateFlags(state_.size()), errCovFlags(errCov_.cols());
    stateFlags.setZero();
    errCovFlags.setZero();

    for (const auto frameId : redundant_frame_ids) {
        auto &poseInfo = framePoseInfos_.at(frameId);
        stateFlags.segment(poseInfo.stateEntry_, 7).setOnes();
        errCovFlags.segment(poseInfo.errCovEntry_, 6).setOnes();
        framePoseInfos_.erase(frameId);
    }
    MathUtil::ReduceEigenVector(state_, stateFlags);
    MathUtil::ReduceEigenMatrix(errCov_, errCovFlags);

    // Update entries
    UpdatePoseInfos(true);
}


void MsckfVio::AugmentState() {
    int stateLen = static_cast<int>(state_.size());
    int errCovLen = static_cast<int>(errCov_.cols());

    // Augment the state vector
    Matrix<double, 7, 1> currQp = state_.head(7);
    state_.conservativeResize(stateLen+7);
    state_.tail(7) = currQp;

    // Augment the error state covariance matrix
    MatrixXd SIGMA_ii = errCov_.topLeftCorner(6, 6);
    MatrixXd SIGMA_ic = errCov_.topRows(6);
    MatrixXd SIGMA_ci = errCov_.leftCols(6);
    errCov_.conservativeResize(errCovLen+6, errCovLen+6);
    errCov_.block(0, errCovLen, errCovLen, 6) = SIGMA_ci;
    errCov_.block(errCovLen, 0, 6, errCovLen) = SIGMA_ic;
    errCov_.bottomRightCorner(6, 6) = SIGMA_ii;

    // Update the frame pose info map: current frame's entry will be changed
    // from the heading IMU state position to the newly added state.
    UpdatePoseInfos(true);

}


void MsckfVio::MsckfUpdate(const vector<size_t> &featureIds, const set<size_t> &involvedFrameIds) {
    if (featureIds.empty() || involvedFrameIds.size() < 2) {
        return;
    }

    // Estimate the rows of the measurement Jacobian. Since some features will be
    // classified as outliers later, the actual Jacobian rows may be smaller than this
    // expected value.
    const bool useStereo = (cam1_ != nullptr);
    int jacobianRows = 0;
    for (const auto featureId : featureIds) {
        const auto &feature = featureServer_.at(featureId);
        int featureJacobianRows = 0;
        for (const auto &obser : feature.observations_) {
            if (involvedFrameIds.count(obser.first) > 0) {
                featureJacobianRows += (useStereo && obser.second.isStereo) ? 3 : 2;
            }
        }
        jacobianRows += featureJacobianRows - 3;
    }

    // cout << "jacobian rows: " << jacobianRows << endl;
    const int jacobianCols = static_cast<int>(errCov_.cols());
    MatrixXd Ho(jacobianRows, jacobianCols);
    VectorXd ro(jacobianRows);
    int rowIdx = 0;
    int numInliers = 0;
    for (const size_t featureId : featureIds) {
        const auto &feature = featureServer_.at(featureId);

        // Involved frames that have observed this feature
        vector<size_t> frameIds;
        for (const auto &obser : feature.observations_) {
            if (involvedFrameIds.count(obser.first) > 0) {
                frameIds.push_back(obser.first);
            }
            if (framePoseInfos_.find(obser.first) == framePoseInfos_.end()) {
                cerr << "This should not happen! Something goes wrong!" << endl;
                exit(-1);
            }
        }
        if (frameIds.size() < 2) {
            cerr << "No enough involved observations for this feature. This should not happen!" << endl;
            continue;
        }

        // Compute measurement Jacobian and residuals for this feature
        MatrixXd Hoi;
        VectorXd roi;
        if (!FeatureJacobian(featureId, frameIds, Hoi, roi) ) {
//            cerr << "feature jacobian calculation failed!" << endl;
            continue;
        }

        // mahalanobis distance test
        if (GatingTest(Hoi, roi)) {
            // then this feature is an inlier
            int iRows = static_cast<int>(Hoi.rows());
            Ho.block(rowIdx, 0, iRows, jacobianCols) = Hoi;
            ro.segment(rowIdx, iRows) = roi;
            numInliers++;
            rowIdx += iRows;
            if (frameIds.size() > 2) {
                lostMapPoints_.push_back(feature.xw_);
            }
        }
    }

    // cout << "num inliers: " << numInliers << endl;
    if (rowIdx < 1) {
        cout << "no inliers!" << endl;
        return;
    }

    Ho.conservativeResize(rowIdx, jacobianCols);
    ro.conservativeResize(rowIdx);

    // Compress if Ho is not a flat matrix
    if (Ho.rows() > Ho.cols()) {
        Eigen::JacobiRotation<double> givensHo;
        for (int n=0; n<Ho.cols(); n++) {
            for (int m=(int)Ho.rows()-1; m>n; m--) {
                // Givens matrix G
                givensHo.makeGivens(Ho(m-1,n), Ho(m,n));
                (Ho.block(m-1,n,2,Ho.cols()-n)).applyOnTheLeft(0, 1, givensHo.adjoint());
                (ro.block(m-1,0,2,1)).applyOnTheLeft(0, 1, givensHo.adjoint());
            }
        }
        Ho.conservativeResize(Ho.cols(), Ho.cols());
        ro.conservativeResize(Ho.cols());
    }

    // Update state and error covariance
    MatrixXd PHt = errCov_ * Ho.transpose();
    MatrixXd S = Ho * PHt;
    for (int k=0; k<(int)S.rows(); k++) {
        S(k,k) += imageFeatureNoise_;
    }

    // Kalman gain K = P * H' / S
    MatrixXd Kt = S.ldlt().solve(PHt.transpose());
    MatrixXd K = Kt.transpose();

    // TODO: Deal with Gauge freedom.
    // Update state
    const size_t refId = framePoseInfos_.begin()->first;
    const int refEntry = framePoseInfos_[refId].stateEntry_;
    const Vector3d ref_p = state_.segment(refEntry + 4, 3); // p_W_I0
    const Quaterniond ref_q(state_(refEntry), state_(refEntry+1), state_(refEntry+2), state_(refEntry+3)); // q_I0_W
    const Matrix3d ref_R = ref_q.matrix(); // R_W_I0

    VectorXd dx = K * ro;
    CalcNewState(dx, state_);

    Vector3d new_ref_p = state_.segment(refEntry + 4, 3); // p_W'_I0
    Quaterniond new_ref_q(state_(refEntry), state_(refEntry+1), state_(refEntry+2), state_(refEntry+3)); // q_I0_W'
    Matrix3d new_ref_R = new_ref_q.matrix(); // R_W'_I0

    Vector3d ypr = MathUtil::R2ypr(ref_R);
    Vector3d new_ypr = MathUtil::R2ypr(new_ref_R);
    double diffYaw = ypr(0) - new_ypr(0);
    Matrix3d diffR = MathUtil::ypr2R(Vector3d(diffYaw, 0, 0));  // R_W_W'
    if (fabs(fabs(ypr(1)) * 180.0 / CV_PI - 90) < 1.0 || fabs(fabs(new_ypr(1)) * 180.0 / CV_PI - 90) < 1.0) {
        cerr << "Warning: singular euler point!" << endl;
        diffR = ref_R * new_ref_R.transpose();
    }

    for (const auto &it : framePoseInfos_) {
        int et = it.second.stateEntry_;
        Matrix3d R_W_Ik = diffR * Quaterniond(state_(et), state_(et+1), state_(et+2), state_(et+3)).matrix(); // R_W_W' * R_W'_Ik
        Vector3d p_W_Ik = ref_p + diffR * (state_.segment(et+4, 3) - new_ref_p); // p_W_I0 + R_W_W'(p_W'_Ik - p_W'_I0);
        Quaterniond q_Ik_W(R_W_Ik);
        state_.segment(et, 4) << q_Ik_W.w(), q_Ik_W.x(), q_Ik_W.y(), q_Ik_W.z();
        state_.segment(et+4, 3) = p_W_Ik;
    }
    Vector3d v_W_Ik = diffR * state_.segment(7, 3); // R_W_W' * v_W'_Ik;
    state_.segment(7, 3) = v_W_Ik;

    // Update error state covariance
    errCov_ -= PHt * K.transpose();

    // Make sure that the covariance matrix is symmetric
    MatrixXd errCovT = errCov_.transpose();
    errCov_ += errCovT;
    errCov_ *= 0.5;

    // Update frame pose infos
    UpdatePoseInfos(false);
}


bool MsckfVio::FeatureJacobian(size_t featureId, const vector<size_t> &frameIds, MatrixXd &Hoi, VectorXd &roi) {
    const bool useStereo = (cam1_ != nullptr);

    const auto &feature = featureServer_.at(featureId);
    const Vector3d xw = feature.xw_;    // 3D position in world frame
    int jacobianRows = 0;
    for (const auto &frameId : frameIds) {
        jacobianRows += (useStereo && feature.observations_.at(frameId).isStereo) ? 3 : 2;
    }

    const int LS = (int)errCov_.cols();

    Matrix<double, Eigen::Dynamic, 3> J_il, H_fil;

    MatrixXd Hif(jacobianRows, 3);    // Jacobian with respect to the feature's 3D position
    MatrixXd Hii = MatrixXd::Zero(jacobianRows, LS); // Jacobian with respect to the error state
    VectorXd ri(jacobianRows);    // residuals

    double stereoBaseline = prl_.norm();
    int rowIdx = 0;
    for (const auto &frameId : frameIds) {
        const auto &obser = feature.observations_.at(frameId);

        const PoseInfo &poseInfo = framePoseInfos_.at(frameId);
        const int entry = poseInfo.errCovEntry_;
        const Matrix3d &R_C_G = poseInfo.R_C_W_;
        const Vector3d &p_G_C = poseInfo.p_W_C_;
        const Vector3d &p_G_I = poseInfo.p_W_I_;

        // 3D position in camera frame
        Vector3d xc = R_C_G * (xw - p_G_C);

#if USE_UNIT_SPHERE_ERROR
        // Observation ray
        Vector3d obserRay = obser.ray0;

        // Tangent base
        Matrix<double, 2, 3> tangentBase;
        Vector3d tmp(0, 0, 1);
        if (obserRay == tmp) {
            tmp << 1, 0, 0;
        }
        Vector3d b1 = (tmp - obserRay * (obserRay.transpose() * tmp)).normalized();
        Vector3d b2 = obserRay.cross(b1);
        tangentBase.row(0) = b1.transpose();
        tangentBase.row(1) = b2.transpose();

        Vector3d rayc = xc.normalized();

        const int dim = (useStereo && obser.isStereo) ? 3 : 2;
        J_il.resize(dim, 3);
        J_il.topRows(2) = tangentBase * MathUtil::NormalizationJacobian(xc);
        ri.segment(rowIdx, 2) = tangentBase * (obserRay - rayc);
        if (dim > 2) {
            J_il.bottomRows(1) = stereoBaseline * MathUtil::InverseNormJacobian(xc);
            ri(rowIdx + 2) = stereoBaseline * ( 1.0 / obser.stereoDepth - 1.0 / xc.norm());
        }

#else
        // Pinhole projection to image plane
        Matrix3d projection_jacobian;
        Vector2d h;
        if (!caim_->PinholeProjection(xc, h, projection_jacobian, true) ) {
            return false;
        }
        J_il = projection_jacobian.topRows(2);

        // residual
        ri.segment(rowIdx, 2) = Vector2d(obser.upt0.x, obser.upt0.y) - h;
		
		const int dim = 2;

#endif

        H_fil = J_il * R_C_G;
        Hif.block(rowIdx, 0, dim, 3) = H_fil;
        Hii.block(rowIdx, entry, dim, 6) << H_fil * MathUtil::VecToSkew(xw-p_G_I), -H_fil;
        rowIdx += dim;
    }

	if (rowIdx < jacobianRows) {
		cerr << "FeatureJacobian: rowIdx < expected jacobian rows!" << endl;
		Hii.conservativeResize(rowIdx, Hii.cols());
		Hif.conservativeResize(rowIdx, 3);
		ri.conservativeResize(rowIdx);
	}
		
    // Project Hii & ri to the left null space of H_fi
    int leftRows = rowIdx - 3;

# if 0
    Eigen::ColPivHouseholderQR<MatrixXd> eigenQr(Hif);
    MatrixXd QrQ = eigenQr.matrixQ();
    MatrixXd ViT = QrQ.rightCols(leftRows).transpose();
    Hoi = ViT * Hii;
    roi = ViT * ri;
#else
    // use Givens rotation
    Eigen::JacobiRotation<double> givensHoi;
    for (int n = 0; n < Hif.cols(); ++n) {
        for (int m = (int) Hif.rows() - 1; m > n; m--) {
            // Givens matrix G
            givensHoi.makeGivens(Hif(m - 1, n), Hif(m, n));
            (Hif.block(m - 1, n, 2, Hif.cols() - n)).applyOnTheLeft(0, 1, givensHoi.adjoint());
            (Hii.block(m - 1, 0, 2, Hii.cols())).applyOnTheLeft(0, 1, givensHoi.adjoint());
            (ri.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, givensHoi.adjoint());
        }
    }

    // The Hif jacobian max rank is 3 if it is a 3d position, thus size of the left nullspace is Hf.rows()-3
    Hoi = Hii.block(Hif.cols(), 0, Hii.rows()-Hif.cols(), Hii.cols());
    roi = ri.block(Hif.cols(),0,ri.rows()-Hif.cols(),ri.cols());
#endif

    return true;
}


bool MsckfVio::GatingTest(const Eigen::MatrixXd &H, const Eigen::VectorXd &r) {
    const int dim = (int)H.rows();

    MatrixXd S = H * errCov_ * H.transpose();
    for (int i = 0; i < dim; i++) {
        S(i, i) += imageFeatureNoise_;
    }
    double mahanalobisDistance = r.transpose() * S.inverse() * r;
    if (mahanalobisDistance < MAHALA95_TABLE[dim-1]) {
        return true;
    }
    return false;
}


void MsckfVio::FindRedundantFrames(vector<size_t> &redundantFrameIds) {
    redundantFrameIds.clear();

    if (framePoseInfos_.size() < 5) {
        return;
    }

    // Select the forth frame from the end as keyframe
    auto key_state_iter = framePoseInfos_.end();
    for (int i = 0; i < 4; ++i) {
        key_state_iter--;
    }

    auto first_state_iter = framePoseInfos_.begin();

    auto state_iter = key_state_iter;
    state_iter++;

    // key camera pose
    const Matrix3d keyRot = key_state_iter->second.R_C_W_;
    const Vector3d keyPos = key_state_iter->second.p_W_C_;

    for (int i = 0; i < 2; ++i) {
        const Matrix3d rot = state_iter->second.R_C_W_;
        const Vector3d pos = state_iter->second.p_W_C_;
        double dist = (pos - keyPos).norm();
        double angle = AngleAxisd(rot.transpose() * keyRot).angle();
        if (dist < translationThresh_ && angle < rotationThresh_ && trackingRate_ > trackingRateThresh_) {
            redundantFrameIds.push_back(state_iter->first);
            state_iter++;
        } else {
            redundantFrameIds.push_back(first_state_iter->first);
            first_state_iter++;
            state_iter--;
            state_iter--;
        }
    }

    std::sort(redundantFrameIds.begin(), redundantFrameIds.end());
}


void MsckfVio::CalcNewState(const VectorXd &dx, VectorXd &x) {
    // Update poses
    for (const auto &it : framePoseInfos_) {
        const size_t frameId = it.first;
        const PoseInfo &poseInfo = it.second;

        // new orientation
        Vector4d qOld = x.segment(poseInfo.stateEntry_, 4);
        Vector3d dq = dx.segment(poseInfo.errCovEntry_, 3);
        Vector4d qNew;
        MathUtil::IncreQuat(qOld, dq, qNew);
        x.segment(poseInfo.stateEntry_, 4) = qNew;

        // new position
        x.segment(poseInfo.stateEntry_ + 4, 3) += dx.segment(poseInfo.errCovEntry_ + 3, 3);
    }

    // Update the heading IMU state (the pose has been updated already)
    x.segment(7, headDim_ - 7) += dx.segment(6, headDim_ - 7);
}


void MsckfVio::UpdatePoseInfos(bool updateEntries) {
    if (updateEntries) {
        // Re-calculate entries
        int stateEntry = headDim_;
        int errCovEntry = errHeadDim_;
        for (auto &iter : framePoseInfos_) {
            if (iter.first == currFrameId_) {
                iter.second.stateEntry_ = 0;
                iter.second.errCovEntry_ = 0;
            } else {
                iter.second.stateEntry_ = stateEntry;
                iter.second.errCovEntry_ = errCovEntry;
                stateEntry += 7;    // quaternion(4) + position(3)
                errCovEntry += 6;   // quaternion_error(3) + position_error(3)
            }
        }
    }

    // Calculate poses
    for (auto &it : framePoseInfos_) {
        size_t frameId = it.first;
        PoseInfo &poseInfo = it.second;
        const Matrix<double, 7, 1> &qp = state_.segment(poseInfo.stateEntry_, 7);

        Quaterniond q_I_G(qp(0), qp(1), qp(2), qp(3));
        poseInfo.R_I_W_ = q_I_G.matrix().transpose();
        poseInfo.p_W_I_ = qp.tail(3);
        poseInfo.R_C_W_ = caim_->Rci_ * poseInfo.R_I_W_;
        poseInfo.p_W_C_ = poseInfo.p_W_I_ + poseInfo.R_I_W_.transpose() * caim_->pic_;
    }
}


void MsckfVio::CollectCameraPoses(map<size_t, Matrix<double, 3, 4>> &camPoses) {
    camPoses.clear();

    UpdatePoseInfos(false);

    for (const auto &it : framePoseInfos_) {
        const PoseInfo &poseInfo = it.second;
        Matrix<double, 3, 4> pose;
        pose << poseInfo.R_C_W_, -poseInfo.R_C_W_ * poseInfo.p_W_C_;
        camPoses[it.first] = pose;
    }
}


vector<Matrix4d> MsckfVio::QuerySlidingWindowPoses() {
    vector<Matrix4d> poses;
    for (const auto &it : framePoseInfos_) {
        Matrix4d pose = Matrix4d::Identity();
        pose.topLeftCorner(3, 3) = it.second.R_C_W_.transpose();
        pose.topRightCorner(3, 1) = it.second.p_W_C_;
        poses.push_back(pose);
    }
    return poses;
}

}//namespace hybrid_msckf {
