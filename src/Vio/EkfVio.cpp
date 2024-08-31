//
//  EkfVio.cpp
//  DearVins

#include <sstream>
#include "Vio/EkfVio.hpp"
#include "Utils/MathUtil.h"
#include <sstream>

using namespace cv;
using namespace std;
using namespace Eigen;

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

namespace inslam {

EkfVio::EkfVio(const nlohmann::json &config, shared_ptr<Caimura> cam0, shared_ptr<Caimura> cam1)
: cam_(std::move(cam0)), cam1_(std::move(cam1))
{
    // Parameters
    slidingWindowSize_ = config["sliding_window_size"];
    maxNumInstateFeatures_ = config["num_instate_features"];
    useUnitSphereError_ = config["use_unit_sphere_error"].get<bool>();
    useOnePointRansac_ = config["use_one_point_ransac"].get<bool>();
    useOosFeatures_ = config["use_oos_features"].get<bool>();
    if ((cam_->IsEquirectangular() || cam_->IsEquidistant()) && !useUnitSphereError_) {
        cerr << "pano camera or equidistant camera only support "
                "unit sphere reprojection errors for now!" << endl;
        useUnitSphereError_ = true;
    }
    cout << "EkfVio config: " << "\n\tsliding window size: " << slidingWindowSize_
           << "\n\tin state features: " << maxNumInstateFeatures_
           << "\n\tuse unit sphere reprojection errors: " << useUnitSphereError_
           << "\n\tuse one point ransac: " << useOnePointRansac_
           << "\n\tuse out-of-state features: " << useOosFeatures_ << endl;

    // Set gravity vector & Nc
    gravity_ << 0, 0, -9.81;
    SetNoiseNc();

    featureMeasurementNoise_ = 1.0; // pixels
    if (useUnitSphereError_) {
        featureMeasurementNoise_ *= std::pow(cam_->GetAngularResolution(), 2);
    }

    // Stereo pose
    if (cam1_ != nullptr) {
        Rrl_ = cam1_->Rci_ * cam_->Ric_;
        prl_ = cam1_->pci_ + cam1_->Rci_ * cam_->pic_;
    }

    // Grid the image to uniform distribute insfeatures
    int cellSize = static_cast<int>(std::sqrt(cam_->width() * cam_->height() / (maxNumInstateFeatures_ * 2.0)));
    cellSize = std::max(cellSize, 1);
    featureGrid_ = std::make_shared<FeatureGrid>(cam_->width(), cam_->height(), cellSize);

    groups_.clear();
    instateFeatures_.clear();
    mvMapPoints.clear();
}


void EkfVio::Initialize(Eigen::Vector4d &q, Eigen::Vector3d &p, Eigen::Vector3d &v, int frame_id, double timestamp)
{
    currFrameId_ = frame_id;
    currTimestamp_ = timestamp;

    // Set state vector
    state_.resize(16);
    state_ << q, p, v, cam_->bg_, cam_->ba_;

    // Set covariance
    errCov_.resize(15, 15);
    errCov_.setZero();

    int i;
    for (i=0; i<3; i++)     errCov_(i,i) = 3e-3;
    for (i=3; i<6; i++)     errCov_(i,i) = 1e-8;
    for (i=6; i<9; i++)     errCov_(i,i) = 4e-4;
    for (i=9; i<12; i++)    errCov_(i,i) = 1e-6;
    for (i=12; i<15; i++)   errCov_(i,i) = 1e-3;
}


void EkfVio::Run(const vector<FeatureObservation> &featureMsgs,
                   double t_img, int frame_id,
                   const std::vector<Eigen::Matrix<double, 7, 1>> &vimu)
{

    currFrameId_ = frame_id;
    currTimestamp_ = t_img;

    // Propagation using IMU data
    Propagate(vimu);

    // Update feature server
    UpdateFeatureServer(featureMsgs);

    // Track instate features
    vector<size_t> ok_indices;
    TrackInsfeatures(ok_indices);
    cout << "Tracked insfeatures " << ok_indices.size() << endl;
    numTrackedInsfeatures_ = static_cast<int>(ok_indices.size());

    // Calculate h, H and S for tracekd instate features
    // Note that only the well predicted insfeatures' indices are kept
    vector<size_t> compatible_indices = PredictSHh(ok_indices);
    numCompatibleInsfeatures_ = static_cast<int>(compatible_indices.size());
    cout << "Compatible insfeatures " << numCompatibleInsfeatures_ << endl;

    // One point ransac
    vector<size_t> inlier_indices = useOnePointRansac_ ?
            OnePointRansac(compatible_indices) : compatible_indices;
    numInlierInsfeatures_ = static_cast<int>(inlier_indices.size());

    // Update the filter with inlier in-state features
    if (useOosFeatures_) {
        HybridUpdate(inlier_indices);
    } else {
        EKFUpdate(inlier_indices);
    }

    // Remove groups & insfeatures from state / err covariance / groups / instate features
    vector<uchar> flags(instateFeatures_.size(), 0);
    for (const auto &idx : inlier_indices) {
        flags[idx] = 255;
    }
    RemoveGFs(flags);

    // Add a new group of features
    numNewInsfeatures_ = AddNewGFs();
    cout << "new insfeatures " << numNewInsfeatures_ << endl;

    // Remove lost features from feature server
    vector<size_t> lostFeatureIds;
    for (auto &it : featureServer_) {
        if (it.second.latestFrameId_ != currFrameId_) {
            lostFeatureIds.push_back(it.first);
        }
        it.second.initialized_ = false;
    }

    for (const auto featureId : lostFeatureIds) {
        featureServer_.erase(featureId);
    }

}


int EkfVio::AddNewGFs() {

    // Do not insert a new group if (the newly added insfeatures < 3)
    int nwanted = maxNumInstateFeatures_ - (int)instateFeatures_.size();
    if (nwanted < 3)
        return 0;

    // Grab some active features (that have not been used by the EkfVio) from the pool.
    vector<size_t> candidates;
    for (const auto &it : featureServer_) {
        if (it.second.isInState_) { // already in state
            continue;
        }
        if (it.second.latestFrameId_ != currFrameId_) { // lost
            continue;
        }
        candidates.push_back(it.first);
    }

    if (candidates.size() < 3) {
        return 0;
    }

    // ---------- Add group pose into state ------------
    int stateLen  = (int)state_.size();
    int errCovLen  = (int)errCov_.cols();

    Group gro{};
    gro.frameId  = currFrameId_;
    gro.count = 0;
    gro.stateEntry = stateLen;  // the entry will be calculated by UpdateEntries later
    gro.errCovEntry = errCovLen;
    groups_[gro.frameId] = gro;

    // Step1. Add group pose and its covariance to the end
    Matrix<double, 7, 1> qp = state_.head(7);
    state_.conservativeResize(stateLen+7);
    state_.tail(7) = qp;

    MatrixXd cov_ii = errCov_.topLeftCorner(6, 6);
    MatrixXd cov_ic = errCov_.topRows(6);
    MatrixXd cov_ci = errCov_.leftCols(6);
    errCov_.conservativeResize(errCovLen+6, errCovLen+6);
    errCov_.block(0, errCovLen, errCovLen, 6) = cov_ci;
    errCov_.block(errCovLen, 0, 6, errCovLen+6) << cov_ic, cov_ii;

    // Step2. Swap this new group with old features
    const int numFeatures = static_cast<int>(instateFeatures_.size());
    VectorXd featureStates = state_.segment(stateLen-numFeatures, numFeatures);
    state_.tail(numFeatures) = featureStates;
    state_.segment(stateLen - numFeatures, 7) = qp;

    const int la = errCovLen - numFeatures;
    const int lb = numFeatures;
    const int lc = 6;

    const MatrixXd cov_aa = errCov_.topLeftCorner(la, la);
    const MatrixXd cov_ab = errCov_.block(0, la, la, lb);
    const MatrixXd cov_ac = errCov_.topRightCorner(la, lc);
    const MatrixXd cov_bb = errCov_.block(la, la, lb, lb);
    const MatrixXd cov_bc = errCov_.block(la, la+lb, lb, lc);
    const MatrixXd cov_cc = errCov_.bottomRightCorner(lc, lc);
    errCov_ << cov_aa, cov_ac, cov_ab,
                    cov_ac.transpose(), cov_cc, cov_bc.transpose(),
                    cov_ab.transpose(), cov_bc, cov_bb;

    UpdateEntries();

    int nnew = AddNewInsfeatures();

    return nnew;

}


int EkfVio::AddNewInsfeatures() {
    int numWant = maxNumInstateFeatures_ - (int)instateFeatures_.size();
    if (numWant < 1) {
        return 0;
    }

    // Put all frame poses in a map structure
    map<size_t, Matrix<double,3,4> > framePoses;
    RefreshPoses();
    for (const auto &it : groups_) {
        const SensorPose &pose = it.second.pose;
        Matrix<double, 3, 4> Pcw;
        Pcw << pose.Rcw, -pose.Rcw * pose.pwc;
        framePoses[it.first] = Pcw;
    }

    /* Select features to add into state:
     * 1. active (still tracked) and hasn't been used as in-state feature (necessary)
     * 2. can be well triangulated (optional)
     */
    // Draw instateFeature grid
    featureGrid_->ResetGrid(false);
    for (auto &insFea : instateFeatures_) {
        if (insFea.measurement_.frameId == static_cast<size_t>(currFrameId_)) {
            featureGrid_->SetGridOccpuancy(insFea.measurement_.pt0);
        }
    }

    bool isFull = false;
    vector<size_t> candidates, occpuyCandidates, poorCandidates;
    for (auto &it : featureServer_) {
        auto &feature = it.second;
        if (static_cast<int>(feature.latestFrameId_) != currFrameId_) { // lost
            continue;
        }
        if (feature.isInState_) { // already in state
            continue;
        }

        vector<size_t> frameIds;
        for (const auto &ot : feature.observations_) {
            if (groups_.find(ot.first) != groups_.end()) {
                frameIds.push_back(ot.first);
            }
        }
        if (frameIds.empty()) { // no group has observed this feature
            continue;
        }

        // set the latest related group as anchor group
        feature.anchorFrameId_ = frameIds.back();

        auto &pt0 = feature.observations_[currFrameId_].pt0;
        if (feature.InitializePosition(framePoses, cam_, cam1_, false)) {
            if(featureGrid_->GetOccupancyState(pt0)) { // Occpuancy
                occpuyCandidates.push_back(it.first);
            }
            else { // Idle
                featureGrid_->SetGridOccpuancy(pt0);
                candidates.push_back(it.first);
                if(static_cast<int>(candidates.size()) >= numWant) {
                    isFull = true;
                    break;
                }
            }
        } else {
            poorCandidates.push_back(it.first);
        }
    }

    if (!isFull) {
        int numRemainWant = numWant - static_cast<int>(candidates.size());
        if (static_cast<int>(occpuyCandidates.size()) >= numRemainWant) { // Add occupied goodCandidate(triangulated) first
            //random_shuffle(occpuyCandidates.begin(), occpuyCandidates.end());
            occpuyCandidates.resize(numRemainWant);
            isFull = true;
        }
        else if (!poorCandidates.empty()) { // Add poorCandidates
            // Add idle feature
            vector<size_t> occpuyPoorCandidates;
            for (int i = 0;i < static_cast<int>(poorCandidates.size()); i++) {
                size_t &feaId = poorCandidates[i];
                auto &pt0 = featureServer_.at(feaId).observations_[currFrameId_].pt0;
                if (!featureGrid_->GetOccupancyState(pt0)) {
                    featureGrid_->SetGridOccpuancy(pt0);
                    occpuyCandidates.push_back(feaId);
                }
                else {
                    occpuyPoorCandidates.push_back(feaId);
                }
                if (static_cast<int>(occpuyCandidates.size()) >= numRemainWant) {
                    isFull = true;
                    break;
                }
            }

            // Add occupied feature
            numRemainWant -= static_cast<int>(occpuyCandidates.size());
            if(!isFull && static_cast<int>(occpuyPoorCandidates.size()) > numRemainWant) {
                random_shuffle(occpuyPoorCandidates.begin(), occpuyPoorCandidates.end());
                occpuyPoorCandidates.resize(numRemainWant);
            }
            candidates.insert(candidates.end(), occpuyPoorCandidates.begin(), occpuyPoorCandidates.end());
        }
        candidates.insert(candidates.end(), occpuyCandidates.begin(), occpuyCandidates.end());
    }

    int nnew = std::min(static_cast<int>(candidates.size()), numWant);
    VectorXd invDepths(nnew);
    VectorXd invDepthSigmas(nnew);

    for (int i = 0; i < nnew; ++i) {
        size_t featureId = candidates[i];
        auto &feature = featureServer_.at(featureId);
        auto &anchorGroup = groups_.at(feature.anchorFrameId_);
        const auto &anchorObser = feature.observations_.at(feature.anchorFrameId_);

        feature.isInState_ = true;
        anchorGroup.count++;

        Insfeature insfeat;
        insfeat.featureId = featureId;
        insfeat.anchorFrameId = feature.anchorFrameId_;
        insfeat.anchorRay = anchorObser.ray0;
        instateFeatures_.push_back(insfeat);

        // inverse depth
        double rho = 0.5, rhoSigma = 1.0;
        if (feature.initialized_) {
            feature.RefinePosition(framePoses, cam_);
            Vector3d xc = anchorGroup.pose.Rcw * (feature.xw_ - anchorGroup.pose.pwc);
            double invDepth = 1.0 / xc.norm();
            rho = min(max(invDepth, cam_->minInverseDepth_), cam_->maxInverseDepth_);
            rhoSigma = min(max(invDepth*3, 0.2), 1.0);
        }
        invDepths(i) = rho;
        invDepthSigmas(i) = rhoSigma;
    }

    // Add features' inverse depths to the tail of state vector
    const auto stateLen = static_cast<int>(state_.size());
    const auto errCovLen = static_cast<int>(errCov_.cols());
    state_.conservativeResize(stateLen+nnew);
    state_.tail(nnew) = invDepths;

    // Augment covariance matrix
    errCov_.conservativeResize(errCovLen+nnew, errCovLen+nnew);
    errCov_.bottomRows(nnew).setZero();
    errCov_.rightCols(nnew).setZero();
    errCov_.bottomRightCorner(nnew, nnew) = invDepthSigmas.asDiagonal();

    UpdateEntries();

    return nnew;
}


void EkfVio::Propagate(const std::vector<Eigen::Matrix<double, 7, 1> > &vimu)
{
    if (vimu.empty())   return;

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
    const int covLen = (int)errCov_.cols();
    MatrixXd cov_ii = errCov_.topLeftCorner(15, 15);
    MatrixXd cov_ic = errCov_.topRightCorner(15, covLen-15);

    Matrix3d PHI_q_bg, PHI_v_bg;
    MatrixXd PHI = Eigen::MatrixXd::Identity(15, 15), Qd;

    Matrix<double,7,1> data0 = vimu[0], data1;

    for (size_t k=0; k<vimu.size()-1; k++)
    {
        data1 = vimu[k+1];
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

        // PHI = partial(x_imu_(l+1) ) / partial(x_imu_l)
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

        cov_ii = PHI * cov_ii * PHI.transpose() + Qd;   // cov(imu state, imu state)
        cov_ic = PHI * cov_ic;  // cov(imu state, other state)
    }

    state_.head(10) << q.w(), q.vec(), p, v;
    errCov_.topLeftCorner(15, 15) = (cov_ii+cov_ii.transpose())/2;  // cov(imu state, imu state)
    errCov_.topRightCorner(15, covLen-15) = cov_ic; // cov(imu state, other state)
    errCov_.bottomLeftCorner(covLen-15, 15) = cov_ic.transpose(); // cov(other state, imu state)

}


void EkfVio::UpdateFeatureServer(const vector<FeatureObservation> &featureMsgs)
{
    for (const auto &it : featureMsgs) {
        const auto featureId = it.id;
        if (featureServer_.find(featureId) == featureServer_.end()) {
            featureServer_.insert(make_pair(featureId, VioFeature(featureId)));
        }
        featureServer_.at(featureId).AddObservation(it);
    }
}


void EkfVio::TrackInsfeatures(std::vector<size_t> &ok_indices)
{
    ok_indices.clear();

    int stereoObserCount = 0, monoObserCount = 0;
    for (size_t i=0; i<instateFeatures_.size(); i++)
    {
        Insfeature &insfeat = instateFeatures_[i];
        const auto &feature = featureServer_.at(insfeat.featureId);
        if (feature.latestFrameId_ == currFrameId_) {
            const auto &obser = feature.observations_.at(currFrameId_);
            ok_indices.push_back(i);
            Vector2d upt(obser.upt0.x, obser.upt0.y);
            insfeat.SetObservation(obser);
            if (obser.isStereo) {
                stereoObserCount++;
            } else {
                monoObserCount++;
            }
        }
    }
    cout << "stereo observations: " << stereoObserCount << ", mono observations: " << monoObserCount << endl;
}


vector<size_t> EkfVio::PredictSHh(std::vector<size_t> &ok_indices)
{
    vector<size_t> compatibleIndices;
    if (ok_indices.empty()) {
        return compatibleIndices;
    }

    // Refresh poses
    RefreshPoses();

    // Frequently used variables
    Vector3d xcj, xw, xcl, rxcl;

    Matrix<double, Eigen::Dynamic, 3> J_il, H_fil;

    double stereoBaseline = prl_.norm();

    const MatrixXd& cov_x_Il = errCov_.leftCols(6);
    const size_t LS = errCov_.cols();

    // For each tracked instate feature
    for (const size_t idx : ok_indices)
    {
        Insfeature &insfeat = instateFeatures_[idx];
        const Group &anchorGroup = groups_.at(insfeat.anchorFrameId);

        // Check inverse depth
        double rho = state_(insfeat.stateEntry);
        if (rho<cam_->minInverseDepth_ || rho>cam_->maxInverseDepth_)
            continue;

        // Anchor sensor pose
        const SensorPose& posej = anchorGroup.pose;

        // Feature's 3D position in anchor camera / world / current camera frame
        xcj = (1.0/rho) * insfeat.anchorRay;
        xw  = posej.Rwc * xcj + posej.pwc;
        xcl = posel_.Rcw * (xw - posel_.pwc);

        // Check 3D position in current camera frame & Projection
        Vector3d rayl = xcl.normalized();
        Vector2d uv;

        cam_->Reproject(xcl, uv, false);
        FeatureObservation predObser;
        predObser.upt0 = Point2f(static_cast<float>(uv(0)), static_cast<float>(uv(1)) );
        predObser.ray0 = rayl;
        if (insfeat.measurement_.isStereo) {
            rxcl = Rrl_ * xcl + prl_;
            Vector2d ruv;
            cam1_->Reproject(rxcl, ruv, false);
            predObser.upt1 = Point2f(static_cast<float>(ruv(0)), static_cast<float>(ruv(1)));
            predObser.ray1 = rxcl.normalized();
            predObser.stereoDepth = xcl.norm();
            predObser.isStereo = true;
        }
        insfeat.SetPrediction(predObser);
        insfeat.CalcInnovation(useUnitSphereError_, compressStereoError_,
                               cam_->fx(), stereoBaseline);

        // Compute projection Jacobian: partial(innovation) / partial
        const size_t dim = insfeat.innovation_.size();
        J_il.resize(dim, 3);
        if (useUnitSphereError_) {
            J_il.topRows(2) = insfeat.tangentBase0_ * MathUtil::NormalizationJacobian(xcl);
        } else {
            J_il.topRows(2) = cam_->ProjectionJacobian(xcl);
        }

        if (dim > 2) {
            if (compressStereoError_) {
                if (useUnitSphereError_) {
                    J_il.bottomRows(1) = stereoBaseline * MathUtil::InverseNormJacobian(xcl);
                } else {
                    J_il.bottomRows(1) << 0, 0, -cam_->fx() * stereoBaseline / (xcl(2) * xcl(2));
                }
            } else {
                if (useUnitSphereError_) {
                    J_il.bottomRows(2) = insfeat.tangentBase1_ * MathUtil::NormalizationJacobian(rxcl) * Rrl_;
                } else {
                    J_il.bottomRows(2) = cam1_->ProjectionJacobian(rxcl) * Rrl_;
                }
            }
        }

        // Measurement Jacobian
        H_fil = J_il * cam_->Rci_ * posel_.Riw;
        insfeat.H_Ij.resize(dim, 6);
        insfeat.H_Il.resize(dim, 6);
        insfeat.H_Ij << -H_fil*MathUtil::VecToSkew(xw - posej.pwi),  H_fil;
        insfeat.H_Il <<  H_fil*MathUtil::VecToSkew(xw - posel_.pwi), -H_fil;
        insfeat.H_rho =  -H_fil * posej.Rwc * xcj / rho;

        // Innovation S = H * SIGMA * H' + R
        const int et2 = anchorGroup.errCovEntry;    // anchor pose error's location in err covariance matrix
        const int et3 = insfeat.errCovEntry;    // inverse depth error's location in err covariance matrix

        insfeat.PHt = cov_x_Il * insfeat.H_Il.transpose()
                    + errCov_.block(0, et2, LS, 6) * insfeat.H_Ij.transpose()
                    + errCov_.col(et3) * insfeat.H_rho.transpose();

        insfeat.S = insfeat.H_Il  * insfeat.PHt.topRows(6)
                  + insfeat.H_Ij  * insfeat.PHt.middleRows(et2, 6)
                  + insfeat.H_rho * insfeat.PHt.row(et3);
        for (int j = 0; j < dim; ++j) {
            insfeat.S(j, j) += featureMeasurementNoise_;
        }

        // Chi-Square error checking
        const VectorXd &err =insfeat.innovation_;
        double chi_err = err.transpose() * insfeat.S.inverse() * err;
        double chi_thresh = MAHALA95_TABLE[dim];
        if (chi_err < chi_thresh) {
            compatibleIndices.push_back(idx);
        }
    }

    return compatibleIndices;
}


vector<size_t> EkfVio::OnePointRansac(const std::vector<size_t> &ok_indices)
{
    vector<uchar> flags(ok_indices.size(), 0);
    size_t LX = state_.size();
    size_t LS = errCov_.cols();

    VectorXd dx(LS);
    VectorXd xhat_new(LX);
    MatrixXd K;

    // Instead of ransac, we adopt brute force searching here since we have at most 50 features to test
    int max_inliers = 0;
    vector<uchar> status;
    for (size_t i=0; i<ok_indices.size(); i++)
    {
        const Insfeature& insfeat = instateFeatures_[ ok_indices[i]];

        // Update state
        K = insfeat.PHt * insfeat.S.inverse();
        dx = K * insfeat.innovation_;
        xhat_new = state_;
        CalcNewXhat(dx, xhat_new);

        // Count supporters (reprojection error < 2 pixels)
        int inliers = CountVotes(xhat_new, ok_indices, status);
        if (inliers > max_inliers) {
            max_inliers = inliers;
            flags = status;
        }
    }

    // Then we update the filter using the so called "Low innovation inliers" (the best
    // feature selected by ransac and its supporters).
    vector<size_t> lowInnovationInliers, outliers;
    for (size_t i = 0; i < ok_indices.size(); ++i) {
        if (status[i]) {
            lowInnovationInliers.push_back(ok_indices[i]);
        } else {
            outliers.push_back(ok_indices[i]);
        }
    }
    cout << "low innovation inliers: " << lowInnovationInliers.size() << endl;

    if (outliers.empty()) {
        return lowInnovationInliers;
    }

    // ====== Rescue high-innovation inliers =========
    // Back up the state, covariance matrix and instate features
    VectorXd state = state_;
    MatrixXd errCov = errCov_;
    vector<Insfeature> instateFeatures = instateFeatures_;

    // Update the filter
    EKFUpdate(lowInnovationInliers);

    // For in-state features whose reprojection errors are larger than 2 pixels,
    // instead of dropping them directly, we use Mahalanobis distance test to choose
    // compatible ones to contribute to the filter.
    vector<size_t> allInliers = lowInnovationInliers;
    vector<size_t> highInnovationInliers = PredictSHh(outliers);
    for (const auto idx : highInnovationInliers) {
        allInliers.push_back(idx);
    }

    // Restore the state, covariance matrix and instate features
    state_ = state;
    errCov_ = errCov;
    instateFeatures_ = instateFeatures;

    return allInliers;
}


int EkfVio::CountVotes(const Eigen::VectorXd &x, const std::vector<size_t> &indices, std::vector<uchar> &status)
{
    int inlier_count = 0;
    status = vector<uchar>(indices.size(), 0);

    // Calculate poses using state vector
    RefreshPoses();

    // Frequently used variables
    double rho;
    Vector3d xcj, xw, xcl;
    Vector2d hi;
    Matrix3d noUseJacobian;

    for (size_t i=0; i<indices.size(); i++)
    {
        const Insfeature& insfeat = instateFeatures_[indices[i]];
        const Group& anchorGroup = groups_.at(insfeat.anchorFrameId);

        // Check inverse depth
        rho = x(insfeat.stateEntry);
        if (rho<cam_->minInverseDepth_ || rho>cam_->maxInverseDepth_)
            continue;

        // The anchor sensor frame
        const SensorPose& posej = anchorGroup.pose;

        // Feature's 3D position in anchor camera / world / current camera frame
        xcj = insfeat.anchorRay / rho;
        xw = posej.Rwc * xcj + posej.pwc;
        xcl = posel_.Rcw * (xw - posel_.pwc);

        // Check the 3D position
        cam_->Reproject(xcl, hi, false);

        // Instantaneous residual testing
        Vector2d dpt = Vector2d(insfeat.measurement_.upt0.x, insfeat.measurement_.upt0.y) - hi;
        if (dpt.norm() < 2)
        {
            status[i] = 255;
            inlier_count++;
        }
    }

    return inlier_count;
}


int EkfVio::CalcOosFeatures(Eigen::MatrixXd &Ho, Eigen::VectorXd &ro) {
    // Put all frame poses in a map structure
    map<size_t, Matrix<double,3,4> > framePoses;
    RefreshPoses();
    for (const auto &it : groups_) {
        const SensorPose &pose = it.second.pose;
        Matrix<double, 3, 4> Pcw;
        Pcw << pose.Rcw, -pose.Rcw * pose.pwc;
        framePoses[it.first] = Pcw;
    }

    // Select qualified lost features
    int jacobianRows = 0;
    vector<MatrixXd> Hois;
    vector<VectorXd> rois;
    for (auto &fit : featureServer_) {
        const size_t featureId = fit.first;
        auto &feature = fit.second;
        if (feature.isInState_) {   // has been used as instate features
            continue;
        }
        if (feature.latestFrameId_ == currFrameId_) {   // still tracking
            continue;
        }

        // Groups that have observed this feature
        vector<size_t> frameIds;
        for (const auto &oit : feature.observations_) {
            const size_t frameId = oit.second.frameId;
            if (groups_.find(frameId) != groups_.end()) {
                frameIds.push_back(frameId);
            }
        }
        if (frameIds.size() < 3) {  // no enough observations
            continue;
        }

        // Try to triangulate
        feature.InitializePosition(framePoses, cam_);
        if (!feature.initialized_) {
            continue;
        }

        // Calculate measurement Jacobian and residuals
        MatrixXd Hoi;
        VectorXd roi;
        if (OosFeatureJacobian(featureId, frameIds, Hoi, roi) ) {
            if (GatingTest(Hoi, roi)) {
                Hois.push_back(Hoi);
                rois.push_back(roi);
                jacobianRows += static_cast<int>(Hoi.rows());
            }
        }
    }

    const int jacobianCols = static_cast<int>(errCov_.cols());
    Ho.resize(jacobianRows, jacobianCols);
    ro.resize(jacobianRows);
    int rowEntry = 0;
    for (size_t i = 0; i < Hois.size(); ++i) {
        const int iRows = static_cast<int>(Hois[i].rows());
        Ho.middleRows(rowEntry, iRows) = Hois[i];
        ro.segment(rowEntry, iRows) = rois[i];
        rowEntry += iRows;
    }

    // TODO: Decomposition

    return static_cast<int>(rois.size());
}


bool EkfVio::OosFeatureJacobian(size_t featureId, const vector<size_t> &frameIds,
                                MatrixXd &Hoi, VectorXd &roi) {
    const auto &feature = featureServer_.at(featureId);
    const Vector3d xw = feature.xw_;    // 3D position in world frame

    const int M2 = 2 * (int)frameIds.size();
    const int LS = (int)errCov_.cols();

    Matrix<double, 2, 3> J_il, H_fil;

    MatrixXd Hif(M2, 3);    // Jacobian with respect to the feature's 3D position
    MatrixXd Hii = MatrixXd::Zero(M2, LS); // Jacobian with respect to the error state
    VectorXd ri(M2);    // residuals

    int rowIdx = 0;
    for (const auto &frameId : frameIds) {
        const auto &obser = feature.observations_.at(frameId);

        const Group &gro = groups_.at(frameId);
        const int entry = gro.errCovEntry;
        const Matrix3d &R_C_G = gro.pose.Rcw;
        const Vector3d &p_G_C = gro.pose.pwc;
        const Vector3d &p_G_I = gro.pose.pwi;

        // 3D position in camera frame
        Vector3d xc = R_C_G * (xw - p_G_C);

        if (useUnitSphereError_) {
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

            // Residual
            Vector3d rayc = xc.normalized();
            ri.segment(rowIdx, 2) = tangentBase * (obserRay - rayc);

            // Projection Jacobian
            Matrix3d J_ray = MathUtil::NormalizationJacobian(xc);
            J_il = tangentBase * J_ray;
        } else {
            // Pinhole projection to image plane
            Matrix3d projection_jacobian;
            Vector2d h;
            if (!cam_->PinholeProjection(xc, h, projection_jacobian, true) ) {
                return false;
            }
            J_il = projection_jacobian.topRows(2);

            // residual
            ri.segment(rowIdx, 2) = Vector2d(obser.upt0.x, obser.upt0.y) - h;
        }


        H_fil = J_il * R_C_G;
        Hif.block(rowIdx, 0, 2, 3) = H_fil;
        Hii.block(rowIdx, entry, 2, 6) << H_fil * MathUtil::VecToSkew(xw-p_G_I), -H_fil;

        rowIdx += 2;
    }

    // Project Hii & ri to the left null space of H_fi
    int leftRows = M2 - 3;
# if 0
    Eigen::ColPivHouseholderQR<MatrixXd> eigenQr(Hif);
    MatrixXd QrQ = eigenQr.matrixQ();
    MatrixXd ViT = QrQ.rightCols(leftRows).transpose();
#else
    JacobiSVD<MatrixXd> svdSolver(Hif, ComputeFullU | ComputeThinV);
    MatrixXd ViT = svdSolver.matrixU().rightCols(leftRows).transpose();
#endif

    Hoi = ViT * Hii;
    roi = ViT * ri;

    return true;

}


bool EkfVio::GatingTest(const MatrixXd &H, const VectorXd &r) {
    const int dim = static_cast<int>(H.rows());
    MatrixXd S = H * errCov_ * H.transpose();
    for (int i = 0; i < dim; i++) {
        S(i, i) += featureMeasurementNoise_;
    }
    double mahanalobisDistance = r.transpose() * S.inverse() * r;
    return mahanalobisDistance < MAHALA95_TABLE[dim-1];
}


bool EkfVio::EKFUpdate(const vector<size_t> &indices)
{
    if (indices.empty())
        return false;
    size_t NDim = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        NDim += instateFeatures_[indices[i]].innovation_.size();
    }

    // Compute P * H' & residuals (just docking)
    size_t M = errCov_.cols();
    MatrixXd PHt(M, NDim);
    VectorXd res(NDim);

    size_t obserIdx = 0;
    for (size_t i=0; i<indices.size(); i++) {
        const Insfeature& insfeat = instateFeatures_[indices[i]];
        const size_t dim = insfeat.innovation_.size();
        PHt.block(0, obserIdx, M, dim) = insfeat.PHt;
        res.segment(obserIdx, dim) = insfeat.innovation_;
        obserIdx += dim;
    }

    // S = H * P * H' + R
    MatrixXd S(NDim, NDim);
    obserIdx = 0;
    for (size_t i=0; i<indices.size(); i++) {
        const size_t& idx_info = indices[i];
        const Insfeature& insfeat = instateFeatures_[idx_info];
        const Group& anchorGroup = groups_.at(insfeat.anchorFrameId);
        const int dim = static_cast<int>(insfeat.innovation_.size());

        int et2 = anchorGroup.errCovEntry;
        int et3 = insfeat.errCovEntry;

        S.block(obserIdx, 0, dim, NDim) = insfeat.H_Il  * PHt.topRows(6)
                              + insfeat.H_Ij * PHt.middleRows(et2, 6)
                              + insfeat.H_rho * PHt.row(et3);
        obserIdx += dim;
    }

    for (size_t i=0; i<NDim; i++)
        S(i,i) += featureMeasurementNoise_;

    // Kalman gain K = P * H' / S
    MatrixXd Kt = S.ldlt().solve(PHt.transpose());
    MatrixXd K = Kt.transpose();

    // TODO: Deal with Gauge freedom.
   // Update state
    const size_t refId = groups_.begin()->first;
    const int refEntry = groups_[refId].stateEntry;
    const Vector3d ref_p = state_.segment(refEntry + 4, 3); // p_W_I0
    const Quaterniond ref_q(state_(refEntry), state_(refEntry + 1), state_(refEntry + 2), state_(refEntry + 3)); // q_I0_W
    const Matrix3d ref_R = ref_q.matrix(); // R_W_I0


    // Update state and covariance
    VectorXd dx = K * res;
    CalcNewXhat(dx, state_);

    Vector3d new_ref_p = state_.segment(refEntry + 4, 3); // p_W'_I0
    Quaterniond new_ref_q(state_(refEntry), state_(refEntry + 1), state_(refEntry + 2), state_(refEntry + 3)); // q_I0_W'
    Matrix3d new_ref_R = new_ref_q.matrix(); // R_W'_I0

    Vector3d ypr = MathUtil::R2ypr(ref_R);
    Vector3d new_ypr = MathUtil::R2ypr(new_ref_R);
    double diffYaw = ypr(0) - new_ypr(0);
    Matrix3d diffR = MathUtil::ypr2R(Vector3d(diffYaw, 0, 0));  // R_W_W'
    if (fabs(fabs(ypr(1)) * 180.0 / CV_PI - 90) < 1.0 || fabs(fabs(new_ypr(1)) * 180.0 / CV_PI - 90) < 1.0) {
        cerr << "Warning: singular euler point!" << endl;
        diffR = ref_R * new_ref_R.transpose();
    }

    for (const auto& it : groups_) {
        int et = it.second.stateEntry;
        Matrix3d R_W_Ik = diffR * Quaterniond(state_(et), state_(et + 1), state_(et + 2), state_(et + 3)).matrix(); // R_W_W' * R_W'_Ik
        Vector3d p_W_Ik = ref_p + diffR * (state_.segment(et + 4, 3) - new_ref_p); // p_W_I0 + R_W_W'(p_W'_Ik - p_W'_I0);
        Quaterniond q_Ik_W(R_W_Ik);
        state_.segment(et, 4) << q_Ik_W.w(), q_Ik_W.x(), q_Ik_W.y(), q_Ik_W.z();
        state_.segment(et + 4, 3) = p_W_Ik;
    }
    Vector3d v_W_Ik = diffR * state_.segment(7, 3); // R_W_W' * v_W'_Ik;
    state_.segment(7, 3) = v_W_Ik;
    int et = 0;
    Matrix3d R_W_Ik = diffR * Quaterniond(state_(et), state_(et + 1), state_(et + 2), state_(et + 3)).matrix(); // R_W_W' * R_W'_Ik
    Vector3d p_W_Ik = ref_p + diffR * (state_.segment(et + 4, 3) - new_ref_p); // p_W_I0 + R_W_W'(p_W'_Ik - p_W'_I0);
    Quaterniond q_Ik_W(R_W_Ik);
    state_.segment(et, 4) << q_Ik_W.w(), q_Ik_W.x(), q_Ik_W.y(), q_Ik_W.z();
    state_.segment(et + 4, 3) = p_W_Ik;
    // Rotate points
    // Depth won't change

    errCov_ -= PHt * Kt;

    MatrixXd SIGMAt = errCov_.transpose();
    errCov_ += SIGMAt;
    errCov_ *= 0.5;

    RefreshPoses();

    return true;
}


bool EkfVio::HybridUpdate(const vector<size_t> &indices) {
    // Stack in-state features' measurement Jacobians and residuals
    const int jacobian_cols = static_cast<int>(errCov_.cols());
    int jacobian_rows = 0;
    for (const size_t idx : indices) {
        jacobian_rows += static_cast<int>(instateFeatures_.at(idx).innovation_.size());
    }
    MatrixXd H_ins(jacobian_rows, jacobian_cols);
    VectorXd r_ins(jacobian_rows);
    H_ins.setZero();

    int rowEntry = 0;
    for (const size_t idx : indices) {
        const Insfeature &insfeat = instateFeatures_.at(idx);
        const Group &anchorGroup = groups_.at(insfeat.anchorFrameId);
        const int iRows = static_cast<int>(insfeat.innovation_.size());
        H_ins.block(rowEntry, 0, iRows, 6) = insfeat.H_Il;
        H_ins.block(rowEntry, anchorGroup.errCovEntry, iRows, 6) = insfeat.H_Ij;
        H_ins.block(rowEntry, insfeat.errCovEntry, iRows, 1) = insfeat.H_rho;
        r_ins.segment(rowEntry, iRows) = insfeat.innovation_;
        rowEntry += iRows;
    }

    // Stack qualified out-of-state features' measurement Jacobians and residuals
    MatrixXd H_oos;
    VectorXd r_oos;
    numOosFeatures_ = CalcOosFeatures(H_oos, r_oos);
    cout << "out of state features: " << numOosFeatures_ << endl;

    // Stack in-state and out-of-state features together
    MatrixXd H_total(H_ins.rows() + H_oos.rows(), jacobian_cols);
    VectorXd r_total(r_ins.size() + r_oos.size());
    H_total << H_ins, H_oos;
    r_total << r_ins, r_oos;
    if (H_total.rows() > 0) {
        UpdateFilter(H_total, r_total);
    }

    RefreshPoses();

    return true;
}


void EkfVio::UpdateFilter(const MatrixXd &H, const VectorXd &r) {
    // Update state and error covariance
    MatrixXd PHt = errCov_ * H.transpose();
    MatrixXd S = H * PHt;
    for (int k=0; k<(int)S.rows(); k++) {
        S(k,k) += featureMeasurementNoise_;
    }

    // Kalman gain K = P * H' / S
    MatrixXd Kt = S.ldlt().solve(PHt.transpose());
    MatrixXd K = Kt.transpose();

    // TODO: Deal with Gauge freedom.
   // Update state
    const size_t refId = groups_.begin()->first;
    const int refEntry = groups_[refId].stateEntry;
    const Vector3d ref_p = state_.segment(refEntry + 4, 3); // p_W_I0
    const Quaterniond ref_q(state_(refEntry), state_(refEntry + 1), state_(refEntry + 2), state_(refEntry + 3)); // q_I0_W
    const Matrix3d ref_R = ref_q.matrix(); // R_W_I0


    // Update state and covariance
    VectorXd dx = K * r;
    CalcNewXhat(dx, state_);

    Vector3d new_ref_p = state_.segment(refEntry + 4, 3); // p_W'_I0
    Quaterniond new_ref_q(state_(refEntry), state_(refEntry + 1), state_(refEntry + 2), state_(refEntry + 3)); // q_I0_W'
    Matrix3d new_ref_R = new_ref_q.matrix(); // R_W'_I0

    Vector3d ypr = MathUtil::R2ypr(ref_R);
    Vector3d new_ypr = MathUtil::R2ypr(new_ref_R);
    double diffYaw = ypr(0) - new_ypr(0);
    Matrix3d diffR = MathUtil::ypr2R(Vector3d(diffYaw, 0, 0));  // R_W_W'
    if (fabs(fabs(ypr(1)) * 180.0 / CV_PI - 90) < 1.0 || fabs(fabs(new_ypr(1)) * 180.0 / CV_PI - 90) < 1.0) {
        cerr << "Warning: singular euler point!" << endl;
        diffR = ref_R * new_ref_R.transpose();
    }

    for (const auto& it : groups_) {
        int et = it.second.stateEntry;
        Matrix3d R_W_Ik = diffR * Quaterniond(state_(et), state_(et + 1), state_(et + 2), state_(et + 3)).matrix(); // R_W_W' * R_W'_Ik
        Vector3d p_W_Ik = ref_p + diffR * (state_.segment(et + 4, 3) - new_ref_p); // p_W_I0 + R_W_W'(p_W'_Ik - p_W'_I0);
        Quaterniond q_Ik_W(R_W_Ik);
        state_.segment(et, 4) << q_Ik_W.w(), q_Ik_W.x(), q_Ik_W.y(), q_Ik_W.z();
        state_.segment(et + 4, 3) = p_W_Ik;
    }
    Vector3d v_W_Ik = diffR * state_.segment(7, 3); // R_W_W' * v_W'_Ik;
    state_.segment(7, 3) = v_W_Ik;
    int et = 0;
    Matrix3d R_W_Ik = diffR * Quaterniond(state_(et), state_(et + 1), state_(et + 2), state_(et + 3)).matrix(); // R_W_W' * R_W'_Ik
    Vector3d p_W_Ik = ref_p + diffR * (state_.segment(et + 4, 3) - new_ref_p); // p_W_I0 + R_W_W'(p_W'_Ik - p_W'_I0);
    Quaterniond q_Ik_W(R_W_Ik);
    state_.segment(et, 4) << q_Ik_W.w(), q_Ik_W.x(), q_Ik_W.y(), q_Ik_W.z();
    state_.segment(et + 4, 3) = p_W_Ik;
    // Rotate points
    // Depth won't change

    // Update error state covariance
    errCov_ -= PHt * Kt;

    // Make sure that the covariance matrix is symmetric
    MatrixXd errCovT = errCov_.transpose();
    errCov_ += errCovT;
    errCov_ *= 0.5;
}


void EkfVio::CalcNewXhat(const VectorXd &dx, VectorXd &x)
{
    Vector4d qnew;

    // Update q, p, v, bg, ba
    MathUtil::IncreQuat(x.head(4), dx.head(3), qnew);
    x.head(4) = qnew;   // q
    x.segment(4, 12) += dx.segment(3, 12);  // p, v, bg, ba

    // Update the sliding window poses
    size_t pose_k = 16, pose_sk = 15;
    for (const auto &it : groups_)
    {
        MathUtil::IncreQuat(x.segment(pose_k, 4), dx.segment(pose_sk, 3), qnew);
        x.segment(pose_k, 4) = qnew;    // q_Ij_G
        x.segment(pose_k+4, 3) += dx.segment(pose_sk+3, 3); // p_G_Ij

        pose_k += 7;
        pose_sk += 6;
    }

    // Update inverse depths
    int nf = static_cast<int>(instateFeatures_.size());
    x.tail(nf) += dx.tail(nf);
}


void EkfVio::RemoveGFs(const vector<uchar> &flags)
{
    // Prepare
    VectorXi xbad(state_.size());    xbad.setZero();
    VectorXi sbad(errCov_.cols());   sbad.setZero();
    vector<size_t> gro_godie;

    vector<int> feats_godie;
    for (int i = 0; i < (int)instateFeatures_.size(); i++) {
        const auto &insfeat = instateFeatures_.at(i);
        auto &anchorGroup = groups_.at(insfeat.anchorFrameId);
        if (!flags[i]) {
            feats_godie.push_back(i);
            xbad(insfeat.stateEntry) = 1;
            sbad(insfeat.errCovEntry) = 1;
            featureServer_.erase(instateFeatures_[i].featureId);    // remove it from feature server
            anchorGroup.count--;
            Vector3d xw;
            if (Calc3DInsfea(insfeat, xw)) {
                mvMapPoints.push_back(xw);
            }
        }
    }

    for (int i = (int)feats_godie.size() - 1; i >=0; i--) {
        instateFeatures_.erase(instateFeatures_.begin() + feats_godie[i]);
    }

    const int numOverflow = static_cast<int>(groups_.size()) - slidingWindowSize_;
    for (auto &it : groups_) {
        if (it.second.count < 1) {
            if (useOosFeatures_ && static_cast<int>(gro_godie.size()) >= numOverflow) {
                continue;
            }
            gro_godie.push_back(it.first);
            xbad.segment(it.second.stateEntry, 7).setOnes();
            sbad.segment(it.second.errCovEntry, 6).setOnes();
        }
    }

    for (const auto &frameId : gro_godie) {
        groups_.erase(frameId);
    }

    // Shrink state vector and covariance matrix
    MathUtil::ReduceEigenVector(state_, xbad);
    MathUtil::ReduceEigenMatrix(errCov_, sbad);

    // Update entries
    UpdateEntries();

}


void EkfVio::UpdateEntries() {

    int stateEntry=16, errCovEntry = 15;

    for (auto &it : groups_) {
        Group& gro = it.second;
        gro.stateEntry = stateEntry;
        gro.errCovEntry = errCovEntry;
        stateEntry += 7;
        errCovEntry += 6;
    }

    for (auto &insfeat : instateFeatures_) {
        insfeat.stateEntry = stateEntry++ ;
        insfeat.errCovEntry = errCovEntry++ ;
    }
}


void EkfVio::RefreshPoses() {
    // Current pose
    Matrix<double, 7, 1> qp = state_.head(7);
    QpToPose(qp, posel_);

    // Sliding window poses
    for (auto &it : groups_) {
        qp = state_.segment(it.second.stateEntry, 7);
        QpToPose(qp, it.second.pose);
    }
}


void EkfVio::QpToPose(const Eigen::Matrix<double, 7, 1> &qp, SensorPose &sp) {
    Quaterniond q_Ij_W(qp(0), qp(1), qp(2), qp(3));
    sp.Rwi = q_Ij_W.matrix();
    sp.Riw = sp.Rwi.transpose();
    sp.Rcw = cam_->Rci_ * sp.Riw;
    sp.Rwc = sp.Rcw.transpose();
    sp.pwi = qp.tail(3);
    sp.pwc = sp.pwi + sp.Rwi * cam_->pic_;
}


void EkfVio::SetNoiseNc() {
    noiseNc_.setZero();

    int i;
    for (i=0; i<3; i++)
        noiseNc_(i,i) = pow(cam_->gyroSigma_, 2);
    for (i=6; i<9; i++)
        noiseNc_(i,i) = pow(cam_->accSigma_, 2);
    for (i=9; i<12; i++)
        noiseNc_(i,i) = pow(cam_->gyroRandomWalk_, 2);
    for (i=12; i<15; i++)
        noiseNc_(i,i) = pow(cam_->accRandomWalk_, 2);
}


void EkfVio::SeparateVector(const std::vector<size_t> &v, const std::vector<uchar> &flags, std::vector<size_t> &vi, std::vector<size_t> &vo)
{
    vi.clear();
    vo.clear();

    size_t N = v.size();
    for (size_t i=0; i<N; i++)
        if (flags[i])
            vi.push_back(v[i]);
        else
            vo.push_back(v[i]);

}


vector<Vector3d> EkfVio::QueryActiveFeatures() {
    vector<Vector3d> activePoints;
    Vector3d xw;
    for (const auto &insfeat : instateFeatures_) {
        if (Calc3DInsfea(insfeat, xw)) {
            activePoints.push_back(xw);
        }
    }
    return activePoints;
}


vector<Matrix4d> EkfVio::QuerySlidingWindowPoses() {
    vector<Matrix4d> poses;
    Matrix4d Twc = Matrix4d::Identity();
    for (const auto &it : groups_) {
        Twc.topRows(3) << it.second.pose.Rwc, it.second.pose.pwc;
        poses.push_back(Twc);
    }
    return poses;
}


void EkfVio::DrawDebugInfo(cv::Mat &img) {
    if (img.empty()) {
        cerr << "Can't draw debug info while the InputOutput image is empty!" << endl;
        return;
    }

    const float r = 3;
    for (const auto &insfeat : instateFeatures_) {
        const auto &pt = insfeat.measurement_.pt0;
        rectangle(img, Point2f(pt.x-r, pt.y-r), Point2f(pt.x+r, pt.y+r), Scalar(0, 255, 0), 1);
    }

    stringstream ss;
    ss << "filter: G=" << groups_.size() << " F trk=" << numTrackedInsfeatures_
        << " com=" << numCompatibleInsfeatures_
        << " inl=" << numInlierInsfeatures_
        << " new=" << numNewInsfeatures_;
    putText(img, ss.str(), Point(5, 40), cv::FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255,255,255), 1);
}


bool EkfVio::Calc3DInsfea(const Insfeature& insfeat, Vector3d &xw)
{
    // Anchor frame pose
    const SensorPose& posej = groups_.at(insfeat.anchorFrameId).pose;

    // Check uncertainty of inverse depth
    if (state_[insfeat.errCovEntry] > 0.1)
        return false;

    double rho = state_(insfeat.stateEntry);
    if (rho<cam_->minInverseDepth_ || rho>cam_->maxInverseDepth_)
        return false;

    // Feature's 3D position in anchor camera / world / current camera frame
    Vector3d xcj = insfeat.anchorRay / rho;
    xw = posej.Rwc * xcj + posej.pwc;
    return true;
}

}//namespace inslam {