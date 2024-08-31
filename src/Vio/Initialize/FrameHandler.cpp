#include "Vio/Initialize/FrameHandler.hpp"
#include "Vio/Initialize/Utils/epnp.h"
#include "Vio/Initialize/Utils/Triangulation.h"
#include "Vio/Initialize/Utils/EssentialRansac.h"
#include "Vio/Initialize/StereoDepthCost.hpp"
#include "Vio/Initialize/Optimization.hpp"
#include "Utils/MathUtil.h"
#include "Vio/Triangulator.h"

#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/problem.h>


namespace inslam {

FrameHandler::FrameHandler(MapPtr map, const nlohmann::json &config, std::shared_ptr<Caimura> leftCaim,
                           std::shared_ptr<Caimura> rightCam) : map_(map), leftCam_(leftCaim),
                           rightCam_(rightCam) {
    maxFeature_ = config["max_corners"];
    radius_ = config["radius"];
    int startAligmentFrameNum = config["start_aligment_frame_num"];
    int border = config["border"];

    featureId_ = 0;
    lastKeyframeFeatureCount_ = 0;
    angularResolution_ = leftCam_->GetAngularResolution();
    state_ = NOT_START;
    initilization_ = std::make_shared<inslam::PoseInitialization>(map_);
    trackFailCnt_ = 0;
    R_ = cv::Mat::eye(3, 3, CV_64F);
    t_ = cv::Mat::zeros(3, 1, CV_64F);
    doStereo_ = rightCam_ == nullptr ? false : true;
    lastInitFeatureCount_ = static_cast<int>(1e6);

    // set cam mask
    int width = leftCam_->width();
    int height = leftCam_->height();
    camMask_ = cv::Mat(height, width, CV_8UC1, 255);
    int xBorder = border, yBorder = border;
    if (leftCam_->IsEquirectangular()) {
        // Mask out the top & bottom region of panoramic image
        yBorder = std::max(yBorder,  static_cast<int>(leftCam_->height() * 0.1) );
    }
    camMask_.rowRange(0, yBorder).setTo(0);
    camMask_.rowRange(leftCam_->height()-yBorder, leftCam_->height()).setTo(0);
    camMask_.colRange(0, xBorder).setTo(0);
    camMask_.colRange(leftCam_->width()-xBorder, leftCam_->width()).setTo(0);

    // static initialize
    staticImgCount_ = 0;

    // imu
    doImuAligment_ = false;
    newVioFlag_ = false;
    imuInitializing_.SetMap(map_);
    imuInitializing_.SetStartAligmentFrameNum(startAligmentFrameNum);
    initBa_.setZero();

    // Calibrate imu
    doCalibrateImuEx_ = false;
    imuCalib_ = std::make_shared<ImuCalibration>(map_, angularResolution_);

    saveSlamPoseFlag_ = false;
}


bool FrameHandler::Run(const cv::Mat &imgL, double imgTime, const cv::Mat &imgR) {

    if (imgL.empty() || imgL.type() != CV_8UC1 || imgL.cols != leftCam_->width()) {
        std::cout << "Frame: image not same size as camera." << std::endl;
        return false;
    }

    if(imgR.empty()) {
        doStereo_ = false;
    }

    currFrame_.reset(new Frame(imgTime, imgL));
    UpdateResult res = UpdateResult::RESULT_FAILURE;
    if (state_ == InitTrackState::INITIALIZED) {
        res = ProcessFrame();
    }else if(state_ == InitTrackState::NOT_INITIALIZED) {
        res = ProcessSecondFrame();
    }else if(state_ == InitTrackState::NOT_START) {
        res = ProcessFirstFrame();
    }else if(state_ == InitTrackState::TRACKING_LOST) {
        std::cout << "track lost" << std::endl;
        Reset();
    }

    PostProcess(res, imgR);
    DebugShow(2);

    if(doCalibrateImuEx_) {
        return TryCalibImu();
    } else if(doImuAligment_) {
        return TryInitialize();
    } else {
        if(saveSlamPoseFlag_ && state_ == InitTrackState::INITIALIZED) {
            frameDataset_.push_back(currFrame_);
        }
        return state_ == InitTrackState::INITIALIZED;
    }
}


UpdateResult FrameHandler::ProcessFirstFrame() {
    // calculate the reference feature of new frame
    GetNewCorners(maxFeature_);
    if(currFrame_->Features().size() < maxFeature_ * 0.4) {
        state_ = InitTrackState::NOT_START;
        std::cout << "To few features generated in first frame: " << currFrame_->Features().size() << std::endl;
        Reset();
        return UpdateResult::RESULT_FAILURE;
    }
    state_ = InitTrackState::NOT_INITIALIZED;
    currFrame_->SetFrameId(0);
    map_->InsertKeyframe(currFrame_);
    lastKeyframeFeatureCount_ = static_cast<int>(currFrame_->Features().size());
    return UpdateResult::RESULT_IS_KEYFRAME;
}


UpdateResult FrameHandler::ProcessSecondFrame() {
    // using KLT track second frame feature
    Track(currFrame_, lastFrame_);

    // tracking lost
    int currFeatCnt = static_cast<int>(currFrame_->Features().size());
    if (currFeatCnt < lastKeyframeFeatureCount_ * 0.3 || currFeatCnt < 20) {
        state_ = InitTrackState::TRACKING_LOST;
        return UpdateResult::RESULT_FAILURE;
    }

    if (currFeatCnt < lastKeyframeFeatureCount_ * 0.8) {
        GetNewCorners(maxFeature_ - currFeatCnt);
        currFrame_->SetFrameId(1);
        map_->InsertKeyframe(currFrame_);
        bool goodInitialized = initilization_->Run();
        if (goodInitialized) {
            map_->GetKeyFrames().back()->GetPose(R_, t_);
            lastKeyframeFeatureCount_ = static_cast<int>(currFrame_->Features().size());
            state_ = InitTrackState::INITIALIZED;
            std::cout << "init pose: r " << R_ << ": t" << t_ << std::endl;
            return UpdateResult::RESULT_IS_KEYFRAME;
        } else {
            state_ = InitTrackState::TRACKING_LOST;
            Reset();
            return UpdateResult::RESULT_FAILURE;
        }
    }
    return UpdateResult::RESULT_NO_KEYFRAME;
}


UpdateResult FrameHandler::ProcessFrame() {
    Track(currFrame_, lastFrame_);
    int N = std::max(int(lastKeyframeFeatureCount_ * 0.8), 8);
    bool lostFeatures = (int)currFrame_->Features().size() < N ? true : false;
    if (lostFeatures) {
        if (currFrame_->Features().size() < 20) {
            trackFailCnt_++;
            Reset();
            return UpdateResult::RESULT_FAILURE;
        }
        GetNewCorners(maxFeature_ - static_cast<int>(currFrame_->Features().size()));
        currFrame_->SetFrameId(static_cast<int>(map_->GetKeyFrames().size()));
        map_->InsertKeyframe(currFrame_);
        bool flag = PnPIncrement();
        if (!flag) {
            map_->PopKeyframe(currFrame_);
            trackFailCnt_++;
            Reset();
            std::cout << "keyframe pose increment failure" << std::endl;
            return UpdateResult::RESULT_FAILURE;
        }
        lastKeyframeFeatureCount_ = static_cast<int>(currFrame_->Features().size());
        return UpdateResult::RESULT_IS_KEYFRAME;
    } else {
        currFrame_->SetFrameId(static_cast<int>(map_->GetKeyFrames().size()));
        map_->InsertKeyframe(currFrame_);
        bool flag = PnPIncrement();
        if (!flag) {
            map_->PopKeyframe(currFrame_);
            trackFailCnt_++;
            Reset();
            std::cout << "normal frame pose increment failure" << std::endl;
            return UpdateResult::RESULT_FAILURE;
        }
        return UpdateResult::RESULT_NO_KEYFRAME;
    }
}


void FrameHandler::PostProcess(UpdateResult res, const cv::Mat &imgR) {
    if(res != UpdateResult::RESULT_FAILURE) {
        if(doStereo_) {
            CalcStereo(imgR);
            LocalMapping(res);
            if(!initFrameList_.empty() || state_ == NOT_INITIALIZED) {
                OptmizeInitMappoint();
            }
        } else {
            LocalMapping(res);
        }

        trackFailCnt_ = 0;
        if(lastFrame_ != nullptr) {
            lastFrame_->ImageRelease();
        }
        lastFrame_ = currFrame_;

        int frameId = 0;
        int keyFrameSize = static_cast<int>(map_->keyframes_.size());
        if(res == UpdateResult::RESULT_IS_KEYFRAME && keyFrameSize > 1) {
            frameId = currFrame_->FrameId();
        } else if (res == UpdateResult::RESULT_NO_KEYFRAME) {
            frameId = keyFrameSize == 1 ? -2 : -1;
        }
        if(frameId != 0 && doImuAligment_) {
            ImuInitInsert(*currFrame_, frameId, currFrame_->Time());
        }
    } else if(trackFailCnt_ > 10) {
        state_ = InitTrackState::TRACKING_LOST;
    }
}


bool FrameHandler::TryInitialize() {
    // try static initialize
    if(staticImgCount_ >= 4) {
        Eigen::Vector3d accMean(0.0, 0.0, 0.0);
        Eigen::Vector3d gyrMean(0.0, 0.0, 0.0);
        double accStd = CalcStd(staticAccDatas_, accMean);
        double gyrStd = CalcStd(staticGyrDatas_, gyrMean);
        if(accStd < 0.8 && gyrStd < 0.05) {
            Eigen::Vector3d gyrBias = gyrMean;
            Eigen::Vector3d GIj = accMean - initBa_;
            Eigen::Quaterniond q_Ij_g = MathUtil::GetQfromA(GIj);
            qv_ << q_Ij_g.w(), q_Ij_g.vec(),  0.0, 0.0, 0.0;
            std::cout << "static initialize completed, accStd: " << accStd  << ", gyrStd: " << gyrStd << std::endl;
            std::cout << "static initialize completed, gyrBias: " << gyrBias << std::endl;
            return true;
        }
    }

    // try dynamic initialize
    if (state_ == InitTrackState::INITIALIZED && !imuInitializing_.IsSuccess() ){
        imuInitializing_.FrameOptimization(angularResolution_);
        bool flag = imuInitializing_.Alignment(doStereo_);

        if (flag) {
            if(imuInitializing_.IsSuccess()) {
                // get the last frame Imu velocity in world
                std::vector<Frame> &frames = imuInitializing_.GetFrames();
                VisualInertialState &imuStateJ = frames.back().imu_state;
                Eigen::Vector3d Gc0 = imuInitializing_.GetGravityVector();
                Eigen::Vector3d Gcj = imuStateJ.Rwc * Gc0;
                Eigen::Vector3d GIj = leftCam_->Ric_ * Gcj;
                Eigen::Quaterniond q_Ij_g = MathUtil::GetQfromA(GIj);
                Eigen::Vector3d Vgj = q_Ij_g.matrix() * leftCam_->Ric_ * imuStateJ.Rwc * imuStateJ.Vj;
                qv_ << q_Ij_g.w(), q_Ij_g.vec(),  Vgj;
                std::cout << "dynamic initialize completed!" << std::endl;
                return true;
            } else {
                Reset();
                std::cout << "Init fail, reset initializer" << std::endl;
                return false;
            }
        }
    }
    return false;
}


double FrameHandler::CalcStd(std::deque<Eigen::Vector3d> &dataList, Eigen::Vector3d &dataMean) {
    if(dataList.empty()) {
        return -1;
    }

    Eigen::Vector3d init = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum = std::accumulate(std::begin(dataList), std::end(dataList), init,
                                          [](const Eigen::Vector3d data0, const Eigen::Vector3d &data1) {
        return data0 + data1;
    });
    dataMean =  sum / dataList.size();

    double accumVar = 0.0;
    std::for_each(std::begin(dataList), std::end(dataList), [&accumVar, &dataMean](const Eigen::Vector3d &d) {
        accumVar  += (d - dataMean).transpose() * (d - dataMean);
    });

    return std::sqrt(accumVar / (dataList.size()-1));
}


void FrameHandler::OptmizeInitMappoint() {
    // optimize mapPoints which generated by stereo
    if (map_->keyframes_.size() < 2 && currFrame_->Features().size() < lastInitFeatureCount_ * 0.98) {
        initFrameList_.push_back(currFrame_);
        //initFrameList_.back().ReleaseImage();
        lastInitFeatureCount_ = static_cast<int>(currFrame_->Features().size());
        if(initilization_->RunStereo(initFrameList_)) {
            state_ = INITIALIZED;
        } else {
            std::cout << "stereo initial fail,try monocular initial" << std::endl;
            std::vector<MapPointPtr> &mapPoints = map_->mapPoints_;
            std::for_each(std::begin(mapPoints), std::end(mapPoints), [](MapPointPtr &mp){
                mp->pt3dFlag = false;
            });
            state_ = NOT_INITIALIZED;
        }
    } else if (map_->keyframes_.size() == 3) {
        initFrameList_.clear();
        lastInitFeatureCount_ = static_cast<int>(1e6);
    }
}


void FrameHandler::LocalMapping(UpdateResult res) {
    if(map_->GetKeyFrames().size() < 2) {
        return;
    }

    std::vector<FramePtr> frames;
    int optimizeNum = 1;
    if(res == UpdateResult::RESULT_IS_KEYFRAME) {
        optimizeNum = 10;
        Update3DPoints();
    }

    int startId = std::max(1, (int)map_->GetKeyFrames().size() - optimizeNum);
    for(int i = startId; i < (int)map_->GetKeyFrames().size(); i++) {
        frames.push_back(map_->keyframes_[i]);
    }

    LocalOptimization(frames, map_, angularResolution_);
    frames.back()->GetPose(R_, t_);

    if(res == UpdateResult::RESULT_NO_KEYFRAME) {
        map_->PopKeyframe(currFrame_);
    }
}


void FrameHandler::Reset() {
    R_ = cv::Mat::eye(3, 3, CV_64F);
    t_ = cv::Mat::zeros(3, 1, CV_64F);
    state_ = NOT_START;
    map_->Clear();
    lastKeyframeFeatureCount_ = 0;
    angularResolution_ = leftCam_->GetAngularResolution();
    initilization_.reset(new PoseInitialization(map_));
    featureId_= 0;
    lastFrame_.reset();
    currFrame_.reset();
    trackFailCnt_ = 0;
    initFrameList_.clear();
    lastInitFeatureCount_ = static_cast<int>(1e6);
    doImuAligment_ = false;
    imuInitializing_.AligmentReset(true);
    imuInitializing_.SetBa(initBa_);
    frameDataset_.clear();
    imuCalib_->Reset();
}


void FrameHandler::Track(FramePtr currFrame, FramePtr lastFrame) {
    if(lastFrame == nullptr || lastFrame->Features().empty()) {
        return;
    }

    const std::vector<Feature> &lastFeas = lastFrame->Features();
    std::vector<cv::Point2f> pts1, pts2, pts21;
    for(auto iter = lastFeas.begin(); iter != lastFeas.end(); iter++) {
        pts1.push_back(iter->pt);
    }

    // run flow
    std::vector<uchar> status1, status2;
    std::vector<float> errs1, errs2;
    cv::calcOpticalFlowPyrLK(lastFrame_->LeftPyramid(), currFrame_->LeftPyramid(), pts1, pts2, status1, errs1);
    cv::calcOpticalFlowPyrLK(currFrame_->LeftPyramid(), lastFrame_->LeftPyramid(), pts2, pts21, status2, errs2);

    mask_ = camMask_.clone();
    std::vector<Feature> &currFeas = currFrame->Features();
    std::vector<int> &idList = currFrame->IDList();
    int i = 0;
    for( auto iter = lastFeas.begin(); iter != lastFeas.end(); iter++, i++) {

        // cross check
        if (cv::norm(pts1[i] - pts21[i]) >= 1) {
            continue;
        }

        if(mask_.at<uchar>(static_cast<int>(pts2[i].y), static_cast<int>(pts2[i].x)) == 0) {
            continue;
        }
        cv::circle(mask_, pts2[i], radius_, 0, -1);

        Feature fea;
        fea.featureID = iter->featureID;
        fea.imageID = static_cast<int>(map_->GetKeyFrames().size());
        fea.pt = pts2[i];
        fea.depth = -1.0;
        Eigen::Vector2d uv(fea.pt.x, fea.pt.y);
        Eigen::Vector3d sp = leftCam_->LiftSphere(uv, true);
        fea.spherePt1 = cv::Point3f(static_cast<float>(sp(0)), static_cast<float>(sp(1)), static_cast<float>(sp(2)));
        currFeas.push_back(fea);
        idList.push_back(fea.featureID);
    }

    if(doImuAligment_) {
        CheckCameraStatic(currFeas, lastFeas);
    }

    DebugShow(1);
}


void FrameHandler::CheckCameraStatic(const std::vector<Feature> &currFeas, const std::vector<Feature> &lastFeas) {
    int j0 = 0;
    std::vector<cv::Point2f> pts1, pts2;
    for (int i = 0; i < static_cast<int>(lastFeas.size()); ++i) {
        for (int j = j0; j < static_cast<int>(currFeas.size()); ++j) {
            if (lastFeas[i].featureID == currFeas[j].featureID) {
                j0 = j;
                pts1.push_back(lastFeas[i].pt);
                pts2.push_back(currFeas[j].pt);
            }
        }
    }
    if (pts1.size() <= 10) {
        staticImgCount_ = staticImgCount_ > 0 ? (staticImgCount_ - 1) : 0;
        return;
    }

    double lowThreshold = 0.6;
    int dynamicPixCnt = 0;
    for (int i = 0; i < static_cast<int>(pts1.size()); ++i) {
        cv::Point2f dif = pts1[i] - pts2[i];
        double difVal = cv::norm(dif);
        if (difVal > lowThreshold) {
            ++dynamicPixCnt;
        }
        if (dynamicPixCnt > 0.6 * pts1.size()) {
            staticImgCount_ = 0;
            staticAccDatas_.clear();
            staticGyrDatas_.clear();
            return;
        }
    }
    if (dynamicPixCnt <= 0.2 * pts1.size()) {
        staticImgCount_++;
        return;
    }
}


void FrameHandler::GetNewCorners(int nNewFeature)
{
    // Detect new corners
    std::vector<Feature> &features = currFrame_->Features();
    std::vector<cv::Point2f> newPts;
    cv::goodFeaturesToTrack(currFrame_->Image(), newPts, nNewFeature, 0.01, radius_, mask_, 3, 0);
    if(newPts.size() < nNewFeature * 0.8) {
        int dis = static_cast<int>(radius_ * 0.5);
        mask_ = camMask_.clone();
        for (auto fea : features) {
            const cv::Point2f& pt = fea.pt;
            cv::circle(mask_, pt, dis, 0, -1);
        }
        cv::goodFeaturesToTrack(currFrame_->Image(), newPts, nNewFeature, 0.005, dis, mask_, 3, 0);
    }

    std::vector<int> &idList = currFrame_->IDList();
    for(auto pt : newPts) {
        Feature fea;
        fea.featureID = featureId_++;
        fea.imageID = static_cast<int>(map_->GetKeyFrames().size());
        fea.pt = pt;
        fea.depth = -1.0;
        Eigen::Vector2d uv(fea.pt.x, fea.pt.y);
        Eigen::Vector3d sp = leftCam_->LiftSphere(uv, true);
        fea.spherePt1 = cv::Point3f(static_cast<float>(sp(0)), static_cast<float>(sp(1)), static_cast<float>(sp(2)));
        features.push_back(fea);
        idList.push_back(fea.featureID);
    }
}


bool FrameHandler::PnPIncrement() {
    int endId = static_cast<int>(map_->keyframes_.size())-1;
    if (endId < 0) {
        return false;
    }
    const std::vector<Feature> &features = map_->keyframes_[endId]->Features();
    const auto &mapPoints = map_->mapPoints_;
    std::vector<cv::Point3f> objects;
    std::vector<cv::Point3f> observes;
    for (int i = 0; i < (int)features.size(); ++i) {
        int feaId = features[i].featureID;
        if (feaId < (int)mapPoints.size() && mapPoints[feaId]->pt3dFlag) {
            objects.push_back(mapPoints[feaId]->pt3d);
            observes.push_back(features[i].spherePt1);
        }
    }
    if (observes.size() < 10) {
        return false;
    }
    inslam::pnp::EPNPSolver epnpSolver;
    double reprojThreshold = 5 * leftCam_->GetAngularResolution();
    cv::Mat R, t;
    bool flag = epnpSolver.EPNPRansac(objects, observes, reprojThreshold, R, t, 30);
    std::cout << "pnp inlier:" << objects.size()<< ":" << epnpSolver.GetRansacInlierCount() << std::endl;
    if (!flag) {
        return false;
    }
    std::vector<FramePtr> &keyframesRef = map_->keyframes_;
    keyframesRef[endId]->SetPose(R, t);

    return true;
}


void FrameHandler::CalcStereo(const cv::Mat &Ir) {
    if (Ir.empty() || rightCam_ == nullptr) {
        //spdlog::warn("Can't calc stereo because right image is empty or right camera is null!");
        return;
    }

    std::vector<cv::Mat> pyramidR;
    cv::buildOpticalFlowPyramid(Ir, pyramidR, cv::Size(21, 21), 3, true,
                                cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);

    // Stereo pose
    Eigen::Matrix3d Rrl = rightCam_->Rci_ * leftCam_->Ric_;
    Eigen::Vector3d prl = rightCam_->pci_ + rightCam_->Rci_ * leftCam_->pic_;
    Eigen::Matrix3d Erl = MathUtil::VecToSkew(prl) * Rrl;  // Essential matrix
    std::vector<Eigen::Matrix<double, 3, 4> > poses(2);
    poses[0].setIdentity();
    poses[1] << Rrl, prl;

    // Predict the points on right image
    std::vector<cv::Point2f> leftPts, rightPts;
    std::vector<Eigen::Vector3d> leftRays;
    Eigen::Vector2d uv;
    for (const auto &feature : currFrame_->Features()) {
        leftPts.push_back(feature.pt);
        auto &spt = feature.spherePt1;
        leftRays.emplace_back(Eigen::Vector3d(spt.x, spt.y, spt.z));
        Eigen::Vector3d rightRay = Rrl * leftRays.back();
        rightCam_->Reproject(rightRay, uv, true);
        rightPts.emplace_back(static_cast<float>(uv(0)), static_cast<float>(uv(1)));
    }

    // Optical flow
    std::vector<uchar> status;
    cv::Mat err;
    cv::calcOpticalFlowPyrLK(currFrame_->LeftPyramid(), pyramidR, leftPts, rightPts, status, err, cv::Size(21, 21), 3,
                         cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),
                         cv::OPTFLOW_USE_INITIAL_FLOW);

    // Set stereo observations
    const double minDepth = 1.0 / leftCam_->maxInverseDepth_;
    const double maxDepth = 1.0 / leftCam_->minInverseDepth_;
    const double angularThresh = 2.0 * leftCam_->GetAngularResolution();
    for (size_t i = 0; i < leftPts.size(); ++i) {
        if (!status[i]) {
            continue;
        }
        // check border
        const auto &pt = rightPts[i];
        if (pt.x < 0 || pt.x >= static_cast<float>(leftCam_->width()) - 1 ||
            pt.y < 0 || pt.y >= static_cast<float>(leftCam_->height()) - 1) {
            status[i] = 0;
            continue;
        }

        // check epipolar constraint
        uv << rightPts[i].x, rightPts[i].y;
        Eigen::Vector3d rightRay = rightCam_->LiftSphere(uv, true);
        const Eigen::Vector3d &leftRay = leftRays[i];
        if (!CheckEpipolar(rightRay, leftRay, Erl, angularThresh)) {
            status[i] = 0;
            continue;
        }

        // triangulation
        Eigen::Vector3d xc;
        std::vector<Eigen::Vector3d> rays = {leftRay, rightRay};
        //ComputePoint3dOnSphere(rotationList, translationList, pts, pt3d, err, disparity);

        if (!Triangulator::Solve(rays, poses, angularThresh, xc, false)) {
            status[i] = 0;
            continue;
        }

        // set observation
        double depth = xc.norm();
        if (depth < minDepth || depth > maxDepth) {
            status[i] = 0;
            continue;
        }

        Feature &feat = currFrame_->Features()[i];
        feat.depth = depth;

        MapPointPtr &mp = map_->mapPoints_[feat.featureID];
        if(mp->features.back().imageID == feat.imageID) {
            mp->features.back().depth = feat.depth;
        }
    }

#if 0
    // Debug plot
    cv::Mat visImg;
    hconcat(currFrame_->Image(), Ir, visImg);
    cvtColor(visImg, visImg, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < leftPts.size(); i++) {
        if (!status[i]) {
            continue;
        }
        circle(visImg, leftPts[i], 2, cv::Scalar(0,255,0), 1);
        circle(visImg, rightPts[i]+cv::Point2f(leftCam_->width(), 0), 2, cv::Scalar(0,0,255), 1);
        line(visImg, leftPts[i], rightPts[i]+cv::Point2f(leftCam_->width(), 0), cv::Scalar(255,0,255), 1);
        std::stringstream ss; ss.precision(3);
        ss << currFrame_->Features()[i].depth;
        putText(visImg, ss.str(), leftPts[i], cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255,255,255), 1);
    }
    cv::imshow("stereo corners", visImg);
    cv::waitKey(1);
#endif

}


bool FrameHandler::CheckEpipolar(const Eigen::Vector3d &ray1, const Eigen::Vector3d &ray2,
                                   const Eigen::Matrix3d &E12, double angularThresh) {
    // ray1' * E12 * ray2 = 0
    // ray1' * E12 = [a, b, c]'
    Eigen::RowVector3d coeff = ray1.transpose() * E12;
    double dist = fabs(coeff * ray2) / coeff.norm();
    return dist < angularThresh;
}


void FrameHandler::Update3DPoints() {
    int viewImageID = int(map_->keyframes_.size() - 1);
    const std::vector<int> &ptIdList = map_->keyframes_[viewImageID]->IDList();
    auto &mapPoints = map_->mapPoints_;
    std::vector<int> idList;
    std::vector<double> errList;
    std::vector<double> dispList;
    std::vector<cv::Point3f> pt3dList;
    for (int ii = 0; ii < (int)ptIdList.size(); ++ii) {
        int ptID = ptIdList[ii];
        if (ptID >= (int)mapPoints.size()) {
            continue;
        }
        int cnt = int(mapPoints[ptID]->features.size());
        bool currView = false;
        std::vector<cv::Point3f> pts;
        std::vector<cv::Mat> rotationList;
        std::vector<cv::Mat> translationList;
        pts.resize(cnt);
        rotationList.resize(cnt);
        translationList.resize(cnt);
        int goodCnt = 0;
        for (int j = 0; j < cnt; ++j) {
            int imageID = mapPoints[ptID]->features[j].imageID;
            if (imageID == viewImageID) {
                currView = true;
            }

            map_->keyframes_[imageID]->GetPose(rotationList[goodCnt], translationList[goodCnt]);
            cv::Point3f &sphere_pt = pts[goodCnt];
            sphere_pt = mapPoints[ptID]->features[j].spherePt1;
            ++goodCnt;
        }
        if (!currView) {
            continue;
        }
        if (goodCnt < 2) {
            continue;
        }
        pts.resize(goodCnt);
        rotationList.resize(goodCnt);
        translationList.resize(goodCnt);
        cv::Point3f pt3d;
        double err = 0;
        double disparity = 0;
        ComputePoint3dOnSphere(rotationList, translationList, pts, pt3d, err, disparity);

        cv::Mat currPt3d = rotationList[0] * cv::Mat_<double>(pt3d) + translationList[0];
        if(currPt3d.dot(cv::Mat_<double>(pts[0])) < 0) {
            continue;
        }

        err /= angularResolution_;
        disparity /= angularResolution_;
        idList.push_back(ptID);
        pt3dList.push_back(pt3d);
        errList.push_back(err);
        dispList.push_back(disparity);
    }

    // analize
    {
        if (errList.size() == 0) {
            return;
        }
        std::vector<double> sortErr = errList;
        std::vector<double> sortDisp = dispList;
        std::sort(sortErr.begin(), sortErr.end());
        std::sort(sortDisp.begin(), sortDisp.end());
        double threshErr = sortErr[int(sortErr.size() * 0.5)] * 2 + 1.0;
        double threshDisp = sortDisp[int(sortDisp.size() * 0.1)];
        threshErr = std::min(threshErr, 3.0);
        threshDisp = std::max(threshDisp, 5.0);

        for (int i = 0; i < (int)errList.size(); ++i) {
            int id = idList[i];
            if (errList[i] > threshErr || dispList[i] < threshDisp) {
                mapPoints[id]->pt3dFlag = false;
            } else {
                mapPoints[id]->pt3dFlag = true;
                mapPoints[id]->pt3d = pt3dList[i];
            }
        }
    }
}


void FrameHandler::GetCameraPose(cv::Mat &R, cv::Mat &t) {
    R_.copyTo(R);
    t_.copyTo(t);
}


std::vector<Eigen::Vector3d>  FrameHandler::GetMapPoint(){
    std::vector<Eigen::Vector3d> mapPoints;
    for(auto point : map_->mapPoints_) {
        mapPoints.emplace_back(point->pt3d.x, point->pt3d.y, point->pt3d.z);
    }
    return mapPoints;
}


void FrameHandler::SetSaveSlamPoseFlag(bool flag) {
    saveSlamPoseFlag_ = flag;
}


void FrameHandler::SaveSlamPose(const std::string &path,
                                Eigen::Matrix4d &T_G_C0,
                                Eigen::Matrix4d &T_C_I) {
    if(frameDataset_.empty()) {
        std::cout << "no pose in dataset" << std::endl;
        return;
    }

    LocalOptimization(frameDataset_, map_, angularResolution_);

    // save slam pose
    std::ofstream ofs(path);
    if (ofs.is_open()) {
        for (size_t i = 0; i < frameDataset_.size(); i++) {
            const cv::Mat resR = frameDataset_[i]->R();
            const cv::Mat resT = frameDataset_[i]->T();

            Eigen::Matrix4d T_C0_C = Eigen::Matrix4d::Identity();
            if (!resR.empty() && !resT.empty()) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j)
                        T_C0_C(i, j) = resR.at<double>(i, j);
                    T_C0_C(i, 3) = resT.at<double>(i);
                }
            }
            Eigen::Matrix4d T_G_I =  T_C_I * T_C0_C * T_G_C0;
            // word TO IMUi
            ofs << std::fixed << std::setprecision(6) << frameDataset_[i]->Time()
                << "," << T_G_I(0, 0) << "," << T_G_I(0, 1) << "," << T_G_I(0, 2)
                << "," << T_G_I(1, 0) << "," << T_G_I(1, 1) << "," << T_G_I(1, 2)
                << "," << T_G_I(2, 0) << "," << T_G_I(2, 1) << "," << T_G_I(2, 2)
                << "," << T_G_I(0, 3) << "," << T_G_I(1, 3) << "," << T_G_I(2, 3) << "\n";
        }
        std::cout << "save slam pose to:" << path << std::endl;
        ofs.close();
    }
}


void FrameHandler::DebugShow(int showType) {
    if(currFrame_ == nullptr || lastFrame_ == nullptr) {
        return;
    }
#if 0
    // show depth
    if(showType == 0) {
        // show depth
        float imgScale = 1.5;
        if(lastFrame_ != nullptr && currFrame_ != nullptr) {
            cv::Mat visImg;
            cvtColor(currFrame_->Image(), visImg, cv::COLOR_GRAY2BGR);
            if(imgScale != 1) {
                cv::resize(visImg, visImg, cv::Size(visImg.cols * imgScale, visImg.rows * imgScale));
            }
            const std::vector<Feature> currFeats = currFrame_->Features();

            for (size_t i = 0; i < currFeats.size(); i++) {
                int currFeatId = currFeats[i].featureID;
                cv::circle(visImg, currFeats[i].pt * imgScale, 2, cv::Scalar(0,255,0), 1);
                if(map_->mapPoints_[currFeatId]->pt3dFlag) {
                    //double depth = cv::norm(R_ * cv::Mat_<double>(map_->mapPoints_[currFeatId]->pt3d) + t_);
                    cv::Mat pt = R_ * cv::Mat_<double>(map_->mapPoints_[currFeatId]->pt3d) + t_;
                    char text[64];
                    sprintf(text, "%.1f", /*pt.at<double>(0), pt.at<double>(1), */pt.at<double>(2));
                    cv::putText(visImg, text, currFeats[i].pt * imgScale, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,0));
                }
            }
            cv::imshow("corners depth", visImg);
            cv::waitKey(1);
        }
    }
#endif

#if 0
    // show track
    if(showType == 1) {
        // show track
        if(lastFrame_ != nullptr && currFrame_ != nullptr) {
            cv::Mat visImg;
            hconcat(currFrame_->Image(), lastFrame_->Image(), visImg);
            cvtColor(visImg, visImg, cv::COLOR_GRAY2BGR);
            const std::vector<Feature> currFeats = currFrame_->Features();
            const std::vector<Feature> lastFeats = lastFrame_->Features();

            for (size_t i = 0; i < lastFeats.size(); i++) {
                int lastFeatId = lastFeats[i].featureID;
                bool tracked = false;
                for(size_t j = 0; j < currFeats.size(); j++) {
                    if(lastFeatId == currFeats[j].featureID) {
                        cv::circle(visImg, currFeats[j].pt, 2, cv::Scalar(0,255,0), 1);
                        cv::circle(visImg, lastFeats[i].pt+cv::Point2f(currFrame_->Width(), 0), 2, cv::Scalar(0,0,255), 1);
                        cv::line(visImg, currFeats[j].pt, lastFeats[i].pt+cv::Point2f(currFrame_->Width(), 0), cv::Scalar(255,0,255), 1);
                        tracked = true;
                    }
                }
                if(!tracked) {
                    cv::circle(visImg, lastFeats[i].pt+cv::Point2f(currFrame_->Width(), 0), 4, cv::Scalar(255,0,0), 1);
                }
            }
            cv::imshow("track1 corners", visImg);
            std::cout << "curr feature size:" << currFrame_->Features().size() << std::endl;
            cv::waitKey(1);
        }
    }
#endif

#if 1
    // Visualize debug info
    if(showType == 2) {
        cv::Mat visImg;
        double scale = 960.0 / static_cast<double>(currFrame_->Image().cols);
        cv::resize(currFrame_->Image(), visImg, cv::Size(960, int(currFrame_->Image().rows * scale)));
        cv::cvtColor(visImg, visImg, cv::COLOR_GRAY2BGR);
        for (const auto &feature : currFrame_->Features()) {
            cv::circle(visImg, feature.pt * scale, 3, cv::Scalar(0,0,255), -1);
        }

        imshow("frame", visImg);
        cv::waitKey(1);
    }
#endif

}


void FrameHandler::InsertImu(double gyro[3], double acc[3], double timestamp) {
    Eigen::Map<Eigen::Vector3d> gyro0(gyro);
    Eigen::Map<Eigen::Vector3d> acc0(acc);
    if(doCalibrateImuEx_) {
        imuCalib_->InsertImu(gyro0, acc0, timestamp);
        return;
    }

    doImuAligment_ = true;
    if (refImustate_.last_imu_time > 0) {
        double dt = timestamp - refImustate_.last_imu_time;
        if(dt < 1e-7) {
            return;
        }
        refImustate_.last_imu_time = timestamp;
        refImustate_.imu0.IntegrateNewImu(acc0, gyro0, dt);
    }

    // for static initialize
    if(staticAccDatas_.size() > 150) {
        staticGyrDatas_.pop_front();
        staticAccDatas_.pop_front();
    }
    staticGyrDatas_.push_back(gyro0);
    staticAccDatas_.push_back(acc0 * 10.0);
}


void FrameHandler::SetFinishedFrame(double timestamp) {
    if (refImustate_.last_imu_time > 0) {
        double dt = timestamp - refImustate_.last_imu_time;
        Eigen::Vector3d gyro0 = refImustate_.imu0.raw_data_.back().gyro_;
        Eigen::Vector3d acc0 = refImustate_.imu0.raw_data_.back().acc_;
        refImustate_.imu0.IntegrateNewImu(acc0, gyro0, dt);
        refImustate_.last_imu_time = timestamp;
        refImustate_.timestamp = timestamp;
        lastImustate_ = refImustate_;
    }

    newVioFlag_ = true;
    refImustate_.imu0 = Preintegrated(imuParam_.ba, imuParam_.bg, imuParam_.acc_n, imuParam_.gyro_n, imuParam_.ba_n, imuParam_.bg_n, 0.98);
    refImustate_.last_imu_time = timestamp;
}


void FrameHandler::SetImuParam(double an, double gn, double ban, double bgn) {
    imuParam_.acc_n = an;
    imuParam_.gyro_n = gn;
    imuParam_.ba_n = ban;
    imuParam_.bg_n = bgn;
    imuParam_.ba.setZero();
    imuParam_.bg.setZero();
}


void FrameHandler::SetImuExtrinsic(const Eigen::Matrix3d &Rcb, const Eigen::Vector3d &tcb) {
    imuInitializing_.SetExtrinsic(Rcb, tcb);
}


void FrameHandler::SetInitBa(const Eigen::Vector3d &ba) {
    initBa_ = ba;
    imuInitializing_.SetBa(initBa_);
}

void FrameHandler::SetInitBg(const Eigen::Vector3d &bg) {
    imuCalib_->SetBg(bg);
}


void FrameHandler::GlobalStateRecovery() {
#if 0
    double scale = imuInitializing_.GetScale();
    for (int i = 0; i < (int)map_->keyframes_.size(); i++) {
        FramePtr frame = map_->keyframes_[i];
        frame->SetTraslationScale(scale);
    }
    for (int i = 0; i < (int)map_->mapPoints_.size(); i++) {
        MapPointPtr &map_pt = map_->mapPoints_[i];
        if (map_pt->pt3dFlag) {
            map_pt->pt3d *= scale;
        }
    }

    imuParam_.ba = imuInitializing_.GetBa();
    imuParam_.bg = imuInitializing_.GetBg();
    imuParam_.Rcb = imuInitializing_.GetRotationExtrinsic();
    imuParam_.tcb.setZero();
    imuParam_.Gw = imuInitializing_.GetGravityVector();

    Eigen::Vector3d Gw = imuInitializing_.GetGravityVector();
    Eigen::Quaterniond q_c0_g = MathUtil::GetQfromA(Gw);
    cv::Mat R_c0_g(3, 3, CV_64F);
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> eig_R_c0_g((double*)R_c0_g.data);
    eig_R_c0_g = q_c0_g.matrix();

    cv::Mat R_g_c0 = R_c0_g.t();
    for(auto kf : map_->keyframes_) {
        cv::Mat R_c0_ci,t_c0_ci;
        kf->GetPose(R_c0_ci, t_c0_ci);
        cv::Mat R_g_ci = R_c0_ci * R_g_c0;
        kf->SetPose(R_g_ci, t_c0_ci);
    }

    for(auto mapPoint : map_->mapPoints_) {
        if(mapPoint->pt3dFlag) {
            cv::Mat pt3d = R_c0_g * cv::Mat_<double>(mapPoint->pt3d);
            mapPoint->pt3d = cv::Point3f(pt3d);
        }
    }
#endif

}


void FrameHandler::ImuInitInsert(Frame frame, int frameId, double timestamp) {
    if (doStereo_) {
        Frame frame0;
        if (imuInitializing_.GetLastFrame(frame0)) {
            cv::Mat R0, R1, t0, t1;
            frame0.GetPose(R0, t0);
            frame.GetPose(R1, t1);
            cv::Mat dt = R1.t() * t1 - R0.t() * t0;
            double move = cv::norm(dt);
            if (move > 0.03) {
                SetFinishedFrame(timestamp);
                frame.imu_state = lastImustate_;
                frame.SetFrameId(frameId);
                imuInitializing_.InsertFrame(frame);
            }
        } else {
            SetFinishedFrame(timestamp);
            frame.imu_state = lastImustate_;
            frame.SetFrameId(frameId);
            if(frame.imu_state.last_imu_time > 0) {
                imuInitializing_.InsertFrame(frame);
            }
        }
        return;
    }
    SetFinishedFrame(timestamp);
    frame.imu_state = lastImustate_;
    std::cout.precision(13);
    frame.SetFrameId(frameId);
    if(frame.imu_state.last_imu_time > 0) {
        imuInitializing_.InsertFrame(frame);
    }
}


void FrameHandler::EnableCalibrateImu() {
    doCalibrateImuEx_ = true;
}


bool FrameHandler::TryCalibImu() {
    if(currFrame_ == nullptr) {
        imuCalib_->Reset();
        return false;
    }
    if(state_ == InitTrackState::INITIALIZED) {
        imuCalib_->InsertFrame(currFrame_);
    }
    if(imuCalib_->Calibrate()) {
        imuCalib_->Reset();
        return true;
    }
    return false;
}


} //namespace inslam
