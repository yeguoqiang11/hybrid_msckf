//
// Created by d on 2021/1/22.
//

#include "Vio/ImageProcessor.h"
#include "Utils/PerformanceTest.h"
#include "Utils/MathUtil.h"
#include "Vio/Triangulator.h"
#include <sstream>
//#include "spdlog/spdlog.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace inslam {

//TODO: move this function into a util class
template <typename T>
void ReduceVector(std::vector<T>& v, std::vector<uchar>& status) {
    int j = 0;
    for (int i=0; i<(int)(v.size()); i++) {
        if (status[i]) {
            v[j++] = v[i];
        }
    }
    v.resize(j);
}


ImageProcessor::ImageProcessor(const nlohmann::json &config, std::shared_ptr<Caimura> cam)
    : cam_(std::move(cam)), status_(NOT_INITIALIZED) {
    maxCorners_ = config["max_corners"];
    radius_ = config["radius"];
    fastThreshold_ = config["fast_threshold"];
    border_ = config["border"];
    pyramidLevels_ = config["pyramid_levels"];
    useImu_ = config["use_imu"].get<bool>();
    string detectorStr = config["detector"].get<string>();
    if (detectorStr == "fast") {
        detectorType_ = DetectorType::FAST;
    } else {
        detectorType_ = DetectorType::GFTT;
    }
    cout << "ImageProcessor created with parameters:"
           << "\n\tmax corners: " << maxCorners_ << "\n\traidus: " << radius_
           << "\n\tdetector type: " << detectorStr << "\n\tfast threshold: " << fastThreshold_
           << "\n\tborder: " << border_ << "\n\tpyramid levels: " << pyramidLevels_
           << "\n\tuse imu: " << useImu_ << endl;

    CreateMask();

    // Grid the image to uniform distribute the features
    if (detectorType_ == DetectorType::FAST) {
        detectGrid_ = std::make_shared<FeatureGrid>(cam_->width(), cam_->height(), radius_);
        if(detectGrid_->GetGridNum() < static_cast<int>(maxCorners_ * 6)) {
            cerr << "Warning: too few grids, please reduce radius" << endl;
        }
        detectGrid_->SetMask(mask_);
    }
    trackGrid_ = std::make_shared<FeatureGrid>(cam_->width(), cam_->height(), radius_);
    trackGrid_->SetMask(mask_);
}


bool ImageProcessor::Process(const Mat &I, size_t frameId, double timestamp,
                                           const Mat &Ir, shared_ptr<Caimura> rightCam) {
    // Reset tracking statistics (only for debug)
    numTrackedFeatures_ = 0;
    numNewFeatures_ = 0;
    trackingRatio_ = 0;

    // Update current image
    currFrameId_ = frameId;
    currTimestamp_ = timestamp;
    CreateImagePyramid(I, currPyramid_);

    // Track from previous frame
    if (status_ == ImageProcessor::TRACKING) {
        Track();
    }

    // Remove lost features
    if (!features_.empty()) {
        vector<uchar> flags(features_.size(), 255);
        for (size_t i = 0; i < features_.size(); ++i) {
            if (features_[i].frameId != currFrameId_) {
                flags[i] = 0;
            }
        }
        ReduceVector(features_, flags);
    }

    // Add new corners
    AddNewFeatures(I);

    // Stereo matching and triangulation
    if (!Ir.empty() && rightCam != nullptr) {
        CalcStereo(Ir, rightCam);
    }

    // Prepare for next round
    prevTimestamp_ = timestamp;
    std::swap(prevPyramid_, currPyramid_);
    if (status_ == ImageProcessor::NOT_INITIALIZED) {
        status_ = ImageProcessor::TRACKING;
    }
    return true;
}


vector<FeatureObservation> ImageProcessor::GetActiveFeatures() {
    return features_;
}


void ImageProcessor::Track() {
    // Use imu integration to calculate the relative rotation between last frame and current frame.
    // And predict locations of feature points on current image.
    if (useImu_) {
        cerr << "TODO: implement IMU-aided tracking!" << endl;
    }

    vector<Point2f> prevPts;
    for (const auto &feature : features_) {
        prevPts.push_back(feature.pt0);
    }

    // forward optical flow
    vector<Point2f> currPts;
    vector<uchar> status;
    Mat err;
    cv::calcOpticalFlowPyrLK(prevPyramid_, currPyramid_, prevPts, currPts, status, err,Size(21, 21), 3,
                             TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01));

    // backward optical flow
    vector<Point2f> prevPts2;
    vector<uchar> status2;
    Mat err2;
    cv::calcOpticalFlowPyrLK(currPyramid_, prevPyramid_, currPts, prevPts2, status2, err2,Size(21, 21), 3,
                             TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01));

    const double camRadius = cam_->radius();
    const Point2f camCenter(static_cast<float>(cam_->cx()), static_cast<float>(cam_->cy()));
    for (size_t i = 0; i <prevPts.size(); ++i) {
        if (!status2[i] || cv::norm(prevPts2[i] - prevPts[i]) > 2
            || currPts[i].x < 0 || currPts[i].x >= static_cast<float>(cam_->width()) - 1
            || currPts[i].y < 0 || currPts[i].y >= static_cast<float>(cam_->height()) - 1) {
            status[i] = 0;
        }
        if (camRadius > 0 && cv::norm(currPts[i] - camCenter) > camRadius) {
            status[i] = 0;
        }
    }

    //TODO: fundamental matrix ransac for sphere points
    if (cam_->IsPerspective() || cam_->IsOpencvFisheye()) {
        vector<size_t> indices;
        vector<Point2f> pts1, pts2;
        for (size_t i = 0; i < features_.size(); ++i) {
            if (status[i]) {
                indices.push_back(i);
                pts1.push_back(cam_->UndistortPoint(prevPts[i]));
                pts2.push_back(cam_->UndistortPoint(currPts[i]));
            }
        }
        if (indices.size() > 10) {
            vector<uchar> rscStatus;
            cv::findFundamentalMat(pts1, pts2, FM_RANSAC, 1, 0.999, rscStatus);
            for (size_t i = 0; i < indices.size(); i++) {
                if (!rscStatus[i]) {
                    status[indices[i]] = 0;
                }
            }
        }
    }

    numTrackedFeatures_ = 0;
    trackGrid_->ResetGrid(false);
    for (size_t i = 0; i < features_.size(); ++i) {
        if (status[i]) {
            // Uniform distribution
            Point2f pt = currPts[i];
            if(trackGrid_->GetOccupancyState(pt)) {
                continue;
            }
            trackGrid_->SetGridOccpuancy(pt);

            Point2f upt = cam_->UndistortPoint(pt);
            Vector3d ray = cam_->LiftSphere(Vector2d(pt.x, pt.y), true);
            FeatureObservation &fo = features_.at(i);
            fo.frameId = currFrameId_;
            fo.pt0 = pt;
            fo.upt0 = upt;
            fo.ray0 = ray;
            numTrackedFeatures_++;
        }
    }
    trackingRatio_ = numTrackedFeatures_ / (static_cast<double>(features_.size()) + 0.001);
}


void ImageProcessor::AddNewFeatures(const cv::Mat &img) {
    int numWanted = maxCorners_ - static_cast<int>(features_.size());
    if (numWanted < 3) {
        return;
    }

    // Detect corners
    vector<Point2f> newPts;
    if (detectorType_ == DetectorType::GFTT) {
        // Mask out the neighbor area of existed features
        cv::Mat mask = mask_.clone();
        for (const auto &feature : features_) {
            const Point2f &pt = feature.pt0;
            cv::circle(mask, pt, radius_, 0, -1);
        }
        cv::goodFeaturesToTrack(img, newPts, numWanted, 0.01, radius_, mask);
    }
    else {
        vector<KeyPoint> keypts;
        cv::FAST(img, keypts, 10, true);

        // draw grid
        detectGrid_->ResetGrid(false);
        detectGrid_->SetExistingFeatures(features_, true);

        vector<pair<float, Point2f>> gridKeypts(detectGrid_->GetGridNum(), make_pair(-1.0f, Point2f()));
        for (const auto &key : keypts) {
            if (!mask_.at<uchar>(static_cast<int>(key.pt.y), static_cast<int>(key.pt.x)) ||
                    detectGrid_->GetOccupancyState(key.pt)) {
                continue;
            }

            int gridCode = detectGrid_->GetCellIndex(key.pt.x, key.pt.y);
            if (key.response > gridKeypts[gridCode].first) {
                gridKeypts[gridCode] = make_pair(key.response, key.pt);
            }
        }

        int i1 = 0;
        for(auto &kp : gridKeypts) {
            if(kp.first > 0) {
                gridKeypts[i1++] = kp;
            }
        }
        gridKeypts.resize(i1);

        sort(gridKeypts.begin(), gridKeypts.end(),
             [&](const pair<float, Point2f>& key1, const pair<float, Point2f> &key2) { return key1.first > key2.first; });

        for (const auto& kp : gridKeypts) {
            const cv::Point2f &pt = kp.second;
            if(detectGrid_->GetOccupancyState(pt)) {
                continue;
            }
            detectGrid_->SetGridOccpuancy(pt, true);
            newPts.push_back(pt);
            if (static_cast<int>(newPts.size()) >= numWanted) {
                break;
            }
        }
    }

    if (newPts.empty()) {
        //spdlog::warn("failed to detect corners on this image!");
        return;
    }

    // Undistort
    vector<Point2f> newPtsUn;
    cam_->UndistortPts(newPts, newPtsUn);

    // Add new features
    for (size_t i = 0; i < newPts.size(); ++i) {
        size_t featureId = featureIdGenerator_++;

        FeatureObservation fo;
        fo.id = featureId;
        fo.frameId = currFrameId_;
        fo.pt0 = newPts[i];
        fo.upt0 = newPtsUn[i];
        fo.ray0 = cam_->LiftSphere(Vector2d(fo.pt0.x, fo.pt0.y), true);

        features_.push_back(fo);
    }
    numNewFeatures_ = static_cast<int>(newPts.size());
}


void ImageProcessor::StereoFlow(const vector<Mat> &pyramid0, const vector<Mat> &pyramid1,
                                const shared_ptr<Caimura> &cam0, const shared_ptr<Caimura> &cam1,
                                const vector<Point2f> &pts0, vector<Point2f> &pts1,
                                vector<uchar> &status) {
    // Stereo Pose
    Matrix3d Rrl = cam1->Rci_ * cam0->Ric_;
    Vector3d prl = cam1->pci_ + cam1->Rci_ * cam0->pic_;

    // Predict points in right image
    pts1.clear();
    for (const auto &pt : pts0) {
        Vector3d ray0 = cam0->LiftSphere(Vector2d(pt.x, pt.y), true);
        Vector3d ray1 = Rrl * ray0;
        Vector2d pt1;
        cam1->Reproject(ray1, pt1, true);
        pts1.emplace_back(static_cast<float>(pt1(0)), static_cast<float>(pt1(1)));
    }

    // Optical flow
    Mat err;
    calcOpticalFlowPyrLK(pyramid0, pyramid1, pts0, pts1, status, err, Size(21, 21), 3,
                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                         cv::OPTFLOW_USE_INITIAL_FLOW);

    // Mark points outside the image
    const int cols = pyramid1[0].cols;
    const int rows = pyramid1[0].rows;
    for (size_t i = 0; i < pts1.size(); ++i) {
        const auto &pt = pts1.at(i);
        if (pt.x < 0 || pt.y < 0 || pt.x > cols - 1 || pt.y > rows - 1) {
            status[i] = 0;
        }
    }

}


void ImageProcessor::CalcStereo(const Mat &Ir, shared_ptr<Caimura> &rightCam) {
    if (Ir.empty() || rightCam == nullptr) {
        //spdlog::warn("Can't calc stereo because right image is empty or right camera is null!");
        return;
    }

    vector<Mat> rightPyramid;
    CreateImagePyramid(Ir, rightPyramid);

    // Stereo pose
    Matrix3d Rrl = rightCam->Rci_ * cam_->Ric_;
    Vector3d prl = rightCam->pci_ + rightCam->Rci_ * cam_->pic_;
    Matrix3d Erl = MathUtil::VecToSkew(prl) * Rrl;  // Essential matrix
    vector<Matrix<double, 3, 4> > poses(2);
    poses[0].setIdentity();
    poses[1] << Rrl, prl;

    // Points on left image
    vector<size_t> indices;
    vector<Point2f> leftPts;
    for (size_t i = 0; i < features_.size(); ++i) {
        const auto &feature = features_.at(i);
        indices.push_back(i);
        leftPts.push_back(feature.pt0);
    }
    if (leftPts.empty()) {
        cerr << "No features on left image!" << endl;
        return;
    }

    // Left->Right optical flow
    vector<uchar> status;
    vector<Point2f> rightPts;
    StereoFlow(currPyramid_, rightPyramid, cam_, rightCam, leftPts, rightPts, status);
    ReduceVector(indices, status);
    ReduceVector(leftPts, status);
    ReduceVector(rightPts, status);
    if (leftPts.empty()) {
        cerr << "No features after left->right optical flow" << endl;
        return;
    }

    // Right->Left optical flow
    status.clear();
    vector<Point2f> leftPts2;
    StereoFlow(rightPyramid, currPyramid_, rightCam, cam_, rightPts, leftPts2, status);
    for (size_t i = 0; i < leftPts.size(); ++i) {
        if (cv::norm(leftPts[i] - leftPts2[i]) > 1) {
            status[i] = 0;
        }
    }

    // Set stereo observations
    const double minDepth = 1.0 / cam_->maxInverseDepth_;
    const double maxDepth = 1.0 / cam_->minInverseDepth_;
    const double angularThresh = 2.0 * cam_->GetAngularResolution();
    for (size_t i = 0; i < leftPts.size(); ++i) {
        if (!status[i]) {
            continue;
        }

        const Point2f &pt0 = leftPts.at(i);
        const Point2f &pt1 = rightPts.at(i);
        Vector3d leftRay = cam_->LiftSphere(Vector2d(pt0.x, pt0.y), true);
        Vector3d rightRay = rightCam->LiftSphere(Vector2d(pt1.x, pt1.y), true);

        // check epipolar constraint
        if (!CheckEpipolar(rightRay, leftRay, Erl, angularThresh)) {
            status[i] = 0;
            continue;
        }

        // triangulation
        Vector3d xc;
        vector<Vector3d> rays = {leftRay, rightRay};
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

        FeatureObservation& fo = features_.at(indices[i]);
        fo.pt1 = pt1;
        fo.upt1 = rightCam->UndistortPoint(pt1);
        fo.ray1 = rightRay;
        fo.stereoDepth = depth;
        fo.isStereo = true;
    }

#if 0
    // Debug plot
    Mat visImg;
    hconcat(currPyramid_[0], Ir, visImg);
    cvtColor(visImg, visImg, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < leftPts.size(); i++) {
        if (!status[i]) {
            continue;
        }
        circle(visImg, leftPts[i], 2, Scalar(0,255,0), 1);
        circle(visImg, rightPts[i]+Point2f(cam_->width(), 0), 2, Scalar(0,0,255), 1);
        line(visImg, leftPts[i], rightPts[i]+Point2f(cam_->width(), 0), Scalar(255,0,255), 1);
        stringstream ss; ss.precision(3);
        ss << features_[indices[i]].stereoDepth;
        putText(visImg, ss.str(), leftPts[i], cv::FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255,255,255), 1);
    }
    cv::imshow("stereo corners", visImg);
    waitKey(2);
#endif

}


void ImageProcessor::CreateImagePyramid(const Mat &I, vector<Mat> &pyramid) {
    int patchSize = 21;
    cv::buildOpticalFlowPyramid(I, pyramid, Size(patchSize, patchSize), pyramidLevels_,
                                true, BORDER_REFLECT_101, BORDER_CONSTANT, false);
}


void ImageProcessor::CreateMask() {
    mask_ = Mat::zeros(cam_->height(), cam_->width(), CV_8U);
    if (cam_->radius() > 0) {
        circle(mask_, Point(static_cast<int>(cam_->cx()), static_cast<int>(cam_->cy())),
               static_cast<int>(cam_->radius()), Scalar(255), -1);
    } else {
        mask_.setTo(255);
    }

    // Mask out border
    int xBorder = border_, yBorder = border_;
    if (cam_->IsEquirectangular()) {
        // Mask out the top & bottom region of panoramic image
        yBorder = std::max(yBorder,  static_cast<int>(cam_->height() * 0.1) );
    }
    mask_.rowRange(0, yBorder).setTo(0);
    mask_.rowRange(cam_->height()-yBorder, cam_->height()).setTo(0);
    mask_.colRange(0, xBorder).setTo(0);
    mask_.colRange(cam_->width()-xBorder, cam_->width()).setTo(0);
}


void ImageProcessor::DrawDebugInfo(cv::Mat &img) {
    if (img.empty()) {
        //spdlog::error("Can't draw debug info while the InputOutput image is empty!");
        return;
    }

    // Draw features
    for (const auto &feature : features_) {
        cv::circle(img, feature.pt0, 2, Scalar(0,0,255), -1);
    }

    // Draw tracking statistics
    std::stringstream ss;
    ss.precision(2);
    ss << "tracking: tracked=" << numTrackedFeatures_ 
        << " ratio=" << trackingRatio_
        << " new=" << numNewFeatures_;
    putText(img, ss.str(), Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255,255,255), 1);
}


bool ImageProcessor::CheckEpipolar(const Vector3d &ray1, const Vector3d &ray2,
                                   const Matrix3d &E12, double angularThresh) {
    // ray1' * E12 * ray2 = 0
    // ray1' * E12 = [a, b, c]'
    RowVector3d coeff = ray1.transpose() * E12;
    double dist = fabs(coeff * ray2) / coeff.norm();
    return dist < angularThresh;
}

}//namespace inslam {