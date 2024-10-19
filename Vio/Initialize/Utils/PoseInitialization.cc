#include <numeric>
//#include "spdlog/spdlog.h"
#include "Vio/Initialize/Utils/PoseInitialization.h"
#include "Vio/Initialize/Utils/EssentialRansac.h"
#include "Vio/Initialize/Utils/Triangulation.h"

namespace featslam{
PoseInitialization::PoseInitialization(const std::shared_ptr<Map> &map) {
    map_ = map;
}



bool PoseInitialization::GoodPoseInit(const std::vector<cv::Point3f> &pts1,
                                      const std::vector<cv::Point3f> &pts2,
                                      const cv::Mat &R,
                                      const cv::Mat &t) {
    std::vector<double> depthList;
    std::vector<double> errList;
    ComputeDepthsOnSphere(pts1, pts2, R, t, depthList);
    ComputeProjectErrorsOnSphere(pts1, depthList, R, t, pts2, errList);
    std::vector<double> tmpErrList;
    for (int i = 0; i < errList.size(); i += 3) {
        double x = errList[i];
        double y = errList[i + 1];
        double z = errList[i + 2];
        double err = sqrt(x * x + y * y + z * z);
        tmpErrList.push_back(err);
    }
    errList = tmpErrList;

    sort(tmpErrList.begin(), tmpErrList.end());
    int id = static_cast<int>(tmpErrList.size() * 0.9);
    if (id <= 0) {
        return false;
    }
    double threshold = tmpErrList[id] / 0.9;
    double angularResolution = CV_PI / map_->keyframes_[0]->Height();
    threshold = std::min(threshold, angularResolution * 5);
    std::vector<double> tmpDList;
    double tNorm = cv::norm(t);
    for (int i = 0; i < depthList.size(); ++i) {
        double d = depthList[i];
        if (d < tNorm) {
            d = 100;
        }
        if (errList[i] > threshold) {
            continue;
        }
        tmpDList.push_back(d);
    }
    if (tmpDList.size() < 10 || tmpDList.size() < depthList.size() * 0.6) {
        //spdlog::info("size1, size2: {}\t{}", tmpDList.size(), depthList.size());
        std::cout << "size1, size2: " <<  tmpDList.size() << ": " << depthList.size() << std::endl;
        return false;
    }
    std::sort(tmpDList.begin(), tmpDList.end());
    int id1 = static_cast<int>(tmpDList.size() * 0.5);
    double avgDepth1 = tmpDList[id1];
    //spdlog::info("average depth: {}\n", avgDepth1);
    std::cout << "average depth: " <<  avgDepth1 << std::endl;
    if (avgDepth1 > 50) {
        return false;
    }
    return true;
}

void GetCommonViewPoints(const std::vector<Feature> &features1,
                         const std::vector<Feature> &features2,
                         std::vector<cv::Point3f> &spherePts1,
                         std::vector<cv::Point3f> &spherePts2,
                         std::vector<int> &idList) {
    idList.resize(features1.size());
    spherePts1.resize(features1.size());
    spherePts2.resize(features1.size());
    int startI2 = 0;
    int id = 0;
    int cnt = 0;
    for (int i1 = 0; i1 < features1.size(); ++i1) {
        for (int i2 = startI2; i2 < features2.size(); ++i2) {
            ++cnt;
            if (features1[i1].featureID == features2[i2].featureID) {
                idList[id] = features1[i1].featureID;
                spherePts1[id] = features1[i1].spherePt1;
                spherePts2[id] = features2[i2].spherePt1;
                startI2 = i2;
                ++id;
                break;
            }
            if (features1[i1].featureID < features2[i2].featureID) {
                startI2 = i2;
                break;
            }
        }
    }
    idList.resize(id);
    spherePts1.resize(id);
    spherePts2.resize(id);
}

bool PoseInitialization::Run() {
    std::vector<int> idList;
    std::vector<cv::Point3f> spherePts1, spherePts2;
    const std::vector<Feature> &features1 = map_->keyframes_[0]->Features();
    const std::vector<Feature> &features2 = map_->keyframes_[1]->Features();
    GetCommonViewPoints(features1, features2, spherePts1, spherePts2, idList);
    cv::Mat R, t;
    float essentialThreshold = static_cast<float>(3 * CV_PI / map_->keyframes_[0]->Height());
    FindRTFromSpherePairs(spherePts1, spherePts2, R, t, essentialThreshold);
    map_->keyframes_[1]->SetPose(R, t);
    bool goodInitFlag = GoodPoseInit(spherePts1, spherePts2, R, t);
    return goodInitFlag;
}

bool PoseInitialization::RunStereo(std::vector<FramePtr> &initFrameList) {
    int validPointCnt = 0;
    auto &mapPoints = map_->mapPoints_;
    int mapPtSize = static_cast<int>(mapPoints.size());
    std::vector<std::vector<double>> allDepths(mapPtSize, std::vector<double>());
    std::vector<cv::Mat> spherePt0List(mapPtSize, cv::Mat());
    for(int i = 0; i < (int)initFrameList.size(); i++) {
        cv::Mat R;
        cv::Mat t;
        initFrameList[i]->GetPose(R, t);
        const std::vector<Feature> &features = initFrameList[i]->Features();
        for(int j = 0; j < (int)features.size(); j++) {
            auto &fea = features[j];
            int featureID = fea.featureID;
            if(featureID > mapPtSize - 1) {
                continue;
            }

            if(fea.depth > 0) {
                cv::Mat pt3d1(3, 1, CV_64F);
                pt3d1.at<double>(0, 0) = fea.spherePt1.x * fea.depth;
                pt3d1.at<double>(1, 0) = fea.spherePt1.y * fea.depth;
                pt3d1.at<double>(2, 0) = fea.spherePt1.z * fea.depth;
                cv::Mat pt3d0 = R.t() * pt3d1 - R.t() * t;
                double depth = cv::norm(pt3d0);

                allDepths[featureID].push_back(depth);
                if(spherePt0List[featureID].empty()) {
                    spherePt0List[featureID] = cv::Mat_<double>(mapPoints[featureID]->features[0].spherePt1);
                }
            }
        }
    }

    for(int i = 0; i < (int)allDepths.size(); i++) {
        std::vector<double> &depths = allDepths[i];
        cv::Mat &spherePt0 = spherePt0List[i];
        if(depths.empty()) {
            continue;
        }

        // remove stereo depth with large variance
        cv::Mat pt3d0;
        if(depths.size() == 1) {
            pt3d0 = spherePt0 * float(depths[0]);
        } else {
            double sum = std::accumulate(std::begin(depths), std::end(depths), 0.0);
            double mean =  sum / depths.size();
            double accum  = 0.0;
            std::for_each(std::begin(depths), std::end(depths), [&](const double d) {
                accum  += (d-mean)*(d-mean);
            });
            double variance = accum / (depths.size()-1);
            if(variance / (mean * mean) > 0.05) {
                mapPoints[i]->pt3dFlag = false;
                continue;
            }  else {
                std::sort(depths.begin(), depths.end());
                double depth = depths[depths.size() / 2];
                pt3d0 = spherePt0 * float(depth);
            }
        }

        mapPoints[i]->pt3dFlag = true;
        mapPoints[i]->pt3d = cv::Point3f(pt3d0);
        validPointCnt++;
    }

    if(validPointCnt < 50) {
        //spdlog::info("Stereo initialize depth count: {}\n", validPointCnt);
        std::cout << "Stereo initialize depth count: " << validPointCnt << std::endl;
        return false;
    }

    return true;
}
}  // namespace inslam
