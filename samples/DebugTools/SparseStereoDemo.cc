//
// Created by d on 2021/3/19.
//

#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "Utils/VideoProvider.h"
#include "Utils//PerformanceTest.h"
#include "Utils/DataIOUtil.h"
#include "Utils/json.hpp"
#include "Utils/MathUtil.h"
#include "Vio/Caimura.hpp"
#include "Vio/Initialize/Utils/EssentialRansac.h"
#include "Vio/Initialize/Utils/Triangulation.h"


using namespace cv;
using namespace std;
using namespace Eigen;
using namespace inslam;

nlohmann::json ParseJsonFile(const std::string &file) {
    ifstream ifs(file);
    if (!ifs.is_open()) {
        cerr << "failed to open config file: " << file << endl;
        exit(-1);
    }
    nlohmann::json config = nlohmann::json::parse(ifs);
    ifs.close();
    return config;
}

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


void StereoFlow(const cv::Mat &imgL, const cv::Mat &imgR,
                const shared_ptr<Caimura> &camL, const shared_ptr<Caimura> &camR,
                const vector<Point2f> &ptsL, vector<Point2f> &ptsR, vector<uchar> &status) {
    // Stereo Pose
    Matrix3d Rrl = camR->Rci_ * camL->Ric_;
    Vector3d prl = camR->pci_ + camR->Rci_ * camL->pic_;

    ptsR.clear();
    for (const auto &pt : ptsL) {
        Vector3d leftRay = camL->LiftSphere(Vector2d(pt.x, pt.y), true);
        Vector3d rightRay = Rrl * leftRay;
        Vector2d ptR;
        camR->Reproject(rightRay, ptR, true);
        ptsR.emplace_back(static_cast<float>(ptR(0)), static_cast<float>(ptR(1)));
    }

    // Optical flow
    Mat err;
    calcOpticalFlowPyrLK(imgL, imgR, ptsL, ptsR, status, err, Size(31, 31), 3,
                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                         cv::OPTFLOW_USE_INITIAL_FLOW);

    // Remove points outside the image
    for (size_t i = 0; i < ptsR.size(); ++i) {
        const auto &pt = ptsR.at(i);
        if (pt.x < 0 || pt.y < 0 || pt.x > imgR.cols -1 || pt.y >= imgR.rows - 1) {
            status[i] = 0;
        }
    }

}


void CalcStereo(const cv::Mat &imgL, const cv::Mat &imgR,
                const shared_ptr<Caimura> &camL, const shared_ptr<Caimura> &camR) {
    // Stereo Pose
    Matrix3d Rrl = camR->Rci_ * camL->Ric_;
    Vector3d prl = camR->pci_ + camR->Rci_ * camL->pic_;
    Matrix3d Erl = MathUtil::VecToSkew(prl) * Rrl;  // Essential matrix
    Mat cv_R(3, 3, CV_64F), cv_t(3, 1, CV_64F);
    eigen2cv(Rrl, cv_R);
    eigen2cv(prl, cv_t);

    // Detect corners on left image
    vector<Point2f> leftPts;
    goodFeaturesToTrack(imgL, leftPts, 300, 0.01, 15);
    if (leftPts.empty()) {
        cerr << "No good features have been detected!" << endl;
        return;
    }

    // Left->Right optical flow
    vector<uchar> status;
    vector<Point2f> rightPts;
    StereoFlow(imgL, imgR, camL, camR, leftPts, rightPts, status);
    ReduceVector(leftPts, status);
    ReduceVector(rightPts, status);
    cout << "left-right optical flow points: " << leftPts.size() << endl;
    if (leftPts.empty()) {
        return;
    }

    // Right->Left optical flow
    vector<Point2f> leftPts2;
    StereoFlow(imgR, imgL, camR, camL, rightPts, leftPts2, status);
    for (size_t i = 0; i < leftPts2.size(); ++i) {
        const auto &pt = leftPts.at(i);
        const auto &pt2 = leftPts2.at(i);
        if (cv::norm(leftPts[i] - leftPts2[i]) > 2) {
            status[i] = 0;
        }
    }
    ReduceVector(leftPts, status);
    ReduceVector(leftPts2, status);
    ReduceVector(rightPts, status);
    cout << "right-left optical flow points: " << leftPts.size() << endl;

    // Convert to sphere points
    vector<Point3f> spherePtsL, spherePtsR;
    for (size_t i = 0; i < leftPts.size(); ++i) {
        Vector3f rayL = camL->LiftSphere(Vector2d(leftPts[i].x, leftPts[i].y), true).cast<float>();
        Vector3f rayR = camR->LiftSphere(Vector2d(rightPts[i].x, rightPts[i].y), true).cast<float>();
        spherePtsL.emplace_back(rayL(0), rayL(1), rayL(2));
        spherePtsR.emplace_back(rayR(0), rayR(1), rayR(2));
    }

    // Calc stereo pose
    const auto angularThresh = static_cast<float>(camL->GetAngularResolution());
    if (leftPts.size() > 10) {
        Mat R, t;
        FindRTFromSpherePairs(spherePtsL, spherePtsR, R, t, angularThresh);
        cout << "R:\n" << R << endl << "t: " << t.t() << endl;
    }

    // Triangulate
    vector<double> depths;
    ComputeDepthsOnSphere(spherePtsL, spherePtsR, cv_R, cv_t, depths);

    // Draw
    // Debug plot
    Mat visImg;
    hconcat(imgL, imgR, visImg);
    cvtColor(visImg, visImg, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < leftPts.size(); i++) {
        float k = static_cast<float>(depths[i]) / 8;
        if (k < 0) {
            continue;
        }
        k=std::min(k, 1.f);
        int b = k * 255;
        int r = 255 - b;
        int g = 255 - fabs(k-0.5) * 2 * 255;
        circle(visImg, leftPts[i], 2, CV_RGB(r, g, b), -1);
        circle(visImg, rightPts[i]+Point2f(imgL.cols, 0), 2, Scalar(0,0,255), 1);
//        line(visImg, leftPts[i], rightPts[i]+Point2f(imgL.cols, 0), Scalar(255,0,255), 1);
        stringstream ss; ss.precision(3);
        ss << depths[i];
        putText(visImg, ss.str(), rightPts[i]+Point2f(imgL.cols, 0),
                cv::FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255,255,255), 1);
    }
    cv::imshow("stereo corners", visImg);
    waitKey();
}


int main(int argc, const char * argv[]) {
    const string argKeys =
            "{dir | /home/d/Downloads/evo/evo_0207_room/ | dataset directory path}"
            "{config | /home/d/work/inslam/samples/Vio/config/evo.json | config file path }";

    cv::CommandLineParser parser(argc, argv, argKeys);
    const string dir = parser.get<string>("dir");
    const string configFile = parser.get<string>("config");

    DatasetConfig dc;
    if (!DataIOUtil::ParseDatasetConfig(configFile, dc)) {
        cerr << "failed to parse dataset config!" << endl;
        return -1;
    }

    // Video data
    VideoProvider vip0(dir + "/" + dc.videoFile0, dir + "/" + dc.timeFile, dc.timeScale);
    VideoProvider vip1(dir + "/" + dc.videoFile1, dir + "/" + dc.timeFile, dc.timeScale);

    // Camera
    auto config = ParseJsonFile(configFile);
    auto cam0 = make_shared<Caimura>(config["cameras"][0]);
    auto cam1 = make_shared<Caimura>(config["cameras"][1]);

    // Loop
    Mat img0, img1;
    double t_img;
    int frameId = 0;
    while (true) {
        if (vip0.Read(img0, t_img) && vip1.Read(img1, t_img)) {
            cout << "frameId: " << frameId++ << endl;
            if (img0.cols != dc.imgWidth || img0.rows != dc.imgHeight) {
                resize(img0, img0, Size(dc.imgWidth, dc.imgHeight));
                resize(img1, img1, Size(dc.imgWidth, dc.imgHeight));
            }
            if (img0.channels() == 3) {
                cvtColor(img0, img0, cv::COLOR_BGR2GRAY);
                cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
            }

            CalcStereo(img0, img1, cam0, cam1);

            cout << endl;
        }

        if (waitKey(5) == 'q') {
            break;
        }

    }

    return 0;
}
