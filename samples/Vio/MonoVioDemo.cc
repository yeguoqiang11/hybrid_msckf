//
// Created by d on 2021/1/1.
//

#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include "Vio/DearSystem.hpp"
#include "Viewer/PangoViewer.h"
#include "Utils/VideoProvider.h"
#include "Utils/MathUtil.h"
#include "Utils//PerformanceTest.h"
#include "Utils/DataIOUtil.h"
#include "Utils/json.hpp"


using namespace cv;
using namespace std;
using namespace Eigen;
using namespace hybrid_msckf;


int main(int argc, const char * argv[]) {
    const string argKeys =
            "{dir | D:/dataset/fpvdataset/vio/V1_02_medium/mav0/ | dataset directory path}"
            "{config | D:/work/hybrid_msckf/samples/Vio/config/euroc.json | config file path }";

    cv::CommandLineParser parser(argc, argv, argKeys);
    const string dir = parser.get<string>("dir");
    const string configFile = parser.get<string>("config");
    const bool equalize = true;
    const int startFrameIdx = 0;

    DatasetConfig dc;
    if (!DataIOUtil::ParseDatasetConfig(configFile, dc)) {
        cerr << "failed to parse dataset config!" << endl;
        return -1;
    }

    // Video data
    VideoProvider vip0(dir + "/" + dc.videoFile0, dir + "/" + dc.timeFile, dc.timeScale);
    vip0.Skip(startFrameIdx);

    // IMU data
    vector<Matrix<double, 7, 1> > imu_interval;
    vector<Matrix<double, 7, 1> > all_imu_data;
     DataIOUtil::ReadImuData(dir + "/" + dc.imuFile, all_imu_data, dc.timeScale, dc.accScale);

    // ------------ VINS ---------------------------------------
    // Create VIO system
    DearSystem vio(configFile, true);

    // ----- Viewer ------------------------------------------
    viewer::PangoViewer pango;
    pango.Start();

    // Loop
    int iImu = 0;
    cv::Mat img;
    double t_img;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    while (true) {
        if (!vip0.Read(img, t_img)) {
            cerr << "failed to read image!" << endl;
            break;
        }
        if (img.cols != dc.imgWidth || img.rows != dc.imgHeight) {
            resize(img, img, Size(dc.imgWidth, dc.imgHeight));
        }
        if (img.channels() == 3) {
            cvtColor(img, img, cv::COLOR_BGR2GRAY);
        }
        if(equalize) {
            clahe->apply(img, img);
        }

        // Get imu data
        while (all_imu_data[iImu](0) <= t_img) {
            vio.InputImu(all_imu_data[iImu]);
            iImu++;
        }
        vio.InputImu(all_imu_data[iImu]);
        iImu++;

        // Tracking
        if (!vio.Track(img, t_img) ) {
            continue;
        }

        // Visualize
        Matrix4d Twi, Twc;
        Vector3d vel;
        vio.GetPoseVel(Twi, Twc, vel);
        pango.AddPosition(Twi.topRightCorner(3, 1).cast<float>());
        pango.SetOdomPose(Twi, Vector3f(1, 0, 0));
        // std::cout << "Twi:\n" << Twi << std::endl;

        // sliding window poses
        vector<Matrix4d> camPoses = vio.GetSlidingWindowPoses();
        vector<Vector3f> camColors(camPoses.size(), Vector3f(1, 0, 1));
        pango.SetCamPoses(camPoses, camColors);
        // std::cout << "Twc:\n" << camPoses.back() << std::endl;

        // history 3D points
        vector<Vector3d> mapPoints = vio.GetMapPoints();
        vector<Vector3d> activePoints = vio.GetActivePoints();
        vector<float> points, colors;
        for (const Vector3d &pt : mapPoints) {
            points.push_back(static_cast<float>(pt(0)));
            points.push_back(static_cast<float>(pt(1)));
            points.push_back(static_cast<float>(pt(2)));
            colors.push_back(1);
            colors.push_back(1);
            colors.push_back(1);
        }
        for (const Vector3d &pt : activePoints) {
            points.push_back(static_cast<float>(pt(0)));
            points.push_back(static_cast<float>(pt(1)));
            points.push_back(static_cast<float>(pt(2)));
            colors.push_back(0);
            colors.push_back(1);
            colors.push_back(0);
        }
        pango.SetPointCloud(points, colors, 2, true);

        // cout << endl;
    }

    vio.SaveVioResults(dir + "/mono_vio_results.txt");

    cout << "press any key to quit" << endl;
    waitKey(0);

    return 0;
}
