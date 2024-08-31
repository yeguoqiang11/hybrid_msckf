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
using namespace inslam;


int main(int argc, const char * argv[]) {
    const string argKeys =
            "{dir | D:/dataset/fpvdataset/vio/V1_02_medium/mav0/ | dataset directory path}"
            "{config | D:/work/inslam/samples/Vio/config/euroc.json | config file path }";
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
    VideoProvider vip1(dir + "/" + dc.videoFile1, dir + "/" + dc.timeFile, dc.timeScale);
    vip0.Skip(startFrameIdx);
    vip1.Skip(startFrameIdx);

    // IMU data
    vector<Matrix<double, 7, 1> > imu_interval;
    vector<Matrix<double, 7, 1> > all_imu_data;
    DataIOUtil::ReadImuData(dir + "/" + dc.imuFile, all_imu_data, dc.timeScale, dc.accScale);

    // ------------ VINS ---------------------------------------
    // Create VIO system
    DearSystem vio(configFile, false);

    // ----- Viewer ------------------------------------------
    viewer::PangoViewer pango;
    pango.Start();

    // Loop
    Mat imgLeft, imgRight;
    double t_img;
    int iImu = 0;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    while (true) {
        if (!vip0.Read(imgLeft, t_img) || !vip1.Read(imgRight, t_img)) {
            cerr << "failed to read image!" << endl;
            break;
        }

        if (imgLeft.cols != dc.imgWidth || imgLeft.rows != dc.imgHeight) {
            resize(imgLeft, imgLeft, Size(dc.imgWidth, dc.imgHeight));
            resize(imgRight, imgRight, Size(dc.imgWidth, dc.imgHeight));
        }
        if (imgLeft.channels() == 3) {
            cvtColor(imgLeft, imgLeft, cv::COLOR_BGR2GRAY);
            cvtColor(imgRight, imgRight, cv::COLOR_BGR2GRAY);
        }
        if(equalize) {
            clahe->apply(imgLeft, imgLeft);
            clahe->apply(imgRight, imgRight);
        }

        // Get imu data
        while (all_imu_data[iImu](0) <= t_img) {
            vio.InputImu(all_imu_data[iImu]);
            iImu++;
        }
        vio.InputImu(all_imu_data[iImu]);
        iImu++;

        // Tracking
        if (!vio.Track(imgLeft, t_img, imgRight) ) {
            continue;
        }

        // Visualize
        Matrix4d Twi, Twc;
        Vector3d vel;
        vio.GetPoseVel(Twi, Twc, vel);
        pango.AddPosition(Twi.topRightCorner(3, 1).cast<float>());
        pango.SetOdomPose(Twi, Vector3f(1, 0, 0));

        // sliding window poses
        vector<Matrix4d> camPoses = vio.GetSlidingWindowPoses();
        vector<Vector3f> camColors(camPoses.size(), Vector3f(1, 0, 1));
        pango.SetCamPoses(camPoses, camColors);

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

        cout << endl;
    }

    vio.SaveVioResults(dir + "/stereo_vio_results.txt");

    cout << "press any key to quit" << endl;
    cv::waitKey();

    return 0;
}
