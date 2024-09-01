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


using namespace cv;
using namespace std;
using namespace Eigen;
using namespace hybrid_msckf;

void ReadImgTimes(const string &file, vector<double> &times);


int main(int argc, const char * argv[]) {
    const string argKeys =
            "{dir | /home/d/Downloads/indoor_forward_3_snapdragon_with_gt/ | dataset directory path}"
            "{config | /home/d/work/hybrid_msckf/samples/Vio/config/uzh_fpv.json | config file path }";
    cv::CommandLineParser parser(argc, argv, argKeys);
    const string dataset = parser.get<string>("dir");
    const string configFile = parser.get<string>("config");
    const double timeOffset = -0.015;

    /* Start frame idx:
     * indoor forward 3: 930,
     * indoor forward 5: 900
     * indoor forward 9: 880
     */
    int startFrameIdx = 0;

    // Video data
    vector<double> imgTimes;
    ReadImgTimes(dataset + "/left_images.txt", imgTimes);
    const int totalFrames = static_cast<int>(imgTimes.size());

    // IMU data
    vector<Matrix<double, 7, 1> > imu_interval;
    vector<Matrix<double, 7, 1> > all_imu_data;
    DataIOUtil::ReadImuData(dataset + "/imu.txt", all_imu_data, 1.0, 1.0, false, true);

    // Ground truth data
    vector<Matrix<double, 8, 1> > gtDatas;
    DataIOUtil::ReadGroundTruthUzh(dataset + "groundtruth.txt", gtDatas);

    // ------------ VINS ---------------------------------------
    // Create VIO system
    DearSystem vio(configFile, false);

    // ----- Viewer ------------------------------------------
    viewer::PangoViewer pango;
    pango.Start();

//    // show ground truth
//    std::vector<Eigen::Vector3f> trajectory;
//    for(auto gt : gtDatas) {
//        Vector3f p_G_I = gt.block<3, 1>(1, 0).cast<float>();
//        trajectory.push_back(p_G_I);
//    }
//    Eigen::Vector3f color = Vector3f(1, 0, 1);
//    pango.SetTrajectory(trajectory, color);

    // Loop
    Mat Il, Ir;
    double t_img;
    int iframe = startFrameIdx;
    int iImu = 0;
    while (true) {
        if (iframe < totalFrames) {
            iframe++;
            Il = imread(dataset + "/img/image_0_" + to_string(iframe) + ".png", cv::IMREAD_GRAYSCALE);
            Ir = imread(dataset + "/img/image_1_" + to_string(iframe) + ".png", cv::IMREAD_GRAYSCALE);
            t_img = imgTimes[iframe] + timeOffset;
            if (Il.empty() || Ir.empty()) {
                continue;
            }

            // Get imu data
            while (all_imu_data[iImu](0) <= t_img) {
                vio.InputImu(all_imu_data[iImu]);
                iImu++;
            }
            vio.InputImu(all_imu_data[iImu]);
            iImu++;

            // Tracking
            bool trackOk = false;
#if 0
            trackOk = vio.Track(Il, t_img);
#else
            trackOk = vio.Track(Il, t_img, Ir);
#endif

            // Visualize
            if (!trackOk) {
                continue;
            }
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
            pango.SetPointCloud(points, colors, 3, true);

            cout << endl;
        } else {
            break;
        }
    }

    vio.SaveVioResults(dataset + "/stamped_traj_estimate.txt");

    cout << "press any key to quit!" << endl;
    waitKey();

    return 0;
}


void ReadImgTimes(const string &file, vector<double> &times) {
    ifstream ifs(file);
    if (!ifs.is_open()) {
        cerr << "failed to open image timestamps file: " << file << endl;
        exit(-1);
    }

    // skip header
    string aline;
    getline(ifs, aline);

    int id;
    double t;
    string name;
    while (!ifs.eof()) {
        ifs >> id >> t >> name;
        times.push_back(t);
    }
    cout << "Done reading " << times.size() << " image timestamps!" << endl;
}

