#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include "Vio/DearSystem.hpp"
#include "Viewer/PangoViewer.h"
#include "Utils/VideoProvider.h"
#include "Utils/PerformanceTest.h"
#include "Utils/DataIOUtil.h"
#include "Utils/json.hpp"
#include "Vio/Initialize/Imu/AttitudeFilter.h"
#include "Vio/Caimura.hpp"


using namespace cv;
using namespace std;
using namespace Eigen;
using namespace inslam;

void GetInitCamPoseInGravity(double t_img,
                             Matrix4d &T_G_C0,
                             Matrix4d &T_C_I,
                             const vector<Matrix<double, 7, 1> > &all_imu_data,
                             const string &configFile) {
    // Camera
    ifstream ifs(configFile);
    if (!ifs.is_open()) {
        cerr << "failed to open config file: " << configFile << endl;
        exit(-1);
    }
    nlohmann::json config = nlohmann::json::parse(ifs);
    ifs.close();
    nlohmann::json cam_node = config["cameras"][0];
    auto cam = make_shared<Caimura>(cam_node);
    T_C_I.topLeftCorner(3, 3) = cam->Rci_;
    T_C_I.topRightCorner(3, 1) = cam->pci_;

    Vector3d bg, ba;
    for (int i = 0; i < 3; i++) {
        bg(i) = config["gyro_bias"][i];
        ba(i) = config["acc_bias"][i];
    }

    // Create Attitude filter
    AttitudeFilter filter(bg, ba);

    // get first imu data
    int iImu = 0;
    Matrix<double, 7, 1> imuData0;
    if(all_imu_data[0](0) < t_img) {
        iImu = DataIOUtil::GetInitImuData(all_imu_data, imuData0, t_img);
    } else {
        imuData0 = all_imu_data[0];
    }

    filter.FeedImu(imuData0(0), imuData0.segment(1, 3), imuData0.tail(3), true);
    Quaterniond q_I0_G = filter.GetQuat();
    Matrix4d T_I0_G = Matrix4d::Identity();
    T_I0_G.topLeftCorner(3, 3) = q_I0_G.matrix();
    T_G_C0 = (T_C_I * T_I0_G).inverse();
}


int main(int argc, const char * argv[]) {
    const string argKeys =
            "{dir | /media/dd/F/DATA/evo/evo_0207_loop1/ | dataset directory path}"
            "{config | /home/dd/work/inslam/samples/Vio/config/evo.json | config file path }"
            "{slamPosePath | ./slamPose.txt | path to save slam pose}";
    cv::CommandLineParser parser(argc, argv, argKeys);
    const string dir = parser.get<string>("dir");
    const string configFile = parser.get<string>("config");
    const string slamPosePath = parser.get<string>("slamPosePath");
    const bool equalize = true;
    const int startFrameIdx = 0;
    const bool doStereo = false;
    const bool alignGravity = true;
    bool saveSlamPose = true;

    DatasetConfig dc;
    if (!DataIOUtil::ParseDatasetConfig(configFile, dc)) {
        cerr << "failed to parse dataset config!" << endl;
        return -1;
    }

    // Video data
    VideoProvider vip0(dir + "/" + dc.videoFile0, dir + "/" + dc.timeFile, dc.timeScale);
    vip0.Skip(startFrameIdx);

    std::shared_ptr<VideoProvider> vipPtr1 = nullptr;
    if(doStereo) {
        vipPtr1 = make_shared<VideoProvider>(dir + "/" + dc.videoFile1, dir + "/" + dc.timeFile, dc.timeScale);
        vipPtr1->Skip(startFrameIdx);
    }

    // slam pose align gravity, transfor to imu coordinate
    Matrix4d T_G_C0 = Matrix4d::Identity();
    Matrix4d T_C_I = Matrix4d::Identity();
    Mat imgLeft, imgRight;
    double t_img;
    if(alignGravity) {
        double t_img;
        if (!vip0.Read(imgLeft, t_img) || (doStereo && !vipPtr1->Read(imgRight, t_img))) {
            cerr << "failed to read image!" << endl;
            return -1;
        }

        vector<Matrix<double, 7, 1> > all_imu_data;
        DataIOUtil::ReadImuData(dir + "/" + dc.imuFile, all_imu_data, dc.timeScale, dc.accScale);
        GetInitCamPoseInGravity(t_img, T_G_C0, T_C_I, all_imu_data, configFile);
    }

    // ------------ VINS ---------------------------------------
    // Create VIO system
    DearSystem vio(configFile, true);
    vio.SetSaveSlamPoseFlag(saveSlamPose);

    // ----- Viewer ------------------------------------------
    viewer::PangoViewer pango;
    pango.Start();

    // Loop
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    while (true) {
        if (vip0.Read(imgLeft, t_img) && (!doStereo || vipPtr1->Read(imgRight, t_img)))
        {
            if (imgLeft.cols != dc.imgWidth || imgLeft.rows != dc.imgHeight) {
                resize(imgLeft, imgLeft, Size(dc.imgWidth, dc.imgHeight));
            }
            if (imgLeft.channels() == 3) {
                cvtColor(imgLeft, imgLeft, cv::COLOR_BGR2GRAY);
            }
            if(equalize) {
                clahe->apply(imgLeft, imgLeft);
            }

            // Tracking
            if(!doStereo) {
                // monocular
                vio.RunInitVSlam(imgLeft, t_img);
            }
            else {
                // stereo
                if (imgRight.cols != dc.imgWidth || imgRight.rows != dc.imgHeight) {
                    resize(imgRight, imgRight, Size(dc.imgWidth, dc.imgHeight));
                }
                if (imgRight.channels() == 3) {
                    cvtColor(imgRight, imgRight, cv::COLOR_BGR2GRAY);
                }
                if(equalize) {
                    clahe->apply(imgRight, imgRight);
                }
                vio.RunInitVSlam(imgLeft, t_img, imgRight);
            }

            // Visualize
            Matrix4d T_C_C0 = Matrix4d::Identity();
            cv::Mat resR, resT;
            vio.GetInitCameraPose(resR, resT);
            if (!resR.empty() && !resT.empty()) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j)
                        T_C_C0(i, j) = resR.at<double>(i, j);
                    T_C_C0(i, 3) = resT.at<double>(i);
                }
            }
            Matrix4d T_G_I = T_G_C0 * T_C_C0.inverse() * T_C_I;
            vector<Matrix4d> camPoses(1, T_G_I);
            vector<Vector3f> camColors(1, Vector3f(1, 0, 1));
            pango.SetCamPoses(camPoses, camColors);
            pango.AddPosition(T_G_I.topRightCorner<3, 1>().cast<float>());

            vector<Vector3d> mapPoints = vio.GetInitMapPoint();
            vector<float> points, colors;
            for (const Vector3d &pt_c0 : mapPoints) {
                const Vector3d pt_g = T_G_C0.topLeftCorner(3, 3) * pt_c0 + T_G_C0.topRightCorner(3, 1);
                points.push_back(static_cast<float>(pt_g(0)));
                points.push_back(static_cast<float>(pt_g(1)));
                points.push_back(static_cast<float>(pt_g(2)));
                colors.push_back(1);
                colors.push_back(1);
                colors.push_back(1);
            }
            pango.SetPointCloud(points, colors, 2, true);

            cout << endl;
        } else if(saveSlamPose) {
            saveSlamPose = false;
            //r, t:from cam0 to cami,format: time r00 r01 r02 r10 r11 r12 r20 r21 r22 t0 t1 t2
            Eigen::Matrix4d T_C0_G = T_G_C0.inverse();
            Eigen::Matrix4d T_I_C = T_C_I.inverse();
            vio.SaveSlamPose(slamPosePath, T_C0_G, T_I_C);
        }

        if (pango.ShouldStop()) {
            break;
        }
    }

    return 0;
}
