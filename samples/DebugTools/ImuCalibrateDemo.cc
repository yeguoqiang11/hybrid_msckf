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
    vip0.Skip(startFrameIdx);

    // IMU data
    vector<Matrix<double, 7, 1> > imu_interval;
    vector<Matrix<double, 7, 1> > all_imu_data;
    DataIOUtil::ReadImuData(dir + "/" + dc.imuFile, all_imu_data, dc.timeScale, dc.accScale);

    // ----- Initial condition ---------------------------------------
    Mat img;
    double t_img;
    if (!vip0.Read(img, t_img)) {
        cerr << "failed to read image!" << endl;
        return -1;
    }
    double t_img_last = t_img;

    // ------------ VINS ---------------------------------------
    // Create VIO system
    DearSystem vio(configFile, true);

    // ----- Viewer ------------------------------------------
    viewer::PangoViewer pango;
    pango.Start();

    // Loop
    bool calibFlag = false;
    int iImu = 0;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    while (true) {
        if (vip0.Read(img, t_img) && !calibFlag) {
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
            iImu = DataIOUtil::GetImuInterval(all_imu_data, imu_interval, t_img_last, t_img, iImu);
            t_img_last = t_img;
            cout << "imu interval: " << imu_interval.size() << endl;

            // Tracking
            calibFlag = vio.RunImuCalibrationEx(img, t_img, imu_interval);

            // Visualize
            Matrix4d T_I_G = Matrix4d::Identity();
            cv::Mat resR, resT;
            vio.GetInitCameraPose(resR, resT);
            if (!resR.empty() && !resT.empty()) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j)
                        T_I_G(i, j) = resR.at<double>(i, j);
                    T_I_G(i, 3) = resT.at<double>(i);
                }
            }
            Matrix4d T_G_I = T_I_G.inverse();
            vector<Matrix4d> camPoses(1, T_G_I);
            vector<Vector3f> camColors(1, Vector3f(1, 0, 1));
            pango.SetCamPoses(camPoses, camColors);
            pango.AddPosition(T_G_I.topRightCorner<3, 1>().cast<float>());

            vector<Vector3d> mapPoints = vio.GetInitMapPoint();
            vector<float> points, colors;
            for (const Vector3d &pt : mapPoints) {
                points.push_back(static_cast<float>(pt(0)));
                points.push_back(static_cast<float>(pt(1)));
                points.push_back(static_cast<float>(pt(2)));
                colors.push_back(1);
                colors.push_back(1);
                colors.push_back(1);
            }
            pango.SetPointCloud(points, colors, 2, true);

            cout << endl;
        } else if(!calibFlag) {
            std::cout << "valid frame less than 500, no enough frame for calibrate" << std::endl;
            break;
        }

        if (pango.ShouldStop()) {
            break;
        }
    }

    return 0;
}
