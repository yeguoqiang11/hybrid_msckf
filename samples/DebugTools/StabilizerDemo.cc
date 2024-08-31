//
// Created by d on 2021/3/18.
//

#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include "Utils/VideoProvider.h"
#include "Utils//PerformanceTest.h"
#include "Utils/DataIOUtil.h"
#include "Utils/json.hpp"
#include "Vio/Initialize/Imu/AttitudeFilter.h"
#include "Vio/Caimura.hpp"


using namespace cv;
using namespace std;
using namespace Eigen;
using namespace inslam;


void PanoToFisheyeMap(shared_ptr<Caimura> cam,
                      const int panoWidth,
                      const int panoHeight,
                      const cv::Mat& R,
                      cv::Mat& map) {
    map = cv::Mat::zeros(panoHeight, panoWidth, CV_32FC2);
    float* mapData = (float*)map.data;
    double* rData = (double*)R.data;
    for (int i = 0; i < panoHeight; ++i) {
        double angleZ = i * CV_PI / panoHeight;
        double rxy = sin(angleZ);
        double z = cos(angleZ);
        for (int j = 0; j < panoWidth; ++j) {
            double angleX = (panoWidth - j) * CV_2PI / panoWidth;
            double x = cos(angleX) * rxy;
            double y = -sin(angleX) * rxy;
            double x2 = rData[0] * x + rData[1] * y + rData[2] * z;
            double y2 = rData[3] * x + rData[4] * y + rData[5] * z;
            double z2 = rData[6] * x + rData[7] * y + rData[8] * z;
            double angleZ2 = acos(z2);

            Vector2d uv;
            if (cam->IsOpencvFisheye() || cam->IsEquidistant()) {
                double distortAngleZ2 = cam->IsEquidistant() ?
                                        cam->DistortThetaEquidistant(angleZ2) :cam->DistortThetaOpencvFisheye(angleZ2);
                double rxy2 = sqrt(x2 * x2 + y2 * y2);
                rxy2 = std::max(rxy2, 1e-12);
                double nx = x2 / rxy2;
                double ny = y2 / rxy2;
                double radius = distortAngleZ2 * cam->fx();
                uv(0) = nx * radius + cam->cx();
                uv(1) = ny * radius + cam->cy();
            } else {
                cam->Reproject(Vector3d(x2, y2, z2), uv, true);
                if (angleZ2 > cam->fovAngle() / 2) {
                    uv << -1, -1;
                }
            }

            mapData[(i * panoWidth + j) * 2] = static_cast<float>(uv(0));
            mapData[(i * panoWidth + j) * 2 + 1] = static_cast<float>(uv(1));
        }
    }
}


int main(int argc, const char * argv[]) {
    const string argKeys =
            "{dir | /home/d/Downloads/evo/evo_0207_outdoor/ | dataset directory path}"
            "{config | /home/d/work/inslam/samples/Vio/config/evo.json | config file path }";

    cv::CommandLineParser parser(argc, argv, argKeys);
    const string dir = parser.get<string>("dir");
    const string configFile = parser.get<string>("config");
    const int startFrameIdx = 0;
    const bool useGravity = false;

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
    const Matrix3d R_I_C = cam->Ric_;

    Vector3d bg, ba;
    for (int i = 0; i < 3; i++) {
        bg(i) = config["gyro_bias"][i];
        ba(i) = config["acc_bias"][i];
    }

    // Create Attitude filter
    AttitudeFilter filter(bg, ba);

    // ----- Initial condition ---------------------------------------
    Mat img;
    double t_img;
    if (!vip0.Read(img, t_img)) {
        cerr << "failed to read image!" << endl;
        return -1;
    }
    double t_img_last = t_img;


    int iImu = 0;
    Matrix<double, 7, 1> imuData0;
    iImu = DataIOUtil::GetInitImuData(all_imu_data, imuData0, t_img_last);
    filter.FeedImu(imuData0(0), imuData0.segment(1, 3), imuData0.tail(3), useGravity);
    Quaterniond q_I0_G = filter.GetQuat();


    // Loop
    int frameId = 0;
    while (true) {
        if (vip0.Read(img, t_img)) {
            cout << "frameId: " << frameId++ << endl;
            if (img.cols != dc.imgWidth || img.rows != dc.imgHeight) {
                resize(img, img, Size(dc.imgWidth, dc.imgHeight));
            }
            if (img.channels() == 3) {
                cvtColor(img, img, cv::COLOR_BGR2GRAY);
            }

            // Get imu data
            iImu = DataIOUtil::GetImuInterval(all_imu_data, imu_interval, t_img_last, t_img, iImu);
            t_img_last = t_img;
            cout << "imu interval: " << imu_interval.size() << endl;

            for (size_t i = 1; i < imu_interval.size(); i++) {
                Matrix<double, 7, 1> imuData = imu_interval[i];
                filter.FeedImu(imuData(0), imuData.segment(1, 3), imuData.tail(3), useGravity);
            }

            Quaterniond q_I_G = filter.GetQuat();
            Matrix3d R_G_I0 = q_I0_G.matrix();
            Matrix3d R_G_I = q_I_G.matrix();
            Matrix3d R_C_C0 = R_I_C.transpose() * R_G_I.transpose() * R_G_I0 * R_I_C;

            cv::Mat cvR(3, 3, CV_64F);
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    cvR.at<double>(i, j) = R_C_C0(i, j);
                }
            }

            const int panoWidth = 800;
            const int panoHeight = 400;
            cv::Mat map;
            cv::Vec3d rx(CV_PI/2, 0, 0);
            cv::Mat Rx;
            cv::Rodrigues(rx,Rx);
            PanoToFisheyeMap(cam, panoWidth, panoHeight, cvR*Rx, map);
            cv::Mat panoImg;
            cv::remap(img, panoImg, map, cv::Mat(), cv::INTER_LINEAR);
            cv::putText(panoImg, string("left"), Point(50,50), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1);
            imshow("pano image", panoImg);
            imshow("original", img);
            waitKey(2);

            cout << endl;
        }

        if (waitKey(5) == 'q') {
            break;
        }

    }

    return 0;
}
