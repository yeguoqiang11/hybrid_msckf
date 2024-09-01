//
// Created by d on 2021/3/19.
//

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
#include "Vio/Caimura.hpp"


using namespace cv;
using namespace std;
using namespace Eigen;
using namespace hybrid_msckf;


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
            "{dir | D:/dataset/fpvdataset/vio/V1_02_medium/mav0/ | dataset directory path}"
            "{config | D:/work/hybrid_msckf/samples/Vio/config/euroc.json | config file path }";

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


    // Loop
    Mat img;
    double t_img;
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

            Matrix3d R_C_C0 = Matrix3d::Identity();

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
