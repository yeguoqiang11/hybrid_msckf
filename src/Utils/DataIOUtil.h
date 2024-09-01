//
// Created by d on 2020/9/23.
//

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <map>

namespace hybrid_msckf {

struct LidarInfo {
    double minAngle;
    double maxAngle;
    double angleIncrement;
    double timeIncrement;
    double scanTime;
    double minRange;
    double maxRange;
};

struct DatasetConfig {
    std::string videoFile0 = "cam0.mp4";
    std::string videoFile1 = "cam1.mp4";
    std::string timeFile = "timestamps.txt";
    std::string imuFile = "imu.txt";
    double timeScale = 1.0;
    double accScale = 1.0;
    int imgWidth;
    int imgHeight;
};

class DataIOUtil {
public:
    DataIOUtil() = default;

    static bool ReadScanHeader(std::ifstream &ifs, LidarInfo &lidarInfo);

    static bool ReadOneScan(std::ifstream &ifs, const LidarInfo &info, double &timestamp, std::vector<float> &ranges);

    ///each line: timestamp, x, y, z, qx, qy, qz, qw
    static bool ReadOdomPoses(const std::string &file,
                                            std::vector<std::pair<double, Eigen::Matrix4d> > &time_poses,
                                            double time_scale);

    ///each line: timestamp, x, y, z, qw, qx, qy, qz
    static bool ReadSlamPoses(const std::string &file, std::map<long long, Eigen::Matrix4d> &time_poses);

    ///each line: timestamp, x, y, z, qw, qx, qy, qz
    static void SaveSlamPoses(const std::string &file, std::vector<std::pair<double, Eigen::Matrix4d> > &timePoses);

    ///each line: x, y, z
    static void SavePositions(const std::string &file,
                              const std::vector<Eigen::Vector3d> &positions);

    ///each line: x, y, z
    static void ReadPositions(const std::string &file,
                              std::vector<Eigen::Vector3d> &positions);

    static bool ReadOdoCamExtrinsics(const std::string &file, Eigen::Matrix4d &T_b_c);

    static bool ReadGroundTruthUzh(const std::string &file, std::vector<Eigen::Matrix<double, 8, 1>> &all_data);

    static bool ReadImuData(const std::string &imu_file,
                            std::vector<Eigen::Matrix<double, 7, 1> > &all_data,
                            double timeScale = 1.0,
                            double accScale = 1.0,
                            bool hasDelimiter = true,
                            bool skipFirstCol = false);

    static int GetImuInterval(const std::vector<Eigen::Matrix<double, 7, 1>> &all_data,
                              std::vector<Eigen::Matrix<double, 7, 1>> &imu_interval, double t0, double t1,
                              int entry);

    static int GetInitImuData(const std::vector<Eigen::Matrix<double, 7, 1> > &all_data,
                              Eigen::Matrix<double, 7, 1> &data, double t0);

    static bool ParseDatasetConfig(const std::string &configFile, DatasetConfig &dc);

};

}//namespace hybrid_msckf {
