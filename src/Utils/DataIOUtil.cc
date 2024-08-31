//
// Created by d on 2020/9/23.
//

#include "Utils/DataIOUtil.h"
//#include "spdlog/spdlog.h"
#include "Utils/json.hpp"

using namespace std;

namespace inslam {

bool DataIOUtil::ReadScanHeader(ifstream &ifs, LidarInfo &lidarInfo) {
    string oneLine, s;
    getline(ifs, oneLine);  // header
    getline(ifs, oneLine);
    stringstream ss(oneLine);
    getline(ss, s, ',');
    lidarInfo.minAngle = stod(s);
    getline(ss, s, ',');
    lidarInfo.maxAngle = stod(s);
    getline(ss, s, ',');
    lidarInfo.angleIncrement = stod(s);
    getline(ss, s, ',');
    lidarInfo.timeIncrement = stod(s);
    getline(ss, s, ',');
    lidarInfo.scanTime = stod(s);
    getline(ss, s, ',');
    lidarInfo.minRange = stod(s);
    getline(ss, s, ',');
    lidarInfo.maxRange = stod(s);
    getline(ifs, oneLine);  // "timestamp, ranges"
    return true;
}


bool DataIOUtil::ReadOneScan(ifstream &ifs, const LidarInfo &info, double &timestamp, vector<float> &ranges) {
    string oneLine, s;
    getline(ifs, oneLine);
    if (oneLine.empty()) {
        return false;
    }
    stringstream ss(oneLine);
    getline(ss, s, ',');
    timestamp = stod(s);

    const int N = static_cast<int>(std::ceil((info.maxAngle - info.minAngle) / info.angleIncrement));
    ranges.resize(N);
    for (int i = 0; i < N; ++i) {
        getline(ss, s, ',');
        ranges[i] = stof(s);
    }
    return true;
}


bool DataIOUtil::ReadOdomPoses(const string &file,
                                                vector<pair<double, Eigen::Matrix4d> > &time_poses,
                                                double time_scale) {
    ifstream ifs(file);
    if (!ifs.is_open()) {
        cerr << "can't open pose file: " << file << endl;
        return false;
    }

    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
    Eigen::Matrix<double, 8, 1> tpq;
    string aline, s;

    // header
    getline(ifs, aline);

    while (!ifs.eof()) {
        getline(ifs, aline);
        if (aline.empty()) {
            break;
        }

        stringstream ss(aline);
        for (int i = 0; i < 8; ++i) {
            getline(ss, s, ',');
            tpq(i) = stod(s);
        }

        double t = tpq(0) * time_scale;

        Eigen::Quaterniond qcw(tpq(7), tpq(4), tpq(5), tpq(6));
        Eigen::Matrix3d Rwc = qcw.matrix();
        Twc.topLeftCorner(3, 3) = Rwc;
        Twc.topRightCorner(3, 1) = tpq.segment(1, 3);

        time_poses.emplace_back(t, Twc);
    }
    return true;
}


bool DataIOUtil::ReadSlamPoses(const std::string &file, std::map<long long, Eigen::Matrix4d> &time_poses) {
    ifstream ifs(file);
    if (!ifs.is_open()) {
        cerr << "can't open pose file: " << file << endl;
        return false;
    }

    time_poses.clear();
    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
    Eigen::Matrix<double, 8, 1> tpq;
    string aline, s;

    while (!ifs.eof()) {
        getline(ifs, aline);
        if (aline.empty()) {
            break;
        }

        stringstream ss(aline);
        for (int i = 0; i < 8; ++i) {
            getline(ss, s, ',');
            tpq(i) = stod(s);
        }

        auto t = static_cast<long long>(tpq(0) * 1e3);

        Eigen::Quaterniond qcw(tpq(4), tpq(5), tpq(6), tpq(7));
        Eigen::Matrix3d Rwc = qcw.matrix();
        Twc.topLeftCorner(3, 3) = Rwc;
        Twc.topRightCorner(3, 1) = tpq.segment(1, 3);

        time_poses[t] = Twc;
    }

    return true;
}


void DataIOUtil::SaveSlamPoses(const string &file, vector<pair<double, Eigen::Matrix4d>> &timePoses) {
    std::fstream fs;
    fs.open(file, std::istream::out);
    if (!fs) {
        //spdlog::error("Error opening the file: {}", file);
        return;
    }

    for (const auto &it : timePoses) {
        Eigen::Matrix3d R = it.second.topLeftCorner<3, 3>();
        Eigen::Vector3d p = it.second.topRightCorner<3, 1>();
        Eigen::Quaterniond q(R);
        fs << std::setprecision(18) << it.first << ',' << std::setprecision(9)
           << p(0) << ',' << p(1) << ',' << p(2)
           << ',' << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << '\n';
    }

    fs.close();
    //spdlog::info("pose file saved at: {}", file);
}


void DataIOUtil::SavePositions(const std::string &file, const std::vector<Eigen::Vector3d> &positions) {
    ofstream ofs(file);
    if (!ofs.is_open()) {
        cerr << "can't open file: " << file << endl;
        return;
    }

    for (const Eigen::Vector3d &pos : positions) {
        ofs << pos[0] << " " << pos[1] << " " << pos[2] << "\n";
    }

    ofs.close();
}


void DataIOUtil::ReadPositions(const std::string &file, std::vector<Eigen::Vector3d> &positions) {
    positions.clear();

    ifstream ifs(file);
    if (!ifs.is_open()) {
        cerr << "can't open pose file: " << file << endl;
        return;
    }

    string aline;
    while (!ifs.eof()) {
        getline(ifs, aline);
        if (aline.empty()) {
            break;
        }

        Eigen::Vector3d pos;
        stringstream ss(aline);
        ss >> pos(0) >> pos(1) >> pos(2);
        positions.push_back(pos);
    }
}


bool DataIOUtil::ReadOdoCamExtrinsics(const std::string &file, Eigen::Matrix4d &T_b_c) {
    ifstream ifs(file);
    if (!ifs.is_open()) {
        cerr << "failed to open extrinsics file: " << file << endl;
        return false;
    }

    nlohmann::json conf;
    ifs >> conf;
    ifs.close();

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            T_b_c(i, j) = conf["pose_base_cam"][i * 4 + j];
        }
    }

    return true;
}


bool DataIOUtil::ReadGroundTruthUzh(const string &file, vector<Eigen::Matrix<double, 8, 1>> &all_data) {
    ifstream ifs(file);
    if (!ifs.is_open()) {
        cerr << "failed to open ground truth file: " << file << endl;
        return false;
    }

    // skip header
    string aline;
    getline(ifs, aline);

    Eigen::Matrix<double, 8, 1> data;
    while (!ifs.eof()) {
        int id;
        ifs >> id >> data[0] >> data[1] >> data[2] >> data[3]
        >> data[5] >> data[6] >> data[7] >> data[4];
        all_data.push_back(data);
    }
    ifs.close();
    cout << "done reading " << all_data.size() << " ground truth data!" << endl;
    return true;
}


bool DataIOUtil::ReadImuData(const string &imu_file,
                             vector<Eigen::Matrix<double, 7, 1>> &all_data,
                             double timeScale, double accScale, bool hasDelimiter, bool skipFirstCol) {
    ifstream ifs(imu_file);
    if (!ifs.is_open()) {
        cerr << "Could not open imu data file " << imu_file << endl;
        return false;
    }

    string aline, s;
    double d;
    Eigen::Matrix<double, 7, 1> data;
    char delimiter;

    // skip the header line no matter it's header or not
    getline(ifs, aline);

    while (!ifs.eof()) {
        getline(ifs, aline);
        if (aline.empty()) {
            break;
        }
        stringstream ss(aline);

        if (skipFirstCol) {
            ss >> d;
            if (hasDelimiter)
                ss >> delimiter;
        }

        for (int i = 0; i < 7; i++) {
            ss >> data(i);
            if (hasDelimiter) {
                ss >> delimiter;
            }
        }

        data(0) *= timeScale;
        data.tail(3) *= accScale;
        all_data.push_back(data);
    }
    cout << "Done reading " << all_data.size() << " imu data!" << endl;
    return true;
}


int DataIOUtil::GetImuInterval(const std::vector<Eigen::Matrix<double, 7, 1>> &all_data,
                               std::vector<Eigen::Matrix<double, 7, 1>> &imu_interval, double t0, double t1,
                               int entry) {
    imu_interval.clear();

    bool bfirst = true;
    int imax = static_cast<int>(all_data.size());
    double ratio;
    int i = entry;
    for (; i<imax; i++) {
        const Eigen::Matrix<double, 7, 1>& data = all_data[i];
        if (data(0) < t0) {
            continue;
        } else if (bfirst) {
            const Eigen::Matrix<double, 7, 1>& data_a = all_data[i-1];
            ratio = (data(0) - t0) / (data(0) - data_a(0));
            imu_interval.emplace_back(ratio*data_a + (1.0-ratio)*data);
            if (data(0) > t0 + 1e-12) {
                imu_interval.emplace_back(data);
            }
            bfirst = false;
        } else if (data(0) < t1) {
            imu_interval.push_back(data);
        } else {
            const Eigen::Matrix<double, 7, 1>& data_b = all_data[i-1];
            ratio = (data(0) - t1) / (data(0) - data_b(0));
            imu_interval.emplace_back(ratio*data_b + (1.0-ratio)*data);
            break;
        }
    }

    // Make sure that we do not have zero dt values, otherwise it would
    // cause the covariance matrix to be infinity
    if (imu_interval.size() > 1) {
        for (size_t j = 0; j < imu_interval.size() - 1; j++) {
            if (imu_interval[j+1](0) - imu_interval[j](0) < 1e-12) {
                imu_interval.erase(imu_interval.begin() + j);
                j--;
            }
        }
    }

    return i-3;
}


int DataIOUtil::GetInitImuData(const vector<Eigen::Matrix<double, 7, 1> > &all_data,
                               Eigen::Matrix<double, 7, 1> &data, double t0) {
    size_t i = 0;
    double t = all_data[i](0), tmin = t0 - 0.001;
    while (t < tmin) {
        t = all_data[i++](0);
    }
    data = all_data[i-1];

    return static_cast<int>(i) - 1;
}


bool DataIOUtil::ParseDatasetConfig(const string &configFile, DatasetConfig &dc) {
    ifstream ifs(configFile);
    if (!ifs.is_open()) {
        cerr << "failed to open config file: " << configFile << endl;
        return false;
    }
    nlohmann::json config = nlohmann::json::parse(ifs);
    ifs.close();

    if (config.find("video0") != config.end()) {
        dc.videoFile0 = config["video0"].get<string>();
    }
    if (config.find("video1") != config.end()) {
        dc.videoFile1 = config["video1"].get<string>();
    }
    if (config.find("imu") != config.end()) {
        dc.imuFile = config["imu"].get<string>();
    }
    if (config.find("time_scale") != config.end()) {
        dc.timeScale = config["time_scale"];
    }
    if (config.find("acc_scale") != config.end()) {
        dc.accScale = config["acc_scale"];
    }

    dc.imgWidth = config["cameras"][0]["width"];
    dc.imgHeight = config["cameras"][0]["height"];
    if (config["cameras"][0].find("scale") != config["cameras"][0].end()) {
        double camScale = config["cameras"][0]["scale"];
        dc.imgWidth = static_cast<int>(dc.imgWidth * camScale);
        dc.imgHeight = static_cast<int>(dc.imgHeight * camScale);
    }

    cout << "dataset config: " << endl;
    cout << "\tvideo0: " << dc.videoFile0 << endl;
    cout << "\tvideo1: " << dc.videoFile1 << endl;
    cout << "\ttimestamps: " << dc.timeFile << endl;
    cout << "\timu: " << dc.imuFile << endl;
    cout << "\ttime scale: " << dc.timeScale << endl;
    cout << "\tacc scale: " << dc.accScale << endl;
    cout << "\timage size: " << dc.imgWidth << " x " << dc.imgHeight << endl;
    return true;
}

}//namespace inslam {
