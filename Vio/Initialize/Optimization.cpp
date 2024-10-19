#include "Vio/Initialize/Optimization.hpp"
#include "Vio/Initialize/Imu/CamCost.h"
#include "Vio/Initialize/StereoDepthCost.hpp"

#include <Eigen/src/Core/Matrix.h>
#include <ceres/cost_function.h>
#include <ceres/problem.h>
#include <unordered_map>

namespace featslam{


void LocalOptimization(std::vector<FramePtr> &frames, std::shared_ptr<Map> &map, double angularResolution) {
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::CauchyLoss(angularResolution * 2.0);
    std::unordered_map<int, double *> map_pts_w, map_cam_pose;

    int minCameraID = frames[0]->FrameId();
    std::vector<MapPointPtr> &mapPoints = map->mapPoints_;
    std::vector<FramePtr> &keyFrames = map->keyframes_;
    std::vector<ceres::ResidualBlockId> residual_id_vec;
    for (int i = 0; i < static_cast<int>(frames.size()); i++) {
        FramePtr &frame = frames[i];
        const std::vector<int> &idList = frame->IDList();
        for (size_t j = 0; j < idList.size(); j++) {
            const int &pt_id = idList[j];
            if (pt_id >= static_cast<int>(mapPoints.size())) {
                continue;
            }
            MapPointPtr &map_pt = mapPoints[pt_id];
            if (map_pt->pt3dFlag) {
                if (map_pts_w.find(pt_id) != map_pts_w.end()) {
                    continue;
                }
                map_pts_w[pt_id] = new double[3]{map_pt->pt3d.x, map_pt->pt3d.y, map_pt->pt3d.z};
                const std::vector<Feature> &features = map_pt->features;
                for(auto feat : features) {
                    int frameId = feat.imageID;
                    if (map_cam_pose.find(frameId) == map_cam_pose.end()) {
                        const cv::Vec6d pose = keyFrames[frameId]->GetPose();
                        map_cam_pose[frameId] = new double[6]{pose(0), pose(1), pose(2), pose(3), pose(4), pose(5)};
                    }

                    ceres::CostFunction *projCost = StereoDepthCost::Create(feat.spherePt1.x, feat.spherePt1.y, feat.spherePt1.z, feat.depth);
                    ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(projCost, loss_function, map_cam_pose[frameId], map_pts_w[pt_id]);
                    residual_id_vec.push_back(residualBlockId);
                    if(frameId < minCameraID) {
                        problem.SetParameterBlockConstant(map_cam_pose[frameId]);
                    }
                }
            }
        }
    }

    ceres::Problem::EvaluateOptions eval_opts;
    eval_opts.residual_blocks = residual_id_vec;
    eval_opts.apply_loss_function = false;
    double before_cost;
    problem.Evaluate(eval_opts, &before_cost, NULL, NULL, NULL);

    //double t0 = static_cast<double>(cv::getTickCount());
    ceres::Solver::Options options;
    options.max_num_iterations = 5;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //double t1 = static_cast<double>(cv::getTickCount());
    //double dt = (t1 - t0) * 1000.0 / cv::getTickFrequency();
    //std::cout << "global optimization cost time: " << dt << std::endl;

    double after_cost;
    problem.Evaluate(eval_opts, &after_cost, NULL, NULL, NULL);

    std::unordered_map<int, double *>::iterator it;
    for (it = map_pts_w.begin(); it != map_pts_w.end(); it++) {
        MapPointPtr &map_pt = mapPoints[it->first];
        map_pt->pt3d.x = static_cast<float>(it->second[0]);
        map_pt->pt3d.y = static_cast<float>(it->second[1]);
        map_pt->pt3d.z = static_cast<float>(it->second[2]);
        delete[] it->second;
    }

    for (it = map_cam_pose.begin(); it != map_cam_pose.end(); it++) {
        FramePtr frame = keyFrames[it->first];
        cv::Vec6d pose;
        pose(0) = it->second[0];
        pose(1) = it->second[1];
        pose(2) = it->second[2];
        pose(3) = it->second[3];
        pose(4) = it->second[4];
        pose(5) = it->second[5];
        frame->SetPose(pose);
        delete[] it->second;
    }
}


void FrameLocalOptimization(std::vector<Frame> &frames, std::shared_ptr<Map> &map, double angularResolution) {
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::CauchyLoss(angularResolution * 2.0);
    std::unordered_map<int, double *> map_pts_w, map_cam_pose;

    std::vector<MapPointPtr> &mapPoints = map->mapPoints_;
    std::vector<ceres::ResidualBlockId> residual_id_vec;
    for (int i = 0; i < static_cast<int>(frames.size()); i++) {
        Frame &frame = frames[i];
        if (map_cam_pose.find(i) == map_cam_pose.end()) {
            const cv::Vec6d pose = frame.GetPose();
            map_cam_pose[i] = new double[6]{pose(0), pose(1), pose(2), pose(3), pose(4), pose(5)};
        }

        const std::vector<int> &idList = frame.IDList();
        const std::vector<Feature> &features = frame.Features();
        //Eigen::Matrix3d proj_sqrt_info = Eigen::Matrix3d::Identity();
        for (size_t j = 0; j < idList.size(); j++) {
            const int &pt_id = idList[j];
            if (pt_id >= static_cast<int>(mapPoints.size())) {
                continue;
            }
            const Feature &feat = features[j];
            MapPointPtr &map_pt = mapPoints[pt_id];
            if (map_pt->pt3dFlag) {
                if (map_pts_w.find(pt_id) == map_pts_w.end()) {
                    map_pts_w[pt_id] = new double[3]{map_pt->pt3d.x, map_pt->pt3d.y, map_pt->pt3d.z};
                }
                //Eigen::Vector3d obs(feat.spherePt1.x, feat.spherePt1.y, feat.spherePt1.z);
                //ceres::CostFunction *proj_cost = new CamProjError(proj_sqrt_info, obs);
                //problem.AddResidualBlock(proj_cost, loss_function, map_cam_pose[i], map_pts_w[pt_id]);
                ceres::CostFunction *projCost = StereoDepthCost::Create(feat.spherePt1.x, feat.spherePt1.y, feat.spherePt1.z, feat.depth);
                ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(projCost, loss_function, map_cam_pose[i], map_pts_w[pt_id]);
                residual_id_vec.push_back(residualBlockId);
            }
        }

        if (i == 0) {
            problem.SetParameterBlockConstant(map_cam_pose[i]);
        }
    }

    //double t0 = cv::getTickCount();
    ceres::Solver::Options options;
    options.max_num_iterations = 5;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    // options.linear_solver_type = ceres::LinearSolverType::DENSE_SCHUR;
    //options.trust_region_strategy_type = ceres::DOGLEG;
    // options.num_threads = 1;
    options.minimizer_progress_to_stdout = false;
    options.function_tolerance = 1.0e-02;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //double t1 = cv::getTickCount();
    //double dt = (t1 - t0) * 1000.0 / cv::getTickFrequency();
    //std::cout << "global optimization cost time1: " << dt << std::endl;
    ceres::Problem::EvaluateOptions eval_opts;
    eval_opts.residual_blocks = residual_id_vec;
    eval_opts.apply_loss_function = false;
    double init_cost;
    problem.Evaluate(eval_opts, &init_cost, NULL, NULL, NULL);
    //std::cout << "global optimization cost: " << init_cost << std::endl;

    std::unordered_map<int, double *>::iterator it;
    for (it = map_pts_w.begin(); it != map_pts_w.end(); it++) {
        MapPointPtr &map_pt = mapPoints[it->first];
        map_pt->pt3d.x = static_cast<float>(it->second[0]);
        map_pt->pt3d.y = static_cast<float>(it->second[1]);
        map_pt->pt3d.z = static_cast<float>(it->second[2]);
        delete[] it->second;
    }

    for (it = map_cam_pose.begin(); it != map_cam_pose.end(); it++) {
        Frame &frame = frames[it->first];
        cv::Vec6d pose;
        pose(0) = it->second[0];
        pose(1) = it->second[1];
        pose(2) = it->second[2];
        pose(3) = it->second[3];
        pose(4) = it->second[4];
        pose(5) = it->second[5];
        frame.SetPose(pose);
        delete[] it->second;
    }
}

}  // namespace inslam
