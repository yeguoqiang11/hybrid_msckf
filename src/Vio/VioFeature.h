#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "Vio/Caimura.hpp"
#include "Vio/FeatureObservation.h"

namespace inslam {

class VioFeature {
public:
    explicit VioFeature(size_t id);

    void AddObservation(const FeatureObservation &obser);

    bool CheckMotion(const std::map<size_t, Eigen::Matrix<double, 3, 4> > &framePoses,
                             const std::shared_ptr<Caimura> &caim);

    bool Triangulate(const std::map<size_t, Eigen::Matrix<double, 3, 4> > &framePoses,
                     const std::shared_ptr<Caimura>& caim,
                     const std::shared_ptr<Caimura>& cam1 = nullptr);

    bool Refine(const std::map<size_t, Eigen::Matrix<double, 3, 4> > &framePoses,
                const std::shared_ptr<Caimura>& caim,
                const std::shared_ptr<Caimura>& cam1 = nullptr);


    double ComputeError(const std::vector<Eigen::Matrix<double, 3, 4> > &poses,
                        const std::vector<Eigen::Vector2d> &points,
                        double alpha, double beta, double rho);


    bool InitializePosition(const std::map<size_t, Eigen::Matrix<double, 3, 4> > &framePoses,
                            const std::shared_ptr<Caimura>& caim,
                            const std::shared_ptr<Caimura> &cam1 = nullptr,
                            bool refine = false);

    bool RefinePosition(const std::map<size_t, Eigen::Matrix<double, 3, 4> > &framePoses,
                        const std::shared_ptr<Caimura>& caim);

    void SetAnchorFrame(size_t anchorFrameId, const std::shared_ptr<Caimura>& caim);

    size_t id_;
    size_t latestFrameId_ = -1;
    std::map<size_t, FeatureObservation> observations_;

    bool isInState_;
    size_t anchorFrameId_;
    Eigen::Vector3d anchorRay_;
    int stateEntry_;
    int errCovEntry_;

    bool initialized_;
    Eigen::Vector3d xw_;    // position in world frame
};

}//namespace inslam {