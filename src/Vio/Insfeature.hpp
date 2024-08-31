//
//  Insfeature.hpp
//  DearVins
//
#pragma once

#include <Eigen/Dense>
#include "Vio/FeatureObservation.h"
#include "Vio/Caimura.hpp"
#include "Utils/MathUtil.h"

namespace inslam {

struct Insfeature {

    size_t featureId;
    size_t anchorFrameId;

    int stateEntry; // inverse_depth location in state vector
    int errCovEntry;    // inverse_depth_error location in error covariance matrix

    /* unified viewing vector. */
    Eigen::Vector3d anchorRay;

    inline void SetObservation(const FeatureObservation &observation) {
        measurement_ = observation;
    }

    inline void SetPrediction(const FeatureObservation &observation) {
        prediction_ = observation;
    }

    inline void CalcInnovation(bool useUnitSphereError, bool compressStereo, double focal, double baseline) {
        if (useUnitSphereError) {
            MathUtil::CalcTangentBase(measurement_.ray0, tangentBase0_);
            if (measurement_.isStereo && prediction_.isStereo) {
                if (compressStereo) {
                    innovation_.resize(3);
                    innovation_.head(2) = tangentBase0_ * (measurement_.ray0 - prediction_.ray0);
                    innovation_(2) = baseline * (1.0 / measurement_.stereoDepth - 1.0 / prediction_.stereoDepth);
                } else {
                    MathUtil::CalcTangentBase(measurement_.ray1, tangentBase1_);
                    innovation_.resize(4);
                    innovation_.head(2) = tangentBase0_ * (measurement_.ray0 - prediction_.ray0);
                    innovation_.tail(2) = tangentBase1_ * (measurement_.ray1 - prediction_.ray1);
                }
            } else { // Monocular
                innovation_ = tangentBase0_ * (measurement_.ray0 - prediction_.ray0);
            }
        } else { // reprojection error on image plane
            cv::Point2f dpt0 = measurement_.upt0 - prediction_.upt0;
            if (measurement_.isStereo && prediction_.isStereo) {
                if (compressStereo) {
                    double zMeasure = measurement_.stereoDepth * measurement_.ray0(2);
                    double zPredict = prediction_.stereoDepth * prediction_.ray0(2);
                    innovation_ = Eigen::Vector3d(dpt0.x, dpt0.y, focal * baseline * (1.0 / zMeasure - 1.0 / zPredict));
                } else {
                    cv::Point2f dpt1 = measurement_.upt1 - prediction_.upt1;
                    innovation_ = Eigen::Vector4d(dpt0.x, dpt0.y, dpt1.x, dpt1.y);
                }
            } else {
                innovation_ = Eigen::Vector2d(dpt0.x, dpt0.y);
            }
        }
    }

    // Observation in current frame
    FeatureObservation measurement_;

    // Predictions in current frame
    FeatureObservation prediction_;

    // Tangent base of observation ray. If use unit sphere reprojection errors, then
    // the residual = tangentBase * (observed ray - predicted ray)
    Eigen::Matrix<double, 2, 3> tangentBase0_;
    Eigen::Matrix<double, 2, 3> tangentBase1_;

    //
    Eigen::VectorXd innovation_; // innovation = residual = z - h
    Eigen::MatrixXd S; // innovation matrix, S = H * P * H^t + R, where R is measurement noise covariance
    Eigen::MatrixXd PHt; // P * H^t, define it here to save computations

    Eigen::Matrix<double, Eigen::Dynamic, 6> H_Il;  // partial(h) / partial(current pose error)
    Eigen::Matrix<double, Eigen::Dynamic, 6> H_Ij;  // partial(h) / partial(anchor pose error)
    Eigen::Matrix<double, Eigen::Dynamic, 1> H_rho; // partial(h) / partial(inverse depth error)
};

}//namespace inslam {