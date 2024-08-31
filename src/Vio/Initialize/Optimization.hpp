#pragma once
#include <ceres/ceres.h>
#include <opencv2/core.hpp>

#include "Vio/Initialize/InitTypes.hpp"

namespace inslam {
/*
 * Optimize frames by BA, use all observations in map
 * @param frames: frames to be optimized
 * @param map: map with all the keyframes and mappoints
 * @param angularResolution
 */
void LocalOptimization(std::vector<FramePtr> &frames, std::shared_ptr<Map> &map, double angularResolution);


/*
 * Optimize frames by BA, use input frames observations only
 * @param frames: frames to be optimized
 * @param map: with all the keyframes and mappoints
 * @param angularResolution
 */
void FrameLocalOptimization(std::vector<Frame> &frames, std::shared_ptr<Map> &map, double angularResolution)  ;


}  // namespace inslam
