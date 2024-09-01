#ifndef LINEAR_ALIGNMENT_H
#define LINEAR_ALIGNMENT_H

#include <Eigen/Dense>

#include "Vio/Initialize/InitTypes.hpp"
#include "Vio/Initialize/Imu/Imu.h"

using namespace Eigen;
using namespace std;

namespace hybrid_msckf {

bool IMUAlignmentByLinear(std::vector<VisualInertialState> &allFrameState, Vector3d &bg, Vector3d &g,
                        VectorXd &x, const Eigen::Matrix3d &Rcb, const Vector3d &tcb);

}

#endif //LINEAR_ALIGNMENT_H
