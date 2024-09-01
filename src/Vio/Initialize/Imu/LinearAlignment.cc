#include "Vio/Initialize/Imu/LinearAlignment.h"


namespace hybrid_msckf {

static Vector3d Gravity = Vector3d(0, 0, 9.81);

void SolveGyroscopeBias(std::vector<VisualInertialState> &allFrameState, Vector3d &bg, const Eigen::Matrix3d &Rcb)
{
    Matrix3d A;
    Vector3d b;
    Vector3d deltaBg;
    A.setZero();
    b.setZero();
    std::vector<VisualInertialState>::iterator frameI;
    std::vector<VisualInertialState>::iterator frameJ;
    for (frameI = allFrameState.begin(); next(frameI) != allFrameState.end(); frameI++)
    {
        frameJ = next(frameI);
        MatrixXd tmpA(3, 3);
        tmpA.setZero();
        VectorXd tmpb(3);
        tmpb.setZero();
        Eigen::Quaterniond qji(Rcb * frameI->Rwc * frameJ->Rwc.transpose() * Rcb.transpose());
        tmpA = frameJ->imu0.dRg_;
        tmpb = 2 * (Eigen::Quaterniond(frameJ->imu0.Rji_.inverse()) * qji).vec();
        A += tmpA.transpose() * tmpA;
        b += tmpA.transpose() * tmpb;
    }
    deltaBg = A.ldlt().solve(b);
    bg = allFrameState[0].imu0.bg_ + deltaBg;
    //std::cout << "gyroscope bias initial calibration bg: " << bg.transpose() << "delta bg: " << deltaBg.transpose() << std::endl;

    for (auto &state : allFrameState)
    {
        state.imu0.bg_ = bg;
        state.imu0.RePreintegration();
    }
}


MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}


void RefineGravity(std::vector<VisualInertialState> &allFrameState, Vector3d &g,
                   VectorXd &x, const Eigen::Matrix3d &Rcb, const Vector3d& tcb)
{
    Vector3d g0 = g.normalized() * Gravity.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int allFrameCount = static_cast<int>(allFrameState.size());
    int nState = allFrameCount * 3 + 2 + 1;

    MatrixXd A{nState, nState};
    A.setZero();
    VectorXd b{nState};
    b.setZero();

    std::vector<VisualInertialState>::iterator frameI;
    std::vector<VisualInertialState>::iterator frameJ;
    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frameI = allFrameState.begin(); next(frameI) != allFrameState.end(); frameI++, i++)
        {
            frameJ = next(frameI);
            Eigen::Matrix3d Rwbi = Rcb * frameI->Rwc;
            Eigen::Vector3d tciw = -frameI->Rwc.transpose() * frameI->twc;
            Eigen::Matrix3d Rwbj = Rcb * frameJ->Rwc;
            Eigen::Vector3d tcjw = -frameJ->Rwc.transpose() * frameJ->twc;

            MatrixXd tmpA(6, 9);
            tmpA.setZero();
            VectorXd tmpb(6);
            tmpb.setZero();

            double dt = frameJ->imu0.dT_;

            tmpA.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmpA.block<3, 2>(0, 6) = Rwbi * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmpA.block<3, 1>(0, 8) = Rwbi * (tcjw - tciw) / 100.0;
            tmpb.block<3, 1>(0, 0) = frameJ->imu0.GetPij()+ Rwbi * Rwbj.transpose() * tcb - tcb - Rwbi * dt * dt / 2 * g0;
            //cout << "Pij   " << frame_j->imu0.GetPij() << endl;
            tmpA.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmpA.block<3, 3>(3, 3) = Rwbi * Rwbj.transpose();
            tmpA.block<3, 2>(3, 6) = Rwbi * dt * Matrix3d::Identity() * lxly;;
            tmpb.block<3, 1>(3, 0) = frameJ->imu0.GetVij() - Rwbi * dt * Matrix3d::Identity() * g0;

            Matrix<double, 6, 6> covInv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            covInv.setIdentity();

            MatrixXd rA = tmpA.transpose() * covInv * tmpA;
            VectorXd rb = tmpA.transpose() * covInv * tmpb;

            A.block<6, 6>(i * 3, i * 3) += rA.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += rb.head<6>();

            A.bottomRightCorner<3, 3>() += rA.bottomRightCorner<3, 3>();
            b.tail<3>() += rb.tail<3>();

            A.block<6, 3>(i * 3, nState - 3) += rA.topRightCorner<6, 3>();
            A.block<3, 6>(nState - 3, i * 3) += rA.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(nState - 3);
            g0 = (g0 + lxly * dg).normalized() * Gravity.norm();
            //double s = x(nState - 1) / 100.0;
            //std::cout << "refine: scale: " << s << " " << "g0: " << g0.norm() << " " << g0.transpose() << std::endl;
    }
    g = g0;
}


bool LinearAlignment(std::vector<VisualInertialState> &allFrameState, Vector3d &g, VectorXd &x,
                     const Eigen::Matrix3d &Rcb, const Vector3d& tcb)
{
    int allFrameCount = static_cast<int>(allFrameState.size());
    int nState = allFrameCount * 3 + 3 + 1;

    MatrixXd A{nState, nState};
    A.setZero();
    VectorXd b{nState};
    b.setZero();

    std::vector<VisualInertialState>::iterator frameI;
    std::vector<VisualInertialState>::iterator frameJ;
    int i = 0;
    for (frameI = allFrameState.begin(); next(frameI) != allFrameState.end(); frameI++, i++)
    {
        frameJ = next(frameI);
        Eigen::Matrix3d Rwbi = Rcb * frameI->Rwc;
        Eigen::Vector3d tciw = -frameI->Rwc.transpose() * frameI->twc;
        Eigen::Matrix3d Rwbj = Rcb * frameJ->Rwc;
        Eigen::Vector3d tcjw = -frameJ->Rwc.transpose() * frameJ->twc;

        MatrixXd tmpA(6, 10);
        tmpA.setZero();
        VectorXd tmpb(6);
        tmpb.setZero();

        double dt = frameJ->imu0.dT_;

        tmpA.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmpA.block<3, 3>(0, 6) = Rwbi * dt * dt / 2 * Matrix3d::Identity();
        tmpA.block<3, 1>(0, 9) = Rwbi * (tcjw - tciw) / 100.0;
        tmpb.block<3, 1>(0, 0) = frameJ->imu0.GetPij()+ Rwbi * Rwbj.transpose() * tcb - tcb;
        //cout << "Pij   " << frame_j->imu0.GetPij().transpose() << endl;
        tmpA.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmpA.block<3, 3>(3, 3) = Rwbi * Rwbj.transpose();
        tmpA.block<3, 3>(3, 6) = Rwbi * dt * Matrix3d::Identity();
        tmpb.block<3, 1>(3, 0) = frameJ->imu0.GetVij();
        //cout << "Vij " << frame_j->imu0.GetVij().transpose() << endl;

        Matrix<double, 6, 6> covInv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        covInv.setIdentity();

        MatrixXd rA = tmpA.transpose() * covInv * tmpA;
        VectorXd rb = tmpA.transpose() * covInv * tmpb;

        A.block<6, 6>(i * 3, i * 3) += rA.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += rb.head<6>();

        A.bottomRightCorner<4, 4>() += rA.bottomRightCorner<4, 4>();
        b.tail<4>() += rb.tail<4>();

        A.block<6, 4>(i * 3, nState - 4) += rA.topRightCorner<6, 4>();
        A.block<4, 6>(nState - 4, i * 3) += rA.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    double s = x(nState - 1) / 100.0;
    //std::cout << "estimated scale: " << s << std::endl;
    g = x.segment<3>(nState - 4);
    //std::cout << " result g     " << g.norm() << " " << g.transpose() << std::endl;
    if(fabs(g.norm() - Gravity.norm()) > 1.0 || s < 0)
    {
        return false;
    }

    RefineGravity(allFrameState, g, x, Rcb, tcb);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    //std::cout << " refine  " << g.norm() << " " << g.transpose() << std::endl;

    //Eigen::Vector3d vel = x.segment<3>(nState - 7);
    //std::cout << " refine vel " << vel.norm() << " " << vel.transpose() << std::endl;
    if(s < 0.0 )
        return false;
    else
        return true;
}


bool IMUAlignmentByLinear(std::vector<VisualInertialState> &allFrameState, Vector3d &bg, Vector3d &g,
                        VectorXd &x, const Eigen::Matrix3d &Rcb, const Vector3d &tcb)
{
    SolveGyroscopeBias(allFrameState, bg, Rcb);

    if(LinearAlignment(allFrameState, g, x, Rcb, tcb))
        return true;
    else
        return false;
}

}
