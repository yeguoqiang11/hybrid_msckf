#pragma once

#include <opencv2/opencv.hpp>
namespace hybrid_msckf {
namespace pnp {

class EPNPSolver {
public:
    EPNPSolver() = default;
    ~EPNPSolver() {
        if (A1_)
            delete[] A1_;
        if (A2_)
            delete[] A2_;
    }
    int Run(const std::vector<cv::Point3d> &objectPoints,
            const std::vector<cv::Point3d> &spherePoints,
            cv::Mat &rotations,
            cv::Mat &translations);

    bool EPNPRansac(const std::vector<cv::Point3f> &pts3d,
                    const std::vector<cv::Point3f> &spherePts,
                    double threshold,
                    cv::Mat &R,
                    cv::Mat &t,
                    int minIterCnt = 300);

    int GetRansacInlierCount();

private:
    void RequestMemory(int num);
    void Init();
    void ChooseControlPoints();
    void ComputeBarycentricCoordinates();
    void FillM();
    void ComputeCCS(const double *betas, const double *ut);
    void ComputePCS(void);
    void ComputeL6x10(const double *ut, double *l_6x10);
    void ComputeRho(double *rho);

    void findBetas(const double *ut, const cv::Mat &Rho, double *betas);
    void findBetasApprox1(const cv::Mat &L_6x10, const cv::Mat &Rho, double *betas);
    void findBetasApprox2(const cv::Mat &L_6x10, const cv::Mat &Rho, double *betas);
    void findBetasApprox3(const cv::Mat &L_6x10, const cv::Mat &Rho, double *betas);

    void Compute_A_and_b_GaussNewton(const double *l_6x10, const double *rho, const double betas[4], cv::Mat &A, cv::Mat &b);
    void QRSolve(cv::Mat &A, cv::Mat &b, cv::Mat &X);
    void GaussNewton(const cv::Mat &L_6x10, const cv::Mat &Rho, double betas[4]);

    void EstimateRT(double R[3][3], double t[3]);
    void SolveForSign();
    double ReprojectionError(const double R[3][3], const double t[3]);
    double ComputeRT(const double *ut, const double *betas, double R[3][3], double t[3]);

    double Dot(const double *v1, const double *v2);
    double Dist2(const double *p1, const double *p2);

private:
    int numPoint_ = 0;
    int requestNum_ = 0;
    int maxNR_ = 0;
    bool initFlag_ = false;

    std::vector<double> alphas_, pws_, pcs_, spherePts_;

    cv::Mat CC_, invCC_;
    cv::Mat Pw_, PwTPw_, PwTPw_D_, PwTPw_U_, PwTPw_V_T_;
    cv::Mat M_, MTM_, MW_, MU_, MVt_;

    cv::Mat L_6x10_, rho_, L_6x4_, B4_, L_6x3_, B3_, L_6x5_, B5_;
    cv::Mat A_, B_, X_;
    //for compute R,t by 3D-3D pair points
    cv::Mat ABt_, ABt_D_, ABt_U_, ABt_V_;

    double *A1_ = nullptr, *A2_ = nullptr;
    double cws_[4][3] = {}, ccs_[4][3] = {};
    double Betas_[4][4] = {}, rep_errors_[4] = {};
    double Rs_[4][3][3] = {}, ts_[4][3] = {};

    cv::SVD svd_;
    cv::Mat cvR_, v1_, v2_, v3_, v4_;
    int ransacInliersCount_;
};
}  // namespace pnp
}  // namespace hybrid_msckf
