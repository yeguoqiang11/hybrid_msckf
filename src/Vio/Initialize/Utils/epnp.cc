
#include "epnp.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <opencv2/core/eigen.hpp>

namespace hybrid_msckf {
namespace pnp {

int EPNPSolver::Run(const std::vector<cv::Point3d> &objectPoints,
                    const std::vector<cv::Point3d> &spherePoints,
                    cv::Mat &rotations,
                    cv::Mat &translations) {
    numPoint_ = int(objectPoints.size());
    if (numPoint_ < 4) {
        return -1;
    }
    if (!initFlag_) {
        Init();
        initFlag_ = true;
    }
    RequestMemory(numPoint_);

    for (int i = 0; i < numPoint_; ++i) {
        int i3 = i * 3;
        pws_[i3] = objectPoints[i].x;
        pws_[i3 + 1] = objectPoints[i].y;
        pws_[i3 + 2] = objectPoints[i].z;
        spherePts_[i3] = spherePoints[i].x;
        spherePts_[i3 + 1] = spherePoints[i].y;
        spherePts_[i3 + 2] = spherePoints[i].z;
    }
    ChooseControlPoints();
    ComputeBarycentricCoordinates();
    FillM();

    double *ut = MVt_.ptr<double>();
    ComputeL6x10(ut, L_6x10_.ptr<double>());
    ComputeRho(rho_.ptr<double>());

    findBetas(ut, rho_, Betas_[0]);
    GaussNewton(L_6x10_, rho_, Betas_[0]);
    rep_errors_[0] = ComputeRT(ut, Betas_[0], Rs_[0], ts_[0]);

    findBetasApprox1(L_6x10_, rho_, Betas_[1]);
    GaussNewton(L_6x10_, rho_, Betas_[1]);
    rep_errors_[1] = ComputeRT(ut, Betas_[1], Rs_[1], ts_[1]);

    findBetasApprox2(L_6x10_, rho_, Betas_[2]);
    GaussNewton(L_6x10_, rho_, Betas_[2]);
    rep_errors_[2] = ComputeRT(ut, Betas_[2], Rs_[2], ts_[2]);

    findBetasApprox3(L_6x10_, rho_, Betas_[3]);
    GaussNewton(L_6x10_, rho_, Betas_[3]);
    rep_errors_[3] = ComputeRT(ut, Betas_[3], Rs_[3], ts_[3]);

    int N = 0;
    double minErr = rep_errors_[0];
    for (int i = 1; i < 4; ++i) {
        if (minErr > rep_errors_[i]) {
            minErr = rep_errors_[i];
            N = i;
        }
    }

    if (translations.size() != cv::Size(3, 1) || translations.type() != CV_64F) {
        translations.create(3, 1, CV_64F);
    }
    if (rotations.size() != cv::Size(3, 3) || rotations.type() != CV_64F) {
        rotations.create(3, 3, CV_64F);
    }

    for (int i = 0; i < 3; ++i) {
        double *rData = rotations.ptr<double>(i);
        double *tData = translations.ptr<double>(i);
        for (int j = 0; j < 3; ++j) {
            rData[j] = Rs_[N][i][j];
        }
        tData[0] = ts_[N][i];
    }
    return 1;
}

void EPNPSolver::RequestMemory(int num) {
    if (requestNum_ != num) {
        pws_.resize(3 * num);
        pcs_.resize(3 * num);
        Pw_.create(num, 3, CV_64F);
        spherePts_.resize(3 * num);
        alphas_.resize(4 * num);
        M_.create(num * 2, 12, CV_64F);
    }
    requestNum_ = num;
}

void EPNPSolver::Init() {
    CC_.create(3, 3, CV_64F);
    invCC_.create(3, 3, CV_64F);

    MTM_.create(12, 12, CV_64F);
    MW_.create(12, 1, CV_64F);
    MU_.create(12, 12, CV_64F);
    MVt_.create(12, 12, CV_64F);
    PwTPw_.create(3, 3, CV_64F);
    PwTPw_D_.create(3, 1, CV_64F);
    PwTPw_U_.create(3, 3, CV_64F);
    PwTPw_V_T_.create(3, 3, CV_64F);

    L_6x10_.create(6, 10, CV_64F);
    rho_.create(6, 1, CV_64F);

    L_6x4_.create(6, 4, CV_64F);
    B4_.create(4, 1, CV_64F);
    L_6x3_.create(6, 3, CV_64F);
    B3_.create(3, 1, CV_64F);
    L_6x5_.create(6, 5, CV_64F);
    B5_.create(5, 1, CV_64F);

    A_.create(6, 4, CV_64F);
    B_.create(6, 1, CV_64F);
    X_.create(4, 1, CV_64F);

    ABt_.create(3, 3, CV_64F);
    ABt_D_.create(3, 1, CV_64F);
    ABt_U_.create(3, 3, CV_64F);
    ABt_V_.create(3, 3, CV_64F);

    v1_ = cv::Mat::zeros(1, 12, CV_64F);
    v2_ = cv::Mat::zeros(1, 12, CV_64F);
    v3_ = cv::Mat::zeros(1, 12, CV_64F);
    v4_ = cv::Mat::zeros(1, 12, CV_64F);
}

/*****************************************************************************************************************************************************
 *  compute cws and alpha :
 *              cws[0] = avg_pws cws[j] = eigenvalue(pw'*pw)*eigenvector(pw'*pw),j = 1,2,3
 *
 *              CC[i] = cws[i+1] - cws[0], alpha_i(1,2,3) = inv(CC)*(pw_i - cws[0]) ,alpha_i(0) = 1-alpha_i(1)-alpha_i(2)--alpha_i(3)
 *
 ******************************************************************************************************************************************************/

void EPNPSolver::ChooseControlPoints() {
    for (int j = 0; j < 3; ++j) {
        cws_[0][j] = 0.0;
    }
    for (int i = 0; i < numPoint_; ++i) {
        int i3 = i * 3;
        cws_[0][0] += pws_[i3];
        cws_[0][1] += pws_[i3 + 1];
        cws_[0][2] += pws_[i3 + 2];
    }
    for (int j = 0; j < 3; ++j) {
        cws_[0][j] /= numPoint_;
    }

    for (int i = 0; i < numPoint_; ++i) {
        int i3 = i * 3;
        double *pwData = Pw_.ptr<double>(i);
        pwData[0] = pws_[i3] - cws_[0][0];
        pwData[1] = pws_[i3 + 1] - cws_[0][1];
        pwData[2] = pws_[i3 + 2] - cws_[0][2];
    }
    PwTPw_ = Pw_.t() * Pw_;
    svd_.compute(PwTPw_, PwTPw_D_, PwTPw_U_, PwTPw_V_T_, cv::SVD::MODIFY_A);  //

    for (int i = 1; i < 4; i++) {
        double *utData = PwTPw_V_T_.ptr<double>(i - 1);
        double k = sqrt(PwTPw_D_.at<double>(i - 1, 0) / numPoint_);
        for (int j = 0; j < 3; j++) {
            cws_[i][j] = cws_[0][j] + k * utData[j];
        }
    }
}

void EPNPSolver::ComputeBarycentricCoordinates() {
    if (alphas_.size() != 4 * numPoint_) {
        alphas_.resize(4 * numPoint_);
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 1; j < 4; j++) {
            CC_.at<double>(i, j - 1) = cws_[j][i] - cws_[0][i];
        }
    }
    invCC_ = CC_.inv(cv::DECOMP_SVD);
    double *ci = invCC_.ptr<double>();
    for (int i = 0; i < numPoint_; i++) {
        double *a = &alphas_[0] + 4 * i;
        for (int j = 0; j < 3; j++) {
            int j3 = j * 3;
            int i3 = i * 3;
            a[1 + j] =
                ci[j3] * (pws_[i3] - cws_[0][0]) + ci[j3 + 1] * (pws_[i3 + 1] - cws_[0][1]) + ci[j3 + 2] * (pws_[i3 + 2] - cws_[0][2]);
        }
        a[0] = 1.0 - a[1] - a[2] - a[3];
    }

    //for (int i = 0; i < numPoint_; i++) {
    //    double *a = &alphas_[0] + 4 * i;
    //    double *b = &pws_[0] + 3 * i;
    //    double errx = b[0] - (a[0] * cws_[0][0] + a[1] * cws_[1][0] + a[2] * cws_[2][0] + a[3] * cws_[3][0]);
    //    double erry = b[1] - (a[0] * cws_[0][1] + a[1] * cws_[1][1] + a[2] * cws_[2][1] + a[3] * cws_[3][1]);
    //    double errz = b[2] - (a[0] * cws_[0][2] + a[1] * cws_[1][2] + a[2] * cws_[2][2] + a[3] * cws_[3][2]);
    //    std::cout << "err: " << errx << ", " << erry << ", " << errz << "\n";
    //}
}

inline void Schmidt12(cv::Mat& _v1, cv::Mat& _v2, cv::Mat& _v3, cv::Mat& _v4) {
    // normalize v1
    double *v1 = (double *)_v1.data;
    double *v2 = (double *)_v2.data;
    double *v3 = (double *)_v3.data;
    double *v4 = (double *)_v4.data;
    auto Normalize = [](double *v) {
        double invLen = 1.0
                        / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] + v[4] * v[4] + v[5] * v[5] + v[6] * v[6] + v[7] * v[7]
                               + v[8] * v[8] + v[9] * v[9] + v[10] * v[10] + v[11] * v[11]);
        v[0] *= invLen;
        v[1] *= invLen;
        v[2] *= invLen;
        v[3] *= invLen;
        v[4] *= invLen;
        v[5] *= invLen;
        v[6] *= invLen;
        v[7] *= invLen;
        v[8] *= invLen;
        v[9] *= invLen;
        v[10] *= invLen;
        v[11] *= invLen;
    };
    
    auto Dot = [](double *v1, double *v2) -> double {
        return v2[0] * v1[0] + v2[1] * v1[1] + v2[2] * v1[2] + v2[3] * v1[3] + v2[4] * v1[4] + v2[5] * v1[5] + v2[6] * v1[6] + v2[7] * v1[7]
               + v2[8] * v1[8] + v2[9] * v1[9] + v2[10] * v1[10] + v2[11] * v1[11];
    };

    Normalize(v1);
    double v2dotv1 = Dot(v1,v2);
    v2[0] -= v2dotv1 * v1[0];
    v2[1] -= v2dotv1 * v1[1];
    v2[2] -= v2dotv1 * v1[2];
    v2[3] -= v2dotv1 * v1[3];
    v2[4] -= v2dotv1 * v1[4];
    v2[5] -= v2dotv1 * v1[5];
    v2[6] -= v2dotv1 * v1[6];
    v2[7] -= v2dotv1 * v1[7];
    v2[8] -= v2dotv1 * v1[8];
    v2[9] -= v2dotv1 * v1[9];
    v2[10] -= v2dotv1 * v1[10];
    v2[11] -= v2dotv1 * v1[11];
    Normalize(v2);

    double v3dotv1 = Dot(v3, v1);
    double v3dotv2 = Dot(v3, v2);
    v3[0] -= v3dotv1 * v1[0] + v3dotv2 * v2[0];
    v3[1] -= v3dotv1 * v1[1] + v3dotv2 * v2[1];
    v3[2] -= v3dotv1 * v1[2] + v3dotv2 * v2[2];
    v3[3] -= v3dotv1 * v1[3] + v3dotv2 * v2[3];
    v3[4] -= v3dotv1 * v1[4] + v3dotv2 * v2[4];
    v3[5] -= v3dotv1 * v1[5] + v3dotv2 * v2[5];
    v3[6] -= v3dotv1 * v1[6] + v3dotv2 * v2[6];
    v3[7] -= v3dotv1 * v1[7] + v3dotv2 * v2[7];
    v3[8] -= v3dotv1 * v1[8] + v3dotv2 * v2[8];
    v3[9] -= v3dotv1 * v1[9] + v3dotv2 * v2[9];
    v3[10] -= v3dotv1 * v1[10] + v3dotv2 * v2[10];
    v3[11] -= v3dotv1 * v1[11] + v3dotv2 * v2[11];
    Normalize(v3);

    double v4dotv1 = Dot(v4, v1);
    double v4dotv2 = Dot(v4, v2);
    double v4dotv3 = Dot(v4, v3);
    v4[0] -= v4dotv1 * v1[0] + v4dotv2 * v2[0] + v4dotv3 * v3[0];
    v4[1] -= v4dotv1 * v1[1] + v4dotv2 * v2[1] + v4dotv3 * v3[1];
    v4[2] -= v4dotv1 * v1[2] + v4dotv2 * v2[2] + v4dotv3 * v3[2];
    v4[3] -= v4dotv1 * v1[3] + v4dotv2 * v2[3] + v4dotv3 * v3[3];
    v4[4] -= v4dotv1 * v1[4] + v4dotv2 * v2[4] + v4dotv3 * v3[4];
    v4[5] -= v4dotv1 * v1[5] + v4dotv2 * v2[5] + v4dotv3 * v3[5];
    v4[6] -= v4dotv1 * v1[6] + v4dotv2 * v2[6] + v4dotv3 * v3[6];
    v4[7] -= v4dotv1 * v1[7] + v4dotv2 * v2[7] + v4dotv3 * v3[7];
    v4[8] -= v4dotv1 * v1[8] + v4dotv2 * v2[8] + v4dotv3 * v3[8];
    v4[9] -= v4dotv1 * v1[9] + v4dotv2 * v2[9] + v4dotv3 * v3[9];
    v4[10] -= v4dotv1 * v1[10] + v4dotv2 * v2[10] + v4dotv3 * v3[10];
    v4[11] -= v4dotv1 * v1[11] + v4dotv2 * v2[11] + v4dotv3 * v3[11];
    Normalize(v4);
}

inline void MtM(const cv::Mat& m, cv::Mat& mtm) {
    int w = m.cols;
    int h = m.rows;
    double *mData = (double *)m.data;
    double *mtmData = (double *)mtm.data;
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = 0;
            for (int k = 0; k < h; ++k) {
                s += mData[k * w + i] * mData[k * w + j];
            }
            mtmData[i * w + j] = s;
        }
    }
    for (int i = 0; i < w; ++i) {
        for (int j = i + 1; j < w; ++j) {
            mtmData[i * w + j] = mtmData[j * w + i];
        }
    }
}
    //fill M matrix and compute ut
void EPNPSolver::FillM() {
    if (alphas_.size() != 4 * numPoint_) {
        return;
    }
    //M_.setTo(0.0);
    memset(M_.data, 0, sizeof(double) * M_.rows * M_.cols);
    for (int i = 0; i < numPoint_; i++) {
        double *a = &alphas_[0] + 4 * i;
        double *MData1 = M_.ptr<double>(i * 2);
        double *MData2 = M_.ptr<double>(i * 2 + 1);
        double *spt = &spherePts_[0] + 3 * i;
        for (int j = 0; j < 4; ++j, MData1 += 3, MData2 += 3) {
            MData1[0] = a[j] * spt[2];
            MData1[2] = -a[j] * spt[0];
            MData2[1] = a[j] * spt[2];
            MData2[2] = -a[j] * spt[1];
        }
    }
    // MTM_ = M_.t() * M_;
    MtM(M_, MTM_);
    // svd_.compute(MTM_, MW_, MU_, MVt_, cv::SVD::MODIFY_A);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr;
    Eigen::MatrixXd eigenMtM;
    cv::cv2eigen(MTM_, eigenMtM);
    qr.compute(eigenMtM);
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
    // Eigen::MatrixXd Q = qr.householderQ();
    cv::eigen2cv(R, cvR_);
    double *vData1 = (double *)v1_.data;
    double *vData2 = (double *)v2_.data;
    double *vData3 = (double *)v3_.data;
    double *vData4 = (double *)v4_.data;
    memset(vData1, 0, sizeof(double) * 12);
    memset(vData2, 0, sizeof(double) * 12);
    memset(vData3, 0, sizeof(double) * 12);
    memset(vData4, 0, sizeof(double) * 12);
    double *RData = (double *)cvR_.data;
    vData1[11] = 1;
    for (int i = 7; i >= 0; --i) {
        for (int j = 11; j > i; --j) {
            vData1[i] += -RData[i * 12 + j] * vData1[j];
        }
        vData1[i] /= RData[i * 12 + i];
    }

    vData2[10] = 1;
    for (int i = 7; i >= 0; --i) {
        for (int j = 11; j > i; --j) {
            vData2[i] += -RData[i * 12 + j] * vData2[j];
        }
        vData2[i] /= RData[i * 12 + i];
    }

    vData3[9] = 1;
    for (int i = 7; i >= 0; --i) {
        for (int j = 11; j > i; --j) {
            vData3[i] += -RData[i * 12 + j] * vData3[j];
        }
        vData3[i] /= RData[i * 12 + i];
    }

    vData4[8] = 1;
    for (int i = 7; i >= 0; --i) {
        for (int j = 11; j > i; --j) {
            vData4[i] += -RData[i * 12 + j] * vData4[j];
        }
        vData4[i] /= RData[i * 12 + i];
    }
    Schmidt12(v1_, v2_, v3_, v4_);
    memcpy(MVt_.data + 8 * 12 * sizeof(double), v4_.data, sizeof(double) * 12);
    memcpy(MVt_.data + 9 * 12 * sizeof(double), v3_.data, sizeof(double) * 12);
    memcpy(MVt_.data + 10 * 12 * sizeof(double), v2_.data, sizeof(double) * 12);
    memcpy(MVt_.data + 11 * 12 * sizeof(double), v1_.data, sizeof(double) * 12);
    double fus = 1e6 / cv::getTickFrequency();
}

//inner product
double EPNPSolver::Dot(const double *v1, const double *v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

//distance between p1 and p2
double EPNPSolver::Dist2(const double *p1, const double *p2) {
    return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

/*****************************************************************************************************************************************************
 *  compute L6x10 and rho:
 *
 *
 *
 *
 ******************************************************************************************************************************************************/
void EPNPSolver::ComputeL6x10(const double *ut, double *l_6x10) {
    const double *v[4];
    v[0] = ut + 12 * 11;
    v[1] = ut + 12 * 10;
    v[2] = ut + 12 * 9;
    v[3] = ut + 12 * 8;

    double dv[4][6][3] = {};

    for (int i = 0; i < 4; i++) {
        int a = 0, b = 1;
        for (int j = 0; j < 6; j++) {
            dv[i][j][0] = v[i][3 * a] - v[i][3 * b];
            dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
            dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

            b++;
            if (b > 3) {
                a++;
                b = a + 1;
            }
        }
    }
    for (int i = 0; i < 6; i++) {
        double *row = l_6x10 + 10 * i;

        row[0] = Dot(dv[0][i], dv[0][i]);
        row[1] = 2.0f * Dot(dv[0][i], dv[1][i]);
        row[2] = Dot(dv[1][i], dv[1][i]);
        row[3] = 2.0f * Dot(dv[0][i], dv[2][i]);
        row[4] = 2.0f * Dot(dv[1][i], dv[2][i]);
        row[5] = Dot(dv[2][i], dv[2][i]);
        row[6] = 2.0f * Dot(dv[0][i], dv[3][i]);
        row[7] = 2.0f * Dot(dv[1][i], dv[3][i]);
        row[8] = 2.0f * Dot(dv[2][i], dv[3][i]);
        row[9] = Dot(dv[3][i], dv[3][i]);
    }
}

void EPNPSolver::ComputeRho(double *rho) {
    rho[0] = Dist2(cws_[0], cws_[1]);
    rho[1] = Dist2(cws_[0], cws_[2]);
    rho[2] = Dist2(cws_[0], cws_[3]);
    rho[3] = Dist2(cws_[1], cws_[2]);
    rho[4] = Dist2(cws_[1], cws_[3]);
    rho[5] = Dist2(cws_[2], cws_[3]);
}

/*****************************************************************************************************************************************************
 *  compute beta:
 *               beta = argmin( sum( (ccs_i - ccs_j)^2 - (cws_i-cws_j)^2 ) )
 *               ccs_i = sum(beta_i*v) ,   ccs_j = sum(beta_j*v)
 *               when v is the eigenvectors corresponding to the smallest four eigenvalues
 *
 ******************************************************************************************************************************************************/

void EPNPSolver::findBetas(const double *ut, const cv::Mat &Rho, double *betas) {
    double DistCC[6] = {0.0};
    const double *v = &ut[0] + 12 * 11;
    DistCC[0] = Dist2(&v[0], &v[3]);
    DistCC[1] = Dist2(&v[0], &v[6]);
    DistCC[2] = Dist2(&v[0], &v[9]);
    DistCC[3] = Dist2(&v[3], &v[6]);
    DistCC[4] = Dist2(&v[3], &v[9]);
    DistCC[5] = Dist2(&v[6], &v[9]);

    for (int i = 0; i < 4; ++i) {
        betas[i] = 0.0;
    }
    double sum1 = 0.0f, sum2 = 1e-8;
    const double *rhoData = Rho.ptr<double>();
    for (int i = 0; i < 6; ++i, rhoData++) {
        sum1 += sqrt(DistCC[i] * rhoData[0]);
        sum2 += DistCC[i];
    }
    betas[0] = sum1 / sum2;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]
// solve four betas, so only to solve four unkown number
void EPNPSolver::findBetasApprox1(const cv::Mat &L_6x10, const cv::Mat &Rho, double *betas) {
    double *b4 = B4_.ptr<double>();

    for (int i = 0; i < 6; i++) {
        double *l6x4 = L_6x4_.ptr<double>(i);
        double *l6x10 = L_6x10_.ptr<double>(i);
        l6x4[0] = l6x10[0];
        l6x4[1] = l6x10[1];
        l6x4[2] = l6x10[3];
        l6x4[3] = l6x10[6];
    }
    cv::solve(L_6x4_, Rho, B4_, cv::DECOMP_QR);

    if (b4[0] < 0) {
        betas[0] = sqrt(-b4[0]);
        betas[1] = -b4[1] / betas[0];
        betas[2] = -b4[2] / betas[0];
        betas[3] = -b4[3] / betas[0];
    } else {
        betas[0] = sqrt(b4[0]);
        betas[1] = b4[1] / betas[0];
        betas[2] = b4[2] / betas[0];
        betas[3] = b4[3] / betas[0];
    }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]
// solve two betas, so only to solve three unkown number
void EPNPSolver::findBetasApprox2(const cv::Mat &L_6x10, const cv::Mat &Rho, double *betas) {
    double *b3 = B3_.ptr<double>();

    for (int i = 0; i < 6; i++) {
        double *l6x3 = L_6x3_.ptr<double>(i);
        double *l6x10 = L_6x10_.ptr<double>(i);
        for (int j = 0; j < 3; ++j) {
            l6x3[j] = l6x10[j];
        }
    }

    cv::solve(L_6x3_, Rho, B3_, cv::DECOMP_QR);

    if (b3[0] < 0) {
        betas[0] = sqrt(-b3[0]);
        betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
    } else {
        betas[0] = sqrt(b3[0]);
        betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
    }

    if (b3[1] < 0)
        betas[0] = -betas[0];

    betas[2] = 0.0;
    betas[3] = 0.0;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]
// solve three betas, so only to solve five unkown number
void EPNPSolver::findBetasApprox3(const cv::Mat &L_6x10, const cv::Mat &Rho, double *betas) {
    double *b5 = B5_.ptr<double>();

    for (int i = 0; i < 6; i++) {
        double *l6x5 = L_6x5_.ptr<double>(i);
        double *l6x10 = L_6x10_.ptr<double>(i);
        for (int j = 0; j < 5; ++j) {
            l6x5[j] = l6x10[j];
        }
    }
    cv::solve(L_6x5_, Rho, B5_, cv::DECOMP_QR);
    if (b5[0] < 0) {
        betas[0] = sqrt(-b5[0]);
        betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
    } else {
        betas[0] = sqrt(b5[0]);
        betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
    }
    if (b5[1] < 0)
        betas[0] = -betas[0];
    betas[2] = b5[3] / betas[0];
    betas[3] = 0.0;
}

// gauss newton iteration for solve Ax = b
void EPNPSolver::Compute_A_and_b_GaussNewton(const double *l_6x10, const double *rho, const double betas[4], cv::Mat &A, cv::Mat &b) {
    for (int i = 0; i < 6; i++) {
        const double *rowL = l_6x10 + i * 10;
        double *rowA = A.ptr<double>(i);
        double *bData = (double *)b.ptr<double>(i);

        rowA[0] = 2 * rowL[0] * betas[0] + rowL[1] * betas[1] + rowL[3] * betas[2] + rowL[6] * betas[3];
        rowA[1] = rowL[1] * betas[0] + 2 * rowL[2] * betas[1] + rowL[4] * betas[2] + rowL[7] * betas[3];
        rowA[2] = rowL[3] * betas[0] + rowL[4] * betas[1] + 2 * rowL[5] * betas[2] + rowL[8] * betas[3];
        rowA[3] = rowL[6] * betas[0] + rowL[7] * betas[1] + rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

        bData[0] =
            rho[i]
            - (rowL[0] * betas[0] * betas[0] + rowL[1] * betas[0] * betas[1] + rowL[2] * betas[1] * betas[1] + rowL[3] * betas[0] * betas[2]
               + rowL[4] * betas[1] * betas[2] + rowL[5] * betas[2] * betas[2] + rowL[6] * betas[0] * betas[3]
               + rowL[7] * betas[1] * betas[3] + rowL[8] * betas[2] * betas[3] + rowL[9] * betas[3] * betas[3]);
    }
}

void EPNPSolver::QRSolve(cv::Mat &A, cv::Mat &b, cv::Mat &X) {
    const int nr = A.rows;
    const int nc = A.cols;
    if (nc <= 0 || nr <= 0)
        return;

    if (maxNR_ != 0 && maxNR_ < nr) {
        delete[] A1_;
        delete[] A2_;
    }
    if (maxNR_ < nr) {
        maxNR_ = nr;
        if (!A1_) {
            delete[] A1_;
        }
        if (!A2_) {
            delete[] A2_;
        }
        A1_ = new double[nr];
        A2_ = new double[nr];
    }

    double *pA = A.ptr<double>(), *ppAkk = pA;
    for (int k = 0; k < nc; k++) {
        double *ppAik1 = ppAkk, eta = fabs(*ppAik1);
        for (int i = k + 1; i < nr; i++) {
            double elt = fabs(*ppAik1);
            if (eta < elt)
                eta = elt;
            ppAik1 += nc;
        }
        if (eta == 0) {
            A1_[k] = A2_[k] = 0.0;
            //cerr << "God damnit, A is singular, this shouldn't happen." << endl;
            return;
        } else {
            double *ppAik2 = ppAkk, sum2 = 0.0, inv_eta = 1. / eta;
            for (int i = k; i < nr; i++) {
                *ppAik2 *= inv_eta;
                sum2 += *ppAik2 * *ppAik2;
                ppAik2 += nc;
            }
            double sigma = sqrt(sum2);
            if (*ppAkk < 0)
                sigma = -sigma;
            *ppAkk += sigma;
            A1_[k] = sigma * *ppAkk;
            A2_[k] = -eta * sigma;
            for (int j = k + 1; j < nc; j++) {
                double *ppAik = ppAkk, sum = 0;
                for (int i = k; i < nr; i++) {
                    sum += *ppAik * ppAik[j - k];
                    ppAik += nc;
                }
                double tau = sum / A1_[k];
                ppAik = ppAkk;
                for (int i = k; i < nr; i++) {
                    ppAik[j - k] -= tau * *ppAik;
                    ppAik += nc;
                }
            }
        }
        ppAkk += nc + 1;
    }

    // b <- Qt b
    double *ppAjj = pA, *pb = b.ptr<double>();
    for (int j = 0; j < nc; j++) {
        double *ppAij = ppAjj, tau = 0;
        for (int i = j; i < nr; i++) {
            tau += *ppAij * pb[i];
            ppAij += nc;
        }
        tau /= A1_[j];
        ppAij = ppAjj;
        for (int i = j; i < nr; i++) {
            pb[i] -= tau * *ppAij;
            ppAij += nc;
        }
        ppAjj += nc + 1;
    }

    // X = R-1 b
    double *pX = X.ptr<double>();
    pX[nc - 1] = pb[nc - 1] / A2_[nc - 1];
    for (int i = nc - 2; i >= 0; i--) {
        double *ppAij = pA + i * nc + (i + 1), sum = 0;

        for (int j = i + 1; j < nc; j++) {
            sum += *ppAij * pX[j];
            ppAij++;
        }
        pX[i] = (pb[i] - sum) / A2_[i];
    }
}

void EPNPSolver::GaussNewton(const cv::Mat &L_6x10, const cv::Mat &Rho, double betas[4]) {
    const int iterations_number = 5;

    double *x = X_.ptr<double>();

    for (int k = 0; k < iterations_number; k++) {
        Compute_A_and_b_GaussNewton(L_6x10.ptr<double>(), Rho.ptr<double>(), betas, A_, B_);
        QRSolve(A_, B_, X_);
        for (int i = 0; i < 4; i++)
            betas[i] += x[i];
    }
}

/*****************************************************************************************************************************************************
 *  compute R,t:3D-3D points to solve R,t
 *               pcs_i = R*pws_i+t,  when i = 1,2,....,numPoints_
 *
 *
 *
 ******************************************************************************************************************************************************/
//compute ccs by ut and beta: ccs = sum(beta_i*ut_i)
void EPNPSolver::ComputeCCS(const double *betas, const double *ut) {
    //for (int i = 0; i < 4; i++)
    //    ccs_[i][0] = ccs_[i][1] = ccs_[i][2] = 0.0f;
    memset(&ccs_[0][0], 0, sizeof(double) * 12);
    for (int i = 0; i < 4; i++) {
        const double *v = ut + 12 * (11 - i);
        /*for (int j = 0; j < 4; j++)
            for (int k = 0; k < 3; k++)
                ccs_[j][k] += betas[i] * v[3 * j + k];*/
        ccs_[0][0] += betas[i] * v[0];
        ccs_[0][1] += betas[i] * v[1];
        ccs_[0][2] += betas[i] * v[2];

        ccs_[1][0] += betas[i] * v[3];
        ccs_[1][1] += betas[i] * v[4];
        ccs_[1][2] += betas[i] * v[5];

        ccs_[2][0] += betas[i] * v[6];
        ccs_[2][1] += betas[i] * v[7];
        ccs_[2][2] += betas[i] * v[8];

        ccs_[3][0] += betas[i] * v[9];
        ccs_[3][1] += betas[i] * v[10];
        ccs_[3][2] += betas[i] * v[11];
    }    
}

//compute pcs bu ccs and alpha
void EPNPSolver::ComputePCS() {
    for (int i = 0; i < numPoint_; i++) {
        double *a = &alphas_[0] + 4 * i;
        double *pc = &pcs_[0] + 3 * i;

        for (int j = 0; j < 3; j++)
            pc[j] = a[0] * ccs_[0][j] + a[1] * ccs_[1][j] + a[2] * ccs_[2][j] + a[3] * ccs_[3][j];
    }
}

//ICP estimate R,t by pairs of 3D points in world coordinate system and camera coordinate system
void EPNPSolver::EstimateRT(double R[3][3], double t[3]) {
    double pc0[3] = {};

    pc0[0] = pc0[1] = pc0[2] = 0.0;
    //compute avg_pc and avg_pw
    for (int i = 0; i < numPoint_; i++) {
        const double *pc = &pcs_[3 * i];

        for (int j = 0; j < 3; j++) {
            pc0[j] += pc[j];
        }
    }
    for (int j = 0; j < 3; j++) {
        pc0[j] /= numPoint_;
    }

    double *abt = ABt_.ptr<double>();

    //ABt = (pwi-avg_pw)*(pci - avg_pc)'
    // ABt_.setTo(0.0);
    memset(ABt_.data, 0, sizeof(double) * ABt_.rows * ABt_.cols);
    for (int i = 0; i < numPoint_; i++) {
        double *pc = &pcs_[3 * i];
        double *pw = Pw_.ptr<double>(i);

        for (int j = 0; j < 3; j++) {
            abt[3 * j] += (pc[j] - pc0[j]) * pw[0];
            abt[3 * j + 1] += (pc[j] - pc0[j]) * pw[1];
            abt[3 * j + 2] += (pc[j] - pc0[j]) * pw[2];
        }
    }
    svd_.compute(ABt_, ABt_D_, ABt_U_, ABt_V_, cv::SVD::MODIFY_A);
    //Vt = V
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            double temp = ABt_V_.at<double>(i, j);
            ABt_V_.at<double>(i, j) = ABt_V_.at<double>(j, i);
            ABt_V_.at<double>(j, i) = temp;
        }
    }
    //R = U*Vt ======> R(i,j) = U.row(i).dot(Vt.col(j)) = U.row(i).dot(V.row(j))
    double *abt_u = ABt_U_.ptr<double>(), *abt_v = ABt_V_.ptr<double>();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R[i][j] = Dot(abt_u + 3 * i, abt_v + 3 * j);

    const double det = R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] - R[0][2] * R[1][1] * R[2][0]
                       - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

    //if det_R<0 ====> R.row(2) = -R.row(2)
    if (det < 0) {
        R[2][0] = -R[2][0];
        R[2][1] = -R[2][1];
        R[2][2] = -R[2][2];
    }

    //t = avg_pc - R*avg_pw
    t[0] = pc0[0] - Dot(R[0], cws_[0]);
    t[1] = pc0[1] - Dot(R[1], cws_[0]);
    t[2] = pc0[2] - Dot(R[2], cws_[0]);
}

void EPNPSolver::SolveForSign() {
    if (pcs_[2] < 0.0) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                ccs_[i][j] = -ccs_[i][j];

        for (int i = 0; i < numPoint_; i++) {
            pcs_[3 * i] = -pcs_[3 * i];
            pcs_[3 * i + 1] = -pcs_[3 * i + 1];
            pcs_[3 * i + 2] = -pcs_[3 * i + 2];
        }
    }
}

double EPNPSolver::ReprojectionError(const double R[3][3], const double t[3]) {
    double sum2 = 0.0;

    for (int i = 0; i < numPoint_; i++) {
        double *pw = &pws_[3 * i];
        double *spt = &spherePts_[3 * i];
        double Xc = Dot(R[0], pw) + t[0];
        double Yc = Dot(R[1], pw) + t[1];
        double Zc = Dot(R[2], pw) + t[2];
        double inv_len = 1.0 / std::sqrt(Xc * Xc + Yc * Yc + Zc * Zc);
        sum2 += std::sqrt((spt[0] - Xc * inv_len) * (spt[0] - Xc * inv_len) + (spt[1] - Yc * inv_len) * (spt[1] - Yc * inv_len)
                          + (spt[2] - Zc * inv_len) * (spt[2] - Zc * inv_len));
    }

    return sum2 / numPoint_;
}

double EPNPSolver::ComputeRT(const double *ut, const double *betas, double R[3][3], double t[3]) {
    ComputeCCS(betas, ut);
    ComputePCS();   
    SolveForSign();
    EstimateRT(R, t);
    return ReprojectionError(R, t);
}

int Inliers2(const std::vector<cv::Point3f> &pts3d, const std::vector<cv::Point3f> &spherePts, double threshold, cv::Mat &R, cv::Mat &t) {
    double r11 = R.at<double>(0, 0);
    double r12 = R.at<double>(0, 1);
    double r13 = R.at<double>(0, 2);
    double r21 = R.at<double>(1, 0);
    double r22 = R.at<double>(1, 1);
    double r23 = R.at<double>(1, 2);
    double r31 = R.at<double>(2, 0);
    double r32 = R.at<double>(2, 1);
    double r33 = R.at<double>(2, 2);
    double tx = t.at<double>(0, 0);
    double ty = t.at<double>(1, 0);
    double tz = t.at<double>(2, 0);
    cv::Point3f projPt;
    int inliersCnt = 0;
    for (int i = 0; i < pts3d.size(); ++i) {
        double X = pts3d[i].x;
        double Y = pts3d[i].y;
        double Z = pts3d[i].z;
        projPt.x = float(r11 * X + r12 * Y + r13 * Z + tx);
        projPt.y = float(r21 * X + r22 * Y + r23 * Z + ty);
        projPt.z = float(r31 * X + r32 * Y + r33 * Z + tz);
        double len = std::max(cv::norm(projPt), 1e-20);
        projPt /= len;
        double dis = cv::norm(projPt - spherePts[i]);
        if (dis <= threshold) {
            ++inliersCnt;
        }
    }
    return inliersCnt;
}

bool EPNPSolver::EPNPRansac(const std::vector<cv::Point3f> &pts3d,
                            const std::vector<cv::Point3f> &spherePts,
                            double threshold,
                            cv::Mat &R,
                            cv::Mat &t,
                            int minIterCnt) {
    int iterCnt = 0;
    int N = 1000000;
    std::vector<cv::Point3d> objPts(4);
    std::vector<cv::Point3d> obsPts(4);
    cv::Mat bestR = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat bestT = cv::Mat::eye(3, 1, CV_64F);
    bestR.copyTo(R);
    bestT.copyTo(t);
    if (pts3d.size() < 4) {
        return false;
    }
    int maxInliresCnt = 0;
    int maxIterCnt = 200;
    while (iterCnt < minIterCnt || (iterCnt < maxIterCnt && N-- >= 0)) {
        ++iterCnt;
        int id1 = rand() % pts3d.size();
        int id2 = rand() % pts3d.size();
        int id3 = rand() % pts3d.size();
        int id4 = rand() % pts3d.size();
        if (id1 == id2 || id1 == id3 || id2 == id3 || id4 == id1 || id4 == id2 || id4 == id3) {
            --iterCnt;
            continue;
        }
        objPts[0] = cv::Point3d(pts3d[id1].x, pts3d[id1].y, pts3d[id1].z);
        objPts[1] = cv::Point3d(pts3d[id2].x, pts3d[id2].y, pts3d[id2].z);
        objPts[2] = cv::Point3d(pts3d[id3].x, pts3d[id3].y, pts3d[id3].z);
        objPts[3] = cv::Point3d(pts3d[id4].x, pts3d[id4].y, pts3d[id4].z);
        obsPts[0] = cv::Point3d(spherePts[id1].x, spherePts[id1].y, spherePts[id1].z);
        obsPts[1] = cv::Point3d(spherePts[id2].x, spherePts[id2].y, spherePts[id2].z);
        obsPts[2] = cv::Point3d(spherePts[id3].x, spherePts[id3].y, spherePts[id3].z);
        obsPts[3] = cv::Point3d(spherePts[id4].x, spherePts[id4].y, spherePts[id4].z);
        Run(objPts, obsPts, R, t);
        int cnt = Inliers2(pts3d, spherePts, threshold, R, t);
        if (cnt > maxInliresCnt) {
            maxInliresCnt = cnt;
            R.copyTo(bestR);
            t.copyTo(bestT);
        }
        double p = double(maxInliresCnt) / pts3d.size();
        double tmp = 1 - pow(p, 4);
        tmp = std::max(std::min(0.9999, tmp), 0.0001);
        int n = -4 / log10(tmp);
        if (iterCnt > n + minIterCnt) {
            break;
        }
    }
    bestR.copyTo(R);
    bestT.copyTo(t);
    ransacInliersCount_ = maxInliresCnt;
    if (maxInliresCnt < pts3d.size() * 0.25 || maxInliresCnt < 16) {
        return false;
    }
    return true;
}

int EPNPSolver::GetRansacInlierCount() {
    return ransacInliersCount_;
}

}  // namespace pnp
}  // namespace hybrid_msckf
