//
// Created by d on 19-2-19.
//

#include "Viewer/PangoViewer.h"

using namespace std;

namespace hybrid_msckf { namespace viewer {

shared_ptr<PangoViewer> PangoViewer::instance_ = nullptr;

PangoViewer::PangoViewer(int width, int height) : width_(width), height_(height)
{
    backgroundColor_ << 0.2f, 0.2f, 0.2f, 0.1f;
    Eigen::Vector3f origin(0, 0, 0);
    SetAxisHelper(origin, 1.0f);
    SetGridHelper(origin);
    SetTrajectoryColor(Eigen::Vector3f(1.0f, 1.0f, 0.0f));
}


PangoViewer::~PangoViewer()
{
    delete viewerThread_;
}


void PangoViewer::Start()
{
    viewerThread_ = new std::thread(&PangoViewer::RenderProcess, this);
}


void PangoViewer::Stop()
{
    shouldStop_ = true;
    if (viewerThread_->joinable()) {
        viewerThread_->join();
    }
}


void PangoViewer::RenderProcess() {
    float viewPointX = 0, viewPointY = -4, viewPointZ = 2;

    pangolin::CreateWindowAndBind("3D", width_, height_);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glEnable(GL_PROGRAM_POINT_SIZE);

    pangolin::OpenGlRenderState s_cam(
            // camera projection matrix (w, h, fx, fy, cx, cy, zNear, zFar)
            pangolin::ProjectionMatrix(width_, height_, 600, 600, 500, 320, 0.1, 10000),
            // viewing direction: looking from (viewPoint.X, .Y, .Z) to (0,0,0)
            pangolin::ModelViewLookAt(viewPointX, viewPointY, viewPointZ, 0, 0, 0, 0, 0, 1)
    );

    const int uiWidth = 120;
    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(uiWidth), 1.0, -1000.f/640.f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    // menu
    const int menuWith = uiWidth;
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(menuWith));
    pangolin::Var<bool> show_axis("menu.Axis", true, true);
    pangolin::Var<bool> show_grid("menu.Grid", true, true);
    pangolin::Var<bool> show_camPoses("menu.CamPoses", true, true);
    pangolin::Var<bool> show_odomPoses("menu.Odometry", true, true);
    pangolin::Var<bool> show_trajectory("menu.Trajectory", true, true);
    pangolin::Var<bool> show_bboxes("menu.BBox3d", true, true);
    pangolin::Var<bool> show_landmarks("menu.Landmarks", true, true);
    pangolin::Var<bool> show_pointcloud("menu.PointCloud", true, true);
    pangolin::Var<bool> show_voxels("menu.Voxels", true, true);
    pangolin::Var<bool> show_plan("menu.PathPlan", true, true);
    pangolin::Var<bool> show_goal("menu.Goal", true, true);
    pangolin::Var<bool> capture_screen("menu.ScreenCapture", false, true);
    pangolin::Var<bool> terminate("menu.Terminate", false, true);

    // loop
    while (!pangolin::ShouldQuit()) {

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);

        glClearColor(backgroundColor_(0), backgroundColor_(1),
                          backgroundColor_(2), backgroundColor_(3));

        if (show_axis) {
            DrawAxisHelper();
        }

        if (show_grid) {
            DrawGridHelper();
        }

        if (show_camPoses) {
            DrawCamPoses();
        }

        if (show_odomPoses) {
            DrawOdomPoses();
        }

        if (show_trajectory) {
            DrawTrajectory();
        }

        if (show_bboxes) {
            DrawBBoxInfos();
        }

        if (show_landmarks) {
            DrawLandmakrs();
        }

        if (show_pointcloud) {
            DrawPointCloud();
        }

        if (show_voxels) {
            DrawVoxels();
        }

        if (show_plan) {
            DrawPlannedPath();
        }

        if (show_goal) {
            DrawGoals();
        }

        pangolin::FinishFrame();

        if (capture_screen) {
            pangolin::PixelFormat fmt = pangolin::PixelFormatFromString("RGBA32");
            auto &v = pangolin::DisplayBase().v;
            if (screenImageBuffer_.w != v.w || screenImageBuffer_.h != v.h) {
                screenImageBuffer_.Dealloc();
                screenImageBuffer_.Alloc(v.w, v.h, v.w * fmt.bpp / 8);
            }
            glReadBuffer(GL_BACK);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(v.l, v.b, v.w, v.h, GL_RGBA, GL_UNSIGNED_BYTE, screenImageBuffer_.ptr);

            cv::Mat imgBuffer = cv::Mat(height_, width_, CV_8UC4, screenImageBuffer_.ptr);
            screenImageMutex_.lock();
            cv::cvtColor(imgBuffer, screenImage_,  cv::COLOR_RGBA2BGR);
            screenImageMutex_.unlock();
        }

        if (terminate || shouldStop_) {
            shouldStop_ = true;
            break;
        }

        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

}

/* Background color */
void PangoViewer::SetBackground(const Eigen::Vector4f &color) {
    backgroundColor_ = color;
}

/* Axis helper */
void PangoViewer::SetAxisHelper(const Eigen::Vector3f &origin, float length) {
    axisOrigin_ = origin;
    axisLength_ = length;
}


/* Grid helper */
void PangoViewer::SetGridHelper(const Eigen::Vector3f &origin, float grid_size, int grid_num) {
    gridOrigin_ = origin;
    gridSize_ = grid_size;
    gridNum_ = grid_num;
}


/* camera poses */
void PangoViewer::AddCamPose(const Eigen::Matrix4d &Twc, const Eigen::Vector3f &color) {
    camPoseMutex_.lock();
    camPoses_.push_back(Twc);
    camPoseColors_.push_back(color);
    camPoseMutex_.unlock();
}


void PangoViewer::SetCamPoses(const std::vector<Eigen::Matrix4d> &poses, const std::vector<Eigen::Vector3f> &colors) {
    camPoseMutex_.lock();
    camPoses_ = poses;
    camPoseColors_ = colors;
    camPoseMutex_.unlock();
}


void PangoViewer::ClearCamPoses() {
    camPoseMutex_.lock();
    camPoses_.clear();
    camPoseColors_.clear();
    camPoseMutex_.unlock();
}


 void PangoViewer::SetOdomPose(const Eigen::Matrix4d &Twb, const Eigen::Vector3f &color) {
    odomPoseMutex_.lock();
    odomPoses_.push_back(Twb);
    if (odomPoses_.size() > 1) {
        odomPoses_.pop_front();
    }
    odomPoseColor_ = color;
    odomPoseMutex_.unlock();
}


void PangoViewer::ClearOdomPoses() {
    odomPoseMutex_.lock();
    odomPoses_.clear();
    odomPoseMutex_.unlock();
}


/* Trajectory */
void PangoViewer::AddPosition(const Eigen::Vector3f &pos) {
    trajectoryMutex_.lock();
    trajectory_.push_back(pos);
    trajectoryMutex_.unlock();
}


void PangoViewer::SetTrajectory(const std::vector<Eigen::Vector3f> &trajectory, const Eigen::Vector3f &color) {
    trajectoryMutex_.lock();
    trajectory_ = trajectory;
    trajectoryColor_ = color;
    trajectoryMutex_.unlock();
}


void PangoViewer::SetTrajectoryColor(const Eigen::Vector3f &color) {
    trajectoryMutex_.lock();
    trajectoryColor_ = color;
    trajectoryMutex_.unlock();
}


void PangoViewer::ClearTrajectory() {
    trajectoryMutex_.lock();
    trajectory_.clear();
    trajectoryMutex_.unlock();
}


/* Planned path */
void PangoViewer::SetPlannedPath(const std::vector<Eigen::Vector3f> &path, const Eigen::Vector3f &color) {
    plannedPathMutex_.lock();
    plannedPath_ = path;
    plannedPathColor_ = color;
    plannedPathMutex_.unlock();
}

void PangoViewer::ClearPlannedPath() {
    plannedPathMutex_.lock();
    plannedPath_.clear();
    plannedPathMutex_.unlock();
}


/* Goal */
void PangoViewer::SetGoal(const Eigen::Vector3f &pos, const Eigen::Vector3f &color) {
    goalMutex_.lock();
    goalPositions_.push_back(pos);
    if (goalPositions_.size() > 100) {
        goalPositions_.pop_front();
    }
    goalColor_ = color;
    goalMutex_.unlock();
}


void PangoViewer::ClearGoals() {
    goalMutex_.lock();
    goalPositions_.clear();
    goalMutex_.unlock();
}

/* 3D object infos */
void PangoViewer::AddBBoxInfo(const BBoxInfo &bbox) {
    bboxInfoMutex_.lock();
    bboxInfos_[bbox.id_] = bbox;
    bboxInfoMutex_.unlock();
}


void PangoViewer::RemoveBBoxInfo(int id) {
    bboxInfoMutex_.lock();
    if (bboxInfos_.find(id) != bboxInfos_.end()) {
        bboxInfos_.erase(id);
    }
    bboxInfoMutex_.unlock();
}


void PangoViewer::ClearBBoxInfos()  {
    bboxInfoMutex_.lock();
    bboxInfos_.clear();
    bboxInfoMutex_.unlock();
}


/* Landmarks */
void PangoViewer::SetLandmarks(const std::vector<Eigen::Vector3f> &landmarks, const std::vector<Eigen::Vector3f> &colors, float point_size) {
    landmarkMutex_.lock();
    landmarks_ = landmarks;
    landmarkColors_ = colors;
    landmarkSize_ = point_size;
    landmarkMutex_.unlock();
}


void PangoViewer::ClearLandmarks() {
    landmarkMutex_.lock();
    landmarks_.clear();
    landmarkColors_.clear();
    landmarkMutex_.unlock();
}


/* Point cloud */
void PangoViewer::SetPointCloud(std::vector<float> &points, std::vector<float> &colors, float point_size, bool move) {
    std::unique_lock<mutex> lock(pointCloudMutex_);
    if (move) {
        pointCloud_ = std::move(points);
        pointCloudColors_ = std::move(colors);
    } else {
        pointCloud_ = points;
        pointCloudColors_ = colors;
    }
    pointSize_ = point_size;
    pointCloudUpdated_ = true;
}


void PangoViewer::ClearPointCloud() {
    std::unique_lock<mutex> lock(pointCloudMutex_);
    pointCloud_.clear();
    pointCloudColors_.clear();
    pointCloudUpdated_ = true;
}


void PangoViewer::SetVoxels(std::vector<float> &points, std::vector<float> &colors, float voxel_size, bool move) {
    std::unique_lock<mutex> lock(voxelMutex_);
    if (move) {
        voxels_ = std::move(points);
        voxelColors_ = std::move(colors);
    } else {
        voxels_ = points;
        voxelColors_ = colors;
    }
    voxelSize_ = voxel_size;
    voxelUpdated_ = true;
}


void PangoViewer::ClearVoxels() {
    std::unique_lock<mutex> lock(voxelMutex_);
    voxels_.clear();
    voxelColors_.clear();
    voxelUpdated_ = true;
}


cv::Mat PangoViewer::GetScreenShot() {
    cv::Mat img;
    unique_lock<mutex> lock(screenImageMutex_);
    cv::flip(screenImage_, img, 0);
    return img;
}


/* ----------------- Drawing functions -------------------------------*/
void PangoViewer::DrawAxisHelper() {
    Eigen::Vector3f origin = axisOrigin_;
    float len = axisLength_;
    float x0 = origin(0), y0 = origin(1), z0 = origin(2);

    glLineWidth(2);
    glBegin(GL_LINES);

    glColor3f(1.f, 0.f, 0.f);
    glVertex3f(x0, y0, z0);
    glVertex3f(x0 + len, y0, z0);

    glColor3f(0.f, 1.f, 0.f);
    glVertex3f(x0, y0, z0);
    glVertex3f(x0, y0 + len, z0);

    glColor3f(0.f, 0.f, 1.f);
    glVertex3f(x0, y0, z0);
    glVertex3f(x0, y0, z0 + len);

    glEnd();
}


void PangoViewer::DrawGridHelper() {
    Eigen::Vector3f origin = gridOrigin_;
    float grid_size = gridSize_;
    int N = gridNum_;

    const float len = grid_size * static_cast<float>(N);
    const float xmin = origin(0) - len;
    const float xmax = origin(0) + len;
    const float ymin = origin(1) - len;
    const float ymax = origin(1) + len;

    glLineWidth(1);
    glColor3f(0.4f, 0.4f, 0.4f);
    glBegin(GL_LINES);

    for (int i = 0; i < 2 * N + 1; i++)  {
        float li = grid_size * static_cast<float>(i);
        glVertex3f(xmin+li, ymin, 0);
        glVertex3f(xmin+li, ymax, 0);
        glVertex3f(xmin, ymin+li, 0);
        glVertex3f(xmax, ymin+li, 0);
    }

    glEnd();
}


void PangoViewer::DrawCamPoses() {
    const float w = 0.1f;
    const float h = w * 0.75f;
    const float z = w * 0.6f;
    float xs[] = {w, -w, -w, w};
    float ys[] = {h, h, -h, -h};

    // get data
    camPoseMutex_.lock();
    vector<Eigen::Matrix4d> poses = camPoses_;
    vector<Eigen::Vector3f> colors = camPoseColors_;
    camPoseMutex_.unlock();

    // draw
    for (size_t n = 0; n < poses.size(); ++n) {
        const Eigen::Matrix4d &Twc = poses.at(n);
        const Eigen::Vector3f &color = colors.at(n);

        glPushMatrix();
        glMultMatrixd(Twc.data());

        glLineWidth(1);
        glColor3f(color(0), color(1), color(2));
        glBegin(GL_LINES);

        for (int i=0; i<4; i++) {
            glVertex3f(0, 0, 0);
            glVertex3f(xs[i], ys[i], z);
            int j = (i+1) % 4;
            glVertex3f(xs[i], ys[i], z);
            glVertex3f(xs[j], ys[j], z);
        }

        glEnd();

        glPopMatrix();
    }
}


void PangoViewer::DrawOdomPoses() {
    unique_lock<mutex> lock(odomPoseMutex_);
    if (odomPoses_.empty()) {
        return;
    }

    const Eigen::Matrix4d pose = odomPoses_.back();
    float len = 1;
    float x0 = (float)pose(0, 3), y0 = (float)pose(1, 3), z0 = (float)pose(2, 3);

    glLineWidth(3);
    glBegin(GL_LINES);

    glColor3f(1.f, 0.f, 0.f);
    glVertex3f(x0, y0, z0);
    glVertex3f(x0 + (float)pose(0, 0), y0 + (float)pose(1, 0), z0 + (float)pose(2, 0));

    glColor3f(0.f, 1.f, 0.f);
    glVertex3f(x0, y0, z0);
    glVertex3f(x0 + (float)pose(0, 1), y0 + (float)pose(1, 1), z0 + (float)pose(2, 1));

    glColor3f(0.f, 0.f, 1.f);
    glVertex3f(x0, y0, z0);
    glVertex3f(x0 + (float)pose(0, 2), y0 + (float)pose(1, 2), z0 + (float)pose(2, 2));

    glEnd();

}


void PangoViewer::DrawTrajectory() {
    // get data
    trajectoryMutex_.lock();
    vector<Eigen::Vector3f> trajectory = trajectory_;
    Eigen::Vector3f color = trajectoryColor_;
    trajectoryMutex_.unlock();

    // draw
    glLineWidth(2);
    glColor3f(color(0), color(1), color(2));
    glBegin(GL_LINES);

    for (size_t i = 1; i < trajectory.size(); ++i) {
        const Eigen::Vector3f &a = trajectory.at(i - 1);
        const Eigen::Vector3f &b = trajectory.at(i);
        glVertex3f(a(0), a(1), a(2));
        glVertex3f(b(0), b(1), b(2));
    }

    glEnd();
}


void PangoViewer::DrawPlannedPath() {
    // get data
    plannedPathMutex_.lock();
    vector<Eigen::Vector3f> path = plannedPath_;
    Eigen::Vector3f color = plannedPathColor_;
    plannedPathMutex_.unlock();

    // draw
    glLineWidth(2);
    glColor3f(color(0), color(1), color(2));
    glBegin(GL_LINES);

    for (size_t i = 1; i < path.size(); ++i) {
        const Eigen::Vector3f &a = path.at(i - 1);
        const Eigen::Vector3f &b = path.at(i);
        glVertex3f(a(0), a(1), a(2));
        glVertex3f(b(0), b(1), b(2));
    }

    glEnd();
}


void PangoViewer::DrawGoals() {
    unique_lock<mutex> lock(goalMutex_);
    if (goalPositions_.empty()) {
        return;
    }
    const Eigen::Vector3f p = goalPositions_.back();

    // Draw
    glPointSize(10);
    glBegin(GL_POINTS);

    glColor3f(goalColor_(0), goalColor_(1), goalColor_(2));
    glVertex3f(p(0), p(1), p(2));

    glEnd();
}


void PangoViewer::DrawBBoxInfos() {
    // get data
    bboxInfoMutex_.lock();
    std::map<int, BBoxInfo> bboxes = bboxInfos_;
    bboxInfoMutex_.unlock();

    // draw
    vector<Eigen::Vector3f> vertices;

    for (const auto &id_bbox : bboxes) {
        const auto &bbox = id_bbox.second;

        // vertices
        if (bbox.vertices_.empty()) {
            GetBboxVertices(bbox, vertices);
        }
        vertices = bbox.vertices_;

        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;

            const Eigen::Vector3f &bi = vertices[i];
            const Eigen::Vector3f &bj = vertices[j];
            const Eigen::Vector3f &ui = vertices[i+4];
            const Eigen::Vector3f &uj = vertices[j+4];

            glLineWidth(2);
            glBegin(GL_LINES);
            glColor3f(bbox.r_, bbox.g_, bbox.b_);

            glVertex3f(bi(0), bi(1), bi(2));
            glVertex3f(bj(0), bj(1), bj(2));

            glVertex3f(ui(0), ui(1), ui(2));
            glVertex3f(uj(0), uj(1), uj(2));

            glVertex3f(bi(0), bi(1), bi(2));
            glVertex3f(ui(0), ui(1), ui(2));

            glEnd();
        }
    }
}


void PangoViewer::DrawLandmakrs() {
    // get data
    landmarkMutex_.lock();
    vector<Eigen::Vector3f> landmarks = landmarks_;
    vector<Eigen::Vector3f> colors = landmarkColors_;
    float point_size = landmarkSize_;
    landmarkMutex_.unlock();

    // draw
    glPointSize(point_size);
    glBegin(GL_POINTS);

    for (size_t i = 0; i < landmarks.size(); ++i) {
        const Eigen::Vector3f &p = landmarks.at(i);
        const Eigen::Vector3f &c = colors.at(i);
        glColor3f(c(0), c(1), c(2));
        glVertex3f(p(0), p(1), p(2));
    }

    glEnd();
}

void PangoViewer::DrawPointCloud()
{
    glPointSize(pointSize_);

    // upload data to Gpu
    if (pointCloudUpdated_) {
        pointCloudMutex_.lock();
        size_t num_pts = std::min(pointCloud_.size() / 3, (size_t)MAX_POINTS_NUM);
        pointCloudMutex_.unlock();

        if (num_pts != pcVertexBuffer_.num_elements) {
            pcVertexBuffer_.Reinitialise(pangolin::GlArrayBuffer, static_cast<GLuint>(num_pts), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
            pcColorBuffer_.Reinitialise(pangolin::GlArrayBuffer, static_cast<GLuint>(num_pts), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        }

        pointCloudMutex_.lock();
        auto new_num_pts = std::min(pointCloud_.size() / 3, num_pts);
        pcVertexBuffer_.Upload(&(pointCloud_[0]), 3 * sizeof(float) * new_num_pts);
        pcColorBuffer_.Upload(&(pointCloudColors_[0]), 3 * sizeof(float) * new_num_pts);
        pointCloudUpdated_ = false;
        pointCloudMutex_.unlock();
    }

    if (pcVertexBuffer_.num_elements > 0) {
        pangolin::RenderVboCbo(pcVertexBuffer_, pcColorBuffer_);
    }
}


//TODO: improve voxel render
void PangoViewer::DrawVoxels() {
    // upload data to Gpu
    if (voxelUpdated_) {
        voxelMutex_.lock();
        size_t num_voxels = std::min(voxels_.size() / 3, (size_t)MAX_VOXELS_NUM);
        voxelMutex_.unlock();

        if (num_voxels * 24 != voxelVertexBuffer_.num_elements) {
            voxelVertexBuffer_.Reinitialise(pangolin::GlArrayBuffer, static_cast<GLuint>(num_voxels * 24), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
            voxelColorBuffer_.Reinitialise(pangolin::GlArrayBuffer, static_cast<GLuint>(num_voxels * 24), GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        }

        if (voxelQuads_.size() < num_voxels * 72) {
            voxelQuads_.resize(num_voxels * 72);
            voxelQuadColors_.resize(num_voxels * 72);
        }

        voxelMutex_.lock();
        auto new_num_voxels = std::min(voxels_.size() / 3, num_voxels);

        size_t k = 0, ck = 0;
        const float s = voxelSize_ / 2.f;
        for(size_t i = 0; i < new_num_voxels; ++i) {
            auto i3 = i * 3;
            float x = voxels_[i3];
            float y = voxels_[i3 + 1];
            float z = voxels_[i3 + 2];

            // front
            voxelQuads_[k++] = x - s; voxelQuads_[k++] = y - s; voxelQuads_[k++] =  z - s;
            voxelQuads_[k++] = x - s; voxelQuads_[k++] = y + s; voxelQuads_[k++] =  z - s;
            voxelQuads_[k++] = x - s; voxelQuads_[k++] = y + s; voxelQuads_[k++] = z + s;
            voxelQuads_[k++] = x - s; voxelQuads_[k++] = y - s; voxelQuads_[k++] =  z + s;
            // back
            voxelQuads_[k++] = x + s; voxelQuads_[k++] = y - s; voxelQuads_[k++] =  z - s;
            voxelQuads_[k++] = x + s; voxelQuads_[k++] = y + s; voxelQuads_[k++] = z - s;
            voxelQuads_[k++] = x + s; voxelQuads_[k++] = y + s; voxelQuads_[k++] = z + s;
            voxelQuads_[k++] = x + s; voxelQuads_[k++] = y - s; voxelQuads_[k++] = z + s;
            // left
            voxelQuads_[k++] = x - s; voxelQuads_[k++] = y - s; voxelQuads_[k++] = z - s;
            voxelQuads_[k++] = x + s; voxelQuads_[k++] = y - s; voxelQuads_[k++] = z - s;
            voxelQuads_[k++] = x + s; voxelQuads_[k++] = y - s; voxelQuads_[k++] = z + s;
            voxelQuads_[k++] = x - s; voxelQuads_[k++] = y - s; voxelQuads_[k++] =  z + s;
            // right
            voxelQuads_[k++] = x - s; voxelQuads_[k++] = y + s; voxelQuads_[k++] = z - s;
            voxelQuads_[k++] = x + s; voxelQuads_[k++] = y + s; voxelQuads_[k++] = z - s;
            voxelQuads_[k++] = x + s; voxelQuads_[k++] = y + s; voxelQuads_[k++] = z + s;
            voxelQuads_[k++] = x - s; voxelQuads_[k++] = y + s; voxelQuads_[k++] = z + s;
            // up
            voxelQuads_[k++] = x - s; voxelQuads_[k++] = y - s; voxelQuads_[k++] = z + s;
            voxelQuads_[k++] = x - s; voxelQuads_[k++] = y + s; voxelQuads_[k++] = z + s;
            voxelQuads_[k++] = x + s; voxelQuads_[k++] = y + s; voxelQuads_[k++] = z + s;
            voxelQuads_[k++] = x + s; voxelQuads_[k++] = y - s; voxelQuads_[k++] = z + s;
            // bottom
            voxelQuads_[k++] = x - s; voxelQuads_[k++] = y - s; voxelQuads_[k++] = z - s;
            voxelQuads_[k++] = x - s; voxelQuads_[k++] = y + s; voxelQuads_[k++] = z - s;
            voxelQuads_[k++] = x + s; voxelQuads_[k++] = y + s; voxelQuads_[k++] = z - s;
            voxelQuads_[k++] = x + s; voxelQuads_[k++] = y - s; voxelQuads_[k++] = z - s;

            float r = voxelColors_[i3];
            float g = voxelColors_[i3 + 1];
            float b = voxelColors_[i3 + 2];

            for (int j = 0; j < 24; j++) {
                voxelQuadColors_[ck++] = r;
                voxelQuadColors_[ck++] = g;
                voxelQuadColors_[ck++] = b;
            }
        }


        voxelVertexBuffer_.Upload(&(voxelQuads_[0]), 24 * 3 * sizeof(float) * new_num_voxels);
        voxelColorBuffer_.Upload(&(voxelQuadColors_[0]), 24 * 3 * sizeof(float) * new_num_voxels);
        voxelUpdated_ = false;
        voxelMutex_.unlock();
    }

    if (voxelVertexBuffer_.num_elements > 0) {
        pangolin::RenderVboCbo(voxelVertexBuffer_, voxelColorBuffer_, true, GL_QUADS);
    }
}

/* Utils functions */
void PangoViewer::GetBboxVertices(const BBoxInfo &bbox, std::vector<Eigen::Vector3f> &vertices) {
    if (vertices.size() != 8) {
        vertices.resize(8);
    }

    const Eigen::Vector3f pos(bbox.x_, bbox.y_, bbox.z_);
    const float half_l = bbox.l_ * 0.5f;
    const float half_w = bbox.w_ * 0.5f;
    const float h = bbox.h_ * 0.5f;

    vertices[0] = pos + Eigen::Vector3f(-half_l, -half_w, 0);
    vertices[1] = pos + Eigen::Vector3f(-half_l, half_w, 0);
    vertices[2] = pos + Eigen::Vector3f(half_l, half_w, 0);
    vertices[3] = pos + Eigen::Vector3f(half_l, -half_w, 0);
    for (int i = 4; i < 8; i++) {
        vertices[i] = vertices[i-4] + Eigen::Vector3f(0, 0, h);
    }
}

}}//namespace hybrid_msckf { namespace viewer {


