//
//  FeatureGrid.hpp
//  DearVins
//
#pragma once

#include<vector>
#include<math.h>
#include<algorithm>
#include<Eigen/Core>
#include <opencv2/opencv.hpp>
#include "Vio/FeatureObservation.h"

namespace hybrid_msckf {

class FeatureGrid {
public:
    FeatureGrid(
            const int imgWidth,
            const int imgHeight,
            const int cellSize) :
        cellSize_(cellSize),
        nGridCols_(static_cast<int>(ceil(static_cast<double>(imgWidth)/cellSize_))),
        nGridRows_(static_cast<int>(ceil(static_cast<double>(imgHeight)/cellSize_))),
        gridOccupancy_(nGridCols_ * nGridRows_, false)
    {
        if(cellSize_ < 1) {
            std::cerr << "Warning: cellSize less than 1" << std::endl;
            exit(1);
        }
    }


    void ResetGrid(bool applyMask = true) {
        if(applyMask && !gridOccupancyMask_.empty()) {
            gridOccupancy_.assign(gridOccupancyMask_.begin(), gridOccupancyMask_.end());
        }
        else {
            std::fill(gridOccupancy_.begin(), gridOccupancy_.end(), false);
        }
    }


    void SetMask(cv::Mat &mask) {
        if(gridOccupancyMask_.empty()) {
            gridOccupancyMask_.resize(nGridCols_ * nGridRows_, false);
        }
        for(int i = 0; i < nGridRows_; i++) {
            int idx = i * nGridCols_;
            for(int j = 0; j < nGridCols_; j++) {
                int row = std::min(i * cellSize_ + cellSize_ / 2, mask.rows);
                int col = std::min(j * cellSize_ + cellSize_ / 2, mask.cols);
                if(mask.at<uchar>(row, col) == 0 ) {
                    gridOccupancyMask_.at(idx + j) = true;
                }
            }
        }
    }


    bool GetOccupancyState(const cv::Point2f& pt) {
        return gridOccupancy_[GetCellIndex(pt.x, pt.y)];
    }


    int GetGridNum() {
        return static_cast<int>(gridOccupancy_.size());
    }


    void SetExistingFeatures(const std::vector<FeatureObservation>& features, bool occupyAdjacent = false) {
        std::for_each(features.begin(), features.end(), [&](const FeatureObservation& i){
            SetGridOccpuancy(i.pt0, occupyAdjacent);
        });
    }


    void SetGridOccpuancy(const Eigen::Vector2d& px, bool occupyAdjacent = false) {
        int index = GetCellIndex(px[0], px[1]);
        gridOccupancy_[index] = true;
        if(occupyAdjacent) {
            SetAdjacentOccupy(index);
        }
    }


    void SetGridOccpuancy(const cv::Point2f& pt, bool occupyAdjacent = false) {
        int index = GetCellIndex(pt.x, pt.y);
        gridOccupancy_[index] = true;
        if(occupyAdjacent) {
            SetAdjacentOccupy(index);
        }
    }


    template<class T>
    inline int GetCellIndex(T x, T y) {
        return static_cast<int>(y)/cellSize_*nGridCols_ + static_cast<int>(x)/cellSize_;
    }

    // debug show
    void DrawMask(int width, int height) {
        cv::Mat mask = cv::Mat_<uchar>(height, width);
        mask.setTo(0);
        for(int i = 0; i < nGridRows_; i++) {
            int idx = i * nGridCols_;
            for(int j = 0; j < nGridCols_; j++) {
                if(gridOccupancy_.at(idx + j) == true) {
                    int rowMin = i * cellSize_;
                    int rowMax = std::min((i + 1) * cellSize_, height);
                    int colMin = j * cellSize_;
                    int colMax = std::min((j + 1) * cellSize_, width);
                    mask.colRange(colMin, colMax).rowRange(rowMin, rowMax).setTo(255);
                }
            }
        }
        cv::imshow("gridMask", mask);
        cv::waitKey(1);
    }

private:
    void SetAdjacentOccupy(int index) {
        int id1 = index + nGridCols_;
        int id2 = index - nGridCols_;
        std::vector<int> ids = {index+1, index-1,
                                id1, id1+1, id1-1,
                                id2, id2+1, id2-1};
        for(auto id : ids) {
            if(id >= 0 && id < static_cast<int>(gridOccupancy_.size())) {
                gridOccupancy_[id] = true;
            }
        }
    }

    const int cellSize_;
    const int nGridCols_;
    const int nGridRows_;
    std::vector<bool> gridOccupancy_;
    std::vector<bool> gridOccupancyMask_;
};

}//namespace hybrid_msckf {