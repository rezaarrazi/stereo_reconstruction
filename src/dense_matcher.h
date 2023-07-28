#ifndef _DENSE_MATCHER
#define _DENSE_MATCHER


#include <array>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/imgproc/types_c.h>
#include "stereo_dataset.h"


class DenseMatcher
{

    private:
        int BM_DISPARITY_NUMBER_ = 256;
        int BM_BLOCK_SIZE_ = 21;

        int SGBM_MIN_DISPARITY_ = 0;
        int SGBM_DISPARITY_NUMBER_ = 256;
        int SGBM_BLOCK_SIZE_ = 5;
        int SGBM_DISP12_MAX_DIFF_ = -1;
        int SGBM_SPECKLE_WINDOW_SIZE_ = 100;
        int SGBM_SPECKLE_RANGE_ = 32;
        int PRE_FILTER_CAP_ = 63;
        int UNIQUENESS_RATIO_ = 5;

        std::array<cv::Mat, 2> images_;
        cv::Size image_size_;
        std::array<cv::Mat, 2> camera_intrinsics_;
        cv::Mat rotation_;
        cv::Mat translation_;
        std::array<cv::Mat, 2> rectified_images_;
        cv::Mat disparity_map_;
        cv::Mat colorful_disparity_map_;
        cv::Mat projection_matrix_;        

    public:
        DenseMatcher() {}

        void LoadData(const StereoDataset& stereo_dataset, const cv::Mat& rotation, const cv::Mat& translation);

        void LoadDataDirectly(const StereoDataset& stereo_dataset);

        cv::Mat GetDisparityMap() const;

        cv::Mat GetColorfulDisparityMap() const;

        cv::Mat GetProjectionMatrix() const;

        void RectifyImages();

        void ComputeDisparityMap(std::size_t type);

        void ComputeDisparityMapDirectly(std::size_t type);

        void ComputeDisparityMapWithoutConversion(std::size_t type);

        void FilterMedian();

};


#endif


