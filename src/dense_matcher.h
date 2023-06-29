#ifndef _DENSE_MATCHER
#define _DENSE_MATCHER


#include <array>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
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

        std::array<cv::Mat, 2> rectified_images_;
        cv::Mat disparity_map_;
        cv::Mat colorful_disparity_map_;

        void FillHoles(const StereoDataset& stereo_dataset, std::size_t window_size, std::size_t type);

    public:
        DenseMatcher() {}

        cv::Mat GetDisparityMap() const;

        cv::Mat GetColorfulDisparityMap() const;

        void RectifyImages(const StereoDataset& stereo_dataset, const cv::Mat& rotation, const cv::Mat& translation);

        void ComputeDisparityMap(const StereoDataset& stereo_dataset, const cv::Mat& rotation, const cv::Mat& translation, std::size_t type);

};


#endif



