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
        int bm_disparity_number_ = 256;
        int bm_block_size_ = 21;
        int sgbm_min_disparity_ = 0;
        int sgbm_disparity_number_ = 256;
        int sgbm_block_size_ = 5;
        int sgbm_disp12_max_diff_ = -1;
        int sgbm_speckle_window_size_ = 100;
        int sgbm_speckle_range_ = 32;
        int pre_filter_cap_ = 63;
        int uniqueness_ratio_ = 5;
        std::array<cv::Mat, 2> rectified_images_;
        cv::Mat disparity_map_;
        cv::Mat colorful_disparity_map_;
        void FillHolesMaximum(const StereoDataset& stereo_dataset, std::size_t window_size);
        void FillHolesAverage(const StereoDataset& stereo_dataset, std::size_t window_size);

    public:
        DenseMatcher() {}

        cv::Mat GetDisparityMap() const;

        cv::Mat GetColorfulDisparityMap() const;

        void RectifyImages(const StereoDataset& stereo_dataset, const cv::Mat& rotation, const cv::Mat& translation);

        void ComputeDisparityMap(const StereoDataset& stereo_dataset, const cv::Mat& rotation, const cv::Mat& translation, std::size_t type);

};


#endif



