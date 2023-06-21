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
        std::array<cv::Mat, 2> rectified_images_;
        cv::Mat disparity_map_;

    public:
        DenseMatcher() {}

        cv::Mat GetDisparityMap() const;

        void ComputeDisparityMap(const StereoDataset& stereo_dataset, const cv::Mat& rotation, const cv::Mat& translation);

};


#endif