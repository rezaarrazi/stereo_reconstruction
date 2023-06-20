#ifndef _DISPARITY
#define _DISPARITY


#include <array>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include "stereo_dataset.h"

class Disparity
{

    private:
        StereoDataset dataset_;
        cv::Mat R_;
        cv::Mat t_;
        cv::Mat image_left_rec_;
        cv::Mat image_right_rec_;
        cv::Mat disparity_;

    public:
        Disparity(StereoDataset dataset, cv::Mat R, cv::Mat t) : dataset_(dataset), R_(R), t_(t) {}

        cv::Mat GetDisparity() const;

        void ComputeDisparity();

};


#endif