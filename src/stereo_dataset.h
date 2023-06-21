#ifndef _STEREO_DATASET
#define _STEREO_DATASET


#include <dirent.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <cstring>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/highgui/highgui.hpp>


class StereoDataset
{

    private:
        const std::string DATA_PATH = "../Data/Middlebury";
        std::vector<std::string> folder_names_;
        std::size_t image_pair_number_ = 0;

        std::array<cv::Mat, 2> images_;
        std::array<cv::Mat, 2> disparity_maps_;

        std::array<cv::Mat, 2> camera_intrinsics_ = {cv::Mat_<float>(3, 3, 0.0), cv::Mat_<float>(3, 3, 0.0)};
        float doffs_ = 0.0;
        float baseline_ = 0.0;
        std::size_t image_width_ = 0;
        std::size_t image_height_ = 0;
        std::size_t disparity_num_ = 0;
        std::size_t min_disparity_ = 0;
        std::size_t max_disparity_ = 0;

        void ReadIntrinsics(const std::string& line, std::size_t camera_index);

    public:
        StereoDataset();

        std::size_t GetImagePairNumber() const;

        std::array<cv::Mat, 2> GetImages() const;

        std::array<cv::Mat, 2> GetDisparityMaps() const;

        std::array<cv::Mat, 2> GetCameraIntrinsics() const;

        std::size_t GetImageWidth() const;
        std::size_t GetImageHeight() const;
        cv::Size GetImageSize() const;

        std::size_t GetMinDisparity() const;
        std::size_t GetMaxDisparity() const;

        void SetImages(std::size_t image_id);

        void SetDisparityMaps(std::size_t image_id);

        void SetCalibrations(std::size_t image_id);

};


#endif








