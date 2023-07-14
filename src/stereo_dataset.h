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
#include "PFMReadWrite.h"


class StereoDataset
{

    private:
        std::string data_path_ = "../Data/Middlebury";
        std::vector<std::string> folder_names_;
        std::size_t image_pair_number_ = 0;

        std::array<cv::Mat, 2> images_;
        std::array<cv::Mat, 2> disparity_maps_;

        std::array<cv::Mat, 2> camera_intrinsics_ = {cv::Mat_<double>(3, 3, 0.0), cv::Mat_<double>(3, 3, 0.0)};

        double doffs_ = 0.0;
        double baseline_ = 0.0;
        std::size_t image_width_ = 0;
        std::size_t image_height_ = 0;
        std::size_t disparity_num_ = 0;
        std::size_t min_disparity_ = 0;
        std::size_t max_disparity_ = 0;

        void ReadIntrinsics(const std::string& line, std::size_t camera_index);

    public:
        StereoDataset(const std::string& data_path = "../Data/Middlebury");

        std::size_t GetImagePairNumber() const;

        std::array<cv::Mat, 2> GetImages() const;

        cv::Size GetImageSize() const;

        std::array<cv::Mat, 2> GetDisparityMaps() const;

        std::array<cv::Mat, 2> GetCameraIntrinsics() const;

        double GetDoffs() const;

        double GetBaseline() const;

        std::size_t GetImageWidth() const;

        std::size_t GetImageHeight() const;

        std::size_t GetMinDisparity() const;

        std::size_t GetMaxDisparity() const;

        void SetImages(std::size_t image_id);

        void SetDisparityMaps(std::size_t image_id);

        void SetCalibrations(std::size_t image_id);

};


#endif








