#ifndef _FEATURE_EXTRACTOR
#define _FEATURE_EXTRACTOR


#include <iostream>
#include <vector>
#include <array>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>


class FeatureExtractor
{

    private:
        std::array<cv::Mat, 2> images_;
        std::array<std::vector<cv::KeyPoint>, 2> keypoints_;
        std::array<cv::Mat, 2> features_;

    public:
        FeatureExtractor() {}

        const std::array<std::vector<cv::KeyPoint>, 2>& GetKeypoints() const;

        const std::array<cv::Mat, 2>& GetFeatures() const;

        void SetImages(const std::array<cv::Mat, 2>& images);

        void ExtractFeaturesORB();

        void ExtractFeaturesSIFT();

        void ExtractFeaturesSURF();

        void ExtractFeaturesBRISK();

};


#endif








