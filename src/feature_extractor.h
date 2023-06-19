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
        cv::Mat image0_;
        cv::Mat image1_;
        std::array<std::vector<cv::KeyPoint>, 2> keypoints_;
        std::array<cv::Mat, 2> features_;

    public:
        FeatureExtractor() {}

        FeatureExtractor(const cv::Mat& image0, const cv::Mat& image1): image0_(image0), image1_(image1) {}

        const std::array<std::vector<cv::KeyPoint>, 2>& GetKeypoints() const;

        const std::array<cv::Mat, 2>& GetFeatures() const;

        void SetImagePair(const cv::Mat& image0, const cv::Mat& image1);

        void ExtractFeaturesORB();

        void ExtractFeaturesSIFT();

        void ExtractFeaturesSURF();

        void ExtractFeaturesBRISK();

};


#endif








