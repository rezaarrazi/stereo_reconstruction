#include "feature_extractor.h"


const std::array<std::vector<cv::KeyPoint>, 2>& FeatureExtractor::GetKeypoints() const
{
    return keypoints_;
}


const std::array<cv::Mat, 2>& FeatureExtractor::GetFeatures() const
{
    return features_;
}


void FeatureExtractor::SetImages(const std::array<cv::Mat, 2>& images)
{
    images_ = images;
}


void FeatureExtractor::ExtractFeaturesORB()
{
    cv::Ptr<cv::ORB> orb_extractor = cv::ORB::create();
    orb_extractor->detectAndCompute(images_[0], cv::noArray(), keypoints_[0], features_[0]);
    orb_extractor->detectAndCompute(images_[1], cv::noArray(), keypoints_[1], features_[1]);
    features_[0].convertTo(features_[0], CV_32F);
    features_[1].convertTo(features_[1], CV_32F);
}








