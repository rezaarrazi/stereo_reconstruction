#include "feature_extractor.h"


const std::array<std::vector<cv::KeyPoint>, 2>& FeatureExtractor::GetKeypoints() const
{
    return keypoints_;
}


const std::array<cv::Mat, 2>& FeatureExtractor::GetFeatures() const
{
    return features_;
}


std::size_t FeatureExtractor::GetAverageKeypointNumber() const
{
    std::size_t keypoint_number0 = keypoints_[0].size();
    std::size_t keypoint_number1 = keypoints_[1].size();
    return static_cast<std::size_t>((keypoint_number0 + keypoint_number1) / 2.0);
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

void FeatureExtractor::ExtractFeaturesSIFT()
{
    cv::Ptr<cv::SIFT> sift_extractor = cv::SIFT::create();
    sift_extractor->detectAndCompute(images_[0], cv::noArray(), keypoints_[0], features_[0]);
    sift_extractor->detectAndCompute(images_[1], cv::noArray(), keypoints_[1], features_[1]);
    features_[0].convertTo(features_[0], CV_32F);
    features_[1].convertTo(features_[1], CV_32F);
}

void FeatureExtractor::ExtractFeaturesSURF()
{
    cv::Ptr<cv::SURF> surf_extractor = cv::SURF::create();
    surf_extractor->detectAndCompute(images_[0], cv::noArray(), keypoints_[0], features_[0]);
    surf_extractor->detectAndCompute(images_[1], cv::noArray(), keypoints_[1], features_[1]);
    features_[0].convertTo(features_[0], CV_32F);
    features_[1].convertTo(features_[1], CV_32F);
}






