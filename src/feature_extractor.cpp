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


void FeatureExtractor::ExtractFeatures(std::size_t type)
{

    if (type >= 0 && type < 4)
    {
        if (type == 0)
        {
            cv::Ptr<cv::ORB> orb_extractor = cv::ORB::create();
            orb_extractor->detectAndCompute(images_[0], cv::noArray(), keypoints_[0], features_[0]);
            orb_extractor->detectAndCompute(images_[1], cv::noArray(), keypoints_[1], features_[1]);
        }
        else if (type == 1)
        {
            cv::Ptr<cv::SIFT> sift_extractor = cv::SIFT::create();
            sift_extractor->detectAndCompute(images_[0], cv::noArray(), keypoints_[0], features_[0]);
            sift_extractor->detectAndCompute(images_[1], cv::noArray(), keypoints_[1], features_[1]);
        }
        else if (type == 2)
        {
            cv::Ptr<cv::xfeatures2d::SURF> surf_extractor = cv::xfeatures2d::SURF::create();
            surf_extractor->detectAndCompute(images_[0], cv::noArray(), keypoints_[0], features_[0]);
            surf_extractor->detectAndCompute(images_[1], cv::noArray(), keypoints_[1], features_[1]);
        }
        else
        {
            cv::Ptr<cv::BRISK> brisk_extractor = cv::BRISK::create();
            brisk_extractor->detectAndCompute(images_[0], cv::noArray(), keypoints_[0], features_[0]);
            brisk_extractor->detectAndCompute(images_[1], cv::noArray(), keypoints_[1], features_[1]);
        }

        features_[0].convertTo(features_[0], CV_32F);
        features_[1].convertTo(features_[1], CV_32F);
    }
    else
        std::cout << "type should be >= 0 and < 4.\n";

}




