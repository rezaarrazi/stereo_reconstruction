#include "sparse_matcher.h"


bool SparseMatcher::CompareMatches(const cv::DMatch& match0, const cv::DMatch& match1)
{
    return match0.distance < match1.distance;
}


SparseMatcher::SparseMatcher()
{
    feature_matcher_ = cv::DescriptorMatcher::create("BruteForce");
}


void SparseMatcher::SetMatchNum(std::size_t match_num)
{
    match_num_ = match_num;
}


void SparseMatcher::MatchSparsely(const std::array<std::vector<cv::KeyPoint>, 2>& keypoints, const std::array<cv::Mat, 2>& features)
{

    feature_matcher_->match(features[0], features[1], matches_);
    std::sort(matches_.begin(), matches_.end(), CompareMatches);

    cv::KeyPoint keypoint;
    cv::Point2f point;

    for (std::size_t i = 0; i < match_num_; i++)
    {
        keypoint = keypoints[0][matches_[i].queryIdx];
        matched_points0_.push_back(cv::Point2f(keypoint.pt.x, keypoint.pt.y));

        keypoint = keypoints[1][matches_[i].trainIdx];
        matched_points1_.push_back(cv::Point2f(keypoint.pt.x, keypoint.pt.y));

        selected_matches_.push_back(matches_[i]);
    }

}


void SparseMatcher::DisplayMatchings(const cv::Mat& image0, const cv::Mat& image1, const std::array<std::vector<cv::KeyPoint>, 2>& keypoints)
{
    cv::Mat matched_image;
    cv::drawMatches(image0, keypoints[0], image1, keypoints[1], selected_matches_, matched_image);
    cv::imshow("matched image", matched_image);
    cv::waitKey(0);
}








