#include "sparse_matcher.h"


bool SparseMatcher::CompareMatches(const cv::DMatch& match0, const cv::DMatch& match1)
{
    return match0.distance < match1.distance;
}


SparseMatcher::SparseMatcher()
{
    bf_matcher_ = cv::DescriptorMatcher::create("BruteForce");
    flann_based_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
}


std::array<std::vector<cv::Point2f>, 2> SparseMatcher::GetMatchedPoints() const
{
    return matched_points_;
}


void SparseMatcher::MatchSparselyBFSortTop(const std::array<std::vector<cv::KeyPoint>, 2>& keypoints,
                                           const std::array<cv::Mat, 2>& features, std::size_t keypoint_number)
{

    bf_matcher_->match(features[0], features[1], matches_);
    std::sort(matches_.begin(), matches_.end(), CompareMatches);

    cv::KeyPoint keypoint;
    cv::Point2f point;

    for (std::size_t i = 0; i < keypoint_number; i++)
    {
        keypoint = keypoints[0][matches_[i].queryIdx];
        matched_points_[0].push_back(cv::Point2f(keypoint.pt.x, keypoint.pt.y));

        keypoint = keypoints[1][matches_[i].trainIdx];
        matched_points_[1].push_back(cv::Point2f(keypoint.pt.x, keypoint.pt.y));

        selected_matches_.push_back(matches_[i]);
    }

}


void SparseMatcher::MatchSparselyBFMinDistance(const std::array<std::vector<cv::KeyPoint>, 2>& keypoints,
                                               const std::array<cv::Mat, 2>& features, double distance_ratio)
{

    bf_matcher_->match(features[0], features[1], matches_);
    std::sort(matches_.begin(), matches_.end(), CompareMatches);

    cv::KeyPoint keypoint;
    cv::Point2f point;

    for (std::size_t i = 0; i < matches_.size(); i++)
    {

        if (matches_[i].distance >= distance_ratio * matches_[0].distance)
            continue;

        keypoint = keypoints[0][matches_[i].queryIdx];
        matched_points_[0].push_back(cv::Point2f(keypoint.pt.x, keypoint.pt.y));

        keypoint = keypoints[1][matches_[i].trainIdx];
        matched_points_[1].push_back(cv::Point2f(keypoint.pt.x, keypoint.pt.y));

        selected_matches_.push_back(matches_[i]);
    }

}


void SparseMatcher::MatchSparselyFLANNBased(const std::array<std::vector<cv::KeyPoint>, 2>& keypoints,
                                            const std::array<cv::Mat, 2>& features, double ratio)
{

    std::vector<std::vector<cv::DMatch>> flann_based_matches;
    flann_based_matcher_->knnMatch(features[0], features[1], flann_based_matches, 2);

    cv::KeyPoint keypoint;
    cv::Point2f point;

    for (std::size_t i = 0; i < flann_based_matches.size(); i++)
    {
        if (flann_based_matches[i][0].distance >= flann_based_matches[i][1].distance * ratio)
            continue;

        keypoint = keypoints[0][flann_based_matches[i][0].queryIdx];
        matched_points_[0].push_back(cv::Point2f(keypoint.pt.x, keypoint.pt.y));

        keypoint = keypoints[1][flann_based_matches[i][0].trainIdx];
        matched_points_[1].push_back(cv::Point2f(keypoint.pt.x, keypoint.pt.y));

        selected_matches_.push_back(flann_based_matches[i][0]);
        
    }

}


void SparseMatcher::DisplayMatchings(const std::array<cv::Mat, 2>& images, const std::array<std::vector<cv::KeyPoint>, 2>& keypoints,
                                     bool save_matched_images)
{

    cv::Mat matched_images;
    cv::drawMatches(images[0], keypoints[0], images[1], keypoints[1], selected_matches_, matched_images);
    cv::imshow("matched images", matched_images);

    if (save_matched_images == true)
        cv::imwrite("../matched_images1.png", matched_images);

    cv::waitKey(0);

}








