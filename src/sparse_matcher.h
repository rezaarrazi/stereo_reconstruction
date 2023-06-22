#ifndef _SPARSE_MATCHER
#define _SPARSE_MATCHER


#include <vector>
#include <array>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>


class SparseMatcher
{

    private:
        cv::Ptr<cv::DescriptorMatcher> bf_matcher_;
        cv::Ptr<cv::DescriptorMatcher> flann_based_matcher_;
        std::vector<cv::DMatch> matches_;
        std::array<std::vector<cv::Point2f>, 2> matched_points_;
        std::vector<cv::DMatch> selected_matches_;

        static bool CompareMatches(const cv::DMatch& match0, const cv::DMatch& match1);

    public:
        SparseMatcher();

        std::array<std::vector<cv::Point2f>, 2> GetMatchedPoints() const;

        void MatchSparselyBFSortTop(const std::array<std::vector<cv::KeyPoint>, 2>& keypoints, const std::array<cv::Mat, 2>& features,
                                    std::size_t keypoint_number);

        void MatchSparselyBFMinDistance(const std::array<std::vector<cv::KeyPoint>, 2>& keypoints, const std::array<cv::Mat, 2>& features,
                                        float distance_ratio);
        
        void MatchSparselyFLANNBased(const std::array<std::vector<cv::KeyPoint>, 2>& keypoints, const std::array<cv::Mat, 2>& features,
                                     float ratio);

        void DisplayMatchings(const std::array<cv::Mat, 2>& images, const std::array<std::vector<cv::KeyPoint>, 2>& keypoints);

};


#endif






