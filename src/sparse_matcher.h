#ifndef _SPARSE_MATCHER
#define _SPARSE_MATCHER


#include <vector>
#include <array>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>


class SparseMatcher
{

    private:
        std::size_t match_num_ = 300;
        cv::Ptr<cv::DescriptorMatcher> feature_matcher_;
        std::vector<cv::DMatch> matches_;
        std::array<std::vector<cv::Point2f>, 2> matched_points_;
        std::vector<cv::DMatch> selected_matches_;
        static bool CompareMatches(const cv::DMatch& match0, const cv::DMatch& match1);

    public:
        SparseMatcher();

        std::array<std::vector<cv::Point2f>, 2> GetMatchedPoints() const;

        void SetMatchNum(std::size_t match_num);

        void MatchSparsely(const std::array<std::vector<cv::KeyPoint>, 2>& keypoints, const std::array<cv::Mat, 2>& features);

        void DisplayMatchings(const std::array<cv::Mat, 2>& images, const std::array<std::vector<cv::KeyPoint>, 2>& keypoints);

};


#endif






